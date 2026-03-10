"""
Precompute and persist calibration and null caches to disk.

Usage
-----
    python scripts/precompute_caches.py [OPTIONS]

Options
-------
    --generators  comma-separated list  (default: nonparametric)
                  e.g. nonparametric,copula,empirical
    --n-cal       INT   calibration samples per scenario (default: config N_CAL)
    --null              also build and save the null cache
    --n-pre       INT   permutations per null entry (default: config PVALUE_PRECOMPUTED_N_PRE)
    --n-jobs      INT   parallel workers (default: 1)
    --output-dir  PATH  directory to write .pkl files (default: cache/)

Output files
------------
    {output_dir}/calibration_ncal{n_cal}.pkl
    {output_dir}/null_npre{n_pre}.pkl   (only when --null is given)

Memory note
-----------
At n_cal=500,000 each worker holds ~640 MB–1 GB of arrays.
With 4 workers: 2.5–4 GB RAM required.
Verify available RAM before using high n_cal with high n_jobs.

Numba note
----------
This script runs a quick Numba warmup at the start (one small power + CI run)
so the main process compiles JIT kernels before building caches.  Workers
spawn fresh processes and load the JIT cache from disk (~1–2 s per worker
once the cache exists).

Quick reference — using the caches in Python
---------------------------------------------
After precomputing, pass ``disk_cache_dir`` to the run functions so they
load caches from disk (pre_warm is then instant on a hit)::

    from power_simulation import run_all_scenarios

    results = run_all_scenarios(
        generator="nonparametric",
        n_cal=96400,
        n_sims=222050,
        n_jobs=4,
        disk_cache_dir="cache/",
    )

    from confidence_interval_calculator import run_all_ci_scenarios

    results = run_all_ci_scenarios(
        generator="nonparametric",
        n_cal=96400,
        disk_cache_dir="cache/",
    )

To persist caches built in-process when a file was missing or stale, set::

    save_cache_to_disk=True   # or config.SAVE_CACHE_TO_DISK = True
"""

import argparse
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Worker functions (must be top-level for pickling on Windows)
# ---------------------------------------------------------------------------

def _calibration_worker(generator, cases_subset, n_distinct_values,
                        dist_types, calibration_mode, n_cal, seed):
    """Build calibration cache for a subset of cases, return snapshots."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from data_generator import (warm_calibration_cache,
                                get_calibration_cache_snapshots)
    warm_calibration_cache(
        generator,
        cases=cases_subset,
        n_distinct_values=n_distinct_values,
        dist_types=dist_types,
        calibration_mode=calibration_mode,
        n_cal=n_cal,
        seed=seed,
    )
    return get_calibration_cache_snapshots()


def _null_worker(cases_subset, n_distinct_values, dist_types, n_pre, seed):
    """Build null cache for a subset of cases, return snapshot."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from permutation_pvalue import warm_precomputed_null_cache, get_null_cache_snapshot
    warm_precomputed_null_cache(
        cases=cases_subset,
        n_distinct_values=n_distinct_values,
        dist_types=dist_types,
        n_pre=n_pre,
        seed=seed,
    )
    return get_null_cache_snapshot()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _partition_cases(cases, n_jobs):
    """Split a dict of cases into n_jobs approximately equal chunks."""
    items = list(cases.items())
    size = math.ceil(len(items) / n_jobs)
    chunks = []
    for i in range(0, len(items), size):
        chunks.append(dict(items[i: i + size]))
    return chunks


def _merge_cal_snapshots(snapshots):
    """Merge a list of calibration snapshot dicts into one."""
    merged = {k: {} for k in ("mp", "mp_cop", "mp_emp", "mp_lin",
                               "sp", "sp_cop", "sp_emp", "sp_lin")}
    for snap in snapshots:
        for k in merged:
            merged[k].update(snap.get(k, {}))
    return merged


def _merge_null_snapshots(snapshots):
    """Merge a list of null snapshot dicts into one."""
    merged = {}
    for snap in snapshots:
        merged.update(snap)
    return merged


def _probe_output_dir_writable(output_dir):
    """Verify we can write to output_dir by creating and removing a probe file.

    Raises SystemExit with a clear message if the directory is not writable,
    so users see save problems before any heavy precomputation.
    """
    probe = Path(output_dir) / ".precompute_caches_write_probe"
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as e:
        print(
            f"Error: cannot write to output directory {output_dir.resolve()!s}.\n"
            f"  {e}\n"
            "Fix permissions or choose a different --output-dir before precomputing.",
            file=sys.stderr,
        )
        sys.exit(1)


def _fmt_size(path):
    size = os.path.getsize(path)
    if size >= 1_073_741_824:
        return f"{size / 1_073_741_824:.2f} GB"
    if size >= 1_048_576:
        return f"{size / 1_048_576:.1f} MB"
    return f"{size / 1024:.1f} KB"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute and persist calibration (and optionally null) caches."
    )
    parser.add_argument(
        "--generators",
        default="nonparametric",
        metavar="GEN[,GEN,...]",
        help="Comma-separated generators to calibrate (default: nonparametric). "
             "Choices: nonparametric, copula, empirical, linear.",
    )
    parser.add_argument(
        "--n-cal",
        type=int,
        default=None,
        metavar="N_CAL",
        help="Calibration samples per scenario (default: config N_CAL)",
    )
    parser.add_argument(
        "--null",
        action="store_true",
        help="Also build and save the precomputed null cache",
    )
    parser.add_argument(
        "--n-pre",
        type=int,
        default=None,
        metavar="N_PRE",
        help="Permutations per null entry (default: config PVALUE_PRECOMPUTED_N_PRE)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        metavar="N_JOBS",
        help="Parallel worker processes (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        default="cache/",
        metavar="OUTPUT_DIR",
        help="Directory to write .pkl files (default: cache/)",
    )
    args = parser.parse_args()

    # Parse --generators (comma-separated)
    valid_gens = ("nonparametric", "copula", "empirical", "linear")
    raw_gens = [g.strip().lower() for g in args.generators.split(",") if g.strip()]
    invalid = [g for g in raw_gens if g not in valid_gens]
    if invalid:
        parser.error(f"Unknown generator(s): {', '.join(invalid)}. "
                     f"Choose from: {', '.join(valid_gens)}.")
    generators = [g for g in raw_gens if g in valid_gens]
    if not generators:
        parser.error("--generators must be one or more of: "
                     "nonparametric, copula, empirical, linear.")

    # Add project root to path so imports work when called from any directory.
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from config import (CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES,
                        N_CAL, CALIBRATION_MODE, PVALUE_PRECOMPUTED_N_PRE)
    from data_generator import (save_calibration_caches_to_disk,
                                get_calibration_cache_snapshots,
                                _CALIBRATION_CACHE_MULTIPOINT,
                                _CALIBRATION_CACHE_MULTIPOINT_COPULA,
                                _CALIBRATION_CACHE_MULTIPOINT_EMP,
                                _CALIBRATION_CACHE_MULTIPOINT_LINEAR,
                                _CALIBRATION_CACHE,
                                _CALIBRATION_CACHE_COPULA,
                                _CALIBRATION_CACHE_EMP,
                                _CALIBRATION_CACHE_LINEAR)
    from permutation_pvalue import save_null_cache_to_disk, _NULL_CACHE
    from power_simulation import min_detectable_rho
    from confidence_interval_calculator import bootstrap_ci_averaged

    n_cal = args.n_cal if args.n_cal is not None else N_CAL
    n_pre = args.n_pre if args.n_pre is not None else PVALUE_PRECOMPUTED_N_PRE
    n_jobs = max(1, args.n_jobs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _probe_output_dir_writable(output_dir)

    # --- Numba JIT warmup (one small run so cache is populated before cache build) ---
    print("Warming up Numba (one small run)...")
    case = CASES[3]
    n_single = case["n"]
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
    rho_obs = case["observed_rho"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        min_detectable_rho(
            n_single, 4, "heavy_center", y_params,
            generator="nonparametric", n_sims=50, seed=0, direction="positive",
            calibration_mode="multipoint", n_cal=1,
        )
        bootstrap_ci_averaged(
            n_single, 4, "heavy_center", rho_obs, y_params,
            generator="nonparametric", n_reps=5, n_boot=10, seed=0,
            batch_bootstrap=True, calibration_mode="multipoint",
        )
    print("Warmup done.\n")

    print(f"Generators : {', '.join(generators)}")
    print(f"n_cal      : {n_cal:,}")
    print(f"n_jobs     : {n_jobs}")
    print(f"Output dir : {output_dir.resolve()}")
    if args.null:
        print(f"n_pre      : {n_pre:,}")
    print()

    # -----------------------------------------------------------------------
    # Calibration cache (one pass per generator; all merged into one file)
    # -----------------------------------------------------------------------
    cal_path = output_dir / f"calibration_ncal{n_cal}.pkl"
    print(f"Building calibration cache → {cal_path}")
    t0 = time.perf_counter()

    if n_jobs == 1:
        from data_generator import warm_calibration_cache
        for gen in generators:
            warm_calibration_cache(
                gen,
                calibration_mode=CALIBRATION_MODE,
                n_cal=n_cal,
            )
        save_calibration_caches_to_disk(cal_path, n_cal)
    else:
        for gen in generators:
            chunks = _partition_cases(CASES, n_jobs)
            futures_snapshots = []
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futures = {
                    pool.submit(
                        _calibration_worker,
                        gen,
                        chunk,
                        N_DISTINCT_VALUES,
                        DISTRIBUTION_TYPES,
                        CALIBRATION_MODE,
                        n_cal,
                        42 + i,
                    ): i
                    for i, chunk in enumerate(chunks)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    snap = fut.result()
                    futures_snapshots.append(snap)
                    print(f"  {gen} worker {idx} done")

            merged = _merge_cal_snapshots(futures_snapshots)
            _CALIBRATION_CACHE_MULTIPOINT.update(merged["mp"])
            _CALIBRATION_CACHE_MULTIPOINT_COPULA.update(merged["mp_cop"])
            _CALIBRATION_CACHE_MULTIPOINT_EMP.update(merged["mp_emp"])
            _CALIBRATION_CACHE_MULTIPOINT_LINEAR.update(merged["mp_lin"])
            _CALIBRATION_CACHE.update(merged["sp"])
            _CALIBRATION_CACHE_COPULA.update(merged["sp_cop"])
            _CALIBRATION_CACHE_EMP.update(merged["sp_emp"])
            _CALIBRATION_CACHE_LINEAR.update(merged["sp_lin"])
        save_calibration_caches_to_disk(cal_path, n_cal)

    elapsed_cal = time.perf_counter() - t0
    print(f"  Done in {elapsed_cal:.1f}s — {_fmt_size(cal_path)}\n")

    # -----------------------------------------------------------------------
    # Null cache (optional)
    # -----------------------------------------------------------------------
    if args.null:
        null_path = output_dir / f"null_npre{n_pre}.pkl"
        print(f"Building null cache → {null_path}")
        t1 = time.perf_counter()

        if n_jobs == 1:
            from permutation_pvalue import warm_precomputed_null_cache
            warm_precomputed_null_cache(n_pre=n_pre)
            save_null_cache_to_disk(null_path, n_pre)
        else:
            chunks = _partition_cases(CASES, n_jobs)
            null_snapshots = []
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futures = {
                    pool.submit(
                        _null_worker,
                        chunk,
                        N_DISTINCT_VALUES,
                        DISTRIBUTION_TYPES,
                        n_pre,
                        42 + i,
                    ): i
                    for i, chunk in enumerate(chunks)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    snap = fut.result()
                    null_snapshots.append(snap)
                    print(f"  Worker {idx} done")

            merged_null = _merge_null_snapshots(null_snapshots)
            _NULL_CACHE.update(merged_null)
            save_null_cache_to_disk(null_path, n_pre)

        elapsed_null = time.perf_counter() - t1
        print(f"  Done in {elapsed_null:.1f}s — {_fmt_size(null_path)}\n")


if __name__ == "__main__":
    main()
