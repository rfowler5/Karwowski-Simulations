"""
Benchmark realistic runtimes per generator (power + CI) at small params,
then scale to +/-0.01, +/-0.002, +/-0.001 accuracy tiers.

Run one at a time per benchmarking rule. Use --quick for single-scenario only,
--skip-parallel to skip parallel (n_jobs=-1) runs.

Flags:
  --power-only        Benchmark power only (skip CI timing and tables).
  --ci-only           Benchmark CI only (skip power timing and tables).
  --show-cache-costs  Print per-generator calibration and null cache costs with
                      tier scaling; useful for diagnosing slow generators.

Usage:
  python benchmarks/benchmark_realistic_runtimes.py [--quick] [--skip-parallel]
    [--generators ...] [--power-only] [--ci-only] [--show-cache-costs]
"""
import sys
import os
import time
import argparse
import warnings
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import (
    CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES,
    POWER_TIERS, CI_TIERS,
    N_PRE_BENCH, N_PRE_TIERS,
)
from power_simulation import min_detectable_rho, run_all_scenarios
from confidence_interval_calculator import bootstrap_ci_averaged, run_all_ci_scenarios
from data_generator import digitized_available, warm_calibration_cache
from permutation_pvalue import warm_precomputed_null_cache

# Benchmark params (small, for measurement)
N_SIMS_BENCH = 50
N_CAL_BENCH = 300
N_REPS_BENCH = 200
N_BOOT_BENCH = 500
SEED = 42
CASE_ID = 3
DIST_TYPE_SINGLE = "heavy_center"
N_DISTINCT_SINGLE = 4
EFFICIENCY = 0.5

_TIER_LABELS = ["+/-0.01", "+/-0.002", "+/-0.001"]


def _parse_args():
    p = argparse.ArgumentParser(description="Benchmark realistic runtimes and scale to precision tiers.")
    p.add_argument("--generators", default="all",
                   help="Comma-separated: all, nonparametric, copula, linear, empirical (default: all)")
    p.add_argument("--quick", action="store_true", help="Single-scenario only (skip full grid)")
    p.add_argument("--skip-parallel", action="store_true", help="Skip n_jobs=-1 benchmarks")
    p.add_argument("--power-only", action="store_true",
                   help="Benchmark power only (skip CI timing and tables)")
    p.add_argument("--ci-only", action="store_true",
                   help="Benchmark CI only (skip power timing and tables)")
    p.add_argument("--show-cache-costs", action="store_true",
                   help="Print per-generator calibration and null cache build costs with tier scaling")
    args = p.parse_args()
    if args.power_only and args.ci_only:
        p.error("--power-only and --ci-only are mutually exclusive")
    return args


def _resolve_generators(args):
    """Return (power_generators, ci_generators). CI excludes linear."""
    req = [x.strip().lower() for x in args.generators.split(",")]
    if "all" in req:
        power_list = ["nonparametric", "copula", "linear", "empirical"]
        if not digitized_available():
            power_list = [g for g in power_list if g != "empirical"]
    else:
        power_list = [g for g in req if g in ("nonparametric", "copula", "linear", "empirical")]
        if not digitized_available():
            power_list = [g for g in power_list if g != "empirical"]
    ci_list = [g for g in power_list if g != "linear"]
    return power_list, ci_list


def _machine_info():
    logical = os.cpu_count() or 1
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
    except Exception:
        physical = "?"
    return logical, physical


def _fmt_time_sec(t_sec):
    if t_sec < 60:
        return f"{t_sec:.1f}s"
    if t_sec < 3600:
        return f"{t_sec/60:.1f} min"
    return f"{t_sec/3600:.2f} hrs"


def run():
    args = _parse_args()
    power_gens, ci_gens = _resolve_generators(args)

    # Apply --power-only / --ci-only: zero out the skipped list so all loops no-op
    if args.power_only:
        ci_gens = []
    elif args.ci_only:
        power_gens = []

    if not power_gens and not ci_gens:
        print("No generators to run. Check --generators and digitized data.")
        return

    logical_cores, physical_cores = _machine_info()

    case = CASES[CASE_ID]
    n_single = case["n"]
    rho_obs = case["observed_rho"]
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}

    print("Realistic runtimes benchmark (small params, scaled to precision tiers)")
    print("=" * 70)
    print(f"Params: n_sims={N_SIMS_BENCH}, n_cal={N_CAL_BENCH}, n_reps={N_REPS_BENCH}, n_boot={N_BOOT_BENCH}")
    print(f"Power generators: {power_gens}.  CI generators: {ci_gens}.")
    print(f"Measured on: {logical_cores}-logical-core machine (physical: {physical_cores}), Numba enabled.")
    print()

    # --- JIT warmup (n_cal=1 so warmup cache entries don't pollute cold-cache timing) ---
    # This triggers Numba JIT compilation of simulation kernels; warm_calibration_cache
    # does not exercise those paths. n_cal=1 means the warmup calibration cache key
    # (n, k, dt, all_distinct, 1, "multipoint") is distinct from production keys at
    # n_cal=N_CAL_BENCH, so all 88 production entries are timed from cold below.
    # We use n_cal=1 so warmup cache keys stay distinct from production (n_cal=N_CAL_BENCH).
    # With n_cal=1 the bisection often hits the search boundary (too few samples to converge);
    # that triggers a UserWarning. We suppress it here because the warmup is only for JIT
    # compilation — we never use the warmup result. If you see the warning elsewhere it is
    # harmless; it just means that run hit the boundary.
    print("Warming up (one small run per generator)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for gen in power_gens:
            min_detectable_rho(
                n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, y_params,
                generator=gen, n_sims=50, seed=0, direction="positive",
                calibration_mode="multipoint", n_cal=1)
        for gen in ci_gens:
            bootstrap_ci_averaged(
                n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, rho_obs, y_params,
                generator=gen, n_reps=5, n_boot=10, seed=0,
                batch_bootstrap=True, calibration_mode="multipoint")
    print("Warmup done.")

    # --- Calibration cache warmup (timed) ---
    # Collapse power+CI to unique generators; CI shares calibration with power.
    unique_cal_gens = list(dict.fromkeys(
        g for g in (power_gens + ci_gens) if g != "linear"
    ))
    t_cal_build = {}
    for gen in unique_cal_gens:
        t0 = time.perf_counter()
        warm_calibration_cache(gen, y_params, n_cal=N_CAL_BENCH, seed=99,
                               calibration_mode="multipoint")
        t_cal_build[gen] = time.perf_counter() - t0
    print("Calibration cache (all scenario types) warmed.")

    # --- Null cache warmup (timed, always-on; fixes power vs CI null asymmetry) ---
    t0 = time.perf_counter()
    warm_precomputed_null_cache(n_pre=N_PRE_BENCH, seed=SEED)
    t_null_build = time.perf_counter() - t0
    print("Null cache warmed.\n")

    # --- Optional: print per-generator cache cost table with tier scaling ---
    if args.show_cache_costs:
        _n_cal_tiers = [pt[1] for pt in POWER_TIERS]
        print("=== Cache Precompute Costs (88 scenario types) ===")
        print()
        print("  Null cache (scales with n_pre):")
        pre_labels = [f"@bench {N_PRE_BENCH//1000}k"] + \
                     [f"@{lbl} {N_PRE_TIERS[i]//1000}k" for i, lbl in enumerate(_TIER_LABELS)]
        hdr = "  {:<8} | {:>13} | {:>14} | {:>16} | {:>16}".format(
            "n_pre", *pre_labels)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        null_vals = [t_null_build] + [t_null_build * (N_PRE_TIERS[i] / N_PRE_BENCH)
                                      for i in range(3)]
        print("  {:8} | {:>13} | {:>14} | {:>16} | {:>16}".format(
            "88 keys",
            _fmt_time_sec(null_vals[0]),
            _fmt_time_sec(null_vals[1]),
            _fmt_time_sec(null_vals[2]),
            _fmt_time_sec(null_vals[3]),
        ))
        print(f"  (Tier columns = t_null_build × (n_pre_tier / N_PRE_BENCH), linear scaling)\n")

        print("  Calibration (scales with n_cal):")
        cal_n_labels = [f"@bench {N_CAL_BENCH}"] + \
                       [f"@{lbl} {_n_cal_tiers[i]:,}" for i, lbl in enumerate(_TIER_LABELS)]
        hdr2 = "  {:<15} | {:>14} | {:>14} | {:>20} | {:>20}".format(
            "Generator", *cal_n_labels)
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for gen in unique_cal_gens:
            tb = t_cal_build[gen]
            vals = [tb] + [tb * (_n_cal_tiers[i] / N_CAL_BENCH) for i in range(3)]
            print("  {:<15} | {:>14} | {:>14} | {:>20} | {:>20}".format(
                gen,
                _fmt_time_sec(vals[0]),
                _fmt_time_sec(vals[1]),
                _fmt_time_sec(vals[2]),
                _fmt_time_sec(vals[3]),
            ))
        print(f"  (Tier columns = t_cal_build[gen] × (n_cal_tier / N_CAL_BENCH), linear scaling)\n")

    # --- Power: single (seq), grid seq, grid par ---
    # With n_sims=N_SIMS_BENCH (50), bisection can hit the search boundary by chance
    # (lucky seed gives power >= target at rho=0.25). We suppress the boundary
    # UserWarning here because this run is for timing only — accuracy is not the goal.
    # In production (n_sims from POWER_TIERS), the warning remains active.
    power_single = {}
    power_grid_seq = {}
    power_grid_par = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for gen in power_gens:
            print(f"Power [{gen}] single scenario (seq)...", end=" ", flush=True)
            t0 = time.perf_counter()
            min_detectable_rho(
                n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, y_params,
                generator=gen, n_sims=N_SIMS_BENCH, seed=SEED, direction="positive",
                calibration_mode="multipoint", n_cal=N_CAL_BENCH)
            power_single[gen] = time.perf_counter() - t0
            print(f"{power_single[gen]:.2f}s")

        if not args.quick:
            for gen in power_gens:
                print(f"Power [{gen}] full grid seq...", end=" ", flush=True)
                t0 = time.perf_counter()
                run_all_scenarios(
                    generator=gen, n_sims=N_SIMS_BENCH, seed=SEED, n_jobs=1,
                    calibration_mode="multipoint", n_cal=N_CAL_BENCH)
                power_grid_seq[gen] = time.perf_counter() - t0
                print(f"{power_grid_seq[gen]:.2f}s")

            if not args.skip_parallel:
                for gen in power_gens:
                    print(f"Power [{gen}] full grid par...", end=" ", flush=True)
                    t0 = time.perf_counter()
                    run_all_scenarios(
                        generator=gen, n_sims=N_SIMS_BENCH, seed=SEED, n_jobs=-1,
                        calibration_mode="multipoint", n_cal=N_CAL_BENCH)
                    power_grid_par[gen] = time.perf_counter() - t0
                    print(f"{power_grid_par[gen]:.2f}s")
        else:
            for gen in power_gens:
                power_grid_seq[gen] = 0.0
                power_grid_par[gen] = 0.0

    # --- CI: single (seq), grid seq, grid par ---
    ci_single = {}
    ci_grid_seq = {}
    ci_grid_par = {}

    for gen in ci_gens:
        print(f"CI [{gen}] single scenario (seq)...", end=" ", flush=True)
        t0 = time.perf_counter()
        bootstrap_ci_averaged(
            n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, rho_obs, y_params,
            generator=gen, n_reps=N_REPS_BENCH, n_boot=N_BOOT_BENCH, seed=SEED,
            batch_bootstrap=True, calibration_mode="multipoint")
        ci_single[gen] = time.perf_counter() - t0
        print(f"{ci_single[gen]:.2f}s")

    if not args.quick:
        for gen in ci_gens:
            print(f"CI [{gen}] full grid seq...", end=" ", flush=True)
            t0 = time.perf_counter()
            run_all_ci_scenarios(
                generator=gen, n_reps=N_REPS_BENCH, n_boot=N_BOOT_BENCH, seed=SEED,
                n_jobs=1, calibration_mode="multipoint", batch_bootstrap=True)
            ci_grid_seq[gen] = time.perf_counter() - t0
            print(f"{ci_grid_seq[gen]:.2f}s")

        if not args.skip_parallel:
            for gen in ci_gens:
                print(f"CI [{gen}] full grid par...", end=" ", flush=True)
                t0 = time.perf_counter()
                run_all_ci_scenarios(
                    generator=gen, n_reps=N_REPS_BENCH, n_boot=N_BOOT_BENCH, seed=SEED,
                    n_jobs=-1, calibration_mode="multipoint", batch_bootstrap=True)
                ci_grid_par[gen] = time.perf_counter() - t0
                print(f"{ci_grid_par[gen]:.2f}s")
    else:
        for gen in ci_gens:
            ci_grid_seq[gen] = 0.0
            ci_grid_par[gen] = 0.0

    # --- Totals and best config ---
    power_grid_seq_tot = sum(power_grid_seq[g] for g in power_gens)
    power_grid_par_tot = sum(power_grid_par.get(g, 0) for g in power_gens)
    power_best_tot = min(power_grid_seq_tot, power_grid_par_tot) if not args.quick else power_grid_seq_tot
    power_best_is_par = (not args.quick and not args.skip_parallel and power_grid_par_tot < power_grid_seq_tot)

    ci_grid_seq_tot = sum(ci_grid_seq[g] for g in ci_gens)
    ci_grid_par_tot = sum(ci_grid_par.get(g, 0) for g in ci_gens)
    ci_best_tot = min(ci_grid_seq_tot, ci_grid_par_tot) if not args.quick else ci_grid_seq_tot
    ci_best_is_par = (not args.quick and not args.skip_parallel and ci_grid_par_tot < ci_grid_seq_tot)

    combined_best = power_best_tot + ci_best_tot

    # --- Scaling helpers ---
    def scale_power(t_sec, n_sims_tgt, n_cal_tgt):
        return t_sec * (n_sims_tgt / N_SIMS_BENCH)

    def scale_ci(t_sec, n_reps_tgt, n_boot_tgt):
        return t_sec * (n_reps_tgt * n_boot_tgt) / (N_REPS_BENCH * N_BOOT_BENCH)

    t_cal_all = sum(t_cal_build.values())

    # Estimated cold-cache wall time in one parallel run.
    # Workers split 88 scenarios across logical_cores processes; each worker builds cache
    # for its share (~88 / n_workers scenarios) concurrently.  Wall time for cache inside
    # the parallel run ≈ (total cache time) / n_workers.
    # This is subtracted from the measured parallel time before linear scaling, so that
    # tier projections represent simulation-only (hot-cache) cost and stay comparable to
    # the hot-sequential case where cache was pre-warmed before the clock started. 
    t_cache_par_bench = (t_cal_all + t_null_build) / logical_cores

    def null_scaled(tier_i):
        return t_null_build * (N_PRE_TIERS[tier_i] / N_PRE_BENCH)

    def cal_scaled(n_cal_tgt):
        return t_cal_all * (n_cal_tgt / N_CAL_BENCH)

    # When parallel wins, the measured time includes cold-cache overhead that does not
    # scale with n_sims / n_reps.  Subtract the estimated cache wall time before scaling.
    # When sequential wins it is already hot-cache, so no correction is needed.
    power_sim_for_scaling = (max(power_grid_par_tot - t_cache_par_bench, 0.0)
                             if power_best_is_par else power_best_tot)
    ci_sim_for_scaling    = (max(ci_grid_par_tot    - t_cache_par_bench, 0.0)
                             if ci_best_is_par    else ci_best_tot)

    power_scaled = [scale_power(power_sim_for_scaling, POWER_TIERS[i][0], POWER_TIERS[i][1])
                    for i in range(3)]
    ci_scaled = [scale_ci(ci_sim_for_scaling, CI_TIERS[i][0], CI_TIERS[i][1])
                 for i in range(3)]
    combined_scaled = [power_scaled[i] + ci_scaled[i] for i in range(3)]

    # High-core estimates (power only, using parallel time if it was measured and wins).
    # Cache cost at N target cores: (t_cal_all + t_null_build) / N (workers share scenarios).
    # Simulation scales as power_sim_for_scaling * (this_cores / N) / efficiency.
    if not args.quick and not args.skip_parallel and power_best_is_par and power_grid_par_tot > 0:
        t_cache_8  = (t_cal_all + t_null_build) / 8
        t_cache_16 = (t_cal_all + t_null_build) / 16
        est_8  = power_sim_for_scaling * (logical_cores / 8)  / EFFICIENCY + t_cache_8  + ci_sim_for_scaling
        est_16 = power_sim_for_scaling * (logical_cores / 16) / EFFICIENCY + t_cache_16 + ci_sim_for_scaling
        est_8_scaled  = [power_sim_for_scaling * (logical_cores / 8)  / EFFICIENCY * (POWER_TIERS[i][0] / N_SIMS_BENCH) + t_cache_8  + ci_scaled[i] for i in range(3)]
        est_16_scaled = [power_sim_for_scaling * (logical_cores / 16) / EFFICIENCY * (POWER_TIERS[i][0] / N_SIMS_BENCH) + t_cache_16 + ci_scaled[i] for i in range(3)]
    else:
        est_8 = est_16 = None
        est_8_scaled = est_16_scaled = [None] * 3

    # --- Print tables ---
    print()

    if power_gens:
        print("=== POWER: Per-Generator Runtimes (full grid, all generators) ===")
        print()
        print("Generator       | Single (seq) | Grid seq | Grid par | Grid +/-0.01 | Grid +/-0.002 | Grid +/-0.001")
        print("-" * 95)
        for gen in power_gens:
            single_s = f"{power_single[gen]:.2f}s"
            gs = f"{power_grid_seq[gen]:.2f}s" if not args.quick else "-"
            gp = f"{power_grid_par.get(gen, 0):.2f}s" if not args.quick and not args.skip_parallel else "-"
            _par_g = power_grid_par.get(gen, float("inf"))
            _seq_g = power_grid_seq[gen]
            best_per_gen = min(_seq_g, _par_g) if not args.skip_parallel and not args.quick else _seq_g
            _par_wins_g = (not args.skip_parallel and not args.quick and _par_g < _seq_g)
            sim_for_scaling_g = (max(_par_g - t_cache_par_bench, 0.0) if _par_wins_g else best_per_gen)
            if args.quick:
                scale_01 = scale_002 = scale_001 = "-"
            else:
                s1 = scale_power(sim_for_scaling_g, POWER_TIERS[0][0], POWER_TIERS[0][1])
                s2 = scale_power(sim_for_scaling_g, POWER_TIERS[1][0], POWER_TIERS[1][1])
                s3 = scale_power(sim_for_scaling_g, POWER_TIERS[2][0], POWER_TIERS[2][1])
                scale_01 = _fmt_time_sec(s1)
                scale_002 = _fmt_time_sec(s2)
                scale_001 = _fmt_time_sec(s3)
            print(f"{gen:<15} | {single_s:>12} | {gs:>8} | {gp:>8} | {scale_01:>12} | {scale_002:>13} | {scale_001:>12}")
        tot_single = "-"
        tot_gs = _fmt_time_sec(power_grid_seq_tot) if not args.quick else "-"
        tot_gp = _fmt_time_sec(power_grid_par_tot) if not args.quick and not args.skip_parallel else "-"
        tot_01 = _fmt_time_sec(power_scaled[0]) if not args.quick else "-"
        tot_002 = _fmt_time_sec(power_scaled[1]) if not args.quick else "-"
        tot_001 = _fmt_time_sec(power_scaled[2]) if not args.quick else "-"
        print(f"{'ALL GENERATORS':<15} | {tot_single:>12} | {tot_gs:>8} | {tot_gp:>8} | {tot_01:>12} | {tot_002:>13} | {tot_001:>12}")
        print("    Single = single scenario, n_jobs=1. Grid seq = full grid, n_jobs=1. Grid par = full grid, n_jobs=-1.")
        print("    Scaled columns use whichever of seq/par is faster on this machine (noted below table).")
        print("    When par wins, estimated cold-cache wall time is subtracted before scaling (see Notes).")
        print()

    if ci_gens:
        print("=== CI: Per-Generator Runtimes (full grid, all generators, batch bootstrap) ===")
        print()
        print("Generator       | Single (seq) | Grid seq | Grid par | Grid +/-0.01 | Grid +/-0.002 | Grid +/-0.001")
        print("-" * 95)
        for gen in ci_gens:
            single_s = f"{ci_single[gen]:.2f}s"
            gs = f"{ci_grid_seq[gen]:.2f}s" if not args.quick else "-"
            gp = f"{ci_grid_par.get(gen, 0):.2f}s" if not args.quick and not args.skip_parallel else "-"
            _par_g = ci_grid_par.get(gen, float("inf"))
            _seq_g = ci_grid_seq[gen]
            best_per_gen = min(_seq_g, _par_g) if not args.skip_parallel and not args.quick else _seq_g
            _par_wins_g = (not args.skip_parallel and not args.quick and _par_g < _seq_g)
            sim_for_scaling_g = (max(_par_g - t_cache_par_bench, 0.0) if _par_wins_g else best_per_gen)
            if args.quick:
                scale_01 = scale_002 = scale_001 = "-"
            else:
                s1 = scale_ci(sim_for_scaling_g, CI_TIERS[0][0], CI_TIERS[0][1])
                s2 = scale_ci(sim_for_scaling_g, CI_TIERS[1][0], CI_TIERS[1][1])
                s3 = scale_ci(sim_for_scaling_g, CI_TIERS[2][0], CI_TIERS[2][1])
                scale_01 = _fmt_time_sec(s1)
                scale_002 = _fmt_time_sec(s2)
                scale_001 = _fmt_time_sec(s3)
            print(f"{gen:<15} | {single_s:>12} | {gs:>8} | {gp:>8} | {scale_01:>12} | {scale_002:>13} | {scale_001:>12}")
        tot_gs = _fmt_time_sec(ci_grid_seq_tot) if not args.quick else "-"
        tot_gp = _fmt_time_sec(ci_grid_par_tot) if not args.quick and not args.skip_parallel else "-"
        tot_01 = _fmt_time_sec(ci_scaled[0]) if not args.quick else "-"
        tot_002 = _fmt_time_sec(ci_scaled[1]) if not args.quick else "-"
        tot_001 = _fmt_time_sec(ci_scaled[2]) if not args.quick else "-"
        print(f"{'ALL GENERATORS':<15} | {'-':>12} | {tot_gs:>8} | {tot_gp:>8} | {tot_01:>12} | {tot_002:>13} | {tot_001:>12}")
        print("    Scaled columns use whichever of seq/par is faster on this machine.")
        print("    When par wins, estimated cold-cache wall time is subtracted before scaling (see Notes).")
        print()

    if power_gens and ci_gens:
        print("=== COMBINED (Power + CI) Full Grid, All Generators ===")
        print('    "Best config" = for each of power and CI, use whichever of seq/par was faster on this machine, then sum.')
        print()
        print("Tier      | Best config (this machine) | Est. 8-core | Est. 16-core")
        print("-" * 75)
        meas_best = _fmt_time_sec(combined_best) if not args.quick else "-"
        e8_meas = _fmt_time_sec(est_8) if est_8 is not None else "-"
        e16_meas = _fmt_time_sec(est_16) if est_16 is not None else "-"
        print(f"Measured  | {meas_best:>28} | {e8_meas:>11} | {e16_meas:>11}")
        for i, label in enumerate(_TIER_LABELS):
            cb = _fmt_time_sec(combined_scaled[i]) if not args.quick else "-"
            e8 = _fmt_time_sec(est_8_scaled[i]) if est_8_scaled[i] is not None else "-"
            e16 = _fmt_time_sec(est_16_scaled[i]) if est_16_scaled[i] is not None else "-"
            print(f"{label:<9} | {cb:>28} | {e8:>11} | {e16:>11}")
        print()
        pw_best = "par" if power_best_is_par else "seq"
        ci_best_str = "par" if ci_best_is_par else "seq"
        if args.quick:
            print("    Power best: -. CI best: -. (Run without --quick for full grid and best-config summary.)")
        else:
            pw_note = (f" (sim-only: {_fmt_time_sec(power_sim_for_scaling)}, after subtracting ~{_fmt_time_sec(t_cache_par_bench)} est. cache)"
                       if power_best_is_par else "")
            ci_note  = (f" (sim-only: {_fmt_time_sec(ci_sim_for_scaling)}, after subtracting ~{_fmt_time_sec(t_cache_par_bench)} est. cache)"
                       if ci_best_is_par else "")
            print(f"    Power best: {pw_best} ({_fmt_time_sec(power_best_tot)}){pw_note}.")
            print(f"    CI best: {ci_best_str} ({_fmt_time_sec(ci_best_tot)}){ci_note}.")
        print("    Higher-core scaling (power only, if par wins): sim scaled by (this_cores / target_cores) / efficiency; cache scaled by 1 / target_cores.")
        print("    CI with batch bootstrap: if seq is faster at small params, parallel may still win at high n_reps (see one-off benchmark).")
        print()

    # --- Cold-start first-run cost block ---
    print("=== Cold-start first-run cost (simulation + cache build, no disk cache) ===")
    print('    "Hot"  = simulation only (disk-precomputed caches available at startup).')
    print('    "Cold" = simulation + null build + calibration build at production n_cal / n_pre.')
    print()

    run_power = bool(power_gens)
    run_ci = bool(ci_gens)

    if run_power and run_ci:
        print("  Scenario A: power then CI in one run — cache built once, shared")
        print(f"  {'Tier':<9} | {'Power+CI sim':>14} | {'Cache (null+cal)':>17} | {'Total (single run)':>20}")
        print("  " + "-" * 67)
        for i, label in enumerate(_TIER_LABELS):
            sim = power_scaled[i] + ci_scaled[i]
            cache = null_scaled(i) + cal_scaled(POWER_TIERS[i][1])
            total = sim + cache
            if args.quick:
                print(f"  {label:<9} | {'(--quick, no grid)':>14} | {_fmt_time_sec(cache):>17} | {'-':>20}")
            else:
                print(f"  {label:<9} | {_fmt_time_sec(sim):>14} | {_fmt_time_sec(cache):>17} | {_fmt_time_sec(total):>20}")
        print()
        print("  Scenario B: power and CI in separate runs — cache built twice")
        print(f"  {'Tier':<9} | {'Power-only (cold)':>18} | {'CI-only (cold)':>16} | {'Sum (two runs)':>16}")
        print("  " + "-" * 61)
        for i, label in enumerate(_TIER_LABELS):
            t_pow_cold = power_scaled[i] + null_scaled(i) + cal_scaled(POWER_TIERS[i][1])
            t_ci_cold = ci_scaled[i] + null_scaled(i) + cal_scaled(CI_TIERS[i][2])
            t_sum = t_pow_cold + t_ci_cold
            if args.quick:
                print(f"  {label:<9} | {'(--quick, no grid)':>18} | {'(--quick, no grid)':>16} | {'-':>16}")
            else:
                print(f"  {label:<9} | {_fmt_time_sec(t_pow_cold):>18} | {_fmt_time_sec(t_ci_cold):>16} | {_fmt_time_sec(t_sum):>16}")
        print()
    elif run_power:
        print("  Power-only run (cold):")
        print(f"  {'Tier':<9} | {'Simulation (hot)':>18} | {'Cache (null+cal)':>17} | {'Total (cold)':>14}")
        print("  " + "-" * 60)
        for i, label in enumerate(_TIER_LABELS):
            cache = null_scaled(i) + cal_scaled(POWER_TIERS[i][1])
            total = power_scaled[i] + cache
            if args.quick:
                print(f"  {label:<9} | {'(--quick, no grid)':>18} | {_fmt_time_sec(cache):>17} | {'-':>14}")
            else:
                print(f"  {label:<9} | {_fmt_time_sec(power_scaled[i]):>18} | {_fmt_time_sec(cache):>17} | {_fmt_time_sec(total):>14}")
        print()
    elif run_ci:
        print("  CI-only run (cold):")
        print(f"  {'Tier':<9} | {'Simulation (hot)':>18} | {'Cache (null+cal)':>17} | {'Total (cold)':>14}")
        print("  " + "-" * 60)
        for i, label in enumerate(_TIER_LABELS):
            cache = null_scaled(i) + cal_scaled(CI_TIERS[i][2])
            total = ci_scaled[i] + cache
            if args.quick:
                print(f"  {label:<9} | {'(--quick, no grid)':>18} | {_fmt_time_sec(cache):>17} | {'-':>14}")
            else:
                print(f"  {label:<9} | {_fmt_time_sec(ci_scaled[i]):>18} | {_fmt_time_sec(cache):>17} | {_fmt_time_sec(total):>14}")
        print()

    print("Notes:")
    print("  - Simulation columns (hot): cache pre-warmed; measures simulation cost only.")
    print("  - Grid seq times: in-process hot cache (calibration + null built before clock starts).")
    print("  - Grid par times: joblib workers spawn fresh processes with cold caches — each worker")
    print("    rebuilds calibration and null independently. This is a hot-seq vs cold-par comparison.")
    print("    Parallel would match hot-seq if workers loaded disk-precomputed caches at startup.")
    print("  - Cold-start cache cost scales with n_cal (calibration) and n_pre (null) at each tier.")
    print("    Use --show-cache-costs for a per-generator breakdown.")
    print("  - Scaled tier columns: represent hot-cache simulation time, so cache is not double-counted")
    print("    when the Cold-start table is added. When par wins, estimated cold-cache wall time")
    print(f"    (~(t_cal + t_null) / n_workers ≈ {_fmt_time_sec(t_cache_par_bench)} at bench params) is subtracted before scaling.")
    print("    This removes the fixed overhead that would otherwise be scaled up linearly (~20% overestimate at ±0.001).")
    print("  - Power parallel scaling to N cores: sim scaled by (this_cores / N) / 0.5; cache by 1/N (workers split scenarios).")
    print("  - CI with batch bootstrap: parallel is often slower than sequential on this machine (joblib overhead + cold-cache in workers).")
    print("  - On higher-core machines with very high n_reps, the per-scenario time may be large enough for parallel to help; benchmark on your machine.")


if __name__ == "__main__":
    run()
