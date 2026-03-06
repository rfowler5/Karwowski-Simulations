"""
Benchmark realistic runtimes per generator (power + CI) at small params,
then scale to +/-0.01, +/-0.002, +/-0.001 accuracy tiers.

Run one at a time per benchmarking rule. Use --quick for single-scenario only,
--skip-parallel to skip parallel (n_jobs=-1) runs.

Usage:
  python benchmarks/benchmark_realistic_runtimes.py [--quick] [--skip-parallel] [--generators all|nonparametric|copula|linear|empirical]
"""
import sys
import os
import time
import argparse
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import CASES
from power_simulation import min_detectable_rho, run_all_scenarios
from confidence_interval_calculator import bootstrap_ci_averaged, run_all_ci_scenarios
from data_generator import digitized_available

# Benchmark params (small, for measurement)
N_SIMS_BENCH = 500
N_CAL_BENCH = 300
N_REPS_BENCH = 200
N_BOOT_BENCH = 500
SEED = 42
CASE_ID = 3
DIST_TYPE_SINGLE = "heavy_center"
N_DISTINCT_SINGLE = 4

# Precision tier params (from plan / benchmark_precision_params)
POWER_TIERS = [
    (2000, 1000),      # +/-0.01
    (50000, 21000),    # +/-0.002
    (200000, 83000),   # +/-0.001
]
CI_TIERS = [
    (649, 500),        # +/-0.01   (SD_INTER_REP=0.13)
    (16231, 500),      # +/-0.002  (SD_INTER_REP=0.13)
    (64923, 500),      # +/-0.001  (SD_INTER_REP=0.13)
]
EFFICIENCY = 0.5


def _parse_args():
    p = argparse.ArgumentParser(description="Benchmark realistic runtimes and scale to precision tiers.")
    p.add_argument("--generators", default="all",
                   help="Comma-separated: all, nonparametric, copula, linear, empirical (default: all)")
    p.add_argument("--quick", action="store_true", help="Single-scenario only (skip full grid)")
    p.add_argument("--skip-parallel", action="store_true", help="Skip n_jobs=-1 benchmarks")
    return p.parse_args()


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
        import multiprocessing
        # physical not always available on Windows
        physical = getattr(multiprocessing, "cpu_count", lambda: logical)() or logical
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
    if not power_gens:
        print("No power generators to run. Check --generators and digitized data.")
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

    # Warmup: one small run per generator that we will use
    print("Warming up (one small run per generator)...")
    for gen in power_gens:
        min_detectable_rho(
            n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, y_params,
            generator=gen, n_sims=50, seed=0, direction="positive",
            calibration_mode="multipoint", n_cal=N_CAL_BENCH)
    for gen in ci_gens:
        bootstrap_ci_averaged(
            n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, rho_obs, y_params,
            generator=gen, n_reps=5, n_boot=10, seed=0,
            batch_bootstrap=True, calibration_mode="multipoint")
    print("Warmup done.\n")

    # --- Power: single (seq), grid seq, grid par ---
    power_single = {}
    power_grid_seq = {}
    power_grid_par = {}

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

    # --- Scaled times (use best config per component for scaling) ---
    def scale_power(t_sec, n_sims_tgt, n_cal_tgt):
        return t_sec * (n_sims_tgt / N_SIMS_BENCH)

    def scale_ci(t_sec, n_reps_tgt, n_boot_tgt):
        return t_sec * (n_reps_tgt * n_boot_tgt) / (N_REPS_BENCH * N_BOOT_BENCH)

    power_scaled = []
    for (ns, nc) in POWER_TIERS:
        t = scale_power(power_best_tot, ns, nc)
        power_scaled.append(t)
    ci_scaled = []
    for (nr, nb) in CI_TIERS:
        t = scale_ci(ci_best_tot, nr, nb)
        ci_scaled.append(t)
    combined_scaled = [power_scaled[i] + ci_scaled[i] for i in range(3)]

    # High-core estimates (power only, using parallel time if it was measured and wins)
    if not args.quick and not args.skip_parallel and power_best_is_par and power_grid_par_tot > 0:
        est_8 = power_grid_par_tot * (logical_cores / 8) / EFFICIENCY + ci_best_tot
        est_16 = power_grid_par_tot * (logical_cores / 16) / EFFICIENCY + ci_best_tot
        est_8_scaled = [power_grid_par_tot * (logical_cores / 8) / EFFICIENCY * (POWER_TIERS[i][0] / N_SIMS_BENCH) + ci_scaled[i] for i in range(3)]
        est_16_scaled = [power_grid_par_tot * (logical_cores / 16) / EFFICIENCY * (POWER_TIERS[i][0] / N_SIMS_BENCH) + ci_scaled[i] for i in range(3)]
    else:
        est_8 = est_16 = None
        est_8_scaled = est_16_scaled = [None] * 3

    # --- Print tables ---
    print()
    print("=== POWER: Per-Generator Runtimes (full grid, all generators) ===")
    print()
    print("Generator       | Single (seq) | Grid seq | Grid par | Grid +/-0.01 | Grid +/-0.002 | Grid +/-0.001")
    print("-" * 95)
    for gen in power_gens:
        single_s = f"{power_single[gen]:.2f}s"
        gs = f"{power_grid_seq[gen]:.2f}s" if not args.quick else "-"
        gp = f"{power_grid_par.get(gen, 0):.2f}s" if not args.quick and not args.skip_parallel else "-"
        best_per_gen = min(power_grid_seq[gen], power_grid_par.get(gen, float("inf"))) if not args.skip_parallel and not args.quick else power_grid_seq[gen]
        if args.quick:
            scale_01 = scale_002 = scale_001 = "-"
        else:
            s1 = scale_power(best_per_gen, POWER_TIERS[0][0], POWER_TIERS[0][1])
            s2 = scale_power(best_per_gen, POWER_TIERS[1][0], POWER_TIERS[1][1])
            s3 = scale_power(best_per_gen, POWER_TIERS[2][0], POWER_TIERS[2][1])
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
    print()

    print("=== CI: Per-Generator Runtimes (full grid, all generators, batch bootstrap) ===")
    print()
    print("Generator       | Single (seq) | Grid seq | Grid par | Grid +/-0.01 | Grid +/-0.002 | Grid +/-0.001")
    print("-" * 95)
    for gen in ci_gens:
        single_s = f"{ci_single[gen]:.2f}s"
        gs = f"{ci_grid_seq[gen]:.2f}s" if not args.quick else "-"
        gp = f"{ci_grid_par.get(gen, 0):.2f}s" if not args.quick and not args.skip_parallel else "-"
        best_per_gen = min(ci_grid_seq[gen], ci_grid_par.get(gen, float("inf"))) if not args.skip_parallel and not args.quick else ci_grid_seq[gen]
        if args.quick:
            scale_01 = scale_002 = scale_001 = "-"
        else:
            s1 = scale_ci(best_per_gen, CI_TIERS[0][0], CI_TIERS[0][1])
            s2 = scale_ci(best_per_gen, CI_TIERS[1][0], CI_TIERS[1][1])
            s3 = scale_ci(best_per_gen, CI_TIERS[2][0], CI_TIERS[2][1])
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
    print()

    print("=== COMBINED (Power + CI) Full Grid, All Generators ===")
    print('    "Best config" = for each of power and CI, use whichever of seq/par was faster on this machine, then sum.')
    print()
    print("Tier      | Best config (this machine) | Est. 8-core | Est. 16-core")
    print("-" * 75)
    meas_best = _fmt_time_sec(combined_best) if not args.quick else "-"
    e8_meas = _fmt_time_sec(est_8) if est_8 is not None else "-"
    e16_meas = _fmt_time_sec(est_16) if est_16 is not None else "-"
    print(f"Measured  | {meas_best:>28} | {e8_meas:>11} | {e16_meas:>11}")
    for i, label in enumerate(["+/-0.01", "+/-0.002", "+/-0.001"]):
        cb = _fmt_time_sec(combined_scaled[i]) if not args.quick else "-"
        e8 = _fmt_time_sec(est_8_scaled[i]) if est_8_scaled[i] is not None else "-"
        e16 = _fmt_time_sec(est_16_scaled[i]) if est_16_scaled[i] is not None else "-"
        print(f"{label:<9} | {cb:>28} | {e8:>11} | {e16:>11}")
    print()
    pw_best = "par" if power_best_is_par else "seq"
    ci_best = "par" if ci_best_is_par else "seq"
    if args.quick:
        print("    Power best: -. CI best: -. (Run without --quick for full grid and best-config summary.)")
    else:
        print(f"    Power best: {pw_best} ({_fmt_time_sec(power_best_tot)}). CI best: {ci_best} ({_fmt_time_sec(ci_best_tot)}).")
    print("    Higher-core scaling (power only, if par wins): est_time = power_par_time * (this_cores / target_cores) / efficiency.")
    print("    CI with batch bootstrap: if seq is faster at small params, parallel may still win at high n_reps (see one-off benchmark).")
    print()
    print("Notes:")
    print("  - Power parallel scaling to N cores: (sequential_time / N) / 0.5. This assumes joblib overhead is amortized on larger machines.")
    print("  - CI with batch bootstrap: parallel is often slower than sequential on this machine (joblib overhead ~89%, thread oversubscription ~11%).")
    print("  - On higher-core machines with very high n_reps, the per-scenario time may be large enough for parallel to help; benchmark on your machine.")


if __name__ == "__main__":
    run()
