"""
Fit runtime model T = C + k*n (power) and T_ci = C_ci + k_ci*(n_reps*n_boot) (CI)
from single-scenario runs at small params; optionally fit a grid-level model from
full-grid runs to capture cross-scenario heterogeneity.

Use predicted tier runtimes instead of running long empirical full grids.
Run from project root: python benchmarks/estimate_runtime_model.py --generator empirical [options]

Two prediction models:
  - Single-scenario extrapolation = 88 * (C + k*n_tier).  Assumes every scenario
    costs the same as the single representative scenario (case 3).  Used when
    --grid-n-sims is not provided.
  - Grid fit = 88 * (C_g + k_g*n_tier).  Fits C_g, k_g directly from timed
    full-grid runs.  Captures the fact that the grid-average per-sim cost k_g
    differs from the single-scenario k (cross-scenario heterogeneity: different
    sample sizes, tie structures, and bisection depths).

n size for good estimates:
  - Single-scenario fit (--n-sims or --ci-params): Use values large enough that
    each timed run is at least ~1 s so timing noise is small and R^2 is reliable.
    For nonparametric/copula/linear that usually means n_sims >= 2000 (e.g.
    --n-sims 2000,5000,10000). For empirical, defaults (50,75,100) already give
    ~12-24 s per point. For CI, use n_reps*n_boot in the thousands (e.g. 1000+).
  - Grid fit (--grid-n-sims or --grid-ci-params): Provide at least 2 values
    (3+ recommended) so the grid-level linear fit has enough points.  All caches
    (null + calibration) are pre-warmed before grid runs so every grid point
    measures steady-state simulation cost only.

Multi-core extrapolation:
  Without --grid-parallel, multi-core estimates use a rough formula:
  T_par_est = T_seq / (P * 0.5).  Pass --grid-parallel to run the full grid
  once with n_jobs=-1 and extrapolate from the measured parallel time.
"""
import os
import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import CASES, POWER_TIERS, CI_TIERS, N_PRE_BENCH, N_PRE_TIERS
from power_simulation import min_detectable_rho, run_all_scenarios
from confidence_interval_calculator import bootstrap_ci_averaged, run_all_ci_scenarios
from data_generator import digitized_available, warm_calibration_cache
from permutation_pvalue import warm_precomputed_null_cache

# Constants (self-contained; do not import from benchmark script)
N_GRID_SCENARIOS = 88
N_CAL_BENCH = 300
SEED = 42
CASE_ID = 3
DIST_TYPE_SINGLE = "heavy_center"
N_DISTINCT_SINGLE = 4
EFFICIENCY = 0.5  # empirical parallel efficiency (benchmark_realistic_runtimes)

FIT_N_SIMS_BY_GENERATOR = {
    "empirical":     [50, 75, 100],
    "nonparametric": [2000, 5000, 10000],
    "copula":        [2000, 5000, 10000],
    "linear":        [2000, 5000, 10000],
}

FIT_CI_PARAMS_BY_GENERATOR = {
    "empirical":     [(5, 10), (8, 15), (10, 20)],
    "nonparametric": [(20, 50), (40, 100), (60, 150)],
    "copula":        [(20, 50), (40, 100), (60, 150)],
}


def _fmt_time_sec(t_sec):
    if t_sec < 0:
        return "(neg)"
    if t_sec < 60:
        return f"{t_sec:.1f}s"
    if t_sec < 3600:
        return f"{t_sec/60:.1f} min"
    return f"{t_sec/3600:.2f} hrs"


def _machine_info():
    logical = os.cpu_count() or 1
    try:
        import multiprocessing
        physical = getattr(multiprocessing, "cpu_count", lambda: logical)() or logical
    except Exception:
        physical = "?"
    return logical, physical


def _parse_args():
    p = argparse.ArgumentParser(
        description="Fit T=C+k*n runtime model and predict tier runtimes (power + CI)."
    )
    p.add_argument("--generator", required=True,
                   choices=["nonparametric", "copula", "linear", "empirical"],
                   help="Generator to fit")
    p.add_argument("--mode", default="both", choices=["power", "ci", "both"],
                   help="power, ci, or both (default: both; linear -> power only)")
    p.add_argument("--n-sims", default=None,
                   help="Comma-separated n_sims for power fit (overrides per-generator default)")
    p.add_argument("--grid-n-sims", default=None,
                   help="Comma-separated n_sims for full-grid fit (>=2 values recommended)")
    p.add_argument("--ci-params", default=None,
                   help="Space-separated n_reps,n_boot pairs for CI fit (e.g. '20,50 40,100')")
    p.add_argument("--grid-ci-params", default=None,
                   help="Space-separated n_reps,n_boot pairs for full-grid CI fit (>=2 recommended)")
    p.add_argument("--borrow-g-from", default=None, metavar="GENERATOR",
                   help="Estimate grid-level k_g from a faster proxy generator and scale (opt-in)")
    p.add_argument("--repeats", type=int, default=1,
                   help="Repeats per fit point (mean used); default 1")
    p.add_argument("--grid-repeats", type=int, default=1,
                   help="Repeats per full-grid run (mean used); default 1")
    p.add_argument("--case", type=int, default=3,
                   help="Case ID for single-scenario (default 3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed (default 42)")
    p.add_argument("--grid-parallel", action="store_true",
                   help="Run full grid once with n_jobs=-1 for multi-core extrapolation (power and/or CI)")
    return p.parse_args()


def _linear_fit_r2(xs, ys):
    """Fit y = k*x + C via polyfit; return (k, C, R²)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    coefs = np.polyfit(xs, ys, 1)
    k, C = coefs[0], coefs[1]
    y_pred = k * xs + C
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return k, C, r2


def main():
    args = _parse_args()
    generator = args.generator.lower()
    mode = args.mode.lower()
    logical_cores, physical_cores = _machine_info()

    # Resolve mode: linear has no CI
    if generator == "linear":
        mode = "power"
    elif mode == "both":
        mode = "both"

    if generator == "empirical" and not digitized_available():
        print("Error: empirical generator requires digitized data (digitized_available() is False).")
        sys.exit(1)

    if args.borrow_g_from is not None:
        if not args.grid_n_sims and not args.grid_ci_params:
            print("Error: --borrow-g-from requires --grid-n-sims or --grid-ci-params.")
            sys.exit(1)
        proxy = args.borrow_g_from.strip().lower()
        if proxy not in ("nonparametric", "copula", "linear", "empirical"):
            print(f"Error: --borrow-g-from must be a valid generator, got {args.borrow_g_from}")
            sys.exit(1)
        if proxy == generator:
            print("Error: --borrow-g-from must differ from --generator.")
            sys.exit(1)
        if proxy == "empirical":
            print("Warning: --borrow-g-from=empirical is slow; consider nonparametric or copula.")

    # n_sims list for power fit
    if args.n_sims is not None:
        n_sims_list = [int(x.strip()) for x in args.n_sims.split(",")]
    else:
        n_sims_list = FIT_N_SIMS_BY_GENERATOR[generator]
    n_sims_list = sorted(set(n_sims_list))
    if len(n_sims_list) < 2:
        print("Error: Power fit requires at least 2 distinct --n-sims values.")
        sys.exit(1)

    # CI params for CI fit
    if mode in ("ci", "both"):
        if args.ci_params is not None:
            ci_params = []
            for part in args.ci_params.split():
                a, b = part.strip().split(",")
                ci_params.append((int(a), int(b)))
        else:
            ci_params = FIT_CI_PARAMS_BY_GENERATOR.get(generator)
            if not ci_params:
                print("Error: CI not supported for linear; use --mode power.")
                sys.exit(1)
        ci_params = list(dict.fromkeys(ci_params))
        if len(ci_params) < 2:
            print("Error: CI fit requires at least 2 distinct (n_reps*n_boot) values.")
            sys.exit(1)
        n_reps_boot = [nr * nb for (nr, nb) in ci_params]
        if len(set(n_reps_boot)) < 2:
            print("Error: CI fit requires at least 2 distinct n_reps*n_boot values.")
            sys.exit(1)

    grid_n_sims = None
    if args.grid_n_sims:
        grid_n_sims = [int(x.strip()) for x in args.grid_n_sims.split(",")]

    grid_ci_params = None
    if args.grid_ci_params:
        grid_ci_params = []
        for part in args.grid_ci_params.split():
            a, b = part.strip().split(",")
            grid_ci_params.append((int(a), int(b)))

    case = CASES[args.case]
    n_single = case["n"]
    rho_obs = case["observed_rho"]
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}

    # Warmup: JIT compilation + caches that should be warm for all timed runs.
    # n_cal=1 so warmup cache entries (key includes n_cal) are distinct from
    # production entries at n_cal=N_CAL_BENCH — ensures all 88 production entries
    # are timed from cold during the cache warmup below.
    # We use n_cal=1 so warmup cache keys stay distinct from production (n_cal=N_CAL_BENCH).
    # With n_cal=1 the bisection often hits the search boundary (too few samples to converge);
    # that triggers a UserWarning. We suppress it here because the warmup is only for JIT
    # compilation — we never use the warmup result. If you see the warning elsewhere it is
    # harmless; it just means that run hit the boundary.
    print("Warming up (JIT + one small run per generator used)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mode in ("power", "both"):
            min_detectable_rho(
                n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, y_params,
                generator=generator, n_sims=50, seed=0, direction="positive",
                calibration_mode="multipoint", n_cal=1)
        if mode in ("ci", "both"):
            bootstrap_ci_averaged(
                n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, rho_obs, y_params,
                generator=generator, n_reps=5, n_boot=10, seed=0,
                batch_bootstrap=True, calibration_mode="multipoint")
        if args.borrow_g_from and proxy != generator:
            min_detectable_rho(
                n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, y_params,
                generator=proxy, n_sims=50, seed=0, direction="positive",
                calibration_mode="multipoint", n_cal=1)
            if mode in ("ci", "both") and proxy != "linear":
                bootstrap_ci_averaged(
                    n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, rho_obs, y_params,
                    generator=proxy, n_reps=5, n_boot=10, seed=0,
                    batch_bootstrap=True, calibration_mode="multipoint")
    print("Warmup done.")

    # Pre-warm caches for all 88 scenarios so grid runs measure simulation
    # cost only, not one-time cache-building overhead.  Time both null and
    # calibration so we can report cold-start costs in tier predictions.
    need_grid = (grid_n_sims or grid_ci_params)
    t_null_build = None
    t_cal_bench = None
    if need_grid:
        print("Warming null + calibration caches for all 88 scenarios...", flush=True)
        t0 = time.perf_counter()
        warm_precomputed_null_cache(n_pre=N_PRE_BENCH, seed=args.seed)
        t_null_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        warm_calibration_cache(generator, y_params,
                               calibration_mode="multipoint",
                               n_cal=N_CAL_BENCH, seed=99)
        t_cal_bench = time.perf_counter() - t0
        if args.borrow_g_from:
            warm_calibration_cache(proxy, y_params,
                                   calibration_mode="multipoint",
                                   n_cal=N_CAL_BENCH, seed=99)
        print(f"  Null:          {t_null_build:.2f}s  "
              f"(@+/-0.01 {_fmt_time_sec(t_null_build * N_PRE_TIERS[0] / N_PRE_BENCH)}, "
              f"@+/-0.002 {_fmt_time_sec(t_null_build * N_PRE_TIERS[1] / N_PRE_BENCH)}, "
              f"@+/-0.001 {_fmt_time_sec(t_null_build * N_PRE_TIERS[2] / N_PRE_BENCH)})")
        print(f"  Calibration:   {t_cal_bench:.2f}s  "
              f"(@+/-0.01 {_fmt_time_sec(t_cal_bench * POWER_TIERS[0][1] / N_CAL_BENCH)}, "
              f"@+/-0.002 {_fmt_time_sec(t_cal_bench * POWER_TIERS[1][1] / N_CAL_BENCH)}, "
              f"@+/-0.001 {_fmt_time_sec(t_cal_bench * POWER_TIERS[2][1] / N_CAL_BENCH)})")
        print("  (Tier scaling: null × n_pre_tier/N_PRE_BENCH, cal × n_cal_tier/N_CAL_BENCH)")
        print("Caches warm.\n")
    else:
        print()

    has_cold = (t_null_build is not None and t_cal_bench is not None)
    # Outer-scope storage for tier sim times; populated inside power/CI blocks
    # so the combined cold-start summary can reference both regardless of mode.
    t_grid_power = [0.0, 0.0, 0.0]  # filled in power block
    t_grid_ci = [0.0, 0.0, 0.0]     # filled in CI block

    # --- Power fit ---
    if mode in ("power", "both"):
        print(f"=== Power runtime model fit: {generator} ===\n")
        print("Fit points (single scenario, n_jobs=1):")
        Ts = []
        for n in n_sims_list:
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                min_detectable_rho(
                    n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, y_params,
                    generator=generator, n_sims=n, seed=args.seed, direction="positive",
                    calibration_mode="multipoint", n_cal=N_CAL_BENCH)
                times.append(time.perf_counter() - t0)
            t_mean = np.mean(times)
            Ts.append(t_mean)
            rep_str = f" (mean of {args.repeats} rep{'s' if args.repeats != 1 else ''})"
            print(f"  n_sims={n}:   {t_mean:.2f}s{rep_str}")

        k, C, r2 = _linear_fit_r2(n_sims_list, Ts)
        print(f"\nFit: T(n) = C + k*n")
        print(f"  C = {C:.2f} s   k = {k:.2e} s/sim   R^2 = {r2:.4f}")
        if C < 0:
            print("  WARNING: Fitted C < 0 -- model may be mis-specified or noise is large. Interpret with caution.")

        # Grid-level fit: T_grid = S*(C_g + k_g*n) fitted directly from grid timings
        k_g = C_g = r2_g = None
        T_par_power_measured = None
        n_par_power = None
        if args.borrow_g_from and grid_n_sims:
            print("Full-grid fit (via proxy):")
            proxy_n_sims = n_sims_list
            proxy_Ts = []
            for n in proxy_n_sims:
                t0 = time.perf_counter()
                min_detectable_rho(
                    n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, y_params,
                    generator=proxy, n_sims=n, seed=args.seed, direction="positive",
                    calibration_mode="multipoint", n_cal=N_CAL_BENCH)
                proxy_Ts.append(time.perf_counter() - t0)
            k_proxy, C_proxy, _ = _linear_fit_r2(proxy_n_sims, proxy_Ts)

            grid_Ts_proxy = []
            for n in grid_n_sims:
                print(f"  Running proxy [{proxy}] full grid at n_sims={n}...", flush=True)
                times = []
                for _ in range(args.grid_repeats):
                    t0 = time.perf_counter()
                    run_all_scenarios(
                        generator=proxy, n_sims=n, seed=args.seed, n_jobs=1,
                        calibration_mode="multipoint", n_cal=N_CAL_BENCH)
                    times.append(time.perf_counter() - t0)
                grid_Ts_proxy.append(np.mean(times))

            # Fit grid-level model for proxy: T_grid_proxy = S*(C_gp + k_gp*n)
            grid_per_scenario = [t / N_GRID_SCENARIOS for t in grid_Ts_proxy]
            k_gp, C_gp, r2_gp = _linear_fit_r2(grid_n_sims, grid_per_scenario)
            # Scale proxy k_g to target generator via single-scenario k ratio
            ratio = k / k_proxy if k_proxy != 0 else 0
            k_g = k_gp * ratio
            C_g = C_gp * ratio
            r2_g = r2_gp
            print(f"\n  Proxy grid per-scenario fit: C_gp={C_gp:.2f}s, k_gp={k_gp:.2e}  R^2={r2_gp:.4f}")
            print(f"  Scaled to {generator}: k_g = k_gp * (k/k_proxy) = {k_gp:.2e} * {ratio:.2f} = {k_g:.2e}")
            print(f"  C_g = {C_g:.2f}s\n")
        elif grid_n_sims:
            print("Full-grid fit:")
            grid_Ts = []
            for n in grid_n_sims:
                if generator == "empirical":
                    est = N_GRID_SCENARIOS * (C + k * n)
                    est_min = est / 60
                    print(f"\n  WARNING: Empirical full grid at n_sims={n} estimated ~{est_min:.0f} min.")
                    print("  Press Enter to continue or Ctrl-C to abort.")
                    input()
                times = []
                for _ in range(args.grid_repeats):
                    t0 = time.perf_counter()
                    run_all_scenarios(
                        generator=generator, n_sims=n, seed=args.seed, n_jobs=1,
                        calibration_mode="multipoint", n_cal=N_CAL_BENCH)
                    times.append(time.perf_counter() - t0)
                t_grid = np.mean(times)
                grid_Ts.append(t_grid)
                t_per = t_grid / N_GRID_SCENARIOS
                rep_str = f" (mean of {args.grid_repeats} rep{'s' if args.grid_repeats != 1 else ''})" if args.grid_repeats != 1 else ""
                print(f"  n_sims={n}:  grid={t_grid:.1f}s  per_scenario={t_per:.2f}s{rep_str}")

            # Fit per-scenario model: T_per_scenario = C_g + k_g*n
            grid_per_scenario = [t / N_GRID_SCENARIOS for t in grid_Ts]
            if len(grid_n_sims) >= 2:
                k_g, C_g, r2_g = _linear_fit_r2(grid_n_sims, grid_per_scenario)
                print(f"\nGrid fit: T_grid = {N_GRID_SCENARIOS}*(C_g + k_g*n)")
                print(f"  C_g = {C_g:.2f} s   k_g = {k_g:.2e} s/sim   R^2 = {r2_g:.4f}")
                print(f"  vs single-scenario: C = {C:.2f}, k = {k:.2e}")
                if k_g > k:
                    print(f"  k_g/k = {k_g/k:.2f}x (grid-average scenario is harder than case {args.case})")
                elif k_g < k:
                    print(f"  k_g/k = {k_g/k:.2f}x (grid-average scenario is easier than case {args.case})")
            else:
                # Single grid point: use it to estimate a scaling factor on k
                t_per = grid_per_scenario[0]
                n_grid = grid_n_sims[0]
                t_single_pred = C + k * n_grid
                if t_single_pred > 0:
                    scale = t_per / t_single_pred
                    k_g = k * scale
                    C_g = C * scale
                    print(f"\n  Single grid point: scale factor = {scale:.2f} (grid_per_scenario / single_pred)")
                    print(f"  Scaled: C_g = {C_g:.2f}, k_g = {k_g:.2e}")
            print()

        if args.grid_parallel:
            n_par_power = (grid_n_sims[0] if grid_n_sims else n_sims_list[0])
            if not (grid_n_sims or grid_ci_params):
                print("Warming null + calibration caches before parallel grid run...", flush=True)
                warm_precomputed_null_cache(n_pre=N_PRE_BENCH, seed=args.seed)
                warm_calibration_cache(generator, y_params,
                                       calibration_mode="multipoint",
                                       n_cal=N_CAL_BENCH, seed=99)
                print("Caches warm.\n")
            print(f"Measuring full grid (n_jobs=-1) at n_sims={n_par_power} for multi-core extrapolation...", flush=True)
            par_times = []
            for _ in range(args.grid_repeats):
                t0 = time.perf_counter()
                run_all_scenarios(
                    generator=generator, n_sims=n_par_power, seed=args.seed, n_jobs=-1,
                    calibration_mode="multipoint", n_cal=N_CAL_BENCH)
                par_times.append(time.perf_counter() - t0)
            T_par_power_measured = np.mean(par_times)
            rep_str = f" (mean of {args.grid_repeats} rep{'s' if args.grid_repeats != 1 else ''})" if args.grid_repeats != 1 else ""
            print(f"  Parallel grid time = {T_par_power_measured:.1f}s{rep_str}\n")

        # Power tier predictions
        tier_labels = ["+/-0.01", "+/-0.002", "+/-0.001"]
        has_grid_fit = (k_g is not None)
        has_cold = (t_null_build is not None and t_cal_bench is not None)

        print("Tier predictions (hot-cache simulation):")
        if has_grid_fit:
            print("Tier       | Single (hot) | Grid fit (hot) | Grid extrap (hot)")
        else:
            print("Tier       | Single (hot) | Grid extrap (hot)")
        print("-" * (65 if has_grid_fit else 45))
        t_single_power = []
        for i, (n_tier, n_cal_tier) in enumerate(POWER_TIERS):
            t_single = C + k * n_tier
            t_grid_extrap = N_GRID_SCENARIOS * t_single
            t_single_power.append(t_single)
            t_grid_power[i] = t_grid_extrap
            single_str = f"{_fmt_time_sec(t_single):>12}" if t_single >= 0 else f"{'(neg)':>12}"
            row = f"{tier_labels[i]:<10} | {single_str}"
            if has_grid_fit:
                t_grid_fit = N_GRID_SCENARIOS * (C_g + k_g * n_tier)
                fit_str = f"{_fmt_time_sec(t_grid_fit):>14}" if t_grid_fit >= 0 else f"{'(neg)':>14}"
                ext_str = f"{_fmt_time_sec(t_grid_extrap):>17}" if t_grid_extrap >= 0 else f"{'(neg)':>17}"
                row += f" | {fit_str} | {ext_str}"
            else:
                ext_str = f"{_fmt_time_sec(t_grid_extrap):>18}" if t_grid_extrap >= 0 else f"{'(neg)':>18}"
                row += f" | {ext_str}"
            print(row)

        if has_cold:
            print()
            print("Cold-start tier predictions (simulation + null build + calibration build):")
            print("Tier       | Single (cold) | Grid extrap (cold)")
            print("-" * 45)
            for i, (n_tier, n_cal_tier) in enumerate(POWER_TIERS):
                null_cost = t_null_build * (N_PRE_TIERS[i] / N_PRE_BENCH)
                cal_cost = t_cal_bench * (n_cal_tier / N_CAL_BENCH)
                t_cold_single = t_single_power[i] + null_cost + cal_cost if t_single_power[i] >= 0 else -1
                t_cold_grid = t_grid_power[i] + null_cost + cal_cost if t_grid_power[i] >= 0 else -1
                single_str = f"{_fmt_time_sec(t_cold_single):>13}" if t_cold_single >= 0 else f"{'(neg)':>13}"
                grid_str = f"{_fmt_time_sec(t_cold_grid):>18}" if t_cold_grid >= 0 else f"{'(neg)':>18}"
                print(f"{tier_labels[i]:<10} | {single_str} | {grid_str}")
            print("  cache = null × (n_pre_tier/N_PRE_BENCH) + cal × (n_cal_tier/N_CAL_BENCH)")
        print()

        # Multi-core extrapolation (power full grid)
        if T_par_power_measured is not None:
            print("Multi-core extrapolation (power full grid, from measured n_jobs=-1 run):")
            print(f"  This machine: {logical_cores} logical cores. Formula: T_par_tier * (this_cores / N) / {EFFICIENCY}.")
            print("Tier       | Est. 4-core | Est. 8-core | Est. 16-core")
            print("-" * 55)
            for i, (n_tier, _) in enumerate(POWER_TIERS):
                T_par_tier = T_par_power_measured * (n_tier / n_par_power)
                e4 = T_par_tier * (logical_cores / 4) / EFFICIENCY
                e8 = T_par_tier * (logical_cores / 8) / EFFICIENCY
                e16 = T_par_tier * (logical_cores / 16) / EFFICIENCY
                print(f"{tier_labels[i]:<10} | {_fmt_time_sec(e4):>11} | {_fmt_time_sec(e8):>11} | {_fmt_time_sec(e16):>12}")
            print()
        else:
            print("Multi-core extrapolation (power full grid, ROUGH: sequential / P / 0.5):")
            print("  Run with --grid-parallel for measured n_jobs=-1 and reliable extrapolation.")
            print("Tier       | Est. 4-core | Est. 8-core | Est. 16-core")
            print("-" * 55)
            for i, (n_tier, _) in enumerate(POWER_TIERS):
                if has_grid_fit:
                    t_seq = N_GRID_SCENARIOS * (C_g + k_g * n_tier)
                else:
                    t_seq = N_GRID_SCENARIOS * (C + k * n_tier)
                if t_seq <= 0:
                    print(f"{tier_labels[i]:<10} | (neg)       | (neg)       | (neg)")
                else:
                    e4 = t_seq / 4 / EFFICIENCY
                    e8 = t_seq / 8 / EFFICIENCY
                    e16 = t_seq / 16 / EFFICIENCY
                    print(f"{tier_labels[i]:<10} | {_fmt_time_sec(e4):>11} | {_fmt_time_sec(e8):>11} | {_fmt_time_sec(e16):>12}")
            print()

    # --- CI fit ---
    if mode in ("ci", "both"):
        print(f"=== CI runtime model fit: {generator} ===\n")
        print("Fit points (single scenario, n_jobs=1):")
        x_ci = []  # n_reps * n_boot
        T_ci = []
        for (n_reps, n_boot) in ci_params:
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                bootstrap_ci_averaged(
                    n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, rho_obs, y_params,
                    generator=generator, n_reps=n_reps, n_boot=n_boot, seed=args.seed,
                    batch_bootstrap=True, calibration_mode="multipoint")
                times.append(time.perf_counter() - t0)
            t_mean = np.mean(times)
            x_ci.append(n_reps * n_boot)
            T_ci.append(t_mean)
            rep_str = f" (mean of {args.repeats} rep{'s' if args.repeats != 1 else ''})"
            print(f"  n_reps={n_reps}, n_boot={n_boot}:  {t_mean:.2f}s{rep_str}")

        k_ci, C_ci, r2_ci = _linear_fit_r2(x_ci, T_ci)
        print(f"\nFit: T_ci = C_ci + k_ci*(n_reps*n_boot)")
        print(f"  C_ci = {C_ci:.2f} s   k_ci = {k_ci:.2e} s/(rep*boot)   R^2 = {r2_ci:.4f}")
        if C_ci < 0:
            print("  WARNING: Fitted C_ci < 0 -- model may be mis-specified or noise is large. Interpret with caution.")

        # Grid-level CI fit
        k_g_ci = C_g_ci = r2_g_ci = None
        T_par_ci_measured = None
        nr_par_ci = nb_par_ci = None
        if args.borrow_g_from and grid_ci_params:
            print("Full-grid CI fit (via proxy):")
            proxy_x, proxy_T = [], []
            for (nr, nb) in ci_params:
                t0 = time.perf_counter()
                bootstrap_ci_averaged(
                    n_single, N_DISTINCT_SINGLE, DIST_TYPE_SINGLE, rho_obs, y_params,
                    generator=proxy, n_reps=nr, n_boot=nb, seed=args.seed,
                    batch_bootstrap=True, calibration_mode="multipoint")
                proxy_T.append(time.perf_counter() - t0)
                proxy_x.append(nr * nb)
            k_proxy_ci, C_proxy_ci, _ = _linear_fit_r2(proxy_x, proxy_T)

            grid_x_proxy = []
            grid_Ts_ci_proxy = []
            for (nr, nb) in grid_ci_params:
                times = []
                for _ in range(args.grid_repeats):
                    t0 = time.perf_counter()
                    run_all_ci_scenarios(
                        generator=proxy, n_reps=nr, n_boot=nb, seed=args.seed,
                        n_jobs=1, calibration_mode="multipoint", batch_bootstrap=True)
                    times.append(time.perf_counter() - t0)
                grid_Ts_ci_proxy.append(np.mean(times))
                grid_x_proxy.append(nr * nb)

            grid_per_ci = [t / N_GRID_SCENARIOS for t in grid_Ts_ci_proxy]
            k_gp_ci, C_gp_ci, r2_gp_ci = _linear_fit_r2(grid_x_proxy, grid_per_ci)
            ratio_ci = k_ci / k_proxy_ci if k_proxy_ci != 0 else 0
            k_g_ci = k_gp_ci * ratio_ci
            C_g_ci = C_gp_ci * ratio_ci
            r2_g_ci = r2_gp_ci
            print(f"\n  Proxy CI grid per-scenario fit: C_gp={C_gp_ci:.2f}s, k_gp={k_gp_ci:.2e}  R^2={r2_gp_ci:.4f}")
            print(f"  Scaled to {generator}: k_g_ci = {k_g_ci:.2e}, C_g_ci = {C_g_ci:.2f}s\n")
        elif grid_ci_params:
            print("Full-grid CI fit:")
            grid_x_ci = []
            grid_Ts_ci = []
            for (nr, nb) in grid_ci_params:
                if generator == "empirical":
                    est = N_GRID_SCENARIOS * (C_ci + k_ci * nr * nb)
                    print(f"\n  WARNING: Empirical full CI grid at n_reps={nr}, n_boot={nb} estimated ~{_fmt_time_sec(est)}.")
                    print("  Press Enter to continue or Ctrl-C to abort.")
                    input()
                times = []
                for _ in range(args.grid_repeats):
                    t0 = time.perf_counter()
                    run_all_ci_scenarios(
                        generator=generator, n_reps=nr, n_boot=nb, seed=args.seed,
                        n_jobs=1, calibration_mode="multipoint", batch_bootstrap=True)
                    times.append(time.perf_counter() - t0)
                t_grid = np.mean(times)
                grid_Ts_ci.append(t_grid)
                grid_x_ci.append(nr * nb)
                t_per = t_grid / N_GRID_SCENARIOS
                rep_str = f" (mean of {args.grid_repeats} rep{'s' if args.grid_repeats != 1 else ''})" if args.grid_repeats != 1 else ""
                print(f"  n_reps={nr}, n_boot={nb}:  grid={t_grid:.1f}s  per_scenario={t_per:.2f}s{rep_str}")

            grid_per_ci = [t / N_GRID_SCENARIOS for t in grid_Ts_ci]
            if len(grid_ci_params) >= 2:
                k_g_ci, C_g_ci, r2_g_ci = _linear_fit_r2(grid_x_ci, grid_per_ci)
                print(f"\nGrid CI fit: T_ci_grid = {N_GRID_SCENARIOS}*(C_g_ci + k_g_ci*prod)")
                print(f"  C_g_ci = {C_g_ci:.2f} s   k_g_ci = {k_g_ci:.2e} s/(rep*boot)   R^2 = {r2_g_ci:.4f}")
                print(f"  vs single-scenario: C_ci = {C_ci:.2f}, k_ci = {k_ci:.2e}")
            else:
                t_per = grid_per_ci[0]
                prod_grid = grid_x_ci[0]
                t_single_pred = C_ci + k_ci * prod_grid
                if t_single_pred > 0:
                    scale = t_per / t_single_pred
                    k_g_ci = k_ci * scale
                    C_g_ci = C_ci * scale
                    print(f"\n  Single grid point: scale factor = {scale:.2f}")
                    print(f"  Scaled: C_g_ci = {C_g_ci:.2f}, k_g_ci = {k_g_ci:.2e}")
            print()

        if args.grid_parallel:
            nr_par_ci, nb_par_ci = (grid_ci_params[0] if grid_ci_params else ci_params[0])
            if not (grid_n_sims or grid_ci_params):
                print("Warming null + calibration caches before parallel CI grid run...", flush=True)
                warm_precomputed_null_cache(n_pre=N_PRE_BENCH, seed=args.seed)
                warm_calibration_cache(generator, y_params,
                                       calibration_mode="multipoint",
                                       n_cal=N_CAL_BENCH, seed=99)
                print("Caches warm.\n")
            print(f"Measuring full CI grid (n_jobs=-1) at n_reps={nr_par_ci}, n_boot={nb_par_ci}...", flush=True)
            par_ci_times = []
            for _ in range(args.grid_repeats):
                t0 = time.perf_counter()
                run_all_ci_scenarios(
                    generator=generator, n_reps=nr_par_ci, n_boot=nb_par_ci, seed=args.seed,
                    n_jobs=-1, calibration_mode="multipoint", batch_bootstrap=True)
                par_ci_times.append(time.perf_counter() - t0)
            T_par_ci_measured = np.mean(par_ci_times)
            rep_str = f" (mean of {args.grid_repeats} rep{'s' if args.grid_repeats != 1 else ''})" if args.grid_repeats != 1 else ""
            print(f"  Parallel CI grid time = {T_par_ci_measured:.1f}s{rep_str}\n")

        # CI tier predictions
        tier_labels_ci = ["+/-0.01", "+/-0.002", "+/-0.001"]
        has_grid_ci_fit = (k_g_ci is not None)

        print("Tier predictions (CI, hot-cache simulation):")
        if has_grid_ci_fit:
            print("Tier       | Single (hot) | Grid fit (hot) | Grid extrap (hot)")
        else:
            print("Tier       | Single (hot) | Grid extrap (hot)")
        print("-" * (65 if has_grid_ci_fit else 45))
        t_single_ci = []
        for i, (nr, nb, n_cal_tier_ci) in enumerate(CI_TIERS):
            prod = nr * nb
            t_single = C_ci + k_ci * prod
            t_grid_extrap = N_GRID_SCENARIOS * t_single
            t_single_ci.append(t_single)
            t_grid_ci[i] = t_grid_extrap
            single_str = f"{_fmt_time_sec(t_single):>12}" if t_single >= 0 else f"{'(neg)':>12}"
            row = f"{tier_labels_ci[i]:<10} | {single_str}"
            if has_grid_ci_fit:
                t_grid_fit = N_GRID_SCENARIOS * (C_g_ci + k_g_ci * prod)
                fit_str = f"{_fmt_time_sec(t_grid_fit):>14}" if t_grid_fit >= 0 else f"{'(neg)':>14}"
                ext_str = f"{_fmt_time_sec(t_grid_extrap):>17}" if t_grid_extrap >= 0 else f"{'(neg)':>17}"
                row += f" | {fit_str} | {ext_str}"
            else:
                ext_str = f"{_fmt_time_sec(t_grid_extrap):>18}" if t_grid_extrap >= 0 else f"{'(neg)':>18}"
                row += f" | {ext_str}"
            print(row)

        if has_cold:
            print()
            print("Cold-start tier predictions CI (simulation + null build + calibration build):")
            print("Tier       | Single (cold) | Grid extrap (cold)")
            print("-" * 45)
            for i, (nr, nb, n_cal_tier_ci) in enumerate(CI_TIERS):
                null_cost = t_null_build * (N_PRE_TIERS[i] / N_PRE_BENCH)
                cal_cost = t_cal_bench * (n_cal_tier_ci / N_CAL_BENCH)
                t_cold_single = t_single_ci[i] + null_cost + cal_cost if t_single_ci[i] >= 0 else -1
                t_cold_grid = t_grid_ci[i] + null_cost + cal_cost if t_grid_ci[i] >= 0 else -1
                single_str = f"{_fmt_time_sec(t_cold_single):>13}" if t_cold_single >= 0 else f"{'(neg)':>13}"
                grid_str = f"{_fmt_time_sec(t_cold_grid):>18}" if t_cold_grid >= 0 else f"{'(neg)':>18}"
                print(f"{tier_labels_ci[i]:<10} | {single_str} | {grid_str}")
            print("  cache = null × (n_pre_tier/N_PRE_BENCH) + cal × (n_cal_tier/N_CAL_BENCH)")
        print()

        # Multi-core extrapolation (CI full grid)
        if T_par_ci_measured is not None and nr_par_ci is not None and nb_par_ci is not None:
            print("Multi-core extrapolation (CI full grid, from measured n_jobs=-1 run):")
            print(f"  This machine: {logical_cores} logical cores. Formula: T_par_tier * (this_cores / N) / {EFFICIENCY}.")
            print("Tier       | Est. 4-core | Est. 8-core | Est. 16-core")
            print("-" * 55)
            prod_par = nr_par_ci * nb_par_ci
            for i, (nr_tier, nb_tier, _) in enumerate(CI_TIERS):
                prod_tier = nr_tier * nb_tier
                T_par_tier = T_par_ci_measured * (prod_tier / prod_par)
                e4 = T_par_tier * (logical_cores / 4) / EFFICIENCY
                e8 = T_par_tier * (logical_cores / 8) / EFFICIENCY
                e16 = T_par_tier * (logical_cores / 16) / EFFICIENCY
                print(f"{tier_labels_ci[i]:<10} | {_fmt_time_sec(e4):>11} | {_fmt_time_sec(e8):>11} | {_fmt_time_sec(e16):>12}")
            print()
        else:
            print("Multi-core extrapolation (CI full grid, ROUGH: sequential / P / 0.5):")
            print("  Run with --grid-parallel for measured n_jobs=-1 and reliable extrapolation.")
            print("Tier       | Est. 4-core | Est. 8-core | Est. 16-core")
            print("-" * 55)
            for i, (nr_tier, nb_tier, _) in enumerate(CI_TIERS):
                prod_tier = nr_tier * nb_tier
                if has_grid_ci_fit:
                    t_seq = N_GRID_SCENARIOS * (C_g_ci + k_g_ci * prod_tier)
                else:
                    t_seq = N_GRID_SCENARIOS * (C_ci + k_ci * prod_tier)
                if t_seq <= 0:
                    print(f"{tier_labels_ci[i]:<10} | (neg)       | (neg)       | (neg)")
                else:
                    e4 = t_seq / 4 / EFFICIENCY
                    e8 = t_seq / 8 / EFFICIENCY
                    e16 = t_seq / 16 / EFFICIENCY
                    print(f"{tier_labels_ci[i]:<10} | {_fmt_time_sec(e4):>11} | {_fmt_time_sec(e8):>11} | {_fmt_time_sec(e16):>12}")
            print()

    # Combined cold-start block (only when both power and CI were fitted)
    if mode == "both" and has_cold:
        tier_labels_both = ["+/-0.01", "+/-0.002", "+/-0.001"]
        print("=== Cold-start summary: single run vs separate runs (this generator) ===")
        print('  "Single run"  = power then CI in one process — cache built once, shared.')
        print('  "Two runs"    = power and CI in separate processes — cache built twice each.')
        print()
        print(f"  {'Tier':<9} | {'Single run (cold)':>18} | {'Power-only (cold)':>18} | {'CI-only (cold)':>15} | {'Sum (two runs)':>15}")
        print("  " + "-" * 80)
        for i in range(3):
            n_cal_p = POWER_TIERS[i][1]
            n_cal_c = CI_TIERS[i][2]
            null_cost = t_null_build * (N_PRE_TIERS[i] / N_PRE_BENCH)
            cal_cost = t_cal_bench * (n_cal_p / N_CAL_BENCH)
            # Use grid extrap as the reference simulation cost (most relevant for production)
            t_p_sim = t_grid_power[i] if t_grid_power[i] >= 0 else 0
            t_c_sim = t_grid_ci[i] if t_grid_ci[i] >= 0 else 0
            cal_cost_ci = t_cal_bench * (n_cal_c / N_CAL_BENCH)
            t_single_run = t_p_sim + t_c_sim + null_cost + cal_cost
            t_pow_only = t_p_sim + null_cost + cal_cost
            t_ci_only = t_c_sim + null_cost + cal_cost_ci
            t_two_runs = t_pow_only + t_ci_only
            print(f"  {tier_labels_both[i]:<9} | {_fmt_time_sec(t_single_run):>18} | {_fmt_time_sec(t_pow_only):>18} | {_fmt_time_sec(t_ci_only):>15} | {_fmt_time_sec(t_two_runs):>15}")
        print("  (Simulation = full grid extrap. Cache = null + calibration at tier n_pre / n_cal.)")
        print()

    # Notes footer
    print("Notes:")
    print("  - T(n) = C + k*n: single-scenario model. C = fixed overhead, k = per-sim cost.")
    print("  - Grid fit: T_grid = 88*(C_g + k_g*n), fitted directly from full-grid timings.")
    print("    k_g > k when the grid-average scenario is harder than case 3 (typical).")
    print("  - Without --grid-n-sims, 'single extrap' = 88*(C + k*n) (assumes all scenarios ~ case 3).")
    print("  - R^2 < 0.99 suggests measurement noise; re-run with --repeats 3 and/or larger n_sims.")
    print("  - Null + calibration caches are pre-warmed before grid runs; cold-start columns above")
    print("    show the full first-run cost including cache build at each tier's n_pre / n_cal.")
    print("  - Grid par cold-cache: workers spawn fresh — each rebuilds calibration and null.")
    print("    Parallel would match sequential hot if workers loaded disk-precomputed caches.")
    print("  - Multi-core: without --grid-parallel, 4/8/16-core estimates are ROUGH (sequential/P/0.5).")
    print("    Use --grid-parallel to run the grid with n_jobs=-1 once for measured extrapolation.")


if __name__ == "__main__":
    main()
