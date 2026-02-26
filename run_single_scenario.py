"""
Run power and/or CI analysis for a single scenario.

Allows quick testing of a specific (case, k, distribution) combination
without running the full simulation grid.

Typical runtimes (nonparametric, power only):
  - n_sims=50:   ~5s (includes one-time calibration ~3s)
  - n_sims=500:  ~8s
  - n_sims=10000: ~60s
  Adding CIs (n_reps=20, n_boot=500): +5-10s
  Adding CIs (n_reps=200, n_boot=10000): +3-5 min

Programmatic usage
------------------
    from run_single_scenario import main
    result = main(case=3, n_distinct=4, dist_type="heavy_center", n_sims=500)
    result = main(case=3, freq=[19, 18, 18, 18], power_only=True)

CLI usage
---------
Power only (fast):
    python run_single_scenario.py --case 3 --n-distinct 4 --dist-type heavy_center --n-sims 500 --power-only

CI only:
    python run_single_scenario.py --case 1 --n-distinct 4 --dist-type even --ci-only --n-reps 20 --n-boot 500

All-distinct baseline:
    python run_single_scenario.py --case 1 --all-distinct --n-sims 500

Custom frequency distribution:
    python run_single_scenario.py --case 3 --freq 19,18,18,18 --n-sims 500

Skip specific methods:
    python run_single_scenario.py --case 3 --n-distinct 4 --dist-type even --skip-copula --skip-linear
"""

import argparse
import time

import config

_numba_pre = argparse.ArgumentParser(add_help=False)
_numba_pre.add_argument("--no-numba", action="store_true")
_numba_pre.add_argument("--numba", action="store_true")
_pre_args, _ = _numba_pre.parse_known_args()
if _pre_args.no_numba:
    config.USE_NUMBA = False

import numpy as np
from scipy.stats import spearmanr

from config import CASES, N_SIMS, N_BOOTSTRAP, ALPHA, TARGET_POWER, ASYMPTOTIC_TIE_CORRECTION_MODE
from data_generator import generate_cumulative_aluminum, get_generator
from power_simulation import estimate_power, min_detectable_rho, _search_directions
from confidence_interval_calculator import bootstrap_ci_averaged, _asymptotic_ci_results
from power_asymptotic import (asymptotic_results, get_x_counts,
                              min_detectable_rho_asymptotic, asymptotic_ci)


def run_power(case_id, n_distinct, dist_type, generators, n_sims, seed,
              all_distinct=False, freq_dict=None, x_counts=None):
    """Run min-detectable-rho for a single scenario."""
    case = CASES[case_id]
    n = case["n"]
    y_params = {"median": case["median"], "iqr": case["iqr"],
                "range": case["range"]}
    directions = _search_directions(case_id)

    results = []

    for gen in generators:
        for d in directions:
            t0 = time.time()
            md = min_detectable_rho(
                n, n_distinct if not all_distinct else n,
                dist_type if not all_distinct else None,
                y_params, generator=gen, n_sims=n_sims,
                all_distinct=all_distinct, direction=d, seed=seed,
                freq_dict=freq_dict)
            elapsed = time.time() - t0
            results.append({
                "method": gen,
                "direction": d,
                "min_detectable_rho": md,
                "elapsed_s": elapsed,
            })

    _x_counts = x_counts if x_counts is not None else get_x_counts(
        n, n_distinct if not all_distinct else n,
        distribution_type=dist_type if not all_distinct else None,
        all_distinct=all_distinct, freq_dict=freq_dict)
    for d in directions:
        md = min_detectable_rho_asymptotic(
            n, target_power=TARGET_POWER, alpha=ALPHA,
            x_counts=_x_counts, direction=d)
        results.append({
            "method": "asymptotic",
            "direction": d,
            "min_detectable_rho": md,
            "elapsed_s": 0.0,
        })

    return results


def run_ci(case_id, n_distinct, dist_type, generators, n_reps, n_boot,
           seed, all_distinct=False, tie_correction_mode=None,
           freq_dict=None, x_counts=None):
    """Run CI analysis for a single scenario."""
    if tie_correction_mode is None:
        tie_correction_mode = ASYMPTOTIC_TIE_CORRECTION_MODE

    case = CASES[case_id]
    n = case["n"]
    rho_obs = case["observed_rho"]
    y_params = {"median": case["median"], "iqr": case["iqr"],
                "range": case["range"]}

    results = []

    for gen in generators:
        t0 = time.time()
        boot = bootstrap_ci_averaged(
            n, n_distinct if not all_distinct else n,
            dist_type if not all_distinct else None,
            rho_obs, y_params, generator=gen, n_reps=n_reps,
            n_boot=n_boot, all_distinct=all_distinct, seed=seed,
            freq_dict=freq_dict)
        elapsed = time.time() - t0
        results.append({
            "method": f"bootstrap_{gen}",
            "observed_rho": rho_obs,
            "ci_lower": boot["ci_lower"],
            "ci_upper": boot["ci_upper"],
            "mean_rho_hat": boot["mean_rho_hat"],
            "elapsed_s": elapsed,
        })

    _x_counts = x_counts if x_counts is not None else get_x_counts(
        n, n_distinct if not all_distinct else n,
        distribution_type=dist_type if not all_distinct else None,
        all_distinct=all_distinct, freq_dict=freq_dict)
    lo, hi = asymptotic_ci(rho_obs, n, x_counts=_x_counts, tie_correction=True)
    results.append({
        "method": "asymptotic_tc",
        "observed_rho": rho_obs,
        "ci_lower": lo,
        "ci_upper": hi,
        "mean_rho_hat": rho_obs,
        "elapsed_s": 0.0,
    })

    return results


def print_results(scenario_desc, power_results, ci_results):
    """Pretty-print results to console."""
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_desc}")
    print(f"{'='*70}")

    if power_results:
        print("\nMinimum Detectable Rho (80% power, alpha=0.05):")
        print(f"  {'Method':20s} {'Direction':10s} {'Min |rho|':>10s} {'Time':>8s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}")
        for r in power_results:
            print(f"  {r['method']:20s} {r['direction']:10s} "
                  f"{r['min_detectable_rho']:10.4f} {r['elapsed_s']:7.1f}s")

    if ci_results:
        print("\nConfidence Intervals for Observed Rho:")
        print(f"  {'Method':25s} {'rho_obs':>8s} {'CI lower':>10s} {'CI upper':>10s} {'Time':>8s}")
        print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
        for r in ci_results:
            print(f"  {r['method']:25s} {r['observed_rho']:8.4f} "
                  f"{r['ci_lower']:10.4f} {r['ci_upper']:10.4f} "
                  f"{r['elapsed_s']:7.1f}s")
    print()


def _parse_list(s, cast=str):
    return [cast(x.strip()) for x in s.split(",")]


def main(case, n_distinct=None, dist_type=None, freq=None, all_distinct=False,
         n_sims=500, n_boot=500, n_reps=20, seed=42, power_only=False,
         ci_only=False, skip_copula=False, skip_linear=False,
         skip_nonparametric=False, verbose=True, use_numba=None):
    """Run power and/or CI for a single scenario.  Callable without CLI.

    Parameters
    ----------
    case : int
        Case ID (1-4).
    n_distinct : int or None
        Number of distinct x-values (4-10).  Ignored if freq or all_distinct.
    dist_type : str or None
        Distribution type (even, heavy_tail, heavy_center).  Ignored if freq or all_distinct.
    freq : list of int, or str, or None
        Custom frequency counts.  If str, parsed as comma-separated.
        Must sum to case's n.  Overrides n_distinct and dist_type.
    all_distinct : bool
        Use all-distinct x-values.
    n_sims, n_boot, n_reps, seed : int
        Simulation parameters.
    power_only, ci_only : bool
        Run only power or only CI.
    skip_copula, skip_linear, skip_nonparametric : bool
        Exclude Monte Carlo methods.
    verbose : bool
        If True, print results to console.

    Returns
    -------
    dict with keys 'power_results', 'ci_results', 'description'.
    """
    if use_numba is not None:
        config.USE_NUMBA = use_numba
    case_data = CASES[case]
    n = case_data["n"]

    if freq is not None:
        if isinstance(freq, str):
            freq_list = [int(x.strip()) for x in freq.split(",")]
        else:
            freq_list = list(freq)
        if sum(freq_list) != n:
            raise ValueError(f"freq sums to {sum(freq_list)}, case {case} has n={n}")
        custom_freq_dict = {n: {len(freq_list): {"custom": freq_list}}}
        custom_x_counts = np.array(freq_list)
        n_distinct = len(freq_list)
        dist_type = "custom"
        use_custom_freq = True
    else:
        custom_freq_dict = None
        custom_x_counts = None
        use_custom_freq = False
        if not all_distinct and (n_distinct is None or dist_type is None):
            raise ValueError("n_distinct and dist_type required unless all_distinct or freq")

    generators = []
    if not skip_nonparametric:
        generators.append("nonparametric")
    if not skip_copula:
        generators.append("copula")
    if not skip_linear:
        generators.append("linear")
    if not generators:
        generators = ["nonparametric"]

    if all_distinct:
        desc = f"Case {case} ({case_data['label']}), N={n}, all distinct"
    elif use_custom_freq:
        desc = (f"Case {case} ({case_data['label']}), N={n}, "
                f"custom freq {freq_list}")
    else:
        desc = (f"Case {case} ({case_data['label']}), N={n}, "
                f"k={n_distinct}, {dist_type}")

    power_res = None
    ci_res = None

    if not ci_only:
        power_res = run_power(
            case, n_distinct, dist_type, generators, n_sims, seed,
            all_distinct=all_distinct, freq_dict=custom_freq_dict,
            x_counts=custom_x_counts)

    if not power_only:
        ci_res = run_ci(
            case, n_distinct, dist_type, generators, n_reps, n_boot, seed,
            all_distinct=all_distinct, freq_dict=custom_freq_dict,
            x_counts=custom_x_counts)

    if verbose:
        print_results(desc, power_res, ci_res)

    return {"power_results": power_res, "ci_results": ci_res, "description": desc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run power/CI for a single scenario.")
    parser.add_argument("--case", type=int, required=True,
                        help="Case ID (1-4)")
    parser.add_argument("--n-distinct", type=int, default=None,
                        help="Number of distinct x-values (4-10)")
    parser.add_argument("--dist-type", type=str, default=None,
                        help="Distribution type (even, heavy_tail, heavy_center)")
    parser.add_argument("--freq", type=str, default=None,
                        help="Custom frequency counts, comma-separated (e.g. 19,18,18,18). "
                             "Must sum to case's n. Overrides --n-distinct and --dist-type.")
    parser.add_argument("--all-distinct", action="store_true",
                        help="Use all-distinct x-values")
    parser.add_argument("--n-sims", type=int, default=500,
                        help="Monte Carlo sims for power (default: 500)")
    parser.add_argument("--n-boot", type=int, default=500,
                        help="Bootstrap resamples per dataset (default: 500)")
    parser.add_argument("--n-reps", type=int, default=20,
                        help="Datasets to average for bootstrap CI (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--power-only", action="store_true",
                        help="Run power analysis only")
    parser.add_argument("--ci-only", action="store_true",
                        help="Run CI analysis only")
    parser.add_argument("--skip-copula", action="store_true")
    parser.add_argument("--skip-linear", action="store_true")
    parser.add_argument("--skip-nonparametric", action="store_true")
    parser.add_argument("--no-numba", action="store_true",
                        help="Disable Numba JIT (use pure NumPy fallback)")
    parser.add_argument("--numba", action="store_true",
                        help="Enable Numba JIT (default when installed)")
    args = parser.parse_args()

    _use = None
    if args.no_numba:
        _use = False
    elif args.numba:
        _use = True
    main(case=args.case, n_distinct=args.n_distinct, dist_type=args.dist_type,
         freq=args.freq, all_distinct=args.all_distinct, n_sims=args.n_sims,
         n_boot=args.n_boot, n_reps=args.n_reps, seed=args.seed,
         power_only=args.power_only, ci_only=args.ci_only,
         skip_copula=args.skip_copula, skip_linear=args.skip_linear,
         skip_nonparametric=args.skip_nonparametric, use_numba=_use)
