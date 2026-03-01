"""
Main orchestrator for Spearman power / CI simulations.

Runs Monte Carlo (nonparametric, copula, and/or linear), asymptotic analyses,
then produces CSV outputs and a console summary.
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import os
import time
import argparse
import warnings

import config

_numba_pre = argparse.ArgumentParser(add_help=False)
_numba_pre.add_argument("--no-numba", action="store_true")
_numba_pre.add_argument("--numba", action="store_true")
_pre_args, _ = _numba_pre.parse_known_args()
if _pre_args.no_numba:
    config.USE_NUMBA = False

import pandas as pd

from config import (CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES,
                    N_SIMS, N_BOOTSTRAP, ALPHA, TARGET_POWER,
                    POWER_SEARCH_DIRECTION, ASYMPTOTIC_TIE_CORRECTION_MODE,
                    CALIBRATION_MODE)
from power_simulation import run_all_scenarios as mc_scenarios
from power_asymptotic import (asymptotic_results, get_x_counts)
from confidence_interval_calculator import run_all_ci_scenarios
from data_generator import digitized_available
from table_outputs import (build_min_detectable_table, save_min_detectable_table,
                           build_ci_table, save_ci_table,
                           build_all_distinct_table, save_all_distinct_table,
                           print_summary)


def _search_directions(case_id):
    case = CASES[case_id]
    if POWER_SEARCH_DIRECTION == "both_directions":
        return ["positive", "negative"]
    return ["negative"] if case["observed_rho"] < 0 else ["positive"]


def run_asymptotic_power(tie_correction_mode=None, cases=None,
                         n_distinct_values=None, dist_types=None):
    """Run asymptotic min-detectable-rho for all (or filtered) scenarios."""
    if tie_correction_mode is None:
        tie_correction_mode = ASYMPTOTIC_TIE_CORRECTION_MODE

    _cases = {k: v for k, v in CASES.items()
              if cases is None or k in cases}
    _nvals = n_distinct_values if n_distinct_values else N_DISTINCT_VALUES
    _dtypes = dist_types if dist_types else DISTRIBUTION_TYPES

    results = []

    for case_id, case in _cases.items():
        n = case["n"]
        rho_obs = case["observed_rho"]
        directions = _search_directions(case_id)

        for k in _nvals:
            for dt in _dtypes:
                x_counts = get_x_counts(n, k, distribution_type=dt)
                for d in directions:
                    ar = asymptotic_results(
                        n, rho_obs, TARGET_POWER, ALPHA,
                        x_counts, d, tie_correction_mode)
                    for label, vals in ar.items():
                        results.append({
                            "case": case_id,
                            "n": n,
                            "n_distinct": k,
                            "dist_type": dt,
                            "direction": d,
                            "min_detectable_rho": vals["min_detectable_rho"],
                            "method": f"asymptotic_{label}",
                            "all_distinct": False,
                        })

        x_counts = get_x_counts(n, n, all_distinct=True)
        for d in directions:
            ar = asymptotic_results(
                n, rho_obs, TARGET_POWER, ALPHA,
                x_counts, d, tie_correction_mode)
            for label, vals in ar.items():
                results.append({
                    "case": case_id,
                    "n": n,
                    "n_distinct": n,
                    "dist_type": "all_distinct",
                    "direction": d,
                    "min_detectable_rho": vals["min_detectable_rho"],
                    "method": f"asymptotic_{label}",
                    "all_distinct": True,
                })

    return results


def _log(msg):
    print(msg, flush=True)


def main(n_sims=None, n_boot=None, n_reps=None, seed=None, outdir="results",
         tie_correction_mode=None, skip_linear=False, skip_copula=False,
         skip_nonparametric=False, skip_empirical=False, cases=None,
         n_distinct_values=None, dist_types=None, n_jobs=1, use_numba=None,
         calibration_mode=None):
    if use_numba is not None:
        config.USE_NUMBA = use_numba
    if n_sims is None:
        n_sims = N_SIMS
    if n_boot is None:
        n_boot = N_BOOTSTRAP
    if n_reps is None:
        n_reps = 200
    if tie_correction_mode is None:
        tie_correction_mode = ASYMPTOTIC_TIE_CORRECTION_MODE
    if calibration_mode is None:
        calibration_mode = CALIBRATION_MODE

    os.makedirs(outdir, exist_ok=True)

    filter_kw = dict(cases=cases, n_distinct_values=n_distinct_values,
                     dist_types=dist_types)
    all_power = []

    mc_methods = []
    if not skip_nonparametric:
        mc_methods.append("nonparametric")
    if not skip_copula:
        mc_methods.append("copula")
    if not skip_linear:
        mc_methods.append("linear")
    if not skip_empirical:
        mc_methods.append("empirical")

    if "empirical" in mc_methods and not digitized_available():
        warnings.warn(
            "Digitized data not available (data/digitized.py missing or failed to import). "
            "Empirical generator unavailable; using fallback.",
            UserWarning,
            stacklevel=2,
        )
        if len(mc_methods) == 1:
            mc_methods = ["nonparametric"]
        else:
            mc_methods = [g for g in mc_methods if g != "empirical"]

    for gen in mc_methods:
        _log(f"Running {gen} Monte Carlo ({n_sims} sims per scenario)...")
        t0 = time.time()
        pw = mc_scenarios(generator=gen, n_sims=n_sims, seed=seed,
                          n_jobs=n_jobs, calibration_mode=calibration_mode,
                          **filter_kw)
        _log(f"  {gen.title()} done in {time.time() - t0:.1f}s  "
             f"({len(pw)} scenarios)")
        all_power.extend(pw)

    _log("Running asymptotic power analysis...")
    t0 = time.time()
    asym_pow = run_asymptotic_power(tie_correction_mode=tie_correction_mode,
                                    **filter_kw)
    _log(f"  Asymptotic done in {time.time() - t0:.1f}s  "
         f"({len(asym_pow)} scenarios)")
    all_power.extend(asym_pow)

    ci_generators = [g for g in mc_methods if g != "linear"] or ["nonparametric"]
    all_ci = []
    for gen in ci_generators:
        _log(f"Running {gen} averaged bootstrap CIs "
             f"({n_reps} reps x {n_boot} resamples)...")
        t0 = time.time()
        ci = run_all_ci_scenarios(
            generator=gen, n_reps=n_reps, n_boot=n_boot,
            tie_correction_mode=tie_correction_mode, seed=seed,
            n_jobs=n_jobs, calibration_mode=calibration_mode)
        _log(f"  {gen.title()} CIs done in {time.time() - t0:.1f}s  "
             f"({len(ci)} scenarios)")
        all_ci.extend(ci)

    power_df = build_min_detectable_table(all_power)
    ci_df = build_ci_table(all_ci)
    all_distinct_df = build_all_distinct_table(all_power, all_ci)

    p1 = save_min_detectable_table(power_df, os.path.join(outdir, "min_detectable_rho.csv"))
    p2 = save_ci_table(ci_df, os.path.join(outdir, "confidence_intervals.csv"))
    p3 = save_all_distinct_table(all_distinct_df, os.path.join(outdir, "all_distinct_summary.csv"))

    _log(f"\nCSV outputs saved to: {p1}, {p2}, {p3}")

    print_summary(power_df, ci_df, all_distinct_df)

    return power_df, ci_df, all_distinct_df


def _parse_int_list(s):
    """Parse comma-separated integers, e.g. '1,3' -> [1, 3]."""
    return [int(x.strip()) for x in s.split(",")]


def _parse_str_list(s):
    """Parse comma-separated strings."""
    return [x.strip() for x in s.split(",")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Spearman power / CI simulations for Karwowski 2018.")
    parser.add_argument("--n-sims", type=int, default=None,
                        help=f"Monte Carlo iterations (default: {N_SIMS})")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--outdir", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--tie-correction",
                        choices=["with_tie_correction",
                                 "without_tie_correction", "both"],
                        default=None,
                        help="Asymptotic tie-correction mode")
    parser.add_argument("--skip-linear", action="store_true",
                        help="Skip linear Monte Carlo")
    parser.add_argument("--skip-copula", action="store_true",
                        help="Skip copula Monte Carlo")
    parser.add_argument("--skip-nonparametric", action="store_true",
                        help="Skip nonparametric Monte Carlo")
    parser.add_argument("--skip-empirical", action="store_true",
                        help="Skip empirical Monte Carlo")
    parser.add_argument("--n-boot", type=int, default=None,
                        help=f"Bootstrap resamples per dataset (default: {N_BOOTSTRAP})")
    parser.add_argument("--n-reps", type=int, default=None,
                        help="Datasets to average over for bootstrap CIs (default: 200)")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated case IDs to run (e.g. 1,3)")
    parser.add_argument("--n-distinct", type=str, default=None,
                        help="Comma-separated k values (e.g. 4,10)")
    parser.add_argument("--dist-types", type=str, default=None,
                        help="Comma-separated dist types (e.g. even,heavy_center)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs for scenario-level parallelism "
                             "(1=sequential, -1=all cores, default: 1)")
    parser.add_argument("--calibration-mode",
                        choices=["multipoint", "single"],
                        default=None,
                        help="Calibration mode (default: multipoint)")
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
    main(n_sims=args.n_sims, n_boot=args.n_boot, n_reps=args.n_reps,
         seed=args.seed, outdir=args.outdir,
         tie_correction_mode=args.tie_correction,
         skip_linear=args.skip_linear, skip_copula=args.skip_copula,
         skip_nonparametric=args.skip_nonparametric,
         skip_empirical=args.skip_empirical,
         cases=_parse_int_list(args.cases) if args.cases else None,
         n_distinct_values=_parse_int_list(args.n_distinct) if args.n_distinct else None,
         dist_types=_parse_str_list(args.dist_types) if args.dist_types else None,
         n_jobs=args.n_jobs, use_numba=_use,
         calibration_mode=args.calibration_mode)
