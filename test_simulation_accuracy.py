"""
Test whether each data generator achieves the target Spearman correlation.

For each (N, k, distribution, generator, rho_target) combination, simulates
n_sims datasets, computes the Spearman rho of each, and flags scenarios
where the mean realised rho deviates from the target by more than a
configurable threshold (default 0.01).

Typical runtimes:
  - Quick check (1 case, 1 k, 3 generators, 50 sims): ~30s
  - Filtered (1 case, 2 k, nonparametric only, 200 sims): ~30s
  - Full sweep (all cases, all k, all generators, 200 sims): ~10-20 min

Programmatic usage
------------------
    from test_simulation_accuracy import main
    df = main(n_sims=50, cases=[3], n_distinct_values=[4])
    df = main(n_sims=50, cases=[3], custom_freq=[(3, [19, 18, 18, 18])])

CLI usage
---------
Quick check (few sims, worst-case scenario):
    python test_simulation_accuracy.py --n-sims 50 --case 3 --n-distinct 4

Full sweep:
    python test_simulation_accuracy.py --n-sims 200

Custom frequency distribution:
    python test_simulation_accuracy.py --case 3 --freq 19,18,18,18 --n-sims 50

Specific generators:
    python test_simulation_accuracy.py --generators copula,nonparametric
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES, CALIBRATION_MODE
from data_generator import (generate_cumulative_aluminum, get_generator,
                            calibrate_rho, calibrate_rho_copula,
                            calibrate_rho_empirical, generate_y_empirical,
                            get_pool, generate_y_nonparametric, _fit_lognormal)


DEFAULT_RHO_TARGETS = [0.30, -0.30]
FLAG_THRESHOLD = 0.01


def test_scenario(n, n_distinct, distribution_type, rho_target, generator,
                  y_params, n_sims=50, all_distinct=False, seed=None,
                  freq_dict=None, n_cal=300, calibration_mode=None):
    """Generate *n_sims* datasets and return accuracy statistics.

    Parameters
    ----------
    freq_dict : dict or None
        Custom frequency dictionary for distribution_type "custom".

    Returns
    -------
    dict with mean_rho, std_rho, diff, flagged.
    """
    if calibration_mode is None:
        calibration_mode = CALIBRATION_MODE

    gen_fn = get_generator(generator)
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_sims)

    cal_rho = None
    ln_params = _fit_lognormal(y_params["median"], y_params["iqr"])
    if generator == "nonparametric":
        cal_rho = calibrate_rho(
            n, n_distinct, distribution_type, rho_target, y_params,
            all_distinct=all_distinct, freq_dict=freq_dict, n_cal=n_cal,
            calibration_mode=calibration_mode)
    elif generator == "copula":
        cal_rho = calibrate_rho_copula(
            n, n_distinct, distribution_type, rho_target, y_params,
            all_distinct=all_distinct, freq_dict=freq_dict, n_cal=n_cal)
    elif generator == "empirical":
        pool = get_pool(n)
        cal_rho = calibrate_rho_empirical(
            n, n_distinct, distribution_type, rho_target, pool,
            all_distinct=all_distinct, freq_dict=freq_dict, n_cal=n_cal,
            calibration_mode=calibration_mode)

    for i in range(n_sims):
        x = generate_cumulative_aluminum(
            n, n_distinct, distribution_type=distribution_type,
            all_distinct=all_distinct, freq_dict=freq_dict, rng=rng)
        if generator == "nonparametric":
            y = generate_y_nonparametric(x, rho_target, y_params, rng=rng,
                                          _calibrated_rho=cal_rho,
                                          _ln_params=ln_params)
        elif generator == "copula":
            rho_in = cal_rho if cal_rho is not None else rho_target
            y = gen_fn(x, rho_in, y_params, rng=rng)
        elif generator == "empirical":
            y = generate_y_empirical(x, rho_target, y_params, rng=rng,
                                      _calibrated_rho=cal_rho)
        else:
            y = gen_fn(x, rho_target, y_params, rng=rng)
        rhos[i], _ = spearmanr(x, y)

    mean_rho = float(np.mean(rhos))
    diff = mean_rho - rho_target
    return {
        "mean_rho": mean_rho,
        "std_rho": float(np.std(rhos, ddof=1)),
        "min_rho": float(np.min(rhos)),
        "max_rho": float(np.max(rhos)),
        "diff": diff,
        "flagged": abs(diff) > FLAG_THRESHOLD,
    }


def run_accuracy_tests(generators=None, rho_targets=None, n_sims=50,
                       cases=None, n_distinct_values=None, dist_types=None,
                       seed=None, custom_freq=None, n_cal=300,
                       calibration_mode=None):
    """Test accuracy across all (or filtered) scenarios.

    Parameters
    ----------
    custom_freq : list of (case_id, freq_list) or None
        Custom frequency scenarios.  Each freq_list is a list of counts
        that must sum to case["n"].  Adds these scenarios in addition
        to (or instead of, if cases/n_distinct/dist_types filter to empty)
        the standard ones.

    Returns
    -------
    pd.DataFrame
    """
    if generators is None:
        generators = ["nonparametric", "copula", "linear"]
    if rho_targets is None:
        rho_targets = DEFAULT_RHO_TARGETS

    _cases = {k: v for k, v in CASES.items()
              if cases is None or k in cases}
    _nvals = n_distinct_values or N_DISTINCT_VALUES
    _dtypes = dist_types or DISTRIBUTION_TYPES

    rows = []
    n_standard = (len(_cases) * (len(_nvals) * len(_dtypes) + 1)
                 * len(generators) * len(rho_targets))
    n_custom = 0
    if custom_freq:
        n_custom = len(custom_freq) * len(generators) * len(rho_targets)
    total = n_standard + n_custom
    done = 0

    for case_id, case in _cases.items():
        n = case["n"]
        y_params = {"median": case["median"], "iqr": case["iqr"],
                    "range": case["range"]}

        scenarios = []
        for k in _nvals:
            for dt in _dtypes:
                scenarios.append((k, dt, False))
        scenarios.append((n, "all_distinct", True))

        for k, dt, ad in scenarios:
            for gen in generators:
                for rho_t in rho_targets:
                    result = test_scenario(
                        n, k, dt if not ad else None, rho_t, gen,
                        y_params, n_sims=n_sims, all_distinct=ad, seed=seed,
                        n_cal=n_cal, calibration_mode=calibration_mode)
                    rows.append({
                        "case": case_id,
                        "n": n,
                        "n_distinct": k,
                        "dist_type": dt,
                        "generator": gen,
                        "target_rho": rho_t,
                        "all_distinct": ad,
                        **result,
                    })
                    done += 1
                    print(f"\r  {done}/{total} scenarios tested", end="",
                          flush=True)

    if custom_freq:
        for case_id, freq_list in custom_freq:
            case = CASES[case_id]
            n = case["n"]
            k = len(freq_list)
            if sum(freq_list) != n:
                raise ValueError(f"Custom freq for case {case_id} sums to "
                                 f"{sum(freq_list)}, expected {n}")
            freq_dict = {n: {k: {"custom": freq_list}}}
            y_params = {"median": case["median"], "iqr": case["iqr"],
                        "range": case["range"]}
            for gen in generators:
                for rho_t in rho_targets:
                    result = test_scenario(
                        n, k, "custom", rho_t, gen, y_params,
                        n_sims=n_sims, all_distinct=False, seed=seed,
                        freq_dict=freq_dict, n_cal=n_cal,
                        calibration_mode=calibration_mode)
                    rows.append({
                        "case": case_id,
                        "n": n,
                        "n_distinct": k,
                        "dist_type": "custom",
                        "generator": gen,
                        "target_rho": rho_t,
                        "all_distinct": False,
                        **result,
                    })
                    done += 1
                    print(f"\r  {done}/{total} scenarios tested", end="",
                          flush=True)
    print()

    df = pd.DataFrame(rows)
    df = df.sort_values(["case", "generator", "n_distinct", "dist_type",
                         "target_rho"]).reset_index(drop=True)
    return df


def print_report(df):
    """Print summary of flagged scenarios."""
    flagged = df[df["flagged"]]
    n_total = len(df)
    n_flag = len(flagged)

    print(f"\n{'='*80}")
    print(f"ACCURACY TEST SUMMARY  ({n_flag}/{n_total} scenarios flagged, "
          f"threshold={FLAG_THRESHOLD})")
    print(f"{'='*80}")

    if n_flag == 0:
        print("All scenarios within threshold.")
    else:
        cols = ["case", "n_distinct", "dist_type", "generator",
                "target_rho", "mean_rho", "diff"]
        print(flagged[cols].to_string(index=False))

    print(f"\nPer-generator summary:")
    for gen in df["generator"].unique():
        sub = df[df["generator"] == gen]
        n_f = sub["flagged"].sum()
        mean_abs_diff = sub["diff"].abs().mean()
        max_abs_diff = sub["diff"].abs().max()
        print(f"  {gen:15s}: {n_f:3d}/{len(sub)} flagged, "
              f"mean|diff|={mean_abs_diff:.4f}, max|diff|={max_abs_diff:.4f}")


def _parse_list(s, cast=str):
    return [cast(x.strip()) for x in s.split(",")]


def main(n_sims=50, generators=None, rho_targets=None, cases=None,
         n_distinct_values=None, dist_types=None, custom_freq=None,
         seed=42, threshold=0.01, outfile=None, verbose=True, n_cal=300,
         calibration_mode=None):
    """Run accuracy tests.  Callable without CLI.

    Parameters
    ----------
    n_sims : int
        Simulations per scenario.
    generators : list of str or None
        e.g. ["nonparametric", "copula", "linear"].  None = all.
    rho_targets : list of float or None
        Target rhos to test.  None = [0.30, -0.30].
    cases : list of int or None
        Case IDs to include.  None = all.
    n_distinct_values : list of int or None
        k values.  None = all (4-10).
    dist_types : list of str or None
        Distribution types.  None = all.
    custom_freq : list of (case_id, freq_list) or None
        Custom scenarios.  Each freq_list must sum to case["n"].
    seed : int
        RNG seed.
    threshold : float
        Flag when |mean_rho - target| > threshold.
    outfile : str or None
        If set, save results to CSV.
    verbose : bool
        If True, print progress and summary.

    Returns
    -------
    pd.DataFrame
    """
    global FLAG_THRESHOLD
    FLAG_THRESHOLD = threshold

    if verbose:
        print(f"Running accuracy tests ({n_sims} sims/scenario, "
              f"n_cal={n_cal}, threshold={threshold})...")
    df = run_accuracy_tests(
        generators=generators, rho_targets=rho_targets, n_sims=n_sims,
        cases=cases, n_distinct_values=n_distinct_values, dist_types=dist_types,
        seed=seed, custom_freq=custom_freq, n_cal=n_cal,
        calibration_mode=calibration_mode)

    if verbose:
        print_report(df)

    if outfile:
        df.to_csv(outfile, index=False, float_format="%.4f")
        if verbose:
            print(f"\nResults saved to {outfile}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test simulation accuracy for Spearman generators.")
    parser.add_argument("--n-sims", type=int, default=50,
                        help="Simulations per scenario (default: 50)")
    parser.add_argument("--generators", type=str, default=None,
                        help="Comma-separated generators (default: all)")
    parser.add_argument("--rho-targets", type=str, default=None,
                        help="Comma-separated rho targets (default: 0.30,-0.30)")
    parser.add_argument("--case", type=str, default=None,
                        help="Comma-separated case IDs")
    parser.add_argument("--n-distinct", type=str, default=None,
                        help="Comma-separated k values")
    parser.add_argument("--dist-types", type=str, default=None,
                        help="Comma-separated dist types")
    parser.add_argument("--freq", type=str, default=None,
                        help="Custom frequency counts, comma-separated (e.g. 19,18,18,18). "
                             "Requires --case. Adds this custom scenario to the test.")
    parser.add_argument("--n-cal", type=int, default=300,
                        help="Calibration samples per bisection (default: 300)")
    parser.add_argument("--calibration-mode",
                        choices=["multipoint", "single"],
                        default=None,
                        help="Calibration mode (default: multipoint)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outfile", type=str, default=None,
                        help="Save results to CSV")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Flagging threshold (default: 0.01)")
    args = parser.parse_args()

    gens = _parse_list(args.generators) if args.generators else None
    rhos = (_parse_list(args.rho_targets, float)
            if args.rho_targets else None)
    cases_arg = _parse_list(args.case, int) if args.case else None
    nvals = _parse_list(args.n_distinct, int) if args.n_distinct else None
    dtypes = _parse_list(args.dist_types) if args.dist_types else None

    custom_freq_arg = None
    if args.freq:
        if not cases_arg or len(cases_arg) == 0:
            parser.error("--freq requires --case")
        freq_list = [int(x.strip()) for x in args.freq.split(",")]
        n_sum = sum(freq_list)
        matching = [c for c in cases_arg if CASES[c]["n"] == n_sum]
        if not matching:
            parser.error(f"--freq sums to {n_sum}; no case in {cases_arg} has that n")
        custom_freq_arg = [(matching[0], freq_list)]

    main(n_sims=args.n_sims, generators=gens, rho_targets=rhos,
         cases=cases_arg, n_distinct_values=nvals, dist_types=dtypes,
         custom_freq=custom_freq_arg, seed=args.seed, threshold=args.threshold,
         outfile=args.outfile, n_cal=args.n_cal,
         calibration_mode=args.calibration_mode)
