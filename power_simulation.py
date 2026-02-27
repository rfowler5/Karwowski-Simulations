"""
Unified Monte Carlo power simulation supporting all y-generators.

Estimates power for a two-sided Spearman test (H0: rho = 0) via simulation,
then uses bisection to find the minimum detectable |rho| at a target power
level (default 80 %).

Replaces the generator-specific modules (power_simulation_copula.py and
power_simulation_linear.py) with a single parameterised implementation.
The old modules are still importable for backwards compatibility but
delegate to this one internally.

Monte Carlo precision and runtime guidance
------------------------------------------
Power estimate SE ~ sqrt(p(1-p) / n_sims).  At p = 0.80:
  - 10 000 sims -> SE ~ 0.004,  95 % margin ~ +/-0.008
  - 50 000 sims -> SE ~ 0.0018, 95 % margin ~ +/-0.004

Typical runtimes (nonparametric generator):
  - Single scenario: ~5s (500 sims), ~45s (10000 sims); includes calibration
  - Full grid (88 scenarios, 500 sims): ~6 min sequential, ~3 min with n_jobs=4
  - Calibration is cached per (n, k, dist_type) and reused across rho values
  - Copula now uses calibration; linear has no calibration overhead
"""

import numpy as np
from joblib import Parallel, delayed

from config import (CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES,
                    N_SIMS, ALPHA, TARGET_POWER,
                    POWER_SEARCH_DIRECTION, CALIBRATION_MODE)
from data_generator import (generate_cumulative_aluminum, get_generator,
                            calibrate_rho, calibrate_rho_copula,
                            generate_y_nonparametric, _fit_lognormal)
from spearman_helpers import spearman_rho_pvalue_2d


def estimate_power(n, n_distinct, distribution_type, rho_s, y_params,
                   generator="nonparametric", n_sims=None, alpha=None,
                   all_distinct=False, seed=None, freq_dict=None,
                   calibration_mode=None):
    """Estimate power of a two-sided Spearman test via Monte Carlo.

    Parameters
    ----------
    generator : str
        'copula', 'linear', or 'nonparametric'.
    freq_dict : dict or None
        Custom frequency dictionary.  When provided with distribution_type
        "custom", used instead of FREQ_DICT.

    Returns
    -------
    float
        Estimated power (proportion of simulations with p < alpha).
    """
    if n_sims is None:
        n_sims = N_SIMS
    if alpha is None:
        alpha = ALPHA
    if calibration_mode is None:
        calibration_mode = CALIBRATION_MODE

    gen_fn = get_generator(generator)
    rng = np.random.default_rng(seed)

    cal_rho = None
    ln_params = _fit_lognormal(y_params["median"], y_params["iqr"])
    if generator == "nonparametric":
        cal_rho = calibrate_rho(
            n, n_distinct, distribution_type, rho_s, y_params,
            all_distinct=all_distinct, freq_dict=freq_dict,
            calibration_mode=calibration_mode)
    elif generator == "copula":
        cal_rho = calibrate_rho_copula(
            n, n_distinct, distribution_type, rho_s, y_params,
            all_distinct=all_distinct, freq_dict=freq_dict)

    x_all = np.empty((n_sims, n))
    y_all = np.empty((n_sims, n))
    for i in range(n_sims):
        xi = generate_cumulative_aluminum(
            n, n_distinct, distribution_type=distribution_type,
            all_distinct=all_distinct, freq_dict=freq_dict, rng=rng)
        if generator == "nonparametric":
            yi = generate_y_nonparametric(xi, rho_s, y_params, rng=rng,
                                          _calibrated_rho=cal_rho,
                                          _ln_params=ln_params)
        elif generator == "copula":
            rho_in = cal_rho if cal_rho is not None else rho_s
            yi = gen_fn(xi, rho_in, y_params, rng=rng)
        else:
            yi = gen_fn(xi, rho_s, y_params, rng=rng)
        x_all[i] = xi
        y_all[i] = yi

    _, pvals = spearman_rho_pvalue_2d(x_all, y_all, n)
    return np.sum(pvals < alpha) / n_sims


def min_detectable_rho(n, n_distinct, distribution_type, y_params,
                       generator="nonparametric", target_power=None,
                       n_sims=None, alpha=None, all_distinct=False,
                       direction="positive", seed=None, tolerance=0.49e-4,
                       freq_dict=None, calibration_mode=None):
    """Binary search for the minimum |rho| detectable at *target_power*.

    Parameters
    ----------
    direction : str
        'positive' searches [0.25, 0.42]; 'negative' searches [-0.42, -0.25].
    freq_dict : dict or None
        Custom frequency dictionary for distribution_type "custom".
    """
    if target_power is None:
        target_power = TARGET_POWER
    if n_sims is None:
        n_sims = N_SIMS
    if alpha is None:
        alpha = ALPHA
    if calibration_mode is None:
        calibration_mode = CALIBRATION_MODE

    if direction == "positive":
        lo, hi = 0.25, 0.42
    else:
        lo, hi = -0.42, -0.25

    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        pw = estimate_power(n, n_distinct, distribution_type, mid, y_params,
                            generator=generator, n_sims=n_sims, alpha=alpha,
                            all_distinct=all_distinct, seed=seed,
                            freq_dict=freq_dict,
                            calibration_mode=calibration_mode)
        if direction == "positive":
            if pw < target_power:
                lo = mid
            else:
                hi = mid
        else:
            if pw < target_power:
                hi = mid
            else:
                lo = mid

    return (lo + hi) / 2.0


def _search_directions(case_id):
    """Return list of search directions for a given case."""
    case = CASES[case_id]
    if POWER_SEARCH_DIRECTION == "both_directions":
        return ["positive", "negative"]
    return ["negative"] if case["observed_rho"] < 0 else ["positive"]


def _power_one_scenario(case_id, n, y_params, k, dt, all_distinct,
                        direction, generator, n_sims, seed,
                        calibration_mode=None):
    """Run min_detectable_rho for a single (case, k, dt, direction)."""
    md = min_detectable_rho(
        n, k, dt, y_params, generator=generator,
        n_sims=n_sims, all_distinct=all_distinct, direction=direction,
        seed=seed, calibration_mode=calibration_mode)
    return {
        "case": case_id,
        "n": n,
        "n_distinct": k,
        "dist_type": "all_distinct" if all_distinct else dt,
        "direction": direction,
        "min_detectable_rho": md,
        "method": generator,
        "all_distinct": all_distinct,
    }


def run_all_scenarios(generator="nonparametric", n_sims=None, seed=None,
                      cases=None, n_distinct_values=None, dist_types=None,
                      n_jobs=1, calibration_mode=None):
    """Run power analysis for all (or filtered) scenarios.

    Parameters
    ----------
    generator : str
        'copula', 'linear', or 'nonparametric'.
    cases : list of int or None
        If given, restrict to these case IDs.
    n_distinct_values : list of int or None
        If given, restrict to these k-values.
    dist_types : list of str or None
        If given, restrict to these distribution types.
    n_jobs : int
        Number of parallel jobs (1 = sequential, -1 = all cores).

    Returns
    -------
    list of dict
    """
    if n_sims is None:
        n_sims = N_SIMS

    _cases = {k: v for k, v in CASES.items()
              if cases is None or k in cases}
    _nvals = n_distinct_values if n_distinct_values else N_DISTINCT_VALUES
    _dtypes = dist_types if dist_types else DISTRIBUTION_TYPES

    scenarios = []
    scenario_idx = 0
    for case_id, case in _cases.items():
        n = case["n"]
        y_params = {"median": case["median"], "iqr": case["iqr"],
                    "range": case["range"]}
        directions = _search_directions(case_id)

        for k in _nvals:
            for dt in _dtypes:
                for d in directions:
                    sc_seed = (seed + scenario_idx) if seed is not None else None
                    scenarios.append((case_id, n, y_params, k, dt, False,
                                     d, generator, n_sims, sc_seed,
                                     calibration_mode))
                    scenario_idx += 1

        for d in directions:
            sc_seed = (seed + scenario_idx) if seed is not None else None
            scenarios.append((case_id, n, y_params, n, None, True,
                              d, generator, n_sims, sc_seed,
                              calibration_mode))
            scenario_idx += 1

    if n_jobs == 1:
        return [_power_one_scenario(*args) for args in scenarios]

    # n_jobs=-1 uses all available cores
    return Parallel(n_jobs=n_jobs)(
        delayed(_power_one_scenario)(*args) for args in scenarios)
