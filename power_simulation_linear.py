"""
Monte Carlo power simulation using the linear data generator.

Structure mirrors power_simulation_copula.py for direct method comparison.
The linear generator produces y = a + b*x + noise on a log scale, calibrated
so that the Spearman rho between x and y approximates the target value.

.. note::
   This is a legacy module retained for backwards compatibility.  The
   preferred entry point is ``power_simulation.py`` with ``generator="linear"``.

Typical runtimes (linear, no calibration overhead):
  - Single scenario, 500 sims:  ~8s
  - Full grid (88 scenarios), 10000 sims: ~20-40 min
"""

import numpy as np
from scipy.stats import spearmanr

from config import (CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES,
                    N_SIMS, ALPHA, TARGET_POWER,
                    POWER_SEARCH_DIRECTION)
from data_generator import generate_cumulative_aluminum, generate_y_linear


def estimate_power(n, n_distinct, distribution_type, rho_s, y_params,
                   n_sims=None, alpha=None, all_distinct=False, seed=None):
    """Estimate power of a two-sided Spearman test via Monte Carlo (linear y)."""
    if n_sims is None:
        n_sims = N_SIMS
    if alpha is None:
        alpha = ALPHA

    rng = np.random.default_rng(seed)
    rejects = 0

    for _ in range(n_sims):
        x = generate_cumulative_aluminum(
            n, n_distinct, distribution_type=distribution_type,
            all_distinct=all_distinct, rng=rng)
        y = generate_y_linear(x, rho_s, y_params, rng=rng)
        _, p = spearmanr(x, y)
        if p < alpha:
            rejects += 1

    return rejects / n_sims


def min_detectable_rho(n, n_distinct, distribution_type, y_params,
                       target_power=None, n_sims=None, alpha=None,
                       all_distinct=False, direction="positive", seed=None, tolerance = 0.49e-4):
    """Binary search for the minimum |rho| detectable at *target_power*."""
    if target_power is None:
        target_power = TARGET_POWER
    if n_sims is None:
        n_sims = N_SIMS
    if alpha is None:
        alpha = ALPHA

    if direction == "positive":
        lo, hi = 0.25, 0.42
    else:
        lo, hi = -0.42, -0.25

    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        pw = estimate_power(n, n_distinct, distribution_type, mid, y_params,
                            n_sims=n_sims, alpha=alpha,
                            all_distinct=all_distinct, seed=seed)
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


def run_all_scenarios(n_sims=None, seed=None):
    """Run linear power analysis for all tied and all-distinct scenarios.

    Returns
    -------
    list of dict
        One entry per (case, n_distinct, dist_type, direction) combination.
    """
    if n_sims is None:
        n_sims = N_SIMS

    results = []

    for case_id, case in CASES.items():
        n = case["n"]
        y_params = {"median": case["median"], "iqr": case["iqr"],
                    "range": case["range"]}
        directions = _search_directions(case_id)

        for k in N_DISTINCT_VALUES:
            for dt in DISTRIBUTION_TYPES:
                for d in directions:
                    md = min_detectable_rho(
                        n, k, dt, y_params, n_sims=n_sims,
                        direction=d, seed=seed)
                    results.append({
                        "case": case_id,
                        "n": n,
                        "n_distinct": k,
                        "dist_type": dt,
                        "direction": d,
                        "min_detectable_rho": md,
                        "method": "linear",
                        "all_distinct": False,
                    })

        for d in directions:
            md = min_detectable_rho(
                n, n, None, y_params, n_sims=n_sims,
                all_distinct=True, direction=d, seed=seed)
            results.append({
                "case": case_id,
                "n": n,
                "n_distinct": n,
                "dist_type": "all_distinct",
                "direction": d,
                "min_detectable_rho": md,
                "method": "linear",
                "all_distinct": True,
            })

    return results
