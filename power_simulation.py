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

Typical runtimes (nonparametric generator, Numba JIT + precomputed null active):
  - Single scenario: ~0.5s (500 sims), ~4s (10000 sims)
  - Full grid (88 scenarios, 500 sims): ~40s sequential (warm caches), ~20s with n_jobs=4
  - Calibration is cached per (n, k, dist_type) and reused across rho values
  - Copula now uses calibration; linear has no calibration overhead
  - Pre-OPT-2 (no precomputed null): ~5s/500 sims, ~45s/10000 sims, ~6 min full grid
"""

import warnings

import numpy as np
from joblib import Parallel, delayed

from config import (CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES,
                    N_SIMS, ALPHA, TARGET_POWER, N_CAL,
                    RHO_SEARCH_POSITIVE, RHO_SEARCH_NEGATIVE,
                    POWER_SEARCH_DIRECTION, CALIBRATION_MODE,
                    VECTORIZE_DATA_GENERATION,
                    USE_PERMUTATION_PVALUE, N_PERM_DEFAULT, N_PERM_LOW_SIMS,
                    N_SIMS_THRESHOLD_FOR_N_PERM, PVALUE_PRECOMPUTED_N_PRE,
                    PVALUE_MC_ON_CACHE_MISS, EMPIRICAL_USE_PRECOMPUTED_NULL)
from power_asymptotic import get_x_counts
from data_generator import (generate_cumulative_aluminum, get_generator,
                            calibrate_rho, calibrate_rho_copula,
                            calibrate_rho_empirical, digitized_available,
                            generate_y_empirical, generate_y_empirical_batch,
                            get_pool,
                            generate_y_nonparametric, _fit_lognormal,
                            generate_cumulative_aluminum_batch,
                            generate_y_nonparametric_batch,
                            generate_y_copula_batch,
                            generate_y_linear_batch,
                            warm_calibration_cache,
                            get_calibration_cache_snapshots)
from spearman_helpers import spearman_rho_pvalue_2d, spearman_rho_2d
if USE_PERMUTATION_PVALUE:
    from permutation_pvalue import (get_precomputed_null, get_cached_null,
                                    pvalues_from_precomputed_null,
                                    pvalues_mc, _get_n_perm,
                                    warm_precomputed_null_cache,
                                    get_null_cache_snapshot)


def _init_worker_caches(null_snap, cal_snaps):
    """Initialize a worker process with pre-warmed cache snapshots.

    Must be at module top level (not nested) so loky can pickle it by
    reference. Called once per worker at startup before any tasks run.

    Parameters
    ----------
    null_snap : dict
        Snapshot of permutation_pvalue._NULL_CACHE.
    cal_snaps : dict
        Snapshot of all 6 calibration caches from data_generator, keyed
        'mp', 'mp_cop', 'mp_emp', 'sp', 'sp_cop', 'sp_emp'.
    """
    import permutation_pvalue as _pp
    import data_generator as _dg
    _pp._NULL_CACHE.update(null_snap)
    _dg._CALIBRATION_CACHE_MULTIPOINT.update(cal_snaps["mp"])
    _dg._CALIBRATION_CACHE_MULTIPOINT_COPULA.update(cal_snaps["mp_cop"])
    _dg._CALIBRATION_CACHE_MULTIPOINT_EMP.update(cal_snaps["mp_emp"])
    _dg._CALIBRATION_CACHE.update(cal_snaps["sp"])
    _dg._CALIBRATION_CACHE_COPULA.update(cal_snaps["sp_cop"])
    _dg._CALIBRATION_CACHE_EMP.update(cal_snaps["sp_emp"])


def estimate_power(n, n_distinct, distribution_type, rho_s, y_params,
                   generator="nonparametric", n_sims=None, alpha=None,
                   all_distinct=False, seed=None, freq_dict=None,
                   calibration_mode=None, vectorize=None, n_cal=None):
    """Estimate power of a two-sided Spearman test via Monte Carlo.

    Parameters
    ----------
    generator : str
        'copula', 'linear', or 'nonparametric'.
    freq_dict : dict or None
        Custom frequency dictionary.  When provided with distribution_type
        "custom", used instead of FREQ_DICT.
    n_cal : int or None
        Calibration samples per bisection.  When None, uses config N_CAL.

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
    if vectorize is None:
        vectorize = VECTORIZE_DATA_GENERATION
    if n_cal is None:
        n_cal = N_CAL

    if generator == "empirical" and not digitized_available():
        warnings.warn(
            "Digitized data not available (data/digitized.py missing or failed to import). "
            "Falling back to nonparametric generator.",
            UserWarning,
            stacklevel=2,
        )
        generator = "nonparametric"

    gen_fn = get_generator(generator)
    rng = np.random.default_rng(seed)

    cal_rho = None
    ln_params = _fit_lognormal(y_params["median"], y_params["iqr"])
    if generator == "nonparametric":
        cal_rho = calibrate_rho(
            n, n_distinct, distribution_type, rho_s, y_params,
            all_distinct=all_distinct, freq_dict=freq_dict,
            calibration_mode=calibration_mode, n_cal=n_cal)
    elif generator == "copula":
        cal_rho = calibrate_rho_copula(
            n, n_distinct, distribution_type, rho_s, y_params,
            all_distinct=all_distinct, freq_dict=freq_dict, n_cal=n_cal,
            calibration_mode=calibration_mode)
    elif generator == "empirical":
        pool = get_pool(n)
        cal_rho = calibrate_rho_empirical(
            n, n_distinct, distribution_type, rho_s, pool,
            all_distinct=all_distinct, freq_dict=freq_dict,
            calibration_mode=calibration_mode, n_cal=n_cal)

    if vectorize:
        x_all = generate_cumulative_aluminum_batch(
            n_sims, n, n_distinct, distribution_type=distribution_type,
            all_distinct=all_distinct, freq_dict=freq_dict, rng=rng)
        if generator == "nonparametric":
            y_all = generate_y_nonparametric_batch(
                x_all, rho_s, y_params, rng=rng,
                _calibrated_rho=cal_rho, _ln_params=ln_params)
        elif generator == "copula":
            rho_in = cal_rho if cal_rho is not None else rho_s
            y_all = generate_y_copula_batch(x_all, rho_in, y_params, rng=rng)
        elif generator == "empirical":
            y_all = generate_y_empirical_batch(
                x_all, rho_s, y_params, rng=rng,
                _calibrated_rho=cal_rho)
        else:
            y_all = generate_y_linear_batch(x_all, rho_s, y_params, rng=rng)
    else:
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
            elif generator == "empirical":
                yi = generate_y_empirical(xi, rho_s, y_params, rng=rng,
                                          _calibrated_rho=cal_rho)
            else:
                yi = gen_fn(xi, rho_s, y_params, rng=rng)
            x_all[i] = xi
            y_all[i] = yi

    if not USE_PERMUTATION_PVALUE:
        _, pvals = spearman_rho_pvalue_2d(x_all, y_all, n)
        return np.sum(pvals < alpha) / n_sims

    # Permutation-based p-values
    # Non-empirical generators (nonparametric, copula, linear) all produce
    # continuous y with no ties.  Under H0, the y-ranks are a uniform
    # permutation of {1,...,n} regardless of generator, so the permutation
    # null of Spearman rho depends only on n and the x tie structure.  The
    # cache key (n, all_distinct, tuple(x_counts)) correctly captures this:
    # generator is intentionally excluded, and the cached null is shared
    # across all non-empirical generators for the same scenario.
    # Empirical also uses the precomputed null by default (EMPIRICAL_USE_PRECOMPUTED_NULL=True).
    # Approximation error from ignoring y-ties is < 10^-5 on p-values; see README.
    use_precomputed = (generator != "empirical") or EMPIRICAL_USE_PRECOMPUTED_NULL
    if use_precomputed:
        x_counts = get_x_counts(n, n_distinct, distribution_type=distribution_type,
                                all_distinct=all_distinct, freq_dict=freq_dict)
        if PVALUE_MC_ON_CACHE_MISS:
            sorted_abs_null = get_cached_null(n, all_distinct, x_counts)
        else:
            sorted_abs_null = get_precomputed_null(
                n, all_distinct, x_counts, n_pre=PVALUE_PRECOMPUTED_N_PRE, rng=rng)

        if sorted_abs_null is not None:
            # Cache hit: fast binary-search p-values
            rhos_obs = spearman_rho_2d(x_all, y_all)
            pvals = pvalues_from_precomputed_null(rhos_obs, sorted_abs_null)
        else:
            # Cache miss with PVALUE_MC_ON_CACHE_MISS=True: fall back to MC
            n_perm = _get_n_perm(n_sims)
            reject, pvals, rhos_obs = pvalues_mc(x_all, y_all, n_perm, alpha, rng)
    else:
        # Empirical with MC: per-dataset Monte Carlo (exact but much slower)
        n_perm = _get_n_perm(n_sims)
        reject, pvals, rhos_obs = pvalues_mc(x_all, y_all, n_perm, alpha, rng)

    return np.sum(pvals < alpha) / n_sims


# When n_sims <= this value we do not emit the boundary warning (benchmark/timing runs).
# Production tiers use n_sims >= 2220 (POWER_TIERS), so they still get the warning.
N_SIMS_BOUNDARY_WARN_THRESHOLD = 50


def _check_and_warn_boundary(result, lo_bound, hi_bound, boundary_tolerance=1e-4,
                             n_sims=None):
    """Emit UserWarning if bisection result is within tolerance of a search boundary.

    When n_sims is given and <= N_SIMS_BOUNDARY_WARN_THRESHOLD, the warning is
    suppressed so that benchmark/timing runs (e.g. n_sims=50) do not emit it,
    including when run in parallel workers where the caller's warning filter
    does not apply.
    """
    if n_sims is not None and n_sims <= N_SIMS_BOUNDARY_WARN_THRESHOLD:
        return
    if abs(result - lo_bound) < boundary_tolerance or abs(result - hi_bound) < boundary_tolerance:
        warnings.warn(
            f"min_detectable_rho hit search boundary ({result:.4f}). "
            "Consider widening RHO_SEARCH bounds.",
            UserWarning, stacklevel=3)
    return


def min_detectable_rho(n, n_distinct, distribution_type, y_params,
                       generator="nonparametric", target_power=None,
                       n_sims=None, alpha=None, all_distinct=False,
                       direction="positive", seed=None, tolerance=0.49e-4,
                       freq_dict=None, calibration_mode=None, n_cal=None):
    """Binary search for the minimum |rho| detectable at *target_power*.

    Parameters
    ----------
    direction : str
        'positive' searches RHO_SEARCH_POSITIVE; 'negative' searches RHO_SEARCH_NEGATIVE (config).
    freq_dict : dict or None
        Custom frequency dictionary for distribution_type "custom".
    n_cal : int or None
        Calibration samples per bisection.  When None, uses config N_CAL.
    """
    if target_power is None:
        target_power = TARGET_POWER
    if n_sims is None:
        n_sims = N_SIMS
    if alpha is None:
        alpha = ALPHA
    if calibration_mode is None:
        calibration_mode = CALIBRATION_MODE
    if n_cal is None:
        n_cal = N_CAL

    if direction == "positive":
        lo, hi = RHO_SEARCH_POSITIVE
    else:
        lo, hi = RHO_SEARCH_NEGATIVE
    lo_bound, hi_bound = lo, hi

    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        pw = estimate_power(n, n_distinct, distribution_type, mid, y_params,
                            generator=generator, n_sims=n_sims, alpha=alpha,
                            all_distinct=all_distinct, seed=seed,
                            freq_dict=freq_dict,
                            calibration_mode=calibration_mode, n_cal=n_cal)
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

    result = (lo + hi) / 2.0
    _check_and_warn_boundary(result, lo_bound, hi_bound, n_sims=n_sims)
    return result


def _search_directions(case_id):
    """Return list of search directions for a given case."""
    case = CASES[case_id]
    if POWER_SEARCH_DIRECTION == "both_directions":
        return ["positive", "negative"]
    return ["negative"] if case["observed_rho"] < 0 else ["positive"]


def _power_one_scenario(case_id, n, y_params, k, dt, all_distinct,
                        direction, generator, n_sims, seed,
                        calibration_mode=None, n_cal=None):
    """Run min_detectable_rho for a single (case, k, dt, direction)."""
    md = min_detectable_rho(
        n, k, dt, y_params, generator=generator,
        n_sims=n_sims, all_distinct=all_distinct, direction=direction,
        seed=seed, calibration_mode=calibration_mode, n_cal=n_cal)
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
                      n_jobs=1, calibration_mode=None, n_cal=None,
                      pre_warm=True):
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
    n_cal : int or None
        Calibration samples per bisection.  When None, uses config N_CAL.
    pre_warm : bool
        When True (default) and n_jobs != 1, warm the null and calibration
        caches in the main process before snapshotting them into workers.
        Re-warming an already-warm cache costs only dict lookups (~88 checks,
        microseconds).  Set False only if you have pre-warmed manually and
        want to avoid even that overhead.

    Returns
    -------
    list of dict
    """
    if n_sims is None:
        n_sims = N_SIMS
    if n_cal is None:
        n_cal = N_CAL

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
                                     calibration_mode, n_cal))
                    scenario_idx += 1

        # This loops runs the all-distinct scenarios for the given generators
        for d in directions:
            sc_seed = (seed + scenario_idx) if seed is not None else None
            scenarios.append((case_id, n, y_params, n, None, True,
                              d, generator, n_sims, sc_seed,
                              calibration_mode, n_cal))
            scenario_idx += 1

    if n_jobs == 1:
        return [_power_one_scenario(*args) for args in scenarios]

    # Parallel path: warm caches in main process then inject into each worker.
    # Workers (loky spawn on Windows) start with empty module-level dicts;
    # the initializer restores them from the main-process snapshot before
    # any tasks are dispatched.
    _eff_cal_mode = calibration_mode if calibration_mode is not None else CALIBRATION_MODE
    if pre_warm:
        if USE_PERMUTATION_PVALUE:
            warm_precomputed_null_cache(cases=_cases,
                                        n_distinct_values=_nvals,
                                        dist_types=_dtypes)
        warm_calibration_cache(generator,
                               cases=_cases,
                               n_distinct_values=_nvals,
                               dist_types=_dtypes,
                               calibration_mode=_eff_cal_mode,
                               n_cal=n_cal)

    null_snap = get_null_cache_snapshot() if USE_PERMUTATION_PVALUE else {}
    cal_snaps = get_calibration_cache_snapshots()

    return Parallel(
        n_jobs=n_jobs,
        initializer=_init_worker_caches,
        initargs=(null_snap, cal_snaps),
    )(delayed(_power_one_scenario)(*args) for args in scenarios)
