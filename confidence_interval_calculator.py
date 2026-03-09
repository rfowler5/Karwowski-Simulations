"""
Bootstrap confidence intervals for Spearman correlation.

Compatible with all data generators (copula, linear, nonparametric).
Also delegates to power_asymptotic.asymptotic_ci for asymptotic CIs.

Bootstrap method choice
-----------------------
This module uses bootstrap_ci_averaged (averaging CI endpoints over many
simulated datasets) rather than bootstrap_ci_simulated (one dataset).
The single-realization method is unsuitable here because we are simulating
data *from a model*, not resampling observed data.  A single simulated
dataset has rho_hat that may differ substantially from the target rho
(SE ~ 0.11 at n=80), so its bootstrap CI is centered around a random
point rather than the observed correlation.  Averaging over n_reps
independent datasets gives CI endpoints that converge to the expected
CI under the assumed model, properly centered at the target rho.

The single-realization function (bootstrap_ci_simulated) is retained for
diagnostic use -- e.g. to illustrate single-study CI variability -- but
is not used in the main analysis pipeline.

Interpreting tie-corrected vs non-corrected CIs:
  Tie correction widens the CI when x has many ties (fewer distinct values).
  When the two CIs diverge noticeably, the asymptotic approximation is under
  stress and simulation-based CIs should be preferred as the reference.

Typical runtimes (per scenario):
  - Single scenario: ~12s (200 reps x 1000 boot)
  - Full grid (88 scenarios): ~16 min sequential, ~8 min with n_jobs=4
  - Calibration for nonparametric adds ~3s per unique (n, k, dist_type),
    cached for all subsequent rho values within that scenario.

n_reps and second-decimal reliability:
  Inter-rep SD of CI endpoints ≈ 0.13 (conservative worst case with ties, N=73).
  Analytical: SD(endpoint) = (1 - endpoint^2) * sqrt(1.06/(n-3)) * FHP_factor.
  SE of mean = SD/√n_reps. n_reps=200 → SE ≈ 0.009 (borderline for 2-decimal
  precision). SE < 0.005 needs n_reps ≈ 650+; SE < 0.0025 (strong confidence)
  needs ≈ 2600+. See README "Bootstrap CI" for details.

Memory: batch bootstrap path
-----------------------------
The batch path allocates bootstrap index arrays of shape (n_boot, chunk, n)
where chunk = _BATCH_BOOTSTRAP_CHUNK_SIZE (default 2000).  Peak memory per
chunk ≈ n_boot × chunk × n × 4 bytes (int32); at n_boot=500, chunk=2000, n=82
this is ~328 MB.  n_reps is processed in sequential chunks so total n_reps
(e.g. 129,700 at the ±0.001 tier) does not cause a single large allocation.
"""

import warnings

import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel, delayed

from config import (CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES,
                    N_BOOTSTRAP, ALPHA, ASYMPTOTIC_TIE_CORRECTION_MODE,
                    CALIBRATION_MODE, VECTORIZE_DATA_GENERATION,
                    BATCH_CI_BOOTSTRAP, N_CAL)
from data_generator import (generate_cumulative_aluminum, generate_y_copula,
                            digitized_available, generate_y_empirical,
                            generate_y_empirical_batch, generate_y_linear,
                            generate_y_nonparametric, get_generator,
                            calibrate_rho, calibrate_rho_copula,
                            calibrate_rho_empirical, get_pool,
                            _fit_lognormal,
                            generate_cumulative_aluminum_batch,
                            generate_y_nonparametric_batch,
                            generate_y_copula_batch,
                            generate_y_linear_batch,
                            warm_calibration_cache,
                            get_calibration_cache_snapshots)
from power_asymptotic import asymptotic_ci, get_x_counts
from spearman_helpers import (spearman_rho_2d, _fast_spearman_rho,
                              _bootstrap_rhos_jit, _batch_bootstrap_rhos_jit,
                              use_numba)

# Maximum number of n_reps processed per JIT call in the batch bootstrap path.
# Each call allocates (n_boot × chunk × n) int32 indices; at the default values
# (n_boot=500, chunk=2000, n≈82) this is ~328 MB.  Keeps memory bounded
# regardless of total n_reps (e.g. 129,700 at the ±0.001 tier without chunking
# would need ~20 GB).  Override via the private _boot_chunk_size parameter of
# bootstrap_ci_averaged (for testing).
_BATCH_BOOTSTRAP_CHUNK_SIZE = 2000


def _init_ci_worker_caches(cal_snaps):
    """Initialize a CI worker process with pre-warmed calibration cache snapshots.

    Must be at module top level (not nested) so loky can pickle it by
    reference. Called once per worker at startup before any tasks run.
    CI workers never use the permutation null cache, so only calibration
    caches are passed (saves 34-352 MB per worker vs power workers).

    Parameters
    ----------
    cal_snaps : dict
        Snapshot of all 6 calibration caches from data_generator, keyed
        'mp', 'mp_cop', 'mp_emp', 'sp', 'sp_cop', 'sp_emp'.
    """
    import data_generator as _dg
    _dg._CALIBRATION_CACHE_MULTIPOINT.update(cal_snaps["mp"])
    _dg._CALIBRATION_CACHE_MULTIPOINT_COPULA.update(cal_snaps["mp_cop"])
    _dg._CALIBRATION_CACHE_MULTIPOINT_EMP.update(cal_snaps["mp_emp"])
    _dg._CALIBRATION_CACHE.update(cal_snaps["sp"])
    _dg._CALIBRATION_CACHE_COPULA.update(cal_snaps["sp_cop"])
    _dg._CALIBRATION_CACHE_EMP.update(cal_snaps["sp_emp"])


# ---------------------------------------------------------------------------
# Bootstrap CI (core resampling from a single dataset)
# ---------------------------------------------------------------------------

def bootstrap_ci_single(x, y, rho_obs, n_boot=None, alpha=None, rng=None):
    """Percentile bootstrap CI for Spearman rho from a single (x, y) sample.

    Parameters
    ----------
    x, y : ndarray
        Paired observations.
    rho_obs : float
        Point estimate (used only for book-keeping; the bootstrap
        resamples compute their own rho).
    n_boot : int
        Number of bootstrap replicates.
    alpha : float
        Significance level for the CI.
    rng : numpy.random.Generator

    Returns
    -------
    (lower, upper) : tuple of float
    """
    if n_boot is None:
        n_boot = N_BOOTSTRAP
    if alpha is None:
        alpha = ALPHA
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    boot_idx = rng.integers(0, n, size=(n_boot, n), dtype=np.int32)

    if use_numba() and _bootstrap_rhos_jit is not None:
        boot_rhos = _bootstrap_rhos_jit(
            x.astype(np.float64), y.astype(np.float64), boot_idx)
    else:
        x_boot = x[boot_idx]
        y_boot = y[boot_idx]
        boot_rhos = spearman_rho_2d(x_boot, y_boot)

    lo = np.nanpercentile(boot_rhos, 100 * alpha / 2)
    hi = np.nanpercentile(boot_rhos, 100 * (1 - alpha / 2))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Single-realization bootstrap (retained for diagnostics only)
# ---------------------------------------------------------------------------

def bootstrap_ci_simulated(n, n_distinct, distribution_type, rho_s, y_params,
                            generator="nonparametric", n_boot=None, alpha=None,
                            all_distinct=False, seed=None):
    """Generate one (x, y) dataset and compute a bootstrap CI.

    .. warning::
       This function bootstraps from a *single* simulated dataset, so its
       CI is centered around that realization's rho_hat (which may differ
       substantially from rho_s).  Use ``bootstrap_ci_averaged`` for
       stable, properly-centered CIs in the main analysis.

    Returns
    -------
    dict with keys 'ci_lower', 'ci_upper', 'rho_hat'.
    """
    if n_boot is None:
        n_boot = N_BOOTSTRAP
    if alpha is None:
        alpha = ALPHA

    rng = np.random.default_rng(seed)
    x = generate_cumulative_aluminum(
        n, n_distinct, distribution_type=distribution_type,
        all_distinct=all_distinct, rng=rng)

    if generator == "empirical" and not digitized_available():
        warnings.warn(
            "Digitized data not available (data/digitized.py missing or failed to import). "
            "Falling back to nonparametric generator.",
            UserWarning,
            stacklevel=2,
        )
        generator = "nonparametric"

    if generator == "empirical":
        pool = get_pool(n)
        cal_rho = calibrate_rho_empirical(n, n_distinct, distribution_type,
                                          rho_s, pool, all_distinct=all_distinct)
        y = generate_y_empirical(x, rho_s, y_params, rng=rng,
                                  _calibrated_rho=cal_rho)
    else:
        gen_fn = get_generator(generator)
        y = gen_fn(x, rho_s, y_params, rng=rng)

    rho_hat, _ = spearmanr(x, y)
    lo, hi = bootstrap_ci_single(x, y, rho_hat, n_boot=n_boot,
                                  alpha=alpha, rng=rng)
    return {"ci_lower": lo, "ci_upper": hi, "rho_hat": rho_hat}


# ---------------------------------------------------------------------------
# Averaged bootstrap (used in main pipeline)
# ---------------------------------------------------------------------------

def bootstrap_ci_averaged(n, n_distinct, distribution_type, rho_s, y_params,
                           generator="nonparametric", n_reps=200, n_boot=None,
                           alpha=None, all_distinct=False, seed=None,
                           freq_dict=None, calibration_mode=None, vectorize=None,
                           batch_bootstrap=None, n_cal=None,
                           _boot_chunk_size=None):
    """Average bootstrap CI endpoints over *n_reps* independent datasets.

    This is the correct approach when working with simulated (not observed)
    data: it estimates the *expected* bootstrap CI under the assumed model,
    smoothing out the large single-sample variability in rho_hat (SE ≈ 0.11
    at n≈80).  The resulting CI is properly centered near the target rho.

    Parameters
    ----------
    freq_dict : dict or None
        Custom frequency dictionary for distribution_type "custom".
    n_cal : int or None
        Calibration samples.  When None, uses config N_CAL (default 300).
        Calibration noise shifts CI endpoints but not CI width.  For
        accurate absolute endpoint values, set n_cal to the tier value
        from config.CI_TIERS (e.g. 1000 for ±0.01).  See
        docs/UNCERTAINTY_BUDGET.md Part 2 for the full budget.
    _boot_chunk_size : int or None
        Private parameter for testing.  Overrides _BATCH_BOOTSTRAP_CHUNK_SIZE
        for the batch bootstrap path.  Pass a small value (e.g. 2) to force
        chunking even at small n_reps.  Has no effect when batch_bootstrap is
        False or vectorize is False.

    Returns
    -------
    dict with keys 'ci_lower', 'ci_upper' (averaged endpoints),
    'ci_lower_sd', 'ci_upper_sd', and 'mean_rho_hat'.

    Notes
    -----
    n_reps=200 gives SE of mean ≈ 0.007 (inter-rep SD ~0.10). For stronger
    confidence in the second decimal, use n_reps=400+ (SE < 0.005) or 1600+
    (SE < 0.0025). See module docstring and README.
    """
    if n_boot is None:
        n_boot = N_BOOTSTRAP
    if alpha is None:
        alpha = ALPHA
    if calibration_mode is None:
        calibration_mode = CALIBRATION_MODE
    if vectorize is None:
        vectorize = VECTORIZE_DATA_GENERATION
    if batch_bootstrap is None:
        batch_bootstrap = BATCH_CI_BOOTSTRAP
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

    if batch_bootstrap and not vectorize:
        warnings.warn(
            "batch_bootstrap=True requires VECTORIZE_DATA_GENERATION=True; "
            "falling back to per-rep loop.",
            UserWarning,
            stacklevel=2,
        )

    # Use separate RNG streams for data generation and bootstrap resampling so
    # that the n_reps datasets are identical regardless of n_boot.  A shared
    # RNG would advance by n_boot*n extra steps per rep inside
    # bootstrap_ci_single, coupling the dataset draws to n_boot and making
    # results non-comparable across different n_boot values.
    #
    # n_boot choice: For 2-decimal CI precision (third decimal trustworthy for
    # rounding, e.g., knowing if can trust 0.345 and so know that rounds to 0.35), n_boot=1000 is recommended. The SE of the bootstrap quantile
    # estimate scales as ~1/sqrt(n_boot); with n_reps averaging, the combined
    # SE on the mean endpoint with sufficient n_reps is ~0.001-0.002 for n_boot=1000. n_boot=500
    # can differ by ~0.002-0.003 from higher values; n_boot=2000+ adds
    # little beyond 1000.
    
    # However, when n_reps=200, inter-rep variability (SE ~0.007) dominates bootstrap 
    # quantile noise; n_boot=500 is then sufficient and ~2x faster.
    # See README "Bootstrap CI" for verification.
    ss = np.random.SeedSequence(seed)
    data_rng, boot_rng = [np.random.default_rng(s) for s in ss.spawn(2)]

    lowers = np.empty(n_reps)
    uppers = np.empty(n_reps)
    rho_hats = np.empty(n_reps)

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

    if batch_bootstrap and vectorize:
        # --- NEW BATCH PATH ---
        # Step 1: Pre-generate all datasets using data_rng
        x_all = generate_cumulative_aluminum_batch(
            n_reps, n, n_distinct, distribution_type=distribution_type,
            all_distinct=all_distinct, freq_dict=freq_dict, rng=data_rng)

        if generator == "nonparametric":
            y_all = generate_y_nonparametric_batch(
                x_all, rho_s, y_params, rng=data_rng,
                _calibrated_rho=cal_rho, _ln_params=ln_params)
        elif generator == "copula":
            rho_in = cal_rho if cal_rho is not None else rho_s
            y_all = generate_y_copula_batch(x_all, rho_in, y_params, rng=data_rng)
        elif generator == "empirical":
            y_all = generate_y_empirical_batch(
                x_all, rho_s, y_params, rng=data_rng,
                _calibrated_rho=cal_rho)
        else:
            y_all = generate_y_linear_batch(x_all, rho_s, y_params, rng=data_rng)

        # Step 2: Compute rho_hats in one vectorized call
        rho_hats = spearman_rho_2d(x_all, y_all)  # (n_reps,)

        # Step 3-4: Bootstrap resampling — chunked over n_reps to bound
        # memory.  Each chunk allocates (n_boot × cs × n) int32 indices;
        # at n_boot=500, cs=2000, n=82 this is ~328 MB regardless of the
        # total n_reps.  Note: the RNG sequence differs from a single
        # (n_boot, n_reps, n) call because values are consumed in
        # (n_boot, cs, n) C-order blocks rather than one contiguous block.
        # Results are statistically equivalent and reproducible for fixed
        # chunk_size + seed.
        chunk_size = (_boot_chunk_size if _boot_chunk_size is not None
                      else _BATCH_BOOTSTRAP_CHUNK_SIZE)
        boot_rho_matrix = np.empty((n_reps, n_boot), dtype=np.float64)
        for chunk_start in range(0, n_reps, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_reps)
            cs = chunk_end - chunk_start
            boot_idx_chunk = boot_rng.integers(0, n, size=(n_boot, cs, n),
                                               dtype=np.int32)
            if use_numba() and _batch_bootstrap_rhos_jit is not None:
                boot_rho_matrix[chunk_start:chunk_end] = _batch_bootstrap_rhos_jit(
                    x_all[chunk_start:chunk_end].astype(np.float64),
                    y_all[chunk_start:chunk_end].astype(np.float64),
                    boot_idx_chunk)
            else:
                rows = np.arange(cs)[:, None]
                for b in range(n_boot):
                    x_boot = x_all[chunk_start:chunk_end][rows, boot_idx_chunk[b]]
                    y_boot = y_all[chunk_start:chunk_end][rows, boot_idx_chunk[b]]
                    boot_rho_matrix[chunk_start:chunk_end, b] = spearman_rho_2d(
                        x_boot, y_boot)

        # Step 5: Percentiles per rep
        lowers = np.nanpercentile(boot_rho_matrix, 100 * alpha / 2, axis=1)
        uppers = np.nanpercentile(boot_rho_matrix, 100 * (1 - alpha / 2), axis=1)

    elif vectorize:
        # --- EXISTING VECTORIZE PATH (unchanged) ---
        x_reps = generate_cumulative_aluminum_batch(
            n_reps, n, n_distinct, distribution_type=distribution_type,
            all_distinct=all_distinct, freq_dict=freq_dict, rng=data_rng)
        if generator == "nonparametric":
            y_reps = generate_y_nonparametric_batch(
                x_reps, rho_s, y_params, rng=data_rng,
                _calibrated_rho=cal_rho, _ln_params=ln_params)
        elif generator == "copula":
            rho_in = cal_rho if cal_rho is not None else rho_s
            y_reps = generate_y_copula_batch(x_reps, rho_in, y_params, rng=data_rng)
        elif generator == "empirical":
            y_reps = generate_y_empirical_batch(
                x_reps, rho_s, y_params, rng=data_rng,
                _calibrated_rho=cal_rho)
        else:
            y_reps = generate_y_linear_batch(x_reps, rho_s, y_params, rng=data_rng)

        for rep in range(n_reps):
            x, y = x_reps[rep], y_reps[rep]
            rho_hat = _fast_spearman_rho(x, y)
            lo, hi = bootstrap_ci_single(x, y, rho_hat, n_boot=n_boot,
                                          alpha=alpha, rng=boot_rng)
            lowers[rep] = lo
            uppers[rep] = hi
            rho_hats[rep] = rho_hat
    else:
        gen_fn = get_generator(generator) if generator not in ("nonparametric", "empirical") else None
        for rep in range(n_reps):
            x = generate_cumulative_aluminum(
                n, n_distinct, distribution_type=distribution_type,
                all_distinct=all_distinct, freq_dict=freq_dict, rng=data_rng)
            if generator == "nonparametric":
                y = generate_y_nonparametric(x, rho_s, y_params, rng=data_rng,
                                              _calibrated_rho=cal_rho,
                                              _ln_params=ln_params)
            elif generator == "copula":
                rho_in = cal_rho if cal_rho is not None else rho_s
                y = gen_fn(x, rho_in, y_params, rng=data_rng)
            elif generator == "empirical":
                y = generate_y_empirical(x, rho_s, y_params, rng=data_rng,
                                         _calibrated_rho=cal_rho)
            else:
                y = gen_fn(x, rho_s, y_params, rng=data_rng)
            rho_hat = _fast_spearman_rho(x, y)
            lo, hi = bootstrap_ci_single(x, y, rho_hat, n_boot=n_boot,
                                          alpha=alpha, rng=boot_rng)
            lowers[rep] = lo
            uppers[rep] = hi
            rho_hats[rep] = rho_hat

    return {
        "ci_lower": float(np.mean(lowers)),
        "ci_upper": float(np.mean(uppers)),
        "ci_lower_sd": float(np.std(lowers, ddof=1)) if n_reps > 1 else 0.0,
        "ci_upper_sd": float(np.std(uppers, ddof=1)) if n_reps > 1 else 0.0,
        "mean_rho_hat": float(np.mean(rho_hats)),
    }


# ---------------------------------------------------------------------------
# Run CI computation for all scenarios
# ---------------------------------------------------------------------------

def _ci_one_scenario(case_id, case, k, dt, all_distinct, generator,
                     n_reps, n_boot, alpha, tie_correction_mode, seed,
                     calibration_mode=None, batch_bootstrap=None, n_cal=None):
    """Run bootstrap CI + asymptotic CI for a single scenario."""
    n = case["n"]
    rho_obs = case["observed_rho"]
    y_params = {"median": case["median"], "iqr": case["iqr"],
                "range": case["range"]}
    x_counts = get_x_counts(n, k, distribution_type=dt,
                            all_distinct=all_distinct)

    boot = bootstrap_ci_averaged(
        n, k, dt, rho_obs, y_params,
        generator=generator, n_reps=n_reps, n_boot=n_boot,
        alpha=alpha, all_distinct=all_distinct, seed=seed,
        calibration_mode=calibration_mode, batch_bootstrap=batch_bootstrap,
        n_cal=n_cal)

    asym = _asymptotic_ci_results(
        rho_obs, n, alpha, x_counts, tie_correction_mode)

    return {
        "case": case_id,
        "n": n,
        "n_distinct": k,
        "dist_type": "all_distinct" if all_distinct else dt,
        "observed_rho": rho_obs,
        "all_distinct": all_distinct,
        "generator": generator,
        "boot_ci_lower": boot["ci_lower"],
        "boot_ci_upper": boot["ci_upper"],
        "boot_ci_lower_sd": boot["ci_lower_sd"],
        "boot_ci_upper_sd": boot["ci_upper_sd"],
        "boot_mean_rho_hat": boot["mean_rho_hat"],
        **asym,
    }


def run_all_ci_scenarios(generator="nonparametric", n_reps=200, n_boot=None,
                         alpha=None, tie_correction_mode=None, seed=None,
                         n_jobs=1, calibration_mode=None, batch_bootstrap=None,
                         n_cal=None, pre_warm=True):
    """Compute averaged bootstrap and asymptotic CIs for observed rhos.

    Uses bootstrap_ci_averaged (n_reps independent datasets, each
    bootstrapped n_boot times) for stable CI estimates.

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs (1 = sequential, -1 = all cores).
    n_cal : int or None
        Calibration samples per scenario.  When None, uses config N_CAL
        (default 300).  Calibration noise shifts CI endpoints (absolute
        values) but not CI width.  Use the n_cal values from config.CI_TIERS
        for accuracy guarantees; see docs/UNCERTAINTY_BUDGET.md Part 2.
    pre_warm : bool
        When True (default) and n_jobs != 1, warm the calibration cache in
        the main process before snapshotting it into workers.  Re-warming an
        already-warm cache costs only dict lookups (~88 checks, microseconds).
        Set False only if you have pre-warmed manually and want to skip even
        that overhead.

    Returns
    -------
    list of dict
    """
    if n_boot is None:
        n_boot = N_BOOTSTRAP
    if alpha is None:
        alpha = ALPHA
    if tie_correction_mode is None:
        tie_correction_mode = ASYMPTOTIC_TIE_CORRECTION_MODE
    if n_cal is None:
        n_cal = N_CAL

    scenarios = []
    scenario_idx = 0
    for case_id, case in CASES.items():
        n = case["n"]
        for k in N_DISTINCT_VALUES:
            for dt in DISTRIBUTION_TYPES:
                sc_seed = (seed + scenario_idx) if seed is not None else None
                scenarios.append((case_id, case, k, dt, False,
                                  generator, n_reps, n_boot, alpha,
                                  tie_correction_mode, sc_seed,
                                  calibration_mode, batch_bootstrap, n_cal))
                scenario_idx += 1

        sc_seed = (seed + scenario_idx) if seed is not None else None
        scenarios.append((case_id, case, n, None, True,
                          generator, n_reps, n_boot, alpha,
                          tie_correction_mode, sc_seed,
                          calibration_mode, batch_bootstrap, n_cal))
        scenario_idx += 1

    if n_jobs == 1:
        return [_ci_one_scenario(*args) for args in scenarios]

    # Parallel path: warm calibration cache in main process then inject into
    # each worker.  CI workers never use the permutation null cache, so only
    # calibration snapshots are passed.
    _eff_cal_mode = calibration_mode if calibration_mode is not None else CALIBRATION_MODE
    if pre_warm:
        warm_calibration_cache(generator,
                               calibration_mode=_eff_cal_mode,
                               n_cal=n_cal)

    cal_snaps = get_calibration_cache_snapshots()

    return Parallel(
        n_jobs=n_jobs,
        initializer=_init_ci_worker_caches,
        initargs=(cal_snaps,),
    )(delayed(_ci_one_scenario)(*args) for args in scenarios)


def _asymptotic_ci_results(rho_obs, n, alpha, x_counts, tie_correction_mode):
    """Build flat dict of asymptotic CI fields for one scenario."""
    out = {}
    modes = (["with_tie_correction", "without_tie_correction"]
             if tie_correction_mode == "both"
             else [tie_correction_mode])
    for mode in modes:
        tc = mode == "with_tie_correction"
        label = "tc" if tc else "notc"
        lo, hi = asymptotic_ci(rho_obs, n, alpha=alpha,
                               x_counts=x_counts, tie_correction=tc)
        out[f"asym_{label}_ci_lower"] = lo
        out[f"asym_{label}_ci_upper"] = hi
    return out
