"""Unit tests for permutation-based p-value (precomputed null and MC path)."""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def test_precomputed_null_shape_and_stats():
    """Build a null for a known scenario, check shape and that mean≈0, SD is plausible."""
    from permutation_pvalue import get_precomputed_null
    import numpy as np

    # Case 3: n=73, k=4, even → x_counts = [19, 18, 18, 18]
    x_counts = np.array([19, 18, 18, 18])
    rng = np.random.default_rng(42)
    null = get_precomputed_null(73, False, x_counts, n_pre=10_000, rng=rng)

    assert null.shape == (10_000,)
    assert null.dtype == np.float64
    # Sorted absolute values: should be non-decreasing
    assert np.all(null[1:] >= null[:-1])
    # Mean of |rho| under null should be modest (not near 1)
    assert null.mean() < 0.2
    # Max should be < 1
    assert null.max() <= 1.0


def test_precomputed_pvalue_rho_zero():
    """P-value for rho_obs=0 should be near 1 (cannot reject null)."""
    from permutation_pvalue import get_precomputed_null, pvalues_from_precomputed_null
    import numpy as np

    x_counts = np.array([19, 18, 18, 18])
    rng = np.random.default_rng(42)
    null = get_precomputed_null(73, False, x_counts, n_pre=50_000, rng=rng)

    rhos_obs = np.array([0.0])
    pvals = pvalues_from_precomputed_null(rhos_obs, null)
    assert pvals[0] > 0.9  # rho=0 should have high p-value


def test_pvalues_mc_basic():
    """MC p-values should be in (0, 1] and reject should agree with p < alpha."""
    from permutation_pvalue import pvalues_mc
    from config import CASES
    from data_generator import (generate_cumulative_aluminum_batch,
                                 generate_y_nonparametric_batch,
                                 calibrate_rho, _fit_lognormal)
    import numpy as np

    case = CASES[3]
    n, k, dt = case["n"], 4, "even"
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
    rng = np.random.default_rng(42)
    n_sims = 50
    alpha = 0.05

    x_all = generate_cumulative_aluminum_batch(n_sims, n, k, dt, rng=rng)
    cal_rho = calibrate_rho(n, k, dt, 0.35, y_params, calibration_mode="single")
    ln_params = _fit_lognormal(y_params["median"], y_params["iqr"])
    y_all = generate_y_nonparametric_batch(x_all, 0.35, y_params, rng=rng,
                                            _calibrated_rho=cal_rho,
                                            _ln_params=ln_params)

    reject, pvals, rhos_obs = pvalues_mc(x_all, y_all, n_perm=200, alpha=alpha, rng=rng)
    assert pvals.shape == (n_sims,)
    assert np.all(pvals > 0)
    assert np.all(pvals <= 1)
    assert np.array_equal(reject, pvals < alpha)


def test_pvalues_mc_chunked():
    """MC chunked path (n_sims > threshold) should produce same-shape, valid results."""
    from permutation_pvalue import pvalues_mc
    from config import CASES
    from data_generator import (generate_cumulative_aluminum_batch,
                                 generate_y_nonparametric_batch,
                                 calibrate_rho, _fit_lognormal)
    import numpy as np

    case = CASES[3]
    n, k, dt = case["n"], 4, "even"
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
    rng = np.random.default_rng(42)
    n_sims = 80
    alpha = 0.05

    x_all = generate_cumulative_aluminum_batch(n_sims, n, k, dt, rng=rng)
    cal_rho = calibrate_rho(n, k, dt, 0.35, y_params, calibration_mode="single")
    ln_params = _fit_lognormal(y_params["median"], y_params["iqr"])
    y_all = generate_y_nonparametric_batch(x_all, 0.35, y_params, rng=rng,
                                            _calibrated_rho=cal_rho,
                                            _ln_params=ln_params)

    # Force chunking: set threshold=30 so 80 sims triggers the chunked path,
    # and chunk_size=25 so we get multiple chunks.
    reject, pvals, rhos_obs = pvalues_mc(
        x_all, y_all, n_perm=200, alpha=alpha, rng=rng,
        n_sims_batch_threshold=30, n_sims_chunk_size=25)

    assert pvals.shape == (n_sims,)
    assert np.all(pvals > 0)
    assert np.all(pvals <= 1)
    assert rhos_obs.shape == (n_sims,)
    assert np.array_equal(reject, pvals < alpha)


def test_estimate_power_smoke():
    """estimate_power should run without error and return power in [0, 1]."""
    from power_simulation import estimate_power
    from config import CASES

    case = CASES[3]
    n, k, dt = case["n"], 4, "even"
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}

    power = estimate_power(n, k, dt, rho_s=0.35, y_params=y_params,
                           n_sims=50, seed=42, calibration_mode="single")
    assert 0.0 <= power <= 1.0


def test_precomputed_null_cache_hit():
    """Second call with same args should return same object (cache hit)."""
    from permutation_pvalue import get_precomputed_null
    import numpy as np

    x_counts = np.array([20, 20, 20, 20])
    rng = np.random.default_rng(99)
    null1 = get_precomputed_null(80, False, x_counts, n_pre=1000, rng=rng)
    null2 = get_precomputed_null(80, False, x_counts, n_pre=1000, rng=rng)
    assert null1 is null2  # same object from cache


def test_mc_on_cache_miss_cold():
    """With PVALUE_MC_ON_CACHE_MISS=True, a cold cache falls back to MC."""
    from permutation_pvalue import get_cached_null, _NULL_CACHE
    from config import PVALUE_PRECOMPUTED_N_PRE
    import numpy as np

    # Use a unique x_counts unlikely to be in cache from other tests
    x_counts = np.array([11, 11, 11, 12])
    n = int(np.sum(x_counts))  # 45
    key = (n, False, tuple(int(c) for c in x_counts), PVALUE_PRECOMPUTED_N_PRE)
    _NULL_CACHE.pop(key, None)  # ensure cold

    result = get_cached_null(n, False, x_counts)
    assert result is None  # cache miss returns None, not a built array


def test_mc_on_cache_miss_warm():
    """After building, get_cached_null returns the cached array."""
    from permutation_pvalue import get_precomputed_null, get_cached_null
    import numpy as np

    x_counts = np.array([11, 11, 11, 12])
    n = int(np.sum(x_counts))
    rng = np.random.default_rng(7)
    built = get_precomputed_null(n, False, x_counts, n_pre=500, rng=rng)
    cached = get_cached_null(n, False, x_counts, n_pre=500)
    assert cached is built  # same object


def test_estimate_power_mc_on_cache_miss_standard():
    """estimate_power with PVALUE_MC_ON_CACHE_MISS=True, cold cache, standard freq."""
    import power_simulation as ps
    import permutation_pvalue as pp
    from config import CASES
    import numpy as np

    case = CASES[3]
    n, k, dt = case["n"], 4, "even"
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}

    # Ensure cold cache for this scenario
    from power_asymptotic import get_x_counts
    from config import PVALUE_PRECOMPUTED_N_PRE
    x_counts = get_x_counts(n, k, distribution_type=dt)
    key = (n, False, tuple(int(c) for c in x_counts), PVALUE_PRECOMPUTED_N_PRE)
    pp._NULL_CACHE.pop(key, None)

    orig = ps.PVALUE_MC_ON_CACHE_MISS
    ps.PVALUE_MC_ON_CACHE_MISS = True
    try:
        power = ps.estimate_power(n, k, dt, rho_s=0.35, y_params=y_params,
                                  n_sims=30, seed=42, calibration_mode="single")
        assert 0.0 <= power <= 1.0
        # Cache should still be cold (MC path does not populate it)
        assert pp._NULL_CACHE.get(key) is None
    finally:
        ps.PVALUE_MC_ON_CACHE_MISS = orig


def test_estimate_power_mc_on_cache_miss_custom_freq():
    """PVALUE_MC_ON_CACHE_MISS=True works with custom freq_dict (cold cache)."""
    import power_simulation as ps
    import permutation_pvalue as pp
    from config import CASES
    import numpy as np

    case = CASES[3]
    n = case["n"]  # 73
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}

    # Custom freq: 4 groups with unequal sizes
    from config import PVALUE_PRECOMPUTED_N_PRE
    custom_counts = [25, 20, 15, 13]
    freq_dict = {n: {4: {"custom": custom_counts}}}
    x_counts_tuple = tuple(custom_counts)
    key = (n, False, x_counts_tuple, PVALUE_PRECOMPUTED_N_PRE)
    pp._NULL_CACHE.pop(key, None)

    orig = ps.PVALUE_MC_ON_CACHE_MISS
    ps.PVALUE_MC_ON_CACHE_MISS = True
    try:
        power = ps.estimate_power(
            n, 4, "custom", rho_s=0.35, y_params=y_params,
            n_sims=30, seed=42, calibration_mode="single",
            freq_dict=freq_dict)
        assert 0.0 <= power <= 1.0
        # MC path should not have populated the null cache
        assert pp._NULL_CACHE.get(key) is None
    finally:
        ps.PVALUE_MC_ON_CACHE_MISS = orig


def test_null_cache_disk_round_trip():
    """save_null_cache_to_disk / load_null_cache_from_disk round-trips correctly.

    Verifies that:
    - Saved null(s) reload into the correct in-memory 4-tuple key.
    - The reloaded array has the correct length and produces sane p-values.
    - load_null_cache_from_disk returns False (and does not modify cache)
      when called with a mismatched n_pre.
    """
    import os
    import tempfile
    import warnings
    import numpy as np
    from permutation_pvalue import (get_precomputed_null, get_cached_null,
                                    save_null_cache_to_disk,
                                    load_null_cache_from_disk,
                                    pvalues_from_precomputed_null,
                                    _NULL_CACHE)

    n_pre_test = 300
    x_counts = np.array([7, 7, 7, 7])
    n = int(np.sum(x_counts))

    # Build and record the null under a known key
    rng = np.random.default_rng(99)
    original = get_precomputed_null(n, False, x_counts, n_pre=n_pre_test, rng=rng)
    assert len(original) == n_pre_test

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name
    try:
        save_null_cache_to_disk(tmp_path, n_pre_test)

        # Remove only the key used in this test, then reload
        key = (n, False, tuple(int(c) for c in x_counts), n_pre_test)
        _NULL_CACHE.pop(key, None)
        assert get_cached_null(n, False, x_counts, n_pre=n_pre_test) is None

        result = load_null_cache_from_disk(tmp_path, n_pre_test)
        assert result is True

        reloaded = get_cached_null(n, False, x_counts, n_pre=n_pre_test)
        assert reloaded is not None
        assert len(reloaded) == n_pre_test
        # Array content should be identical to what was saved
        assert np.array_equal(reloaded, original)
        # p-values from reloaded null should be sane
        pvals = pvalues_from_precomputed_null(np.array([0.0]), reloaded)
        assert pvals[0] > 0.5

        # Loading with wrong n_pre should return False and not pollute cache
        _NULL_CACHE.pop(key, None)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Disk null cache has n_pre=.*current n_pre=.*",
                category=UserWarning,
            )
            result_wrong = load_null_cache_from_disk(tmp_path, n_pre_test + 1)
        assert result_wrong is False
        assert get_cached_null(n, False, x_counts, n_pre=n_pre_test) is None
    finally:
        os.unlink(tmp_path)


def test_different_n_pre_separate_cache_entries():
    """Different n_pre values produce independent cache entries for the same scenario."""
    from permutation_pvalue import get_precomputed_null
    import numpy as np

    x_counts = np.array([20, 20, 20, 20])
    rng1 = np.random.default_rng(42)
    null_small = get_precomputed_null(80, False, x_counts, n_pre=100, rng=rng1)
    rng2 = np.random.default_rng(43)
    null_large = get_precomputed_null(80, False, x_counts, n_pre=500, rng=rng2)
    assert null_small is not null_large
    assert len(null_small) == 100
    assert len(null_large) == 500
    # A second call with the same n_pre should hit cache and return the same object
    null_small_again = get_precomputed_null(80, False, x_counts, n_pre=100)
    assert null_small_again is null_small


if __name__ == "__main__":
    test_precomputed_null_shape_and_stats()
    test_precomputed_pvalue_rho_zero()
    test_pvalues_mc_basic()
    test_pvalues_mc_chunked()
    test_estimate_power_smoke()
    test_precomputed_null_cache_hit()
    test_mc_on_cache_miss_cold()
    test_mc_on_cache_miss_warm()
    test_estimate_power_mc_on_cache_miss_standard()
    test_estimate_power_mc_on_cache_miss_custom_freq()
    test_null_cache_disk_round_trip()
    test_different_n_pre_separate_cache_entries()
    print("All tests passed.")
