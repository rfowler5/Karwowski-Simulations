"""
Unit tests for the calibration precompute refactoring.

Background
----------
The refactoring split _mean_rho_vec into two functions:
  _precompute_calibration_arrays  -- generates all RNG-based, rho-independent
      arrays (x shuffles, noise ranks, standardized s_x/s_n, sorted lognormal y)
      from a fixed seed.  Called ONCE per bisection probe.
  _eval_mean_rho  -- takes the precomputed arrays and performs only the
      rho-dependent steps (mixed = rho*s_x + sqrt(1-rho^2)*s_n → argsort →
      y-assignment → rank → Pearson-on-ranks).  Called ~27x per bisection.
_mean_rho_vec is now a thin wrapper that calls both in sequence.
The empirical path (_mean_rho_empirical_vec, _bisect_for_probe_empirical)
uses an identical structure with _precompute_calibration_arrays_empirical,
which tiles np.sort(pool) instead of drawing lognormal y values.

Tests
-----
  1. test_precompute_shapes
     Confirms the four arrays come back with shape (n_cal, n).  A shape mismatch
     would cause broadcasting errors in _eval_mean_rho downstream.

  2. test_precompute_y_sorted
     y_batch_sorted must be sorted row-wise so that _eval_mean_rho can assign
     values by argsort(mixed) and produce the intended rank ordering.  An unsorted
     y_batch would silently corrupt the rho estimate.

  3. test_precompute_s_x_zero_mean
     s_x is the standardized x-rank matrix.  Zero row-means and unit row-stds are
     required for the mixing formula rho*s_x + sqrt(1-rho^2)*s_n to produce a
     properly normalised mixed score.  Violations would bias calibration.

  4. test_precompute_s_n_zero_mean
     Same check for s_n (standardized noise ranks).  Noise ranks are always
     permutations of 1..n, so their population mean/std are analytic constants;
     this test guards against off-by-one errors (e.g. using 0..n-1 instead of
     1..n) that would shift the mean away from zero.

  5. test_precompute_deterministic
     _precompute_calibration_arrays must be a pure function of its inputs: calling
     it twice with the same seed must return bit-identical arrays.  Non-determinism
     here would mean different bisection iterations see different landscapes,
     breaking convergence guarantees.

  6. test_eval_equals_vec_wrapper
     THE CORE CORRECTNESS TEST.  Because _mean_rho_vec is now exactly
     `_eval_mean_rho(rho_in, *_precompute_calibration_arrays(..., seed))`,
     both expressions must return the same float.  Exact float equality (not
     just approximate) is asserted because they are the same arithmetic on the
     same values.  Any deviation would indicate the split introduced a bug.

  7. test_empirical_precompute_shapes
     Same shape check as (1) for the empirical path.  Skipped if digitized data
     is unavailable.

  8. test_empirical_y_batch_is_tiled_pool
     For empirical calibration, every row of y_batch_sorted must equal np.sort(pool)
     exactly — the pool is deterministic (no RNG), so tiling must reproduce the
     same sorted values on every row.  A mismatch would mean different rows use
     different y-distributions, invalidating the empirical marginal assumption.

  9. test_eval_equals_empirical_vec_wrapper
     Same float-equality check as (6) but for the empirical wrapper.  Confirms
     _mean_rho_empirical_vec delegates correctly to the precompute + eval pair.

 10. test_eval_monotone_in_rho_in
     _eval_mean_rho must be strictly increasing in rho_in for a given realisation
     of precomputed arrays.  This is the mathematical property that makes bisection
     valid: if the function were non-monotone, the bisection could converge to a
     spurious root.  Uses n_cal=900 (3× default) to suppress MC noise.

 11. test_eval_at_zero_rho
     At rho_in=0 the mixed score equals s_n (pure noise), so the realised rho
     between x and y should be approximately 0.  A value far from 0 would indicate
     a systematic bias in the mixing formula or the standardization step.

 12. test_bisect_achieves_probe
     Verifies that the bisection loop converges and that evaluating _eval_mean_rho
     at the returned rho_in reproduces the probe within MC noise.  Because both
     the bisection and the test use the same precomputed arrays (same seed), the
     check is tight: the bisection tol=5e-5 maps to output error ~1e-4, well
     inside the 0.015 tolerance.

 13. test_bisect_unreachable_returns_none
     With n=73 and k=4 distinct x-values (heavy ties), the maximum achievable
     Spearman rho is ~0.7-0.8.  A probe of 0.99 is genuinely unreachable; the
     bisection must return None rather than silently returning a nonsensical value.

 14. test_bisect_empirical_achieves_probe
     Same convergence check as (12) for the empirical bisection path.  Skipped if
     digitized data is unavailable.

 15. test_mean_rho_vec_output_range
     Sanity check that the public-facing wrapper returns a Python float in (-1, 1)
     across a range of rho_in values.  Guards against dtype accidents (e.g.
     returning a 0-d numpy array) or edge cases that produce NaN/inf.

Usage:
    python tests/test_calibration_precompute.py
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from data_generator import (
    _precompute_calibration_arrays,
    _precompute_calibration_arrays_empirical,
    _eval_mean_rho,
    _mean_rho_vec,
    _mean_rho_empirical_vec,
    _bisect_for_probe,
    _bisect_for_probe_empirical,
    get_pool,
    _get_x_template,
    _fit_lognormal,
    digitized_available,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
# Case 3 parameters (n=73, median=14.3, iqr=13.9)
N = 73
Y_PARAMS = {"median": 14.3, "iqr": 13.9, "range": (11.6, 18.2)}
N_DISTINCT = 4
DIST_TYPE = "even"
N_CAL = 300
SEED = 99

TEMPLATE = _get_x_template(N, N_DISTINCT, DIST_TYPE, None, False)
POOL = get_pool(N) if digitized_available() else None
HAS_POOL = POOL is not None

# MC tolerance for bisection accuracy checks (n_cal=300 → SE ~0.01)
BISECTION_TOL = 0.015


def _arrays():
    """Nonparametric precomputed arrays."""
    return _precompute_calibration_arrays(TEMPLATE, Y_PARAMS, N_CAL, SEED)


def _arrays_emp():
    """Empirical precomputed arrays. Requires digitized data."""
    if not HAS_POOL:
        raise RuntimeError("digitized data not available — empirical test skipped")
    return _precompute_calibration_arrays_empirical(TEMPLATE, POOL, N_CAL, SEED)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_precompute_shapes():
    """All four returned arrays have shape (n_cal, n)."""
    s_x, s_n, y_batch_sorted, x_ranks_batch = _arrays()
    for name, arr in [("s_x", s_x), ("s_n", s_n),
                      ("y_batch_sorted", y_batch_sorted),
                      ("x_ranks_batch", x_ranks_batch)]:
        assert arr.shape == (N_CAL, N), (
            f"{name}: expected shape ({N_CAL}, {N}), got {arr.shape}")
    print("PASS: test_precompute_shapes")


def test_precompute_y_sorted():
    """y_batch_sorted is sorted along axis=1 (each row is non-decreasing)."""
    _, _, y_batch_sorted, _ = _arrays()
    diffs = np.diff(y_batch_sorted, axis=1)
    assert np.all(diffs >= 0), "y_batch_sorted has unsorted rows"
    print("PASS: test_precompute_y_sorted")


def test_precompute_s_x_zero_mean():
    """s_x rows have mean ~0 and std ~1 (standardized x ranks)."""
    s_x, _, _, _ = _arrays()
    row_means = s_x.mean(axis=1)
    row_stds = s_x.std(axis=1, ddof=0)
    assert np.allclose(row_means, 0.0, atol=1e-10), (
        f"s_x row means not ~0: max |mean| = {np.abs(row_means).max():.2e}")
    assert np.allclose(row_stds, 1.0, atol=1e-10), (
        f"s_x row stds not ~1: max |std-1| = {np.abs(row_stds - 1).max():.2e}")
    print("PASS: test_precompute_s_x_zero_mean")


def test_precompute_s_n_zero_mean():
    """s_n rows have mean 0 and std 1 (standardized noise ranks)."""
    _, s_n, _, _ = _arrays()
    row_means = s_n.mean(axis=1)
    row_stds = s_n.std(axis=1, ddof=0)
    assert np.allclose(row_means, 0.0, atol=1e-10), (
        f"s_n row means not ~0: max |mean| = {np.abs(row_means).max():.2e}")
    assert np.allclose(row_stds, 1.0, atol=1e-10), (
        f"s_n row stds not ~1: max |std-1| = {np.abs(row_stds - 1).max():.2e}")
    print("PASS: test_precompute_s_n_zero_mean")


def test_precompute_deterministic():
    """Same seed gives identical precomputed arrays on two calls."""
    a1 = _arrays()
    a2 = _arrays()
    for i, (arr1, arr2) in enumerate(zip(a1, a2)):
        assert np.array_equal(arr1, arr2), (
            f"Array index {i} differs between two calls with same seed")
    print("PASS: test_precompute_deterministic")


def test_eval_equals_vec_wrapper():
    """_eval_mean_rho(rho_in, *precompute(...)) == _mean_rho_vec(rho_in, ...).

    _mean_rho_vec is now a wrapper; they must be float-identical.
    """
    for rho_in in [0.0, 0.10, 0.30, 0.50]:
        arrays = _precompute_calibration_arrays(TEMPLATE, Y_PARAMS, N_CAL, SEED)
        via_eval = _eval_mean_rho(rho_in, *arrays)
        via_vec = _mean_rho_vec(rho_in, TEMPLATE, Y_PARAMS, N_CAL, SEED)
        assert via_eval == via_vec, (
            f"rho_in={rho_in}: _eval_mean_rho={via_eval} != "
            f"_mean_rho_vec={via_vec}")
    print("PASS: test_eval_equals_vec_wrapper")


def test_empirical_precompute_shapes():
    """Empirical precomputed arrays have shape (n_cal, n)."""
    if not HAS_POOL:
        print("SKIP: test_empirical_precompute_shapes (digitized data unavailable)")
        return
    s_x, s_n, y_batch_sorted, x_ranks_batch = _arrays_emp()
    for name, arr in [("s_x", s_x), ("s_n", s_n),
                      ("y_batch_sorted", y_batch_sorted),
                      ("x_ranks_batch", x_ranks_batch)]:
        assert arr.shape == (N_CAL, N), (
            f"{name}: expected shape ({N_CAL}, {N}), got {arr.shape}")
    print("PASS: test_empirical_precompute_shapes")


def test_empirical_y_batch_is_tiled_pool():
    """Empirical y_batch_sorted rows all equal np.sort(pool) exactly."""
    if not HAS_POOL:
        print("SKIP: test_empirical_y_batch_is_tiled_pool (digitized data unavailable)")
        return
    _, _, y_batch_sorted, _ = _arrays_emp()
    expected_row = np.sort(POOL)
    for i in range(N_CAL):
        assert np.array_equal(y_batch_sorted[i], expected_row), (
            f"Row {i} of y_batch_sorted does not equal np.sort(pool)")
    print("PASS: test_empirical_y_batch_is_tiled_pool")


def test_eval_equals_empirical_vec_wrapper():
    """_eval_mean_rho(rho_in, *empirical_precompute(...)) == _mean_rho_empirical_vec(rho_in, ...)."""
    if not HAS_POOL:
        print("SKIP: test_eval_equals_empirical_vec_wrapper (digitized data unavailable)")
        return
    for rho_in in [0.0, 0.10, 0.30, 0.50]:
        arrays = _precompute_calibration_arrays_empirical(TEMPLATE, POOL, N_CAL, SEED)
        via_eval = _eval_mean_rho(rho_in, *arrays)
        via_vec = _mean_rho_empirical_vec(rho_in, TEMPLATE, POOL, N_CAL, SEED)
        assert via_eval == via_vec, (
            f"rho_in={rho_in}: _eval_mean_rho={via_eval} != "
            f"_mean_rho_empirical_vec={via_vec}")
    print("PASS: test_eval_equals_empirical_vec_wrapper")


def test_eval_monotone_in_rho_in():
    """_eval_mean_rho increases with rho_in for fixed precomputed arrays.

    Uses a moderately large n_cal to reduce MC noise, tests over a coarse grid.
    """
    arrays = _precompute_calibration_arrays(TEMPLATE, Y_PARAMS, N_CAL * 3, SEED)
    rho_grid = [0.05, 0.15, 0.30, 0.50, 0.70]
    values = [_eval_mean_rho(r, *arrays) for r in rho_grid]
    for i in range(len(values) - 1):
        assert values[i] < values[i + 1], (
            f"Not monotone: eval({rho_grid[i]})={values[i]:.4f} >= "
            f"eval({rho_grid[i+1]})={values[i+1]:.4f}")
    print(f"PASS: test_eval_monotone_in_rho_in  values={[f'{v:.3f}' for v in values]}")


def test_eval_at_zero_rho():
    """_eval_mean_rho at rho_in=0 should be close to 0."""
    arrays = _precompute_calibration_arrays(TEMPLATE, Y_PARAMS, N_CAL * 3, SEED)
    val = _eval_mean_rho(0.0, *arrays)
    assert abs(val) < 0.05, f"_eval_mean_rho(0.0) = {val:.4f}, expected ~0"
    print(f"PASS: test_eval_at_zero_rho  (val={val:.4f})")


def test_bisect_achieves_probe():
    """_bisect_for_probe returns rho_in whose _eval_mean_rho ≈ probe."""
    probe = 0.30
    arrays = _precompute_calibration_arrays(TEMPLATE, Y_PARAMS, N_CAL, SEED)
    rho_in = _bisect_for_probe(probe, TEMPLATE, Y_PARAMS, N_CAL, SEED)
    assert rho_in is not None, "_bisect_for_probe returned None for probe=0.30"

    achieved = _eval_mean_rho(rho_in, *arrays)
    assert abs(achieved - probe) < BISECTION_TOL, (
        f"Bisection for probe={probe}: achieved={achieved:.4f}, "
        f"diff={abs(achieved - probe):.4f} > tol={BISECTION_TOL}")
    print(f"PASS: test_bisect_achieves_probe  "
          f"(probe={probe}, rho_in={rho_in:.4f}, achieved={achieved:.4f})")


def test_bisect_unreachable_returns_none():
    """_bisect_for_probe returns None when probe is unreachable (too high)."""
    rho_in = _bisect_for_probe(0.99, TEMPLATE, Y_PARAMS, N_CAL, SEED)
    assert rho_in is None, (
        f"Expected None for unreachable probe=0.99, got {rho_in}")
    print("PASS: test_bisect_unreachable_returns_none")


def test_bisect_empirical_achieves_probe():
    """_bisect_for_probe_empirical returns rho_in whose _eval_mean_rho ≈ probe."""
    if not HAS_POOL:
        print("SKIP: test_bisect_empirical_achieves_probe (digitized data unavailable)")
        return
    probe = 0.30
    arrays = _precompute_calibration_arrays_empirical(TEMPLATE, POOL, N_CAL, SEED)
    rho_in = _bisect_for_probe_empirical(probe, TEMPLATE, POOL, N_CAL, SEED)
    assert rho_in is not None, "_bisect_for_probe_empirical returned None for probe=0.30"

    achieved = _eval_mean_rho(rho_in, *arrays)
    assert abs(achieved - probe) < BISECTION_TOL, (
        f"Bisection (empirical) for probe={probe}: achieved={achieved:.4f}, "
        f"diff={abs(achieved - probe):.4f} > tol={BISECTION_TOL}")
    print(f"PASS: test_bisect_empirical_achieves_probe  "
          f"(probe={probe}, rho_in={rho_in:.4f}, achieved={achieved:.4f})")


def test_mean_rho_vec_output_range():
    """_mean_rho_vec output is a float in (-1, 1) for several rho_in values."""
    for rho_in in [0.0, 0.10, 0.30, 0.50, 0.70]:
        val = _mean_rho_vec(rho_in, TEMPLATE, Y_PARAMS, N_CAL, SEED)
        assert isinstance(val, float), f"Expected float, got {type(val)}"
        assert -1.0 < val < 1.0, f"_mean_rho_vec({rho_in}) = {val} out of (-1,1)"
    print("PASS: test_mean_rho_vec_output_range")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_precompute_shapes,
    test_precompute_y_sorted,
    test_precompute_s_x_zero_mean,
    test_precompute_s_n_zero_mean,
    test_precompute_deterministic,
    test_eval_equals_vec_wrapper,
    test_empirical_precompute_shapes,
    test_empirical_y_batch_is_tiled_pool,
    test_eval_equals_empirical_vec_wrapper,
    test_eval_monotone_in_rho_in,
    test_eval_at_zero_rho,
    test_bisect_achieves_probe,
    test_bisect_unreachable_returns_none,
    test_bisect_empirical_achieves_probe,
    test_mean_rho_vec_output_range,
]

if __name__ == "__main__":
    print(f"Running {len(TESTS)} calibration precompute tests...")
    failed = []
    for test_fn in TESTS:
        try:
            test_fn()
        except Exception as exc:
            print(f"FAIL: {test_fn.__name__}  → {exc}")
            failed.append(test_fn.__name__)

    print()
    if failed:
        print(f"FAILED: {len(failed)}/{len(TESTS)} tests failed: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(TESTS)} tests passed.")
