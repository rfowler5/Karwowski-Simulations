"""
Verify skip-y optimisation for non-empirical generators (Issue 4, backlog item 9).

The _return_ranks=True path in non-empirical batch generators must produce
rank arrays such that Spearman correlations (point estimates, bootstrap CIs,
and permutation p-values) are identical to the float-y path.  This tests the
mathematical identities:

  nonparametric: rank(y_final) == rank(mixed)   [monotone marginal assignment]
  copula:        rank(y)       == rank(z_y)      [norm.cdf, lognormal_quantile monotone]
  linear:        rank(y)       == rank(log_y)    [exp monotone]

Used in both the CI batch bootstrap path (confidence_interval_calculator.py)
and the power simulation vectorized path (power_simulation.py).

Steps 1-5 verify rank identity, Spearman rho identity, bootstrap rho matrix
identity, rho=0 edge case, and all-distinct x.  Step 6 verifies that
permutation MC p-values (pvalues_mc) are bit-exact identical for float y vs
rank y, covering the power simulation's MC fallback path.
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from config import CASES
from data_generator import (
    generate_cumulative_aluminum_batch,
    generate_y_nonparametric_batch,
    generate_y_copula_batch,
    generate_y_linear_batch,
    calibrate_rho,
    calibrate_rho_copula,
    _fit_lognormal,
)
from spearman_helpers import (
    _rank_rows,
    spearman_rho_2d,
    _batch_bootstrap_rhos_jit,
    use_numba,
)
from permutation_pvalue import pvalues_mc

SEED = 98765
N_REPS = 30
case = CASES[3]
N = case["n"]
K = 4
DT = "heavy_center"
Y_PARAMS = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
RHO_S = 0.30
RHO_ZERO = 0.0
N_BOOT_TEST = 20

# Shared x_all for all tests (heavy ties: k=4 heavy_center)
x_all = generate_cumulative_aluminum_batch(
    N_REPS, N, K, distribution_type=DT, rng=np.random.default_rng(0))

cal_rho_np = calibrate_rho(N, K, DT, RHO_S, Y_PARAMS, calibration_mode="single")
cal_rho_cop = calibrate_rho_copula(N, K, DT, RHO_S, Y_PARAMS)
ln_params = _fit_lognormal(Y_PARAMS["median"], Y_PARAMS["iqr"])

cal_rho_np_zero = calibrate_rho(N, K, DT, RHO_ZERO, Y_PARAMS, calibration_mode="single")
cal_rho_cop_zero = calibrate_rho_copula(N, K, DT, RHO_ZERO, Y_PARAMS)


def _gen_both(gen_fn, seed, **kwargs):
    """Call gen_fn twice (float y, then rank y) with identical RNG state."""
    rng_f = np.random.default_rng(seed)
    y_float = gen_fn(x_all, rng=rng_f, _return_ranks=False, **kwargs)
    rng_r = np.random.default_rng(seed)
    y_ranks = gen_fn(x_all, rng=rng_r, _return_ranks=True, **kwargs)
    return y_float, y_ranks


# -----------------------------------------------------------------------
# Step 1: Rank identity — _rank_rows(y_float) == y_ranks
# -----------------------------------------------------------------------
print("Step 1: Rank identity (rank(y_float) == y_ranks)")

def check_rank_identity(label, y_float, y_ranks):
    float_ranks = _rank_rows(y_float)
    if not np.array_equal(float_ranks, y_ranks):
        max_diff = np.max(np.abs(float_ranks - y_ranks))
        print(f"  {label}: FAIL — max diff = {max_diff:.2e}")
        return False
    print(f"  {label}: PASS")
    return True

ok = True

yf, yr = _gen_both(lambda x, rng, _return_ranks, **kw:
    generate_y_nonparametric_batch(x, RHO_S, Y_PARAMS, rng=rng,
        _calibrated_rho=cal_rho_np, _ln_params=ln_params,
        _return_ranks=_return_ranks), seed=SEED)
ok &= check_rank_identity("nonparametric (rho=0.30)", yf, yr)

yf, yr = _gen_both(lambda x, rng, _return_ranks, **kw:
    generate_y_copula_batch(x, cal_rho_cop if cal_rho_cop is not None else RHO_S,
        Y_PARAMS, rng=rng, _return_ranks=_return_ranks), seed=SEED+1)
ok &= check_rank_identity("copula (rho=0.30)", yf, yr)

yf, yr = _gen_both(lambda x, rng, _return_ranks, **kw:
    generate_y_linear_batch(x, RHO_S, Y_PARAMS, rng=rng,
        _return_ranks=_return_ranks), seed=SEED+2)
ok &= check_rank_identity("linear (rho=0.30)", yf, yr)

# -----------------------------------------------------------------------
# Step 2: Spearman rho identity — rho(x, y_float) == rho(x, y_ranks)
# -----------------------------------------------------------------------
print("\nStep 2: Spearman rho identity")

def check_rho_identity(label, y_float, y_ranks):
    rho_f = spearman_rho_2d(x_all, y_float)
    rho_r = spearman_rho_2d(x_all, y_ranks)
    if not np.array_equal(rho_f, rho_r):
        max_diff = np.max(np.abs(rho_f - rho_r))
        print(f"  {label}: FAIL — max rho diff = {max_diff:.2e}")
        return False
    print(f"  {label}: PASS")
    return True

yf_np, yr_np = _gen_both(lambda x, rng, _return_ranks, **kw:
    generate_y_nonparametric_batch(x, RHO_S, Y_PARAMS, rng=rng,
        _calibrated_rho=cal_rho_np, _ln_params=ln_params,
        _return_ranks=_return_ranks), seed=SEED+10)
ok &= check_rho_identity("nonparametric", yf_np, yr_np)

yf_cop, yr_cop = _gen_both(lambda x, rng, _return_ranks, **kw:
    generate_y_copula_batch(x, cal_rho_cop if cal_rho_cop is not None else RHO_S,
        Y_PARAMS, rng=rng, _return_ranks=_return_ranks), seed=SEED+11)
ok &= check_rho_identity("copula", yf_cop, yr_cop)

yf_lin, yr_lin = _gen_both(lambda x, rng, _return_ranks, **kw:
    generate_y_linear_batch(x, RHO_S, Y_PARAMS, rng=rng,
        _return_ranks=_return_ranks), seed=SEED+12)
ok &= check_rho_identity("linear", yf_lin, yr_lin)

# -----------------------------------------------------------------------
# Step 3: Bootstrap identity — JIT produces same rho matrix for both
# -----------------------------------------------------------------------
print("\nStep 3: Bootstrap rho matrix identity")

def check_bootstrap_identity(label, y_float, y_ranks):
    boot_idx = np.random.default_rng(777).integers(
        0, N, size=(N_BOOT_TEST, N_REPS, N), dtype=np.int32)

    if use_numba() and _batch_bootstrap_rhos_jit is not None:
        mat_f = _batch_bootstrap_rhos_jit(
            x_all.astype(np.float64), y_float.astype(np.float64), boot_idx)
        mat_r = _batch_bootstrap_rhos_jit(
            x_all.astype(np.float64), y_ranks.astype(np.float64), boot_idx)
        if not np.array_equal(mat_f, mat_r):
            max_diff = np.max(np.abs(mat_f - mat_r))
            print(f"  {label}: FAIL — max bootstrap rho diff = {max_diff:.2e}")
            return False
        print(f"  {label}: PASS (Numba JIT)")
    else:
        mat_f = np.empty((N_REPS, N_BOOT_TEST))
        mat_r = np.empty((N_REPS, N_BOOT_TEST))
        rows = np.arange(N_REPS)[:, None]
        for b in range(N_BOOT_TEST):
            xb = x_all[rows, boot_idx[b]]
            yb_f = y_float[rows, boot_idx[b]]
            yb_r = y_ranks[rows, boot_idx[b]]
            mat_f[:, b] = spearman_rho_2d(xb, yb_f)
            mat_r[:, b] = spearman_rho_2d(xb, yb_r)
        if not np.array_equal(mat_f, mat_r):
            max_diff = np.max(np.abs(mat_f - mat_r))
            print(f"  {label}: FAIL — max bootstrap rho diff = {max_diff:.2e}")
            return False
        print(f"  {label}: PASS (NumPy fallback)")
    return True

ok &= check_bootstrap_identity("nonparametric", yf_np, yr_np)
ok &= check_bootstrap_identity("copula", yf_cop, yr_cop)
ok &= check_bootstrap_identity("linear", yf_lin, yr_lin)

# -----------------------------------------------------------------------
# Step 4: Edge case — rho = 0
# -----------------------------------------------------------------------
print("\nStep 4: Edge case rho=0")

yf, yr = _gen_both(lambda x, rng, _return_ranks, **kw:
    generate_y_nonparametric_batch(x, RHO_ZERO, Y_PARAMS, rng=rng,
        _calibrated_rho=cal_rho_np_zero, _ln_params=ln_params,
        _return_ranks=_return_ranks), seed=SEED+20)
ok &= check_rank_identity("nonparametric rho=0", yf, yr)

yf, yr = _gen_both(lambda x, rng, _return_ranks, **kw:
    generate_y_copula_batch(x, cal_rho_cop_zero if cal_rho_cop_zero is not None else RHO_ZERO,
        Y_PARAMS, rng=rng, _return_ranks=_return_ranks), seed=SEED+21)
ok &= check_rank_identity("copula rho=0", yf, yr)

yf, yr = _gen_both(lambda x, rng, _return_ranks, **kw:
    generate_y_linear_batch(x, RHO_ZERO, Y_PARAMS, rng=rng,
        _return_ranks=_return_ranks), seed=SEED+22)
ok &= check_rank_identity("linear rho=0", yf, yr)

# -----------------------------------------------------------------------
# Step 5: All-distinct x (no x-ties)
# -----------------------------------------------------------------------
print("\nStep 5: All-distinct x (no ties)")
x_distinct = generate_cumulative_aluminum_batch(
    N_REPS, N, N, distribution_type=DT, all_distinct=True,
    rng=np.random.default_rng(1))

cal_d = calibrate_rho(N, N, DT, RHO_S, Y_PARAMS, all_distinct=True,
                       calibration_mode="single")
ln_d = _fit_lognormal(Y_PARAMS["median"], Y_PARAMS["iqr"])

rng_f = np.random.default_rng(SEED+30)
yf_d = generate_y_nonparametric_batch(x_distinct, RHO_S, Y_PARAMS, rng=rng_f,
    _calibrated_rho=cal_d, _ln_params=ln_d, _return_ranks=False)
rng_r = np.random.default_rng(SEED+30)
yr_d = generate_y_nonparametric_batch(x_distinct, RHO_S, Y_PARAMS, rng=rng_r,
    _calibrated_rho=cal_d, _ln_params=ln_d, _return_ranks=True)
ok &= check_rank_identity("nonparametric all-distinct", yf_d, yr_d)

# -----------------------------------------------------------------------
# Step 6: pvalues_mc identity — MC permutation p-values identical for
#         float y vs rank y (covers the power simulation MC fallback path)
# -----------------------------------------------------------------------
print("\nStep 6: pvalues_mc identity (float y vs rank y)")

N_PERM_TEST = 50
ALPHA_TEST = 0.05

def check_pvalues_mc_identity(label, y_float, y_ranks):
    """Assert rhos_obs and pvals are bit-exact identical for float vs rank y."""
    rng_f = np.random.default_rng(12345)
    _, pvals_f, rhos_f = pvalues_mc(x_all, y_float, N_PERM_TEST, ALPHA_TEST, rng_f)
    rng_r = np.random.default_rng(12345)
    _, pvals_r, rhos_r = pvalues_mc(x_all, y_ranks, N_PERM_TEST, ALPHA_TEST, rng_r)
    if not np.array_equal(rhos_f, rhos_r):
        max_diff = np.max(np.abs(rhos_f - rhos_r))
        print(f"  {label}: FAIL — max rhos_obs diff = {max_diff:.2e}")
        return False
    if not np.array_equal(pvals_f, pvals_r):
        max_diff = np.max(np.abs(pvals_f - pvals_r))
        print(f"  {label}: FAIL — max pval diff = {max_diff:.2e}")
        return False
    print(f"  {label}: PASS")
    return True

# Reuse y_float / y_ranks from Step 2 (same seed, same data)
ok &= check_pvalues_mc_identity("nonparametric", yf_np, yr_np)
ok &= check_pvalues_mc_identity("copula", yf_cop, yr_cop)
ok &= check_pvalues_mc_identity("linear", yf_lin, yr_lin)

# -----------------------------------------------------------------------
print()
if ok:
    print("All skip-y rank identity tests passed.")
else:
    print("FAIL: one or more tests failed.")
    sys.exit(1)
