"""Validation tests for spearman_helpers: spearman_rho_2d and spearman_rho_pvalue_2d."""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
_numba_pre = argparse.ArgumentParser(add_help=False)
_numba_pre.add_argument("--no-numba", action="store_true")
_pre_args, _ = _numba_pre.parse_known_args()
if _pre_args.no_numba:
    import config
    config.USE_NUMBA = False

import numpy as np
from scipy.stats import rankdata
from scipy.stats import t as t_dist
from spearman_helpers import spearman_rho_2d, spearman_rho_pvalue_2d


def _fast_spearman_rho(x, y):
    """Reference scalar implementation."""
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    denom = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    if denom < 1e-15:
        return 0.0
    return np.dot(rx, ry) / denom


def _fast_spearman_pvalue(x, y, n):
    """Reference scalar rho + two-sided p-value via t-approximation."""
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    num = np.dot(rx, ry)
    denom = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    if denom < 1e-15:
        return 0.0, 1.0
    rho = num / denom
    t_stat = rho * np.sqrt((n - 2) / max(1.0 - rho * rho, 1e-15))
    pval = 2.0 * t_dist.sf(abs(t_stat), n - 2)
    return rho, pval


# --- Test 1: spearman_rho_2d vs scalar reference ---

np.random.seed(42)
n_rows, n = 100, 73
x = np.random.randn(n_rows, n)
y = np.random.randn(n_rows, n)
x = np.round(x * 3) / 3
y = np.round(y * 2) / 2

ref = np.array([_fast_spearman_rho(x[i], y[i]) for i in range(n_rows)])
vec = spearman_rho_2d(x, y)

assert np.allclose(ref, vec, atol=1e-10), f"rho_2d max diff: {np.max(np.abs(ref - vec))}"
print("Test 1 passed (spearman_rho_2d): max diff =", np.max(np.abs(ref - vec)))


# --- Test 2: spearman_rho_pvalue_2d vs scalar reference ---

ref_rhos = np.empty(n_rows)
ref_pvals = np.empty(n_rows)
for i in range(n_rows):
    r, p = _fast_spearman_pvalue(x[i], y[i], n)
    ref_rhos[i] = r
    ref_pvals[i] = p

vec_rhos, vec_pvals = spearman_rho_pvalue_2d(x, y, n)

assert np.allclose(ref_rhos, vec_rhos, atol=1e-10), \
    f"pvalue_2d rho max diff: {np.max(np.abs(ref_rhos - vec_rhos))}"
assert np.allclose(ref_pvals, vec_pvals, atol=1e-10), \
    f"pvalue_2d pval max diff: {np.max(np.abs(ref_pvals - vec_pvals))}"
print("Test 2 passed (spearman_rho_pvalue_2d): rho max diff =",
      np.max(np.abs(ref_rhos - vec_rhos)),
      ", pval max diff =", np.max(np.abs(ref_pvals - vec_pvals)))
