"""
Vectorized Spearman correlation helpers.

Provides:
- spearman_rho_2d(x, y)          : vectorized rho for 2D arrays (no scipy)
- _fast_spearman_rho(x, y)       : scalar rho (uses scipy.stats.rankdata)
- spearman_rho_pvalue_2d(x, y, n): vectorized (rho, pvalue) for 2D arrays

When Numba is installed and config.USE_NUMBA is True, hot paths use JIT-compiled
functions with inner thread parallelism (prange).  Falls back to pure NumPy
otherwise.
"""

import numpy as np
from scipy.stats import rankdata
from scipy.stats import t as t_dist

import config


# ---------------------------------------------------------------------------
# Numba JIT block (optional -- falls back to NumPy if unavailable)
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def use_numba() -> bool:
    """Return True if Numba is both installed and enabled via config."""
    return _NUMBA_AVAILABLE and config.USE_NUMBA


if _NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def _tied_rank(arr):
        """Average ranks with tie handling -- matches rankdata(method='average')."""
        n = arr.shape[0]
        sorter = np.argsort(arr)
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            while j < n and arr[sorter[j]] == arr[sorter[i]]:
                j += 1
            avg = (i + j + 1) / 2.0
            for k in range(i, j):
                ranks[sorter[k]] = avg
            i = j
        return ranks

    @njit(fastmath=True, cache=True)
    def _pearson_on_ranks_1d(rx, ry):
        """Pearson r on two 1-D rank vectors (scalar return)."""
        mx = np.mean(rx)
        my = np.mean(ry)
        cov = np.mean((rx - mx) * (ry - my))
        varx = np.mean((rx - mx) ** 2)
        vary = np.mean((ry - my) ** 2)
        if varx < 1e-15 or vary < 1e-15:
            return 0.0
        return cov / np.sqrt(varx * vary)

    @njit(fastmath=True, cache=True, parallel=True)
    def _rank_rows_numba(a):
        """Rank every row of a 2-D array (parallel over rows)."""
        m, n = a.shape
        ranks = np.empty((m, n), dtype=np.float64)
        for i in prange(m):
            ranks[i] = _tied_rank(a[i])
        return ranks

    @njit(fastmath=True, cache=True, parallel=True)
    def _bootstrap_rhos_jit(x, y, boot_idx):
        """Spearman rho for each bootstrap resample (parallel over resamples)."""
        n_boot, n = boot_idx.shape
        rhos = np.empty(n_boot, dtype=np.float64)
        for b in prange(n_boot):
            xb = np.empty(n, dtype=np.float64)
            yb = np.empty(n, dtype=np.float64)
            for i in range(n):
                idx = boot_idx[b, i]
                xb[i] = x[idx]
                yb[i] = y[idx]
            rx = _tied_rank(xb)
            ry = _tied_rank(yb)
            rhos[b] = _pearson_on_ranks_1d(rx, ry)
        return rhos

    @njit(fastmath=True, cache=True, parallel=True)
    def _batch_bootstrap_rhos_jit(x_all, y_all, boot_idx_all):
        """Batch bootstrap Spearman rho, parallel over all (rep, boot) pairs.

        Parameters
        ----------
        x_all, y_all : (n_reps, n) float64
        boot_idx_all : (n_boot, n_reps, n) int32
            Pre-generated bootstrap indices.

        Returns
        -------
        result : (n_reps, n_boot) float64
        """
        n_boot, n_reps, n = boot_idx_all.shape
        result = np.empty((n_reps, n_boot), dtype=np.float64)
        total = n_reps * n_boot
        for flat in prange(total):
            rep = flat // n_boot
            b = flat % n_boot
            xb = np.empty(n, dtype=np.float64)
            yb = np.empty(n, dtype=np.float64)
            for i in range(n):
                idx = boot_idx_all[b, rep, i]
                xb[i] = x_all[rep, idx]
                yb[i] = y_all[rep, idx]
            rx = _tied_rank(xb)
            ry = _tied_rank(yb)
            result[rep, b] = _pearson_on_ranks_1d(rx, ry)
        return result
else:
    _bootstrap_rhos_jit = None
    _batch_bootstrap_rhos_jit = None

# ---------------------------------------------------------------------------
# Public ranking and correlation functions
# ---------------------------------------------------------------------------

def _rank_rows(a: np.ndarray) -> np.ndarray:
    """Average ranks for each row of a 2D array, matching rankdata(method="average").

    Uses argsort + bincount for O(m * n * log n) complexity instead of the
    O(m * n^2) pairwise-broadcast approach.
    """
    if use_numba():
        return _rank_rows_numba(a)

    m, n = a.shape
    rows = np.arange(m)[:, None]

    order = np.argsort(a, axis=1, kind='stable')
    sorted_a = a[rows, order]

    group_start = np.ones((m, n), dtype=bool)
    group_start[:, 1:] = (sorted_a[:, 1:] != sorted_a[:, :-1])
    group_id = np.cumsum(group_start, axis=1)

    avg_ranks_sorted = np.empty((m, n), dtype=float)
    for i in range(m):
        gid = group_id[i]
        sizes = np.bincount(gid)[1:]
        ends = np.cumsum(sizes)
        starts = ends - sizes + 1
        avg = (starts + ends) / 2.0
        avg_ranks_sorted[i] = avg[gid - 1]

    ranks = np.empty_like(a, dtype=float)
    ranks[rows, order] = avg_ranks_sorted
    return ranks


def _pearson_on_rank_arrays(rx, ry):
    """Pearson r on already-ranked 2-D arrays (vectorized over rows)."""
    rx_c = rx - np.mean(rx, axis=1, keepdims=True)
    ry_c = ry - np.mean(ry, axis=1, keepdims=True)
    num = np.sum(rx_c * ry_c, axis=1)
    denom = np.sqrt(np.sum(rx_c ** 2, axis=1) * np.sum(ry_c ** 2, axis=1))
    return np.where(denom < 1e-15, 0.0, num / denom)


def spearman_rho_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Vectorized Spearman rho for paired rows of two 2D arrays.
    Matches scipy.stats.spearmanr (and rankdata(method="average")) exactly,
    including proper average-rank tie handling.
    """
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must be 2D arrays with identical shape (n_rows, n)")

    rx = _rank_rows(x)
    ry = _rank_rows(y)
    return _pearson_on_rank_arrays(rx, ry)


def _fast_spearman_rho(x, y):
    """Spearman rho without p-value -- avoids scipy overhead."""
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    denom = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    if denom < 1e-15:
        return 0.0
    return np.dot(rx, ry) / denom


def spearman_rho_pvalue_2d(x: np.ndarray, y: np.ndarray, n: int):
    """
    Vectorized Spearman rho and two-sided p-value for 2D arrays.

    Parameters
    ----------
    x, y : ndarray, shape (n_rows, n)
    n : int
        Number of observations per row (== x.shape[1]).

    Returns
    -------
    rhos : ndarray, shape (n_rows,)
    pvals : ndarray, shape (n_rows,)
    """
    rhos = spearman_rho_2d(x, y)
    rho_sq = rhos * rhos
    denom = np.maximum(1.0 - rho_sq, 1e-15)
    t_stat = rhos * np.sqrt((n - 2) / denom)
    pvals = 2.0 * t_dist.sf(np.abs(t_stat), n - 2)
    return rhos, pvals
