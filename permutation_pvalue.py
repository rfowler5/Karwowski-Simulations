"""
Permutation-based p-values for Spearman correlation.

Provides two paths:
1. Precomputed null — for all generators (including empirical by default).
   A fixed null distribution of Spearman rhos is built once per (n, tie structure)
   and cached. P-values are computed via binary search against sorted |null_rho|.
   For empirical generator, this introduces a negligible approximation from
   ignoring y-ties (bias < 10^-5 on p-values; see README).
2. Monte Carlo — optional fallback for empirical generator when
   config.EMPIRICAL_USE_PRECOMPUTED_NULL is False, or for any generator
   when PVALUE_MC_ON_CACHE_MISS is True and the null is not cached.
   Uses batched Numba parallelism over all (sim, perm) pairs.
"""

import numpy as np

from config import (PVALUE_PRECOMPUTED_N_PRE, N_PERM_DEFAULT, N_PERM_LOW_SIMS,
                    N_SIMS_THRESHOLD_FOR_N_PERM, PVALUE_N_SIMS_BATCH_THRESHOLD,
                    PVALUE_N_SIMS_CHUNK_SIZE, PVALUE_MC_ON_CACHE_MISS)
from spearman_helpers import (spearman_rho_2d, _batch_permutation_rhos_jit,
                              _batch_permutation_rhos_preranked_jit,
                              _rank_rows, _pearson_on_rank_arrays, use_numba)


_NULL_CACHE = {}


def _build_x_midranks(x_counts):
    """Build the midrank vector from group counts.

    For group sizes [c1, c2, ...], assigns midranks per group:
    group 1 occupies positions 1..c1, midrank = (1 + c1) / 2;
    group 2 occupies positions c1+1..c1+c2, midrank = (c1 + 1 + c1 + c2) / 2; etc.

    Returns float64 array of length sum(x_counts).
    """
    n = int(np.sum(x_counts))
    ranks = np.empty(n, dtype=np.float64)
    pos = 0
    for c in x_counts:
        c = int(c)
        midrank = (2 * pos + c + 1) / 2.0  # = (pos+1 + pos+c) / 2
        ranks[pos:pos + c] = midrank
        pos += c
    return ranks


def get_precomputed_null(n, all_distinct, x_counts, n_pre=None, rng=None):
    """Return sorted |null_rho| array for the given tie structure.

    Auto-builds on cache miss (~0.3s for n≈80, n_pre=50k via vectorised
    matmul). Cached for future calls with the same
    (n, all_distinct, tuple(x_counts)) key.

    The generator (nonparametric, copula, linear) is intentionally not part
    of the cache key.  All non-empirical generators produce continuous y with
    no ties, so under H0 the y-ranks are a uniform random permutation of
    {1,...,n} regardless of how y was generated.  The permutation null
    distribution of Spearman rho therefore depends only on n and the x tie
    structure, not on the generator.  The same cached null is correctly reused
    across all non-empirical generators for the same scenario.

    Parameters
    ----------
    n : int
        Sample size.
    all_distinct : bool
        True if x has no ties.
    x_counts : array-like
        Group sizes for tied x-values (from power_asymptotic.get_x_counts).
    n_pre : int or None
        Number of null rhos to generate. Defaults to config.PVALUE_PRECOMPUTED_N_PRE.
    rng : numpy.random.Generator or None
        Used only for building (cache miss). The null is deterministic
        per cache key once built.

    Returns
    -------
    sorted_abs_null : ndarray of shape (n_pre,)
        Sorted absolute values of null Spearman rhos.
    """
    if n_pre is None:
        n_pre = PVALUE_PRECOMPUTED_N_PRE

    key = (n, all_distinct, tuple(int(c) for c in x_counts))

    if key in _NULL_CACHE:
        return _NULL_CACHE[key]

    if rng is None:
        rng = np.random.default_rng()

    x_midranks = _build_x_midranks(x_counts)

    # Standardise x_midranks for Pearson
    x_std = x_midranks - np.mean(x_midranks)
    sx = np.std(x_midranks, ddof=0)
    if sx > 0:
        x_std = x_std / sx

    std_y = np.sqrt((n * n - 1) / 12.0)

    # Vectorised: generate all n_pre permutations at once (argsort of random
    # floats, same pattern as _mc_batch), then a single matmul for all rhos.
    # x_std is zero-mean so dot(x_std, perm_y - mean_y) = dot(x_std, perm_y).
    all_perm_y = np.argsort(rng.random((n_pre, n)), axis=1) + 1.0
    # std_y > 0 for all n >= 2; guard retained for defensive completeness
    if std_y > 0:
        null_rhos = (all_perm_y @ x_std) / (std_y * n)
    else:
        null_rhos = np.zeros(n_pre, dtype=np.float64)

    sorted_abs_null = np.sort(np.abs(null_rhos))
    _NULL_CACHE[key] = sorted_abs_null
    return sorted_abs_null


def get_cached_null(n, all_distinct, x_counts):
    """Return the cached sorted |null_rho| array, or None if not in cache.

    Unlike get_precomputed_null, this never builds the null on a cache miss.
    Use when PVALUE_MC_ON_CACHE_MISS is True so callers can fall back to MC
    instead of paying the build cost.

    Parameters
    ----------
    n : int
    all_distinct : bool
    x_counts : array-like

    Returns
    -------
    sorted_abs_null : ndarray or None
    """
    key = (n, all_distinct, tuple(int(c) for c in x_counts))
    return _NULL_CACHE.get(key, None)


def pvalues_from_precomputed_null(rhos_obs, sorted_abs_null):
    """Compute two-sided permutation p-values from a precomputed null.

    Parameters
    ----------
    rhos_obs : ndarray of shape (n_sims,)
        Observed Spearman rhos.
    sorted_abs_null : ndarray of shape (n_pre,)
        Sorted absolute values of null rhos (from get_precomputed_null).

    Returns
    -------
    pvals : ndarray of shape (n_sims,)
    """
    n_pre = len(sorted_abs_null)
    abs_obs = np.abs(rhos_obs)
    # Number of null rhos with |null_rho| >= |rho_obs|
    insertion = np.searchsorted(sorted_abs_null, abs_obs, side='left')
    count = n_pre - insertion
    pvals = (1 + count) / (1 + n_pre)
    return pvals


def _get_n_perm(n_sims):
    """Return adaptive n_perm based on n_sims."""
    if n_sims >= N_SIMS_THRESHOLD_FOR_N_PERM:
        return N_PERM_DEFAULT
    return N_PERM_LOW_SIMS


def pvalues_mc(x_all, y_all, n_perm, alpha, rng,
               n_sims_batch_threshold=None, n_sims_chunk_size=None):
    """Permutation p-values via per-dataset Monte Carlo.

    Uses batched Numba parallelism. When n_sims exceeds the batch threshold,
    processes sims in chunks to cap memory.

    Parameters
    ----------
    x_all, y_all : ndarray of shape (n_sims, n)
    n_perm : int
    alpha : float
    rng : numpy.random.Generator
    n_sims_batch_threshold : int or None
    n_sims_chunk_size : int or None

    Returns
    -------
    reject : ndarray of shape (n_sims,), dtype bool
        True where permutation p < alpha.
    pvals : ndarray of shape (n_sims,)
    rhos_obs : ndarray of shape (n_sims,)
    """
    if n_sims_batch_threshold is None:
        n_sims_batch_threshold = PVALUE_N_SIMS_BATCH_THRESHOLD
    if n_sims_chunk_size is None:
        n_sims_chunk_size = PVALUE_N_SIMS_CHUNK_SIZE

    n_sims, n = x_all.shape
    rhos_obs = spearman_rho_2d(x_all, y_all)

    if n_sims <= n_sims_batch_threshold:
        pvals = _mc_batch(x_all, y_all, rhos_obs, n_perm, rng)
    else:
        # Chunk processing with deterministic child RNG streams
        n_chunks = (n_sims + n_sims_chunk_size - 1) // n_sims_chunk_size
        child_seeds = np.random.SeedSequence(rng.integers(2**63)).spawn(n_chunks)
        pvals = np.empty(n_sims, dtype=np.float64)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_sims_chunk_size
            end = min(start + n_sims_chunk_size, n_sims)
            chunk_rng = np.random.default_rng(child_seeds[chunk_idx])

            pvals[start:end] = _mc_batch(
                x_all[start:end], y_all[start:end],
                rhos_obs[start:end], n_perm, chunk_rng)

    reject = pvals < alpha
    return reject, pvals, rhos_obs


def _mc_batch(x_chunk, y_chunk, rhos_obs_chunk, n_perm, rng):
    """Run one MC batch: generate perm indices, compute perm rhos, return p-values.

    Precomputes x-ranks once per simulation (x does not change across
    permutations), halving the rank computation vs. ranking x inside the loop.
    """
    chunk_size, n = x_chunk.shape

    # Generate permutation indices: argsort of random floats gives permutations
    perm_idx = np.argsort(
        rng.random((n_perm, chunk_size, n)), axis=2
    ).astype(np.int32)

    if use_numba() and _batch_permutation_rhos_preranked_jit is not None:
        # Precompute x-ranks once per sim (constant across permutations)
        rx_all = _rank_rows(x_chunk.astype(np.float64))
        perm_rhos = _batch_permutation_rhos_preranked_jit(
            rx_all,
            y_chunk.astype(np.float64),
            perm_idx)
    elif use_numba() and _batch_permutation_rhos_jit is not None:
        perm_rhos = _batch_permutation_rhos_jit(
            x_chunk.astype(np.float64),
            y_chunk.astype(np.float64),
            perm_idx)
    else:
        # Pure NumPy fallback: precompute x-ranks once, rank only y per perm
        rx_all = _rank_rows(x_chunk)
        perm_rhos = np.empty((chunk_size, n_perm), dtype=np.float64)
        rows = np.arange(chunk_size)[:, None]
        for b in range(n_perm):
            y_perm = y_chunk[rows, perm_idx[b]]
            ry = _rank_rows(y_perm)
            perm_rhos[:, b] = _pearson_on_rank_arrays(rx_all, ry)

    abs_obs = np.abs(rhos_obs_chunk)[:, np.newaxis]   # (chunk_size, 1)
    abs_perm = np.abs(perm_rhos)                       # (chunk_size, n_perm)
    count = np.sum(abs_perm >= abs_obs, axis=1)        # (chunk_size,)
    pvals = (1 + count) / (1 + n_perm)
    return pvals


def warm_precomputed_null_cache(cases=None, n_distinct_values=None,
                                 dist_types=None, freq_dict=None,
                                 n_pre=None, seed=42):
    """Pre-build and cache precomputed nulls for the given scenario grid.

    Parameters
    ----------
    cases : dict or None
        Case definitions (default: config.CASES).
    n_distinct_values : list of int or None
        Default: config.N_DISTINCT_VALUES.
    dist_types : list of str or None
        Default: config.DISTRIBUTION_TYPES.
    freq_dict : dict or None
        Custom frequency dict (default: config.FREQ_DICT).
    n_pre : int or None
        Default: config.PVALUE_PRECOMPUTED_N_PRE.
    seed : int
        Seed for reproducibility.
    """
    from config import CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES
    from power_asymptotic import get_x_counts

    if cases is None:
        cases = CASES
    if n_distinct_values is None:
        n_distinct_values = N_DISTINCT_VALUES
    if dist_types is None:
        dist_types = DISTRIBUTION_TYPES

    rng = np.random.default_rng(seed)
    built = 0

    for case_id, case in cases.items():
        n = case["n"]
        # Tied scenarios
        for k in n_distinct_values:
            for dt in dist_types:
                x_counts = get_x_counts(n, k, distribution_type=dt,
                                        freq_dict=freq_dict)
                get_precomputed_null(n, False, x_counts, n_pre=n_pre, rng=rng)
                built += 1
        # All-distinct
        x_counts = get_x_counts(n, n, all_distinct=True)
        get_precomputed_null(n, True, x_counts, n_pre=n_pre, rng=rng)
        built += 1

    return built
