# Permutation p-value: Implementation context

Self-contained context for implementing the permutation p-value feature per the [plan](permutation_pvalue_implementation_plan.md). Follow the implementation order in Section 11 of the plan. Each step below provides exact file locations, function signatures, algorithms, and existing code patterns to follow.

---

## Corrections applied to the plan

These corrections have already been applied to the plan file, but are documented here for reference:

1. **Memory table was wrong** — The original example table in Section 7 computed memory for n=10 instead of n≈80. Corrected: 10k sims × 1k perms at n≈80 ≈ 3.3 GB (not 400 MB). Formula: `(4*n + 8) * n_sims_chunk * n_perm` bytes.

2. **Chunk sizes reduced** — `N_SIMS_BATCH_THRESHOLD` lowered from 15k→5k, `N_SIMS_CHUNK_SIZE` from 10k→2k to keep peak memory under ~660 MB per chunk at n≈80, n_perm=1k.

3. **Simplified non-empirical path** — `get_precomputed_null` always auto-builds on cache miss (~1s for n≈80, n_pre=50k). No MC fallback needed for non-empirical. The warning mechanism and MC fallback for non-empirical were removed. Only empirical uses MC.

4. **Permutation index generation** — Added efficient batch method: `np.argsort(rng.random((n_perm, chunk_size, n)), axis=2).astype(np.int32)`.

5. **Reproducibility with chunking** — Changed from `default_rng((seed, chunk_idx))` to `rng.bit_generator.seed_seq.spawn(n_chunks)` since the caller has a Generator not a raw seed.

---

## Step 1: Config constants

**File:** `config.py` — append after the existing `BATCH_CI_BOOTSTRAP = True` line (line 254).

**Add these constants:**

```python
# ---------------------------------------------------------------------------
# Permutation p-value settings
# ---------------------------------------------------------------------------
USE_PERMUTATION_PVALUE = True

N_PERM_DEFAULT = 1000
N_PERM_LOW_SIMS = 2000
N_SIMS_THRESHOLD_FOR_N_PERM = 5000

PVALUE_PRECOMPUTED_N_PRE = 50_000

PVALUE_N_SIMS_BATCH_THRESHOLD = 5_000
PVALUE_N_SIMS_CHUNK_SIZE = 2_000
```

Semantics:
- `USE_PERMUTATION_PVALUE`: When True, `estimate_power` uses permutation-based p-values. When False, uses existing t-based `spearman_rho_pvalue_2d`.
- `N_PERM_DEFAULT` / `N_PERM_LOW_SIMS`: Adaptive n_perm for MC path. Use 1k when n_sims >= threshold, else 2k.
- `PVALUE_PRECOMPUTED_N_PRE`: Number of null rhos for precomputed null (50k gives SE(p) ≈ 0.001 at p=0.05).
- `PVALUE_N_SIMS_BATCH_THRESHOLD` / `PVALUE_N_SIMS_CHUNK_SIZE`: When n_sims exceeds threshold, MC path processes sims in chunks to cap memory. Each chunk of 2k sims × 1k perms × n≈80 uses ~660 MB.

---

## Step 2: `_batch_permutation_rhos_jit` in spearman_helpers.py

**File:** `spearman_helpers.py`

**Pattern to follow:** `_batch_bootstrap_rhos_jit` (lines 92–121). The permutation kernel is structurally identical but uses permutation indices to reorder y (without replacement) instead of bootstrap indices (with replacement).

**What to add:** Inside the `if _NUMBA_AVAILABLE:` block (after `_batch_bootstrap_rhos_jit`, before the `else` at line 122), add:

```python
@njit(fastmath=True, cache=True, parallel=True)
def _batch_permutation_rhos_jit(x_all, y_all, perm_idx_all):
    """Batch permutation Spearman rho, parallel over all (sim, perm) pairs.

    Parameters
    ----------
    x_all, y_all : (n_sims, n) float64
    perm_idx_all : (n_perm, n_sims, n) int32
        Pre-generated permutation indices (each [b, rep, :] is a
        permutation of 0..n-1).

    Returns
    -------
    result : (n_sims, n_perm) float64
    """
    n_perm, n_sims, n = perm_idx_all.shape
    result = np.empty((n_sims, n_perm), dtype=np.float64)
    total = n_sims * n_perm
    for flat in prange(total):
        rep = flat // n_perm
        b = flat % n_perm
        xb = np.empty(n, dtype=np.float64)
        yb = np.empty(n, dtype=np.float64)
        for i in range(n):
            xb[i] = x_all[rep, i]
            idx = perm_idx_all[b, rep, i]
            yb[i] = y_all[rep, idx]
        rx = _tied_rank(xb)
        ry = _tied_rank(yb)
        result[rep, b] = _pearson_on_ranks_1d(rx, ry)
    return result
```

**Key difference from bootstrap:** x is NOT reindexed — `xb[i] = x_all[rep, i]` (direct copy). Only y is permuted via `y_all[rep, perm_idx_all[b, rep, i]]`. Bootstrap reindexes both x and y with the same indices.

**In the `else` branch** (line 122–124), also add `_batch_permutation_rhos_jit = None` alongside the existing `_bootstrap_rhos_jit = None` and `_batch_bootstrap_rhos_jit = None`.

The `else` block should become:
```python
else:
    _bootstrap_rhos_jit = None
    _batch_bootstrap_rhos_jit = None
    _batch_permutation_rhos_jit = None
```

---

## Step 3: New module `permutation_pvalue.py`

**File:** New file `permutation_pvalue.py` at the project root.

**Imports:** Only `numpy`, `config`, and `spearman_helpers`. Does NOT import `power_asymptotic` or `data_generator`. The caller (`power_simulation.estimate_power`) obtains `x_counts` from `power_asymptotic.get_x_counts` and passes it in.

### 3a. Cache and `get_precomputed_null`

```python
"""
Permutation-based p-values for Spearman correlation.

Provides two paths:
1. Precomputed null — for non-empirical generators where y is continuous.
   A fixed null distribution of Spearman rhos is built once per (n, tie structure)
   and cached. P-values are computed via binary search against sorted |null_rho|.
2. Monte Carlo — for empirical generators (y may have ties that vary per dataset).
   Uses batched Numba parallelism over all (sim, perm) pairs.
"""

import numpy as np

from config import (PVALUE_PRECOMPUTED_N_PRE, N_PERM_DEFAULT, N_PERM_LOW_SIMS,
                    N_SIMS_THRESHOLD_FOR_N_PERM, PVALUE_N_SIMS_BATCH_THRESHOLD,
                    PVALUE_N_SIMS_CHUNK_SIZE)
from spearman_helpers import (spearman_rho_2d, _batch_permutation_rhos_jit,
                              use_numba)


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

    Auto-builds on cache miss (~1s for n≈80, n_pre=50k). Cached for
    future calls with the same (n, all_distinct, tuple(x_counts)) key.

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

    # Generate n_pre random permutations of {1, ..., n} and compute Pearson
    # correlation with x_std for each.
    base_ranks = np.arange(1.0, n + 1.0)
    null_rhos = np.empty(n_pre, dtype=np.float64)
    for i in range(n_pre):
        perm_y = rng.permutation(base_ranks)
        y_std = perm_y - np.mean(perm_y)
        sy = np.std(perm_y, ddof=0)
        if sy > 0:
            y_std = y_std / sy
        null_rhos[i] = np.dot(x_std, y_std) / n

    sorted_abs_null = np.sort(np.abs(null_rhos))
    _NULL_CACHE[key] = sorted_abs_null
    return sorted_abs_null
```

**IMPORTANT NOTE on Pearson computation:** The `np.dot(x_std, y_std) / n` above computes `mean((x-mx)/sx * (y-my)/sy)` which equals the Pearson correlation of x_midranks and perm_y. The y permutation is {1,...,n} shuffled, so no ties in y. This matches `_pearson_on_ranks_1d` in the Numba code. Since we're dividing by n (not n-1), this gives the sample Pearson correlation (population formula), matching the existing `_pearson_on_ranks_1d` which uses `np.mean((rx - mx) * (ry - my)) / sqrt(np.mean((rx-mx)**2) * np.mean((ry-my)**2))`.

Actually the formula above needs a correction. Let me reclarify the correct Pearson calculation. `_pearson_on_ranks_1d` computes:
```
cov = mean((rx - mx) * (ry - my))
varx = mean((rx - mx)^2)
vary = mean((ry - my)^2)
return cov / sqrt(varx * vary)
```

After standardisation (x_std = (x - mean) / std, y_std = (y - mean) / std), where std uses ddof=0:
```
cov = mean(x_std * y_std) = dot(x_std, y_std) / n
varx = mean(x_std^2) = 1.0
vary = mean(y_std^2) = 1.0
pearson = cov / sqrt(1 * 1) = dot(x_std, y_std) / n
```

So `np.dot(x_std, y_std) / n` is correct for the Pearson correlation when both vectors are standardised to mean=0 and std(ddof=0)=1.

### 3b. `pvalues_from_precomputed_null`

```python
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
```

Uses the Phipson-Smyth (2010) conservative formula: `p = (1 + count) / (1 + n_pre)`.

### 3c. `pvalues_mc` (Monte Carlo path for empirical)

```python
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
        child_seeds = rng.bit_generator.seed_seq.spawn(n_chunks)
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
    """Run one MC batch: generate perm indices, compute perm rhos, return p-values."""
    chunk_size, n = x_chunk.shape

    # Generate permutation indices: argsort of random floats gives permutations
    perm_idx = np.argsort(
        rng.random((n_perm, chunk_size, n)), axis=2
    ).astype(np.int32)

    if use_numba() and _batch_permutation_rhos_jit is not None:
        perm_rhos = _batch_permutation_rhos_jit(
            x_chunk.astype(np.float64),
            y_chunk.astype(np.float64),
            perm_idx)
    else:
        # Pure NumPy/Python fallback
        perm_rhos = np.empty((chunk_size, n_perm), dtype=np.float64)
        for b in range(n_perm):
            # perm_idx[b] has shape (chunk_size, n)
            rows = np.arange(chunk_size)[:, None]
            y_perm = y_chunk[rows, perm_idx[b]]
            perm_rhos[:, b] = spearman_rho_2d(x_chunk, y_perm)

    abs_obs = np.abs(rhos_obs_chunk)[:, np.newaxis]   # (chunk_size, 1)
    abs_perm = np.abs(perm_rhos)                       # (chunk_size, n_perm)
    count = np.sum(abs_perm >= abs_obs, axis=1)        # (chunk_size,)
    pvals = (1 + count) / (1 + n_perm)
    return pvals
```

**Fallback pattern:** Matches the CI bootstrap fallback in `confidence_interval_calculator.py` lines 301–307 (loop over bootstrap index, apply to all rows via `spearman_rho_2d`). The fallback loops over `n_perm` (not `n_sims`), keeping per-iteration work vectorized over the chunk.

### 3d. `warm_precomputed_null_cache`

```python
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
```

Note: `warm_precomputed_null_cache` imports `CASES`, `get_x_counts` etc. locally to avoid circular imports at module level. `permutation_pvalue.py` top-level imports only `config` and `spearman_helpers`.

---

## Step 4: Wire into `power_simulation.estimate_power`

**File:** `power_simulation.py`

### 4a. Add imports

After the existing imports (line 31–45), add:

```python
from config import (USE_PERMUTATION_PVALUE, N_PERM_DEFAULT, N_PERM_LOW_SIMS,
                    N_SIMS_THRESHOLD_FOR_N_PERM, PVALUE_PRECOMPUTED_N_PRE)
from power_asymptotic import get_x_counts
```

And conditionally:
```python
if USE_PERMUTATION_PVALUE:
    from permutation_pvalue import (get_precomputed_null,
                                     pvalues_from_precomputed_null,
                                     pvalues_mc, _get_n_perm)
```

Note: `spearman_rho_2d` is already available via `spearman_helpers` but not currently imported in power_simulation. Add it:
```python
from spearman_helpers import spearman_rho_pvalue_2d, spearman_rho_2d
```

### 4b. Modify `estimate_power` body

Replace the current final lines (145–146):
```python
    _, pvals = spearman_rho_pvalue_2d(x_all, y_all, n)
    return np.sum(pvals < alpha) / n_sims
```

With:
```python
    if not USE_PERMUTATION_PVALUE:
        _, pvals = spearman_rho_pvalue_2d(x_all, y_all, n)
        return np.sum(pvals < alpha) / n_sims

    # Permutation-based p-values
    if generator != "empirical":
        # Precomputed null (auto-builds on cache miss)
        x_counts = get_x_counts(n, n_distinct, distribution_type=distribution_type,
                                all_distinct=all_distinct, freq_dict=freq_dict)
        sorted_abs_null = get_precomputed_null(
            n, all_distinct, x_counts, n_pre=PVALUE_PRECOMPUTED_N_PRE, rng=rng)
        rhos_obs = spearman_rho_2d(x_all, y_all)
        pvals = pvalues_from_precomputed_null(rhos_obs, sorted_abs_null)
    else:
        # Empirical: per-dataset Monte Carlo
        n_perm = _get_n_perm(n_sims)
        reject, pvals, rhos_obs = pvalues_mc(x_all, y_all, n_perm, alpha, rng)

    return np.sum(pvals < alpha) / n_sims
```

**Parameter threading:** `distribution_type` and `freq_dict` are already available in `estimate_power`'s scope. `n_distinct` is a parameter. `all_distinct` is a parameter. So the `get_x_counts` call has all needed arguments.

**`get_x_counts` signature** (from `power_asymptotic.py` line 269):
```python
def get_x_counts(n, n_distinct, distribution_type=None, all_distinct=False,
                 x_counts=None, freq_dict=None):
```
When `all_distinct=True`, returns `np.ones(n)`. When False, looks up `FREQ_DICT[n][n_distinct][distribution_type]`. When `freq_dict` is provided with `distribution_type="custom"`, looks up `freq_dict[n][n_distinct]["custom"]`.

---

## Step 5: Unit tests

**File:** New file `tests/test_permutation_pvalue.py`

Follow existing test patterns — see `tests/test_power_sanity.py` for the boilerplate:

```python
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
```

### Tests to include:

**Test 1 — Precomputed null shape and statistics:**
```python
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
```

**Test 2 — Precomputed null p-value for rho=0 is near 1:**
```python
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
```

**Test 3 — Monte Carlo p-value returns valid results:**
```python
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
```

**Test 4 — Smoke test for `estimate_power` with permutation p-value:**
```python
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
```

**Test 5 — Cache hit (second call reuses cached null):**
```python
def test_precomputed_null_cache_hit():
    """Second call with same args should return same object (cache hit)."""
    from permutation_pvalue import get_precomputed_null, _NULL_CACHE
    import numpy as np

    x_counts = np.array([20, 20, 20, 20])
    rng = np.random.default_rng(99)
    null1 = get_precomputed_null(80, False, x_counts, n_pre=1000, rng=rng)
    null2 = get_precomputed_null(80, False, x_counts, n_pre=1000, rng=rng)
    assert null1 is null2  # same object from cache
```

### Register in `tests/run_tests.py`

Add to the `QUICK_TESTS` list (before `test_power_sanity.py`):
```python
("test_permutation_pvalue.py", []),
```

---

## Step 6: Benchmark script

**File:** New file `benchmarks/benchmark_permutation_pvalue.py`

Follow the pattern of `benchmarks/benchmark_power.py`: sys.path setup, warmup, time operations, print results.

```python
"""
Benchmark permutation p-value: precomputed null build/lookup, MC path,
and full grid. Run one at a time per benchmarking rule.
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import time
import numpy as np
from config import CASES
from power_simulation import estimate_power
from permutation_pvalue import get_precomputed_null, pvalues_from_precomputed_null, _NULL_CACHE
from power_asymptotic import get_x_counts
from spearman_helpers import spearman_rho_2d

CASE_ID = 3
CASE = CASES[CASE_ID]
Y_PARAMS = {"median": CASE["median"], "iqr": CASE["iqr"], "range": CASE["range"]}
```

Include benchmarks for:
1. **Precomputed null build (cache miss):** Time `get_precomputed_null` for one scenario with cleared cache. Report time and array size.
2. **Precomputed null lookup (cache hit):** Time `pvalues_from_precomputed_null` for 10k observed rhos.
3. **MC path:** Time `estimate_power` with `generator="nonparametric"` and permutation p-value enabled, n_sims=1000.
4. **Comparison with t-based:** Time `estimate_power` with `USE_PERMUTATION_PVALUE` True vs False (may need to toggle the config).

---

## Step 7: README updates

**File:** `README.md`

Add a new subsection after "### Asymptotic" (line ~125) or in "## Performance and Runtime Estimates":

### Content to add:

1. **P-value method subsection** — explaining: power simulation uses permutation-based p-values (precomputed null for non-empirical, per-dataset MC for empirical). Rationale: t-approximation unreliable with heavy ties. Non-empirical auto-builds precomputed null on first access (~1s). Optional `warm_precomputed_null_cache()` for pre-warming before long runs.

2. **Memory subsection** — the corrected table and formula from Section 7 of the plan.

3. **Precision formulas subsection** — the SE formulas from Section 7 of the plan (power binomial SE, bisection SE, calibration SE, combined SE, rounding guidance).

---

## Key existing code patterns to follow

### Import convention
All modules use this pattern for root-relative imports:
```python
from config import ...
from spearman_helpers import ...
```
No `src.` prefix or package structure.

### Numba availability check
```python
from spearman_helpers import use_numba, _batch_permutation_rhos_jit

if use_numba() and _batch_permutation_rhos_jit is not None:
    # Numba JIT path
else:
    # Pure NumPy fallback
```

### Cache pattern
Global dict, keyed by scenario tuple. See `_X_TEMPLATE_CACHE` in `data_generator.py`, `_CALIBRATION_CACHE` in `data_generator.py`, and `_NULL_CACHE` in the new module.

### RNG handling
Functions accept `rng=None` and default to `np.random.default_rng()`. `estimate_power` creates `rng = np.random.default_rng(seed)` at line 86 and passes it through. The new permutation code receives this same `rng`.

### Test boilerplate
```python
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
```

### Benchmark boilerplate
Same sys.path setup. Warmup function. `time.perf_counter()` for timing. Print results. No assertions.

---

## Dependency graph (no circular imports)

```
config.py  (no project imports)
    ↑
spearman_helpers.py  (imports config)
    ↑
permutation_pvalue.py  (imports config, spearman_helpers)
    ↑
power_asymptotic.py  (imports config)
    ↑
power_simulation.py  (imports config, data_generator, spearman_helpers, power_asymptotic, permutation_pvalue)
```

`permutation_pvalue.py` does NOT import `power_asymptotic` or `data_generator` at the top level. The `warm_precomputed_null_cache` function imports them locally to avoid cycles.

---

## Implementation order (from plan Section 11)

1. Config constants in `config.py`
2. `_batch_permutation_rhos_jit` in `spearman_helpers.py`
3. New `permutation_pvalue.py` (cache, precomputed null, MC path, warm function)
4. Wire into `power_simulation.estimate_power`
5. Unit tests in `tests/test_permutation_pvalue.py` + register in `run_tests.py`
6. Benchmark script in `benchmarks/benchmark_permutation_pvalue.py`
7. README updates

After each step, run `python tests/run_tests.py` to verify nothing is broken. After step 5, run `python tests/test_permutation_pvalue.py` specifically.
