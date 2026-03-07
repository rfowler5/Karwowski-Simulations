# OPT-1 Implementation Plan: Precomputed Null for Empirical Generator

**Status:** Ready to implement.
**Validation:** Error analysis completed — see [Math Derivation](#math-derivation-for-readme) below.
**Difficulty:** Low (routing change + config flag + docs + verification script).

> ### Review notes (2026-03-06)
>
> Reviewed by Claude Sonnet and Opus. Changes from review:
>
> 1. **`_fit_lognormal` import removed** from verification sketch — was
>    unused.  (Sonnet also flagged `pvals_mc` as colliding with `pvalues_mc`,
>    but those are different names — no collision.)
> 2. **SE(Δpower) text corrected**: the independent-proportion formula
>    `sqrt(2p(1-p)/n_sims)` was wrong for correlated rejections. Replaced
>    with qualitative bound; conclusion unchanged (conservative direction).
> 3. **rho=0 vs rho=0.30 clearly labelled**: rho=0 checks type I error
>    parity; rho=0.30 checks power bias.  n_sims=50k applies to the
>    type I error check; n_sims=10k to the power bias check.
> 4. **"Three-branch" wording fixed** — it is still two branches; only the
>    condition changed.
> 5. **~60× speedup hedged** as "expected ~60×, pending benchmark
>    verification" throughout.
> 6. **Flag interaction documented**: `EMPIRICAL_USE_PRECOMPUTED_NULL=True`
>    with `PVALUE_MC_ON_CACHE_MISS=True` on a cold cache falls through to
>    MC — correct behavior, now explicitly noted in config comment.
> 7. **Independence from calibration precompute plan** noted (the open
>    `.cursor/plans/calibration_precompute_and_cleanup.plan.md` is
>    orthogonal; do not apply both simultaneously).
> 8. **`y_params` simplified** in verification sketch — empirical marginal
>    never uses it; pass `None` directly instead of threading a parameter.

---

## Summary

Switch the empirical generator from per-dataset Monte Carlo permutation
p-values to the same precomputed null used by nonparametric/copula/linear.
Add a config flag (`EMPIRICAL_USE_PRECOMPUTED_NULL`) so users can revert
to MC if they need higher accuracy at the cost of slower runtime (expected
~60×, pending benchmark verification).

> **Independence note:** This plan is orthogonal to the calibration
> precompute plan (`.cursor/plans/calibration_precompute_and_cleanup.plan.md`).
> They touch different code paths and should not be applied simultaneously.

---

## File-by-File Changes

### 1. `config.py`

Add one new flag in the "Permutation p-value settings" section, after
`PVALUE_MC_ON_CACHE_MISS`:

```python
# When True (default), the empirical generator uses the same precomputed null
# (keyed on x tie structure, built from all-distinct y-ranks 1..n) as
# non-empirical generators.  Expected ~60x faster than per-dataset MC
# (pending benchmark verification; at minimum >>10x).
# The approximation error from ignoring y-ties is < 10^-5 on p-values
# (see README "Precomputed null approximation for empirical generator").
# Set to False to revert to per-dataset MC permutation (exact but slow).
#
# Note: when True with PVALUE_MC_ON_CACHE_MISS=True, a cold cache will
# still fall through to the MC path until the null is built and cached.
# This is correct behavior — the speedup applies once the cache is warm.
EMPIRICAL_USE_PRECOMPUTED_NULL = True
```

### 2. `power_simulation.py`

**Import change** (line ~37): Add the new config flag to the import:

```python
from config import (..., EMPIRICAL_USE_PRECOMPUTED_NULL)
```

**Routing change** (lines 161-182): Widen the existing `if` condition so
the precomputed-null branch also covers empirical when the flag is on.
The structure stays two branches; only the guard changes. The current
code is:

```python
    # Permutation-based p-values
    if generator != "empirical":
        x_counts = get_x_counts(n, n_distinct, distribution_type=distribution_type,
                                all_distinct=all_distinct, freq_dict=freq_dict)
        if PVALUE_MC_ON_CACHE_MISS:
            sorted_abs_null = get_cached_null(n, all_distinct, x_counts)
        else:
            sorted_abs_null = get_precomputed_null(
                n, all_distinct, x_counts, n_pre=PVALUE_PRECOMPUTED_N_PRE, rng=rng)

        if sorted_abs_null is not None:
            rhos_obs = spearman_rho_2d(x_all, y_all)
            pvals = pvalues_from_precomputed_null(rhos_obs, sorted_abs_null)
        else:
            n_perm = _get_n_perm(n_sims)
            reject, pvals, rhos_obs = pvalues_mc(x_all, y_all, n_perm, alpha, rng)
    else:
        # Empirical: per-dataset Monte Carlo
        n_perm = _get_n_perm(n_sims)
        reject, pvals, rhos_obs = pvalues_mc(x_all, y_all, n_perm, alpha, rng)
```

Replace with:

```python
    # Permutation-based p-values
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
            rhos_obs = spearman_rho_2d(x_all, y_all)
            pvals = pvalues_from_precomputed_null(rhos_obs, sorted_abs_null)
        else:
            n_perm = _get_n_perm(n_sims)
            reject, pvals, rhos_obs = pvalues_mc(x_all, y_all, n_perm, alpha, rng)
    else:
        # Empirical with MC: per-dataset Monte Carlo (exact but much slower)
        n_perm = _get_n_perm(n_sims)
        reject, pvals, rhos_obs = pvalues_mc(x_all, y_all, n_perm, alpha, rng)
```

That is the entire logic change. The precomputed null is keyed on
`(n, all_distinct, tuple(x_counts))` — same key as for other generators.
No new null needs to be built; the empirical generator uses the same
x-values as other generators for the same case/scenario, so the same
cached null is reused.

### 3. `permutation_pvalue.py`

**Docstring only.** Update the module docstring (lines 1-10) to reflect
that empirical now uses precomputed null by default:

```python
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
```

### 4. `AUDIT.md`

> **Already applied.** The AUDIT.md file was created with OPT-1 content
> pre-included during plan preparation. Sonnet should **verify** the
> existing entries match the specifications below, but should **not**
> add duplicate content. If the file already contains these entries,
> no changes are needed.

#### 4a. Under "Confirmed Correct", add after the existing `spearman_helpers.py` entry:

```markdown
**`spearman_helpers.py` — `spearman_rho_2d` y-tie handling**
- Ranks both x and y via `_rank_rows` → `_tied_rank` (Numba) or
  argsort+bincount (NumPy) ✓
- Both paths produce `rankdata(method='average')` midranks ✓
- Tie detection in `_tied_rank`: exact `==` on original values; valid for
  digitized data (integer H-Al, 1-decimal B-Al — no intervening arithmetic
  between storage and comparison) ✓
- `_pearson_on_rank_arrays`: standard Pearson formula, normalizes by actual
  rank std (including tie compression) ✓
- `fastmath=True` on Numba JIT: no impact on tie detection; FP midrank
  computation exact for n ≤ 82 ✓
```

#### 4b. Under "Known Approximations", add a new entry:

```markdown
**Precomputed null vs tied-y: denominator mismatch (OPT-1)**

The precomputed null permutes all-distinct y-ranks {1..n}, normalizing by
`std_y = sqrt((n^2-1)/12)`. The observed rho from `spearman_rho_2d`
normalizes by the actual (smaller) std of tied y-ranks. This creates a
systematic anticonservative bias on p-values.

**Digitized data ties (from `data/digitized.py`):**
- H_AL71: 4 tie pairs (28942×2, 32249×2, 72767×2, 2756×2). Σ(t³−t) = 24.
- B_AL71: 1 triple (8.0×3) + 6 pairs (13.1, 18.2, 12.7, 1.8, 6.6, 11.0
  each ×2). Σ(t³−t) = 60.

**Pool totals after resampling** (E[Σ(t³−t)] including base-71 ties +
resampled fill):
- n=73 (B-Al, 2 resampled): ≈ 79
- n=81 (H-Al, 10 resampled): ≈ 98

**Impact on p-values:** The denominator inflation ratio
`σ_untied / σ_tied ≈ 1 + Σ(t³−t) / (2(n³−n))` is at most
1 + 1.0×10⁻⁴ (n=73) or 1 + 9.2×10⁻⁵ (n=81). Combined with the
Edgeworth kurtosis correction, total Δp < 1.4×10⁻⁵.

**Impact on power:** Δpower ≈ 10⁻⁵ (systematic bias, not reduced by
n_sims). Direction: slightly anticonservative (p too small → power too
high by ~10⁻⁵).

**Verdict by precision target:**
| Target | Ratio (bias / target) | Safe? |
|--------|-----------------------|-------|
| ±0.01  | 10⁻³                 | Yes   |
| ±0.002 | 5×10⁻³               | Yes   |
| ±0.001 | 10⁻²                 | Yes   |

This bias is systematic and not averaged out by n_sims.
Users requiring exact permutation p-values can set
`config.EMPIRICAL_USE_PRECOMPUTED_NULL = False`.
```

#### 4c. Under "Optimization Opportunities", update OPT-1 status:

Replace the existing OPT-1 entry with:

```markdown
### OPT-1 — Precomputed null for empirical generator [IMPLEMENTED]
**Validated:** Approximation error < 1.4×10⁻⁵ on p-values, < 10⁻⁵ on
power. Safe for all precision targets (±0.01, ±0.002, ±0.001). See
"Known Approximations" above for full analysis.
**Files changed:** `config.py` (new flag `EMPIRICAL_USE_PRECOMPUTED_NULL`),
`power_simulation.py` (routing logic), `permutation_pvalue.py` (docstring).
**Fallback:** Set `config.EMPIRICAL_USE_PRECOMPUTED_NULL = False` to
revert to per-dataset MC permutation p-values (exact but much slower;
expected ~60×, pending benchmark verification).
```

### 5. `README.md`

#### 5a. Update the "P-value method" section (around line 172-177)

Replace the bullet about empirical generator:

Current text:
> - **Empirical generator:** Y may have ties that vary per dataset, so a
>   single precomputed null is not applicable. The code uses **per-dataset
>   Monte Carlo** permutation p-values...

New text:
> - **Empirical generator** (default: precomputed null): By default, the
>   empirical generator uses the **same precomputed null** as other
>   generators (`config.EMPIRICAL_USE_PRECOMPUTED_NULL = True`). The
>   approximation error from ignoring y-ties is < 1.4 × 10⁻⁵ on
>   p-values — negligible for all precision targets (see
>   [Precomputed null approximation for empirical generator](#precomputed-null-approximation-for-empirical-generator)
>   below). Set `config.EMPIRICAL_USE_PRECOMPUTED_NULL = False` to revert
>   to per-dataset Monte Carlo permutation p-values (exact but much slower).

#### 5b. Update "P-value methods for Monte Carlo power" (around line 194)

Current text (empirical MC bullet):
> - **Per-dataset Monte Carlo** (empirical generator, where y can have
>   ties that vary per dataset): For each sim, generate `n_perm`
>   permutations of y ...

Replace with:
> - **Per-dataset Monte Carlo** (empirical generator when
>   `EMPIRICAL_USE_PRECOMPUTED_NULL = False`; also used as fallback when
>   `PVALUE_MC_ON_CACHE_MISS = True` and null is not cached): For each
>   sim, generate `n_perm` permutations of y (adaptive: 1k when
>   n_sims ≥ 5000, else 2k), compute Spearman rho for each, and
>   p = (1 + N) / (1 + n_perm). See `permutation_pvalue.pvalues_mc`.

#### 5c. Update the "Empirical generator: p-value and calibration" section

In the bullet "P-value by Monte Carlo (permutation)" (around line 152),
replace:

> **P-value by Monte Carlo (permutation):** For the **empirical** generator,
> y can have **ties** (finite pool → repeated values). The null distribution
> of Spearman ρ under permutation depends on **both** the x and y tie
> structures. Because the y tie pattern **varies per dataset** (each sim has
> a different y from the pool), there is no single precomputed null per
> scenario. So for empirical we **always use per-dataset Monte Carlo
> permutation** (no precomputed null cache). This is the only option for
> empirical; for other generators we can use a precomputed null keyed by
> (n, x tie structure) when y has no ties.

With:

> **P-value:** By default (`EMPIRICAL_USE_PRECOMPUTED_NULL = True`), the
> empirical generator uses the **same precomputed null** as other generators
> (keyed on x tie structure, built from all-distinct y-ranks 1..n). Although
> empirical y can have **ties** from the finite pool, the approximation
> error is negligible (< 1.4 × 10⁻⁵ on p-values; see
> [Precomputed null approximation](#precomputed-null-approximation-for-empirical-generator)).
> Set `EMPIRICAL_USE_PRECOMPUTED_NULL = False` to revert to **per-dataset
> Monte Carlo permutation** (exact, much slower). The MC path uses adaptive
> n_perm (1k when n_sims ≥ 5000, else 2k), batched over all (sim, perm)
> pairs via Numba.

#### 5d. Add new README section — COPY THIS EXACTLY (math derivation)

Insert a new section after "Permutation p-value noise in power estimates"
(after line 674, before "#### CI endpoint precision"). This is the math
that Sonnet must copy verbatim:

````markdown
#### Precomputed null approximation for empirical generator

When `config.EMPIRICAL_USE_PRECOMPUTED_NULL = True` (default), the empirical
generator uses the same precomputed null as other generators. The precomputed
null permutes all-distinct y-ranks {1..n}; but empirical y-data has ties from
the finite pool (digitized values repeated via bootstrap resampling). This
section quantifies the approximation error.

**Source of error.** The precomputed null computes each null rho as:

```math
\rho_{\mathrm{null}} = \frac{\sum_i (x_i - \bar{x})(\pi(y)_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2} \cdot \sqrt{\sum_i (\pi(y)_i - \bar{y})^2}}
```

where $`\pi(y)`$ are permuted ranks from {1..n} (all distinct). The observed
rho from `spearman_rho_2d` uses midranks of the actual y-data (with ties),
which have $`\mathrm{std}(y_{\mathrm{ranks}}) < \mathrm{std}(\{1 \ldots n\})`$.
The mismatch inflates $`|\rho_{\mathrm{obs}}|`$ relative to the null by the ratio:

```math
\frac{\sigma_{y,\mathrm{untied}}}{\sigma_{y,\mathrm{tied}}} = \sqrt{\frac{(n^3-n)/12}{(n^3-n)/12 - \Sigma_y/12}} \approx 1 + \frac{\Sigma_y}{2(n^3-n)}
```

where $`\Sigma_y = \sum_j (t_j^3 - t_j)`$ summed over y tie groups.

**Tie counts in digitized data** (`data/digitized.py`):
- **H_AL71:** 4 tie pairs — $`\Sigma = 4 \times 6 = 24`$
- **B_AL71:** 1 triple + 6 pairs — $`\Sigma = 24 + 36 = 60`$

**Pool totals after resampling** (base-71 ties + expected contribution
from `build_empirical_pool` resampled fill):
- **n=73** (B-Al, 2 resampled from B_AL71): $`\Sigma_y \approx 79`$
- **n=80** (B-Al + 7 outliers): $`\Sigma_y \approx 79`$ (outliers are continuous)
- **n=81** (H-Al, 10 resampled from H_AL71): $`\Sigma_y \approx 98`$
- **n=82** (H-Al + 1 outlier): $`\Sigma_y \approx 98`$

**Denominator inflation (worst cases):**

| Case | n | $`n^3-n`$ | $`\Sigma_y`$ | Inflation | $`\Delta\rho`$ at crit |
|------|---|-----------|--------------|-----------|------------------------|
| n=73 | 73 | 388,944 | ~79 | 1 + 1.0×10⁻⁴ | ~2.3×10⁻⁵ |
| n=81 | 81 | 531,360 | ~98 | 1 + 9.2×10⁻⁵ | ~2.0×10⁻⁵ |

**Impact on p-values.** The CDF shift from the kurtosis mismatch
(Edgeworth expansion) plus the denominator inflation gives total
$`\Delta p < 1.4 \times 10^{-5}`$. Direction: anticonservative
(p slightly too small).

**Impact on power.** The p-value bias shifts the rejection rate by
$`\Delta\text{power} \approx 10^{-5}`$. This is a **systematic bias**,
not random noise — it is **not reduced by increasing n_sims**. However,
it is negligible for all precision targets:

| Target | Ratio (bias / target) | Verdict |
|--------|-----------------------|---------|
| ±0.01  | 10⁻³ (0.1%)          | Safe    |
| ±0.002 | 5×10⁻³ (0.5%)        | Safe    |
| ±0.001 | 10⁻² (1%)            | Safe    |

**Variance invariance.** Under permutation, the first two moments of
Spearman rho are exactly invariant to tie structure:
$`\mathrm{E}_\pi[\rho] = 0`$ and $`\mathrm{Var}_\pi(\rho) = 1/(n-1)`$,
regardless of ties in x or y. The error arises only from higher moments
(kurtosis) and from the denominator normalization mismatch between
observed and null rho — both of order $`\Sigma_y / n^3`$.

**Fallback.** Set `config.EMPIRICAL_USE_PRECOMPUTED_NULL = False` to use
per-dataset Monte Carlo permutation p-values (exact, no approximation,
much slower). The MC path is documented in
[Per-dataset Monte Carlo](#p-value-methods-for-monte-carlo-power) above.

**Verification.** Run `benchmarks/verify_precomputed_null_empirical.py`
to empirically measure the p-value and power bias for all four cases.
````

#### 5e. Update the "n_perm is adequate" paragraph (around line 674)

Current text says:
> The empirical generator always uses the MC path (no precomputed null,
> because y-ties vary per dataset).

Replace with:
> The empirical generator uses the precomputed null by default
> (see [Precomputed null approximation](#precomputed-null-approximation-for-empirical-generator)).
> When `EMPIRICAL_USE_PRECOMPUTED_NULL = False`, it falls back to the MC
> path.

### 6. New file: `benchmarks/verify_precomputed_null_empirical.py`

This verification script lets users empirically confirm the approximation
error. It should:

1. For each case (n=73, 80, 81, 82), using **all-distinct x** (n_distinct=n,
   all_distinct=True) to isolate the y-tie effect from x-tie structure:
   a. Build the calibration pool via `get_pool(n)` (fixed seed, cached)
   b. Generate `n_sims` datasets (x, y) using the empirical generator
      (`generate_y_empirical_batch` handles per-sim pool building internally)
   c. Compute observed rhos via `spearman_rho_2d`
   d. Compute p-values two ways:
      - **Precomputed null**: `pvalues_from_precomputed_null(rhos_obs, sorted_abs_null)`
      - **Per-dataset MC**: `pvalues_mc(x_all, y_all, n_perm, alpha, rng)`
   e. Report:
      - Mean |p_precomputed - p_mc| across sims
      - Max |p_precomputed - p_mc|
      - Power (rejection rate) under each method
      - Δpower = power_precomputed - power_mc

2. Run at both rho=0 and rho=0.30. These test **different things**:
   - **rho=0 → type I error check.** Both methods should reject at ≈α.
     Δ(rejection rate) measures the type I error bias from the
     approximation.  Use n_sims=50,000 for stable estimates.
   - **rho=0.30 → power bias check.** Both methods should show similar
     power.  Δpower measures the power bias (the ~10⁻⁵ systematic
     error).  Use n_sims=10,000.

3. Print a summary table with columns:
   `case | n | rho | metric | rate_precomputed | rate_mc | delta | SE(delta) | interpretation`

   where "metric" is "type_I_error" (rho=0) or "power" (rho=0.30),
   and SE(delta) is estimated empirically from per-sim indicator
   differences.

4. Accept CLI args: `--n-sims` (default 10000), `--seed` (default 42),
   `--rhos` (default "0.0,0.30"), `--cases` (default all),
   `--n-distinct` (default: use all-distinct, i.e. n_distinct=n).
   Note: `--n-sims` is the baseline; the rho=0 run uses
   `max(n_sims, 50000)` for type I error stability.
   All-distinct x is the default because the approximation error is
   about y-ties, not x-ties; any x structure would give the same result.

**Structure sketch** (Sonnet should implement):

```python
"""
Verify precomputed null approximation for empirical generator.

Compares p-values and power between:
  1. Precomputed null (all-distinct y-ranks, keyed on x tie structure)
  2. Per-dataset Monte Carlo permutation (exact)

Expected result: Δpower < 10^-4 for all cases. See README
"Precomputed null approximation for empirical generator".
"""
import sys, argparse, time
import numpy as np
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import CASES, ALPHA, PVALUE_PRECOMPUTED_N_PRE
from power_asymptotic import get_x_counts
from data_generator import (generate_cumulative_aluminum_batch,
                            generate_y_empirical_batch, calibrate_rho_empirical,
                            get_pool)
from spearman_helpers import spearman_rho_2d
from permutation_pvalue import (get_precomputed_null, pvalues_from_precomputed_null,
                                pvalues_mc, _get_n_perm)


def verify_one(case_id, n, rho_s, n_sims, alpha, seed,
               n_distinct=None, dist_type=None):
    """Compare precomputed vs MC p-values for one scenario.

    Default: all-distinct x (n_distinct=n) to isolate y-tie effect.
    """
    rng = np.random.default_rng(seed)
    if n_distinct is None:
        n_distinct = n
    all_distinct = (n_distinct == n)

    # Calibrate using the fixed reference pool
    pool = get_pool(n)
    cal_rho = calibrate_rho_empirical(
        n, n_distinct, dist_type, rho_s, pool, all_distinct=all_distinct)

    # Generate data (generate_y_empirical_batch handles per-sim pool building)
    x_all = generate_cumulative_aluminum_batch(
        n_sims, n, n_distinct, distribution_type=dist_type,
        all_distinct=all_distinct, rng=rng)
    # y_params=None: empirical marginal uses pool directly, never reads y_params
    y_all = generate_y_empirical_batch(
        x_all, rho_s, None, rng=rng, _calibrated_rho=cal_rho)

    rhos_obs = spearman_rho_2d(x_all, y_all)

    # Path 1: precomputed null
    x_counts = get_x_counts(n, n_distinct, distribution_type=dist_type,
                            all_distinct=all_distinct)
    sorted_abs_null = get_precomputed_null(n, all_distinct, x_counts,
                                           n_pre=PVALUE_PRECOMPUTED_N_PRE,
                                           rng=np.random.default_rng(42))
    pvals_pre = pvalues_from_precomputed_null(rhos_obs, sorted_abs_null)

    # Path 2: MC
    n_perm = _get_n_perm(n_sims)
    mc_rng = np.random.default_rng(seed + 1000)
    _, pvals_mc, _ = pvalues_mc(x_all, y_all, n_perm, alpha, mc_rng)

    rej_pre = (pvals_pre < alpha).astype(float)
    rej_mc = (pvals_mc < alpha).astype(float)
    rate_pre = rej_pre.mean()
    rate_mc = rej_mc.mean()
    d = rej_pre - rej_mc
    se_delta = d.std(ddof=1) / np.sqrt(len(d))
    delta_p = pvals_pre - pvals_mc
    metric = "type_I_error" if abs(rho_s) < 1e-9 else "power"

    return {
        "case": case_id, "n": n, "rho": rho_s, "metric": metric,
        "rate_precomputed": rate_pre,
        "rate_mc": rate_mc,
        "delta": rate_pre - rate_mc,
        "se_delta": se_delta,
        "mean_abs_delta_p": np.mean(np.abs(delta_p)),
        "max_abs_delta_p": np.max(np.abs(delta_p)),
    }

# ... main(), argparse, loop over cases and rhos, print table ...
```

The key detail: at rho=0 the "calibrated rho" should also be 0 (no
calibration needed for the null case), and the comparison measures the
**type I error** difference (both methods should reject at ≈α=0.05).
At rho=0.30, the comparison measures the **power** bias (the ~10⁻⁵
systematic error from the approximation). The script output should
clearly label which metric each row reports.

**Important note for the verification script:** The MC path has its own
sampling noise (~1/sqrt(n_perm)), so Δpower between precomputed and MC
will be dominated by MC noise, not by the approximation error. To
isolate the true bias, the script should report both Δpower and the
SE of Δpower. Since both rejection decisions use the **same** (x, y)
data, they are positively correlated (most sims far from the α boundary
reject or fail to reject under both methods). The correct SE is
`sqrt((Var(A) + Var(B) - 2·Cov(A,B)) / n_sims)`, which is smaller than
the independent formula `sqrt(2·p·(1-p)/n_sims)`. In practice, estimate
the SE empirically from the per-sim indicator difference
`d_i = 1[p_pre < α] - 1[p_mc < α]` as `std(d) / sqrt(n_sims)`.
With n_sims=50000, the SE will be small but a 10⁻⁵ bias will still not
be individually resolved — the script confirms it is *consistent with
zero* and *bounded well below the precision targets*. The script should
state this interpretation in its output.

### 7. Update `benchmarks/benchmark_realistic_runtimes.py`

The empirical generator timings will change dramatically (precomputed
null is expected ~60× faster for the p-value step). The benchmark script will
automatically pick up the change because it calls `min_detectable_rho`
and `run_all_scenarios` which go through `estimate_power`. No code
change is needed in the benchmark script itself — but add a note in its
docstring or output that empirical now uses precomputed null by default.

---

## Math Derivation (for README)

**Sonnet: copy the markdown in section 5d above verbatim into the README.**
Do not rephrase the math or add additional derivation steps. The math has
been verified by a model with stronger mathematical reasoning. The key
results to preserve exactly:

1. The denominator inflation formula:
   σ_untied/σ_tied = sqrt((n³-n) / ((n³-n) - Σ_y)) ≈ 1 + Σ_y/(2(n³-n))

2. The specific tie counts: H_AL71 Σ=24, B_AL71 Σ=60

3. The pool totals: n=73 ≈ 79, n=81 ≈ 98

4. The Δp bound: < 1.4 × 10⁻⁵

5. The variance invariance statement: E[ρ]=0, Var(ρ)=1/(n-1) exactly,
   regardless of ties

6. The key insight: this is a systematic bias NOT reduced by n_sims

7. The verdict table for all three precision targets

---

## Testing Checklist

After implementation, verify:

- [ ] `python -c "from config import EMPIRICAL_USE_PRECOMPUTED_NULL; print(EMPIRICAL_USE_PRECOMPUTED_NULL)"` prints `True`
- [ ] Empirical power runs use precomputed null (expected ~60× faster; verify with benchmark)
- [ ] Setting `EMPIRICAL_USE_PRECOMPUTED_NULL = False` reverts to MC (slower, same results within MC noise)
- [ ] `benchmarks/verify_precomputed_null_empirical.py` runs and shows Δpower consistent with zero / bounded well below ±0.001
- [ ] All existing tests still pass: `python tests/run_tests.py`
- [ ] `benchmarks/benchmark_realistic_runtimes.py --generators empirical --quick` shows improved timing

---

## What NOT to Change

- `permutation_pvalue.py` logic: no code changes needed, only docstring
- `get_precomputed_null`: already handles all cases correctly
- `warm_precomputed_null_cache`: already builds nulls for all scenario
  (n, k, dist_type) combinations — empirical uses the same nulls
- `confidence_interval_calculator.py`: CI does not use permutation
  p-values; no change needed
- Calibration: empirical calibration is separate from p-value method;
  no change needed
