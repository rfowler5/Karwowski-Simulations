# Statistical Audit — KarwowskiSpearmanPowerSims

Conducted: March 2026. Full codebase reviewed (all 20 source files).
Purpose: document findings for Cursor AI sessions so previously-audited
code does not need to be re-verified from scratch.

---

## Confirmed Correct — Do Not Re-Audit

The following have been verified against published sources. Cursor sessions
should treat these as ground truth and not flag them as suspicious.

**`power_asymptotic.py`**
- FHP variance formula: `(1/(n-1)) * ((n³-n)/12)² / (Dx * Dy)` ✓
- Noncentrality parameter: `rho * sqrt(n-2) / sqrt(1-rho²)` ✓
- Tie correction direction: nc scaled DOWN for power (ties reduce information);
  SE scaled UP for CI (ties widen interval) ✓
- Bonett-Wright SE: `sqrt(1.06/(n-3))` applied to CI only, not power ✓
- Fisher z CI: arctanh → ±z*SE → tanh back-transform, guaranteed in [-1,1] ✓
- FHP applies to x only (y treated as no-ties); intentional and documented ✓
- Bisection direction for negative rho: `hi = mid` when `pw < target`
  (moves toward more-negative lo) ✓

**`permutation_pvalue.py`**
- Null construction: permutes integer ranks 1..n; computes Pearson of
  x_midranks vs permuted y-ranks ✓
- P-value formula: `(1 + count)/(1 + n_pre)` — correct conservative
  Davison-Hinkley formula ✓
- Midrank formula: `avg = (i + j + 1)/2.0` for 0-indexed positions ✓

**`spearman_helpers.py`**
- `_tied_rank` midrank matches `rankdata(method='average')` ✓
- `spearman_rho_2d` computes Pearson on ranks (correct definition) ✓
- t-approximation: `t = rho*sqrt((n-2)/(1-rho²))`, df=n-2 ✓

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

**`data_generator.py`**
- Rank-mixing formula: `mixed = rho_c * s_x + sqrt(1-rho_c²) * s_n`
  (correct Cholesky in rank space) ✓
- `_spearman_to_pearson`: `2*sin(π*rho/6)` — exact for bivariate normal,
  used correctly for copula and linear contexts only ✓
- Calibration caching and ratio logic ✓
- Bisection for calibration ✓

**`confidence_interval_calculator.py`**
- RNG separation via `SeedSequence.spawn(2)` ✓
- Percentile bootstrap: correct choice for model-based (not observed-data)
  CI estimation ✓
- Averaging CI endpoints over n_reps: statistically valid; correctly
  explains reasoning in module docstring ✓
- Batch path bootstrap index layout: `boot_idx_all[b, rep, :]` ✓
- `nanpercentile` at `100*alpha/2` and `100*(1-alpha/2)` ✓

**`benchmark_precision_params.py`**
- Quadrature formula `SE_total = sqrt(SE_bis² + SE_cal²)` ✓
- Balanced allocation `se_each = se_target/sqrt(2)` ✓
- n_reps formula: `(SD_INTER_REP * 1.96 / halfwidth)²` ✓

---

## Known Approximations — Intentional, Not Bugs

**`_fit_lognormal` symmetric IQR split**
```python
q75_target = median + iqr / 2.0   # assumes IQR is symmetric around median
sigma = (np.log(q75_target) - mu) / _NORM_PPF_075
```
For a true log-normal the IQR is asymmetric, so sigma is slightly
underestimated. Has **no effect on rank-based methods** (nonparametric,
linear) because Spearman only depends on ordering. Affects copula's
y-marginal fidelity only. Copula is retained for comparison, not as the
primary method. Documented in README.

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

---

## Fixes Required

### FIX-1 — Silent boundary return in `min_detectable_rho` [MEDIUM] [DONE]
**File:** `power_simulation.py`
**Issue:** Bisection searches in `[0.25, 0.42]` for positive direction and
`[-0.42, -0.25]` for negative. If the true min detectable rho falls outside
these bounds, bisection silently returns the boundary value with no
indication that the answer is wrong. For current parameters (n=73-82,
α=0.05, 80% power) the answer is ~0.30-0.33, safely inside bounds. But
fragile if parameters change.
**Fix:** After bisection, check if result is within `tolerance` of either
boundary and emit a `warnings.warn`.
```python
result = (lo + hi) / 2.0
tolerance = 1e-4
if abs(result - lo_bound) < tolerance or abs(result - hi_bound) < tolerance:
    warnings.warn(
        f"min_detectable_rho hit search boundary ({result:.4f}). "
        "Consider widening RHO_SEARCH bounds.",
        UserWarning, stacklevel=2)
return result
```

### FIX-2 — Dead config values [LOW] [DONE]
**File:** `config.py`
**Issue:** `RHO_SEARCH_POSITIVE = (0.0, 0.6)` and
`RHO_SEARCH_NEGATIVE = (-0.6, 0.0)` are defined but never used.
`min_detectable_rho` in `power_simulation.py` hardcodes `(0.25, 0.42)`.
**Fix:** Either wire these into `min_detectable_rho` as the search bounds
(preferred — makes bounds configurable), or delete them and document the
hardcoded values with a comment explaining why they're tighter than the
config range.

### FIX-3 — Double `.sort()` calls [LOW] [DONE]
**File:** `data_generator.py`
**Issue:** In `_raw_rank_mix` (around line 440-443) and `_raw_rank_mix_batch`
(around line 996-999), the empirical branch sorts the pool array, then the
common code path sorts it again. Harmless but wasteful and confusing.
**Fix:** Remove the second `.sort()` / `.sort(axis=1)` call in each function.
In `_raw_rank_mix`, line 443 `y_values.sort()` is redundant when the
empirical branch already called `np.sort(pool)` at line 440.
In `_raw_rank_mix_batch`, line 999 `y_values.sort(axis=1)` is redundant
when the empirical branch already called `np.sort(pool, axis=1)` at line 996.

### FIX-4 — Unused import [LOW] [DONE]
**File:** `benchmarks/benchmark_full_grid.py`
**Issue:** `bootstrap_ci_single` is imported but never called in the script.
**Fix:** Remove from the import line.

---

## Missing Tests

### TEST-1 — Boundary warning fires
**Test file:** `tests/test_power_sanity.py` or new `test_bisection_bounds.py`
**What to test:** Call `min_detectable_rho` with parameters engineered to
push the answer near 0.42 (e.g., very small n or very high target power).
Assert that `warnings.warn` was emitted. Use `pytest.warns(UserWarning)`.
**Why:** Ensures FIX-1 actually fires; prevents regression where the warning
is accidentally removed.

### TEST-2 — `spearman_var_h0` monotonicity
**Test file:** `tests/test_asymptotic_formulas.py`
**What to test:** For any tie structure, `spearman_var_h0(n, x_counts)` with
ties should be ≥ `spearman_var_h0(n, None)` (no ties). Ties inflate variance.
**Why:** A sign error in the FHP formula would flip this relationship and
produce optimistic (anticonservative) CIs.

### TEST-3 — Negative rho calibration symmetry
**Test file:** `tests/test_calibration_accuracy.py`
**What to test:**
```python
pos = calibrate_rho(n, k, dt, +0.30, y_params)
neg = calibrate_rho(n, k, dt, -0.30, y_params)
assert abs(pos + neg) < 1e-6   # should be exactly negatives of each other
```
**Why:** The calibration uses only positive probes and applies sign afterward.
If the sign logic has a bug, this test catches it. Currently not tested.

### TEST-4 — `_fit_lognormal` median recovery
**Test file:** `tests/` (new `test_data_generator.py` or add to existing)
**What to test:**
```python
mu, sigma = _fit_lognormal(median=2.5, iqr=1.2)
assert abs(np.exp(mu) - 2.5) < 1e-10   # mu = log(median) exactly
assert sigma > 0
```
**Why:** Verifies the basic contract of the function. Simple regression guard.

### TEST-5 — `_interp_with_extrapolation` edge cases
**Test file:** new `test_data_generator.py`
**What to test:** Three cases:
1. `x < xp[0]` (below first probe): result should follow slope from first two points
2. `x > xp[-1]` (above last probe): result should follow slope from last two points
3. `len(xp) == 1` (single point): should return `fp[0]` without IndexError
**Why:** The multipoint calibration falls back to extrapolation for rho
targets outside [0.10, 0.50]. Currently not unit-tested.

### TEST-6 — Asymptotic formulas across all four cases
**Test file:** `tests/test_asymptotic_formulas.py`
**What to test:** The existing test only runs Case 3. Parameterize with
`@pytest.mark.parametrize` over all four case IDs and verify CI contains
observed rho, CI width > 0, and CI is within [-1, 1] for each.
**Why:** Cases 1, 2, 4 have different n and observed rho; edge cases in
the FHP correction could manifest differently.

---

## Optimization Opportunities

### OPT-1 — Precomputed null for empirical generator [IMPLEMENTED]
**Validated:** Approximation error < 1.4×10⁻⁵ on p-values, < 10⁻⁵ on
power. Safe for all precision targets (±0.01, ±0.002, ±0.001). See
"Known Approximations" above for full analysis.
**Files changed:** `config.py` (new flag `EMPIRICAL_USE_PRECOMPUTED_NULL`),
`power_simulation.py` (routing logic), `permutation_pvalue.py` (docstring).
**Fallback:** Set `config.EMPIRICAL_USE_PRECOMPUTED_NULL = False` to
revert to per-dataset MC permutation p-values (exact but much slower;
expected ~60×, pending benchmark verification).
**Verification note:** `benchmarks/verify_precomputed_null_empirical.py`
confirms the empirical generator behaves identically to non-empirical
generators using the same null (all deltas < ±0.01). It does NOT isolate
the 10⁻⁵ y-tie approximation error, which is ~100× smaller than the
dominant noise source (precomputed null realization variance, ~SD 0.001
at n_pre=50k). The analytic derivation in README is the proper validation
of the 10⁻⁵ claim. Mixed-sign deltas across cases (case 3 type_I_error
negative) confirm there is no systematic directional bias from OPT-1.

### OPT-2 — Vectorize `get_precomputed_null` construction [IMPLEMENTED]
**Validated:** Vectorized build in `get_precomputed_null`: all `n_pre`
permutations generated via `np.argsort(rng.random((n_pre, n)), axis=1) + 1.0`;
single matmul `(all_perm_y @ x_std) / (std_y * n)` for null rhos; `std_y` from
closed-form `sqrt((n²−1)/12)`. Statistically equivalent to original loop (KS
tests pass). Cold build ~0.3s per key (~32s for 88 keys); ~16–25× speedup vs
loop. Peak memory ~100 MB (n_pre×n arrays).
**Files changed:** `permutation_pvalue.py` (`get_precomputed_null`).
**Verification:** `benchmarks/verify_vectorized_null.py`.

---

## Known Precision Limitations

### PREC-1 — Precomputed null realization variance limits ±0.001 target

**Affects:** All generators using `get_precomputed_null` (nonparametric,
copula, linear, and empirical with `EMPIRICAL_USE_PRECOMPUTED_NULL=True`).

**Mechanism:** The precomputed null is one random draw of `n_pre`
permutations (seeded at build time). The 95th-percentile critical value
`c = quantile(|rho_null|, 0.95)` has sampling variance:

```
SD(c) = sqrt(0.05 × 0.95) / (f(c) × sqrt(n_pre))
      ≈ 0.218 / (1.04 × sqrt(n_pre))
```

This critical value shift propagates 1:1 to the bisection output `rho*`
(the power-slope and critical-value-slope cancel). So the realization
variance contributes directly to the precision of min-detectable-rho:

| n_pre | SD(c) | 95% CI contribution to rho* | Adequate for |
|-------|-------|-----------------------------|-------------|
| 50,000 (current) | ~0.00094 | ±0.0018 | ±0.01 (comfortable), ±0.002 (marginal) |
| 200,000 | ~0.00047 | ±0.0009 | ±0.001 (just sufficient) |
| 500,000 | ~0.00030 | ±0.0006 | ±0.001 (comfortable) |

**Why this does NOT average out with n_sims — and why the MC path's noise does.**

The critical value `c` is determined once when the null is built, then held
fixed for all n_sims in the session. Every sim's rejection decision is shifted
by the same amount in the same direction. Increasing n_sims reduces sampling
variance of the rejection rate *given* a fixed `c`, but cannot reduce the
offset caused by `c` being slightly wrong. It is a session-persistent bias.

The **MC p-value path** (`EMPIRICAL_USE_PRECOMPUTED_NULL = False`, or
`PVALUE_MC_ON_CACHE_MISS = True` on a cold cache) behaves differently: each
sim builds its own mini-null from `n_perm` fresh, independent permutations.
Each sim's critical value is an independent draw — the errors are IID with
zero mean across sims. Their contribution to the mean rejection rate averages
out as 1/√n_sims. With n_perm=1000, the per-sim critical value noise is
~0.218/√1000 ≈ 0.0069, which averages to ~0.0069/√n_sims on the rejection
rate. At n_sims=10,000 this is ~7×10⁻⁵ — well below ±0.001.

In summary:

| P-value path | Per-sim critical value noise | Correlated across sims? | Averages out with n_sims? |
|---|---|---|---|
| Precomputed null | Shared; fixed at build time | Yes (100%) | No — increase n_pre instead |
| MC per-dataset | IID; fresh permutations each sim | No | Yes — scales as 1/√n_sims |

**Practical implication for ±0.001:** Both paths can reach ±0.001 — either
increase n_pre to 500,000 (precomputed null; fast p-values, one-time build
cost), or use the MC path with n_sims ≈ 10,000–15,000 (no n_pre concern,
but ~60× slower per sim). Neither path has a systematic directional bias;
both have zero-mean noise with different reduction mechanisms.

**Empirical confirmation** (from `verify_precomputed_null_empirical.py`,
seed=42, n_pre=50k): type I error deltas across 4 cases ranged from
-0.0003 to +0.0013 (SD ≈ 0.0007), consistent with predicted SD ~0.001.
Mixed signs confirm this is realization variance, not systematic bias.

**Current setting:** `PVALUE_PRECOMPUTED_N_PRE = 50_000` in `config.py`.
Adequate for ±0.01 and marginally for ±0.002. **Insufficient for ±0.001.**

**Fix for ±0.001 target:** Set `PVALUE_PRECOMPUTED_N_PRE` to 200,000 or 500,000:
- **200,000** (just sufficient): ~1.2s/key; ~106s full warm (~1.8 min); ~141 MB peak
- **500,000** (comfortable): ~3s/key; ~264s full warm (~4.4 min); ~352 MB peak
- p-value lookup: O(log n_pre) — negligible slowdown in both cases

**Relationship to OPT-1:** This limitation is pre-existing and applies to
all generators, not specific to the empirical generator or OPT-1.

**Future work:** See `.cursor/plans/auto_select_n_pre_precision_tier.plan.md`
for the idea of auto-selecting n_pre based on the precision tier being run.

---

## Architecture Notes (for Cursor context)

- **Calibration caching:** `calibrate_rho` and variants cache per
  `(n, k, dist_type, all_distinct, n_cal)`. Cache is module-level dict,
  not persisted across runs. Reused across all rho values within a bisection
  sweep — this is the ~60× speedup vs original design.
- **RNG discipline:** All functions take explicit `rng` parameter.
  `bootstrap_ci_averaged` uses `SeedSequence.spawn(2)` to create independent
  `data_rng` and `boot_rng` streams. This is intentional — do not merge them.
- **Numba JIT:** `_bootstrap_rhos_jit`, `_batch_bootstrap_rhos_jit`,
  `_batch_permutation_rhos_jit` in `spearman_helpers.py`. Pure NumPy fallback
  active when Numba unavailable. `fastmath=True` causes FP jitter — tests use
  tolerance `1e-9` not exact equality for reproducibility checks. **If
  test_reproducibility fails after code changes**, clear the Numba cache first:
  remove `__pycache__/*.nbc` and `__pycache__/*.nbi` (or at least
  `spearman_helpers.*.nbc`/`.nbi`); stale cached bytecode can cause spurious
  failures.
- **FHP applies to x only:** `power_asymptotic.py` receives x_counts but
  not y_counts. y is treated as all-distinct. Intentional for this study
  (x = vaccine aluminum has heavy ties; y = blood aluminum typically does not).
- **Copula warning:** `generate_y_copula` is retained for comparison only.
  With k=4 (heavy ties), jittering attenuates realized Spearman rho by
  0.01-0.06 and underestimates power. Use nonparametric for primary results.
