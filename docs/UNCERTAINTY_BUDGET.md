# Uncertainty Budget — Power and Confidence Interval Precision

This document consolidates all sources of numerical error in the two main
outputs of this framework: (1) minimum detectable rho (power bisection) and
(2) bootstrap CI endpoints. It provides a single reference so users can
verify that their parameter choices meet a target accuracy.

For derivations of the individual terms, see the README
([Equations to compute Precision](../README.md#equations-to-compute-precision))
and AUDIT.md ([Known Precision Limitations](AUDIT.md#known-precision-limitations)).
For adapting these budgets to new data or heavier ties, see
[PRECISION\_WHEN\_DATA\_CHANGES.md](PRECISION_WHEN_DATA_CHANGES.md).

---

## Part 1: Power — min detectable rho

### Error sources

The estimated minimum detectable rho, rho*, is subject to the following
independent error sources.

| # | Source | Correlated across sims? | Controlled by | Formula |
|---|--------|------------------------|---------------|---------|
| 1 | **Bisection MC noise** | No — averages as 1/√n_sims | n_sims | SE = c / √n_sims |
| 2 | **Calibration MC noise** | Yes — one calibration shared by all sims | n_cal | SE = k / √n_cal |
| 3 | **Precomputed null realization** | Yes — one null shared by all sims | n_pre | SD(crit) ≈ 0.21 / √n_pre → ±0.0018 at n_pre=50k |
| 4 | **Calibration interpolation** | Yes — fixed for a given target rho | probe count | ≤ 0.00062 within probes [0.10, 0.50], jittered curve |
| 5 | **Permutation p-value noise** (MC path only) | No — averages | n_perm | +0.17%–1.2% SE inflation |
| 6 | **Precomputed null y-tie approx** (empirical + precomputed null only) | Yes — systematic bias | Switch to MC path to eliminate | Δpower ≈ 10⁻⁵; zero on MC path |

**Source 1 — bisection MC noise.**
By the stochastic root-finding theorem (Waeber, Frazier & Henderson, 2013),
bisection on a Monte Carlo power function converges with SE:

```
SE_bisection = c / √n_sims,   c = √(π(1−π)) / |power′(ρ*)|
```

The coefficient c is determined by the slope of the power curve at the 80%
crossing. This result applies at all n_sims — including n_sims=2,220 (the
±0.01 tier). The formula does not break down at low n_sims; the SE is simply
larger. **Important caveat on re-estimating c:** the script
`scripts/estimate_bisection_c.py` estimates c by a finite-difference slope
between two power evaluations. At n_sims=2,000 the finite-difference slope is
very noisy — empirical c-hat ranges from ~0.10 to ~0.18 depending on seed.
This is the estimation procedure being noisy, not the formula being wrong. Use
n_sims ≥ 5,000 in that script when re-estimating c for new data or heavier
ties; the reported benchmark value c = 0.17 is the analytical (no-tie) result,
which is stable.

**Source 2 — calibration MC noise.**
A single calibration (n_cal samples, seed=99) is computed and cached per
scenario. All n_sims share the same calibrated rho_in, so calibration noise
does not average out with n_sims. The SE contribution is k / √n_cal, where
k = (1 − ρ²) √(1.06/(n−3)) ≈ 0.112 (worst case n=73, ρ=0.30).

**Source 3 — precomputed null realization variance.**
The precomputed null critical value is fixed at build time; its offset is
the same for every sim and does not average out. See AUDIT.md PREC-1 for
the full derivation. This source is **not included** in the tier formulas in
`benchmark_precision_params.py`; it must be managed by setting
`PVALUE_PRECOMPUTED_N_PRE`.

**Source 4 — calibration interpolation.**
After the jitter fix (DERIVATIONS.md §11–13), the 5-probe multipoint
calibration achieves ≤ 0.00062 interpolation error within the probe range
[0.10, 0.50]. Extrapolation beyond 0.50 diverges rapidly and should be avoided.
This is a fixed, structural error floor — it cannot be reduced by increasing
n_cal. For the current Karwowski study (min detectable rho ≈ 0.28–0.34), all
bisection evaluations fall within the probe range.

**Source 5 — permutation p-value noise (MC path only).**
When `EMPIRICAL_USE_PRECOMPUTED_NULL = False`, each sim draws a fresh set of
n_perm permutations to estimate its critical value. These per-sim critical
value errors are IID across sims and therefore average out with n_sims —
unlike the precomputed null (Source 3), which shares one fixed critical value
across all sims. The residual SE inflation is +0.17%–1.2% depending on
n_perm (see README for derivation). This source is absent when using the
precomputed null.

**Source 6 — precomputed null y-tie approximation (empirical + precomputed null only).**
The precomputed null is built once from all-distinct y-ranks {1..n}, using the
no-ties denominator σ_untied = √((n²−1)/12). The empirical generator, however,
resamples from a finite pool of 71 digitized values (which already contain
repeated measurements) into n = 73–81 draws. By the pigeonhole principle
alone this guarantees repeated values; combined with the pre-existing
duplicates in the pool, the empirical y-data **always** has ties. The
denominator σ_tied < σ_untied. The resulting rho is slightly inflated relative
to the threshold, giving an anticonservative bias (power systematically too
high by ~10⁻⁵). For derivation see AUDIT.md "Precomputed null vs tied-y:
denominator mismatch (OPT-1)".

**This error is entirely eliminated** by setting
`config.EMPIRICAL_USE_PRECOMPUTED_NULL = False`, which switches the empirical
generator to per-dataset MC permutation p-values. Each permutation test then
permutes the actual y-data, so no distinct-y assumption is made. The trade-off
is runtime: the MC path uses n_perm = 1,000–2,000 permutations per sim and is
substantially slower than the precomputed null lookup. For the current study
the 10⁻⁵ bias is negligible (< 1% of the ±0.001 tier target), so the default
precomputed null is appropriate.

### Numerical values per tier (current Karwowski data)

Benchmark coefficients: c = 0.17 (analytical, conservative), k = 0.112
(analytical worst case, n=73), n_pre = 50,000 (default config).

| Source | ±0.01 | ±0.002 | ±0.001 |
|--------|-------|--------|--------|
| 1. Bisection MC (95% HW) | 0.0071 (n_sims = 2,220) | 0.0014 (n_sims = 55,520) | 0.00071 (n_sims = 222,050) |
| 2. Calibration MC (95% HW) | 0.0069 (n_cal = 1,000) | 0.0014 (n_cal = 24,100) | 0.00070 (n_cal = 96,400) |
| 3. Null realization (95%) | 0.0018 (n_pre = 50k) | **0.0018** (n_pre = 50k) ✗ | **0.0018** (n_pre = 50k) ✗ |
| 4. Cal interpolation | < 0.001 | < 0.001 | 0.00062 |
| 5. Permutation noise | 0.000012 | 0.0000024 | 0.0000012 |
| 6. y-tie bias (empirical) | 10⁻⁵ | 10⁻⁵ | 10⁻⁵ |
| **Combined 1+2 (RSS)** | **0.010** | **0.0020** | **0.0010** |
| **Combined 1+2+3 (RSS)** | **0.010** | **0.0027** ✗ | **0.0021** ✗ |

The tier formulas (rows 1+2) balance bisection and calibration noise equally.
Row 3 shows that the null realization variance at n_pre = 50,000 is acceptable
for ±0.01, **insufficient for ±0.002** (increases combined HW from 0.0020 to
0.0027, a 35% overshoot), and **insufficient for ±0.001** (total HW ≈ 0.0021
instead of 0.0010).

**n_pre requirements:**

| Target | n_pre needed | Approx build time (88 keys) | Peak memory |
|--------|--------------|-----------------------------|-------------|
| ±0.01 | 50,000 (default) | ~32s | ~100 MB |
| ±0.002 | 200,000 | ~1.8 min | ~141 MB |
| ±0.001 | 500,000 | ~4.4 min | ~352 MB |

---

## Part 2: CI endpoints — bootstrap CI

### Error sources

The reported CI endpoint is the mean of n_reps independent bootstrap-quantile
estimates. Its precision depends on the following sources.

| # | Source | Correlated across reps? | Controlled by | Formula |
|---|--------|------------------------|---------------|---------|
| 1 | **Inter-rep sampling noise** | No — averages as 1/√n_reps | n_reps | SE = σ_rep / √n_reps |
| 2 | **Bootstrap quantile noise** | No — averages | n_reps × n_boot | SE = √(p(1−p) / (n_reps × n_boot × f²)) |
| 3 | **Calibration MC noise** | Yes — one calibration shared by all reps | n_cal | SE = k / √n_cal |
| 4 | **Calibration interpolation** | Yes — fixed for the observed rho | probe count | ≤ 0.00062 within probes [0.10, 0.50] |

Note: the precomputed null (Source 3 in Part 1) is **not** used in the CI
pipeline. CI uses bootstrap resampling, not permutation p-values.

**Source 1 — inter-rep sampling noise.**
Each rep produces one CI endpoint L_i (e.g. the 2.5th percentile of n_boot
bootstrap rhos). The SD of L_i across datasets is:

```
σ_rep = (1 − ρ²) × √(1.06/(n−3)) × FHP_factor
```

Worst case: Case 3 (n=73, upper endpoint ≈ 0.11, k=4 heavy_center):
σ_rep ≈ 0.129. Conservative benchmark: **σ_rep = 0.13**.

**Source 2 — bootstrap quantile noise.**
For the p-th sample quantile (p = 0.025) from n_boot i.i.d. draws, the
asymptotic variance is p(1−p) / (n_boot × f²), where f = f(q_p) is the
PDF of the bootstrap rho distribution evaluated at the quantile. For a
roughly normal bootstrap distribution with SD = σ_boot, the density at the
p-th quantile is:

```
f = φ(Φ⁻¹(p)) / σ_boot
```

where φ is the standard normal PDF and Φ⁻¹ is the quantile function. For
p = 0.025: Φ⁻¹(0.025) = −1.96, φ(−1.96) = 0.0584, so **f = 0.0584 / σ_boot**.

**Calculating σ_boot.** By the bootstrap principle, σ_boot (the SD of
Spearman rho across bootstrap resamples of a single dataset) equals the
sampling distribution SD of ρ̂_s, given by the same Bonett-Wright + FHP
formula used for k and σ_rep:

```
σ_boot = (1 − ρ²) × √(1.06/(n−3)) × FHP_factor
```

evaluated at the observed ρ for the scenario in question. This is the same
formula as σ_rep (Part 2, Source 1) and k (Part 1, Source 2), just evaluated
at different ρ values:

| Coefficient | Evaluated at | Typical value | Formula instance |
|-------------|-------------|---------------|------------------|
| k | ρ = 0.30 (calibration probe) | 0.112 (n=73) | (1−0.30²)√(1.06/70) |
| σ_rep | CI endpoint ρ ≈ 0.11 (worst case) | 0.129 (n=73, FHP) | (1−0.11²)√(1.06/70)×1.057 |
| σ_boot | observed ρ for the scenario | 0.112 (ρ≈0.30, n=73) | same formula |

For the typical case (ρ ≈ 0.30, n = 73):

```
σ_boot = (1 − 0.09) × √(1.06/70) = 0.91 × 0.123 = 0.112
f = 0.0584 / 0.112 = 0.52
```

For the worst case (ρ ≈ 0.11, n = 73, with FHP):

```
σ_boot = (1 − 0.012) × √(1.06/70) × 1.057 = 0.129
f = 0.0584 / 0.129 = 0.45
```

The tables below use the simplification **f = 1**, which *underestimates*
bootstrap noise by a factor of 1/f² ≈ 3.6–4.9× (depending on scenario).
Even at the worst case (f = 0.45, so 1/f² ≈ 4.9), the bootstrap term
contributes < 1.5% of variance — negligible at all tiers.

```
p(1-p) = 0.025 × 0.975 = 0.024375

SE_boot = √(0.024375 / (n_reps × n_boot × f²))

With f = 1:    SE_boot = √(0.024375 / (n_reps × 500))    [tables below]
With f = 0.45: SE_boot ≈ 2.2× larger                      [still negligible]
```

**Universal bound (parameter-free).** The variance ratio (bootstrap /
inter-rep) can be bounded without knowing σ_boot at all. The tier formulas
size n_reps using the worst-case σ_rep = SD_INTER_REP (= 0.13 for current
data). This worst case arises at a CI endpoint near ρ ≈ 0, where (1 − ρ²) is
maximized. At that same scenario, σ_boot is evaluated at the same small ρ, so
σ_boot ≈ σ_rep. Substituting SD_INTER_REP for σ_boot in the ratio:

```
ratio = p(1−p) × SD_INTER_REP² / (n_boot × φ(z_p)² × SD_INTER_REP²)
      = p(1−p) / (n_boot × φ(−1.96)²)
      = 0.024375 / (500 × 0.003412)
      = 0.0143   (1.43% of variance, 12.0% of SE)
```

SD_INTER_REP cancels. The bound **depends only on n_boot and p**, not on
n, ρ, tie structure, or any coefficient that changes with the data. For
n_boot = 500 and p = 0.025, bootstrap quantile noise adds at most 1.43% to
the worst-case inter-rep variance — always negligible, with no recalculation
needed.

*Edge case:* for upper CI endpoints of larger ρ, σ_boot can slightly exceed
the endpoint-specific σ_rep (the upper endpoint has larger |ρ|, reducing
(1 − ρ²)). But those endpoints have σ_rep well below the worst-case 0.13, so
the tier formula already over-allocates n_reps for them, and the total SE
remains within budget.

**When to recalculate f.** In practice, the universal bound above makes
recalculation unnecessary: the 1.43% figure holds for any data as long as
n_boot ≥ 500 and p = 0.025. If you want a scenario-specific f anyway (e.g.
for a smaller n_boot), recompute σ_boot from the formula above and then
f = 0.0584 / σ_boot, or simply use SD_INTER_REP as a conservative plug-in
for σ_boot (since σ_boot ≤ SD_INTER_REP at the worst-case scenario).

**Source 3 — calibration MC noise.**
`bootstrap_ci_averaged` calls `calibrate_rho` once per scenario (same as
power). The result is cached and used for all n_reps datasets. Calibration
noise shifts the CI center but does not affect CI width. With the default
n_cal = 300, SE_cal = 0.112 / √300 = 0.00647 — larger than the ±0.01 tier
target. The corrected tier parameters (see table below) size n_cal to balance
calibration noise against inter-rep noise.

**Calibration noise affects CI center (absolute endpoints), not CI width.**
If the use case is comparing CI widths between methods (bootstrap vs
asymptotic), calibration noise cancels and the uncorrected tiers (n_cal = 300)
are adequate. If the use case is evaluating absolute endpoint values (e.g.
whether the CI excludes zero), the corrected tiers apply.

**Source 4 — calibration interpolation.**
The four observed rhos are ±0.06 and ±0.13. |ρ| = 0.06 falls below the
first probe (0.10) — in the extrapolation region where interpolation error
is ≈ 0.0008–0.0011. |ρ| = 0.13 falls between probes 0.10 and 0.20, where
error ≤ 0.00021. Both are small relative to the ±0.01 tier.

### Bootstrap quantile noise: derivation and numerical values

The variance of the mean endpoint decomposes as (see README §`n_boot` choice):

```math
Var(L̄) = σ_rep² / n_reps + p(1−p) / (n_reps × n_boot × f²)
```

The bootstrap term relative to the inter-rep term:

```
variance ratio = p(1−p) × σ_boot² / (n_boot × φ(z_p)² × σ_rep²)
```

**With f = 1 (simplified):** ratio = 0.024375 / (500 × 1 × 0.0169) ≈ 0.0029
(0.29% of variance, 5.4% of inter-rep SE).

**With scenario-specific f (see Source 2 above):**

| Scenario | f | 1/f² | Variance ratio | SE ratio |
|----------|---|------|----------------|----------|
| f = 1 (simplified) | 1.00 | 1.0 | 0.29% | 5.4% |
| ρ ≈ 0.30, typical (f = 0.52) | 0.52 | 3.7 | 1.1% | 10.3% |
| ρ ≈ 0.11, worst case (f = 0.45) | 0.45 | 4.9 | 1.4% | 11.9% |
| **Universal bound** (σ_boot = σ_rep) | — | — | **1.43%** | **12.0%** |

The universal bound (derived in Source 2 above) is p(1−p) / (n_boot × φ(−1.96)²)
= 0.024375 / (500 × 0.003412) = **1.43%**. It holds for any data and does not
depend on σ_rep, n, ρ, or tie structure — only on n_boot and p. All scenarios
fall at or below this bound.

Bootstrap quantile SE (f = 1, n_boot = 500) per tier, using corrected
n_reps (balanced including calibration):

| Tier | n_reps | SE_boot | 95% HW_boot | % of inter-rep SE |
|------|--------|---------|-------------|-------------------|
| ±0.01 | 1,300 | 0.000194 | 0.000380 | 5.4% |
| ±0.002 | 32,500 | 0.0000387 | 0.0000759 | 5.4% |
| ±0.001 | 129,700 | 0.0000194 | 0.0000380 | 5.4% |

The 5.4% ratio is constant across tiers because both SE_boot and SE_rep
scale as 1/√n_reps. This ratio is independent of tier.

### Numerical values per tier (corrected, including calibration)

Benchmark coefficients: σ_rep = 0.13, k = 0.112, n_boot = 500, f = 1
(simplification; actual f = 0.45–0.52 depending on scenario — see Source 2 above).

| Source | ±0.01 | ±0.002 | ±0.001 |
|--------|-------|--------|--------|
| 1. Inter-rep (95% HW) | 0.0071 (n_reps = 1,300) | 0.0014 (n_reps = 32,500) | 0.00071 (n_reps = 129,700) |
| 2. Bootstrap quantile (95% HW) | 0.000380 (n_boot = 500) | 0.0000759 | 0.0000380 |
| 3. Calibration MC (95% HW) | 0.0069 (n_cal = 1,000) | 0.0014 (n_cal = 24,100) | 0.00070 (n_cal = 96,400) |
| 4. Cal interpolation | < 0.001 | < 0.001 | 0.00062 |
| **Combined 1+2+3 (RSS)** | **0.010** | **0.0020** | **0.0010** |

The n_reps values here are approximately double the uncorrected values (which
ignored calibration). The n_cal values are identical to the power tier values
(same formula, same k).

**For comparison — uncorrected tiers (old CI_TIERS, calibration not accounted for):**

| Tier | n_reps (old) | n_cal (old, implicit default) | SE_total (actual) | Actual 95% HW |
|------|-------------|-------------------------------|-------------------|---------------|
| ±0.01 | 650 | 300 (config default) | 0.00824 | 0.016 |
| ±0.002 | 16,240 | 300 | 0.00655 | 0.013 |
| ±0.001 | 64,930 | 300 | 0.00649 | 0.013 |

With n_cal = 300 (config default), calibration noise (SE_cal = 0.112/√300 =
0.00647) dominates and makes all three tiers effectively equivalent and far
from their stated targets. Increasing n_reps alone cannot fix this — n_cal
must be increased alongside.

---

## Part 3: Quick-reference parameter table

Parameters required to meet each precision target (current Karwowski data,
nonparametric or empirical generator):

| Target | n_sims | n_cal (power) | n_pre | n_reps | n_cal (CI) | n_boot |
|--------|--------|---------------|-------|--------|------------|--------|
| ±0.01 | 2,220 | 1,000 | 50,000 | 1,300 | 1,000 | 500 |
| ±0.002 | 55,520 | 24,100 | 200,000* | 32,500 | 24,100 | 500 |
| ±0.001 | 222,050 | 96,400 | 500,000 | 129,700 | 96,400 | 500 |

(*n_pre = 50,000 gives ±0.0027 combined HW for ±0.002 tier — insufficient;
200,000 brings it to ±0.0022, which is acceptable)

Note: n_cal is the same for power and CI because both use the same formula
k / √n_cal and the same k = 0.112.

---

## Part 4: Which errors matter for which output

| Parameter | Affects min detectable rho (power)? | Affects CI endpoint (absolute)? | Affects CI width? |
|-----------|-------------------------------------|---------------------------------|-------------------|
| n_sims | Yes (source 1) | No | No |
| n_cal | Yes (source 2) | Yes (source 3) | No |
| n_pre | Yes (source 3) | No | No |
| n_reps | No | Yes (source 1) | No |
| n_boot | No | Negligible (source 2) | No |

CI width depends on the underlying data variability (n, ρ, tie structure) and
is not controlled by these simulation parameters.

---

## Part 5: Re-estimating coefficients for new data

The numerical values in this document are for the Karwowski study
(n ≈ 73–82, current tie structures in config.CASES). For different data:

1. **Re-estimate c** via `scripts/estimate_bisection_c.py` using **n_sims ≥ 5,000**.
   At n_sims = 2,000 the finite-difference slope estimator is highly variable
   (c-hat ranges from ~0.10 to ~0.18 depending on seed for the same scenario).
   This is a property of the estimation procedure — the underlying formula
   SE = c/√n_sims is valid at all n_sims. The analytical formula gives c = 0.17
   at worst-case ρ* ≈ 0.33 and is suitable as a conservative upper bound when
   simulation data is unavailable.

2. **Re-estimate k** via `scripts/estimate_calibration_k.py --analytical` (instant).
   Formula: k = (1 − ρ²) √(1.06/(n−3)) at the calibration probe ρ = 0.30.

3. **Re-estimate σ_rep** via `scripts/estimate_interrep_sd.py --analytical --with-ties`.
   Formula: σ_rep = (1 − endpoint²) × √(1.06/(n−3)) × FHP_factor.

4. Update `C_BISECTION`, `C_CAL`, `SD_INTER_REP` in
   `benchmarks/benchmark_precision_params.py` and re-run it.

See [PRECISION\_WHEN\_DATA\_CHANGES.md](PRECISION_WHEN_DATA_CHANGES.md) for
detailed guidance and sensitivity tables.

---

## Appendix: Common Questions

### Q: Does c change depending on n_sims? Does using n_sims = 2,220 (the ±0.01 tier) give a different c than n_sims = 55,520?

**No. c is a property of the population power curve, not of the simulation.**

The true power function is:

```
power(ρ) = P(reject H₀ | true Spearman correlation = ρ)
```

This is a fixed, smooth, deterministic function of ρ. It depends on the study
design (n, α, x tie structure, p-value method) but **not** on n_sims. The slope
power′(ρ*) at the 80% crossing is a single number, so c = √(π(1−π)) / |power′(ρ*)|
is a single number. n_sims is the number of Monte Carlo draws — it controls how
precisely you *estimate* points on this curve, but it does not change the curve
itself.

**Why does c-hat vary from ~0.10 to ~0.18 at n_sims = 2,000 (same scenario,
different seeds)?**

The script `scripts/estimate_bisection_c.py` estimates c by a finite-difference
slope: evaluate power at ρ* ± δ (two calls at n_sims = 2,000), then divide the
difference by 2δ. At n_sims = 2,000, each power evaluation has SE ≈ √(0.16/2000)
≈ 0.009. The slope estimate is the difference of two noisy values divided by a
small step 2δ ≈ 0.02:

```
SE(slope-hat) ≈ √2 × 0.009 / 0.02 ≈ 0.64
```

The true slope is ~2.5, so relative SE ≈ 26%. Propagating through c-hat:

```
SE(c-hat) / c ≈ SE(slope-hat) / slope ≈ 26%
```

With c ≈ 0.14, a ±26% spread gives a range of roughly 0.10 to 0.18 — exactly
what is observed. This is pure estimation noise in the slope measurement. The
true c is constant; it is the *estimator* that is noisy at low n_sims.

**Why is the formula SE(ρ*) = c / √n_sims still valid at n_sims = 2,220?**

The Waeber, Frazier & Henderson (2013) result is asymptotically exact in the
usual Monte Carlo sense: the bisection output ρ* is an unbiased estimator of
the true crossing point with variance c² / n_sims. This holds for every n_sims,
including 2,220. There are higher-order curvature corrections of order O(1/n_sims)
(not O(1/√n_sims)), which at n_sims = 2,220 contribute roughly 10⁻⁴ to 10⁻⁵ to
the bias — more than two orders of magnitude below the 0.004 SE at that tier and
completely negligible.

With Common Random Numbers (same seed passed to every `estimate_power` call inside
the bisection loop), the power function is deterministic for a given seed: bisection
converges cleanly to the exact crossing point of that realization. The variance of
ρ* across seeds is still c² / n_sims, consistent with the general result.

**Summary:**

| Quantity | Depends on n_sims? | Notes |
|---|---|---|
| True c | No | Property of the power curve shape |
| c-hat from `estimate_bisection_c.py` | Yes (noisy at low n_sims) | Estimation procedure, not the true value |
| SE(ρ*) = c / √n_sims | Yes (gets smaller) | Formula valid at all n_sims |
| Guidance "use n_sims ≥ 5000 to estimate c" | N/A | Applies only to the script, not to running power sims |

Use n_sims ≥ 5,000 only when **re-estimating c** for new data via the script.
The tier n_sims values (2,220 for ±0.01, etc.) are sized by the formula
SE = c / √n_sims and are correct — they do not need to satisfy the
"n_sims ≥ 5,000 for stable c estimation" requirement.

---

## References

- Waeber, R., Frazier, P. I., & Henderson, S. G. (2013). Bisection search with
  noisy responses. *SIAM Journal on Control and Optimization*, 51(3), 2261–2279.
  — Provides the theoretical basis for SE_bisection = c / √n_sims as a
  standard result in stochastic root-finding.
- Bonett, D. G., & Wright, T. A. (2000). Sample size requirements for estimating
  Pearson, Kendall and Spearman correlations. *Psychometrika*, 65(1), 23–28.
  — Source of the 1.06 constant used in k and σ_rep formulas.
- Fieller, E. C., Hartley, H. O., & Pearson, E. S. (1957). Tests for rank
  correlation coefficients. I. *Biometrika*, 44(3/4), 470–481.
  — FHP tie-correction variance formula used in σ_rep.
