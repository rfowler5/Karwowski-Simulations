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

1. **Re-estimate k** via `scripts/estimate_calibration_k.py --analytical` (instant).
   Formula: k = (1 − ρ²) √(1.06/(n−3)) at the calibration probe ρ = 0.30.
   Depends on n only (probe ρ is a fixed design constant). Take worst case
   across cases (smallest n).

2. **Re-estimate σ_rep** via `scripts/estimate_interrep_sd.py --analytical --with-ties`.
   Formula: σ_rep = (1 − endpoint²) × √(1.06/(n−3)) × FHP_factor.
   Depends on n, observed ρ (from case data), and tie structure. Does not
   depend on n_boot. Take worst case across cases.

3. **Re-estimate c** via `scripts/estimate_bisection_c.py --analytical --rho <rho*>`.
   Formula: c = √(π(1−π)) / |power′(ρ*)|. This is the only coefficient that
   requires ρ* (a simulation output). Use a conservative ρ* guess initially,
   then refine after a quick ±0.01 run. The analytical (no-tie) formula is
   generator-independent and conservative. c increases with ρ*, so round ρ*
   **up** for conservatism. For empirical re-estimation, use n_sims ≥ 5,000
   (the finite-difference slope estimator is highly variable at n_sims = 2,000).

4. Update `C_BISECTION`, `C_CAL`, `SD_INTER_REP` in
   `benchmarks/benchmark_precision_params.py` and re-run it.

See [PRECISION\_WHEN\_DATA\_CHANGES.md](PRECISION_WHEN_DATA_CHANGES.md) for
the full step-by-step process, coefficient dependency table, and sensitivity
analysis (including k variation across multipoint calibration probes).

---

## Part 6: Runtime optimization — Neyman allocation and disk-persistent calibration

This section analyzes how precomputing calibration to disk affects the
runtime–precision trade-off and derives optimized tier parameters for use
with disk persistence. It covers two configurations:

- **Option A (conservative):** current balanced tiers + disk persistence
  (saves calibration warmup time only).
- **Option B (aggressive):** disk persistence + reduced n\_sims / n\_reps
  (saves up to ~45% of per-run time at ±0.001).

All cost estimates use the nonparametric/empirical fast-path generators.
Copula costs differ (see caveats at end of section).

### Why the balanced design is not runtime-optimal

The balanced tier design (Parts 1–3) equalizes variance contributions:

```
σ² = C_B²/n_sims + C_K²/n_cal = S,    with C_B²/n_sims = C_K²/n_cal = S/2
```

This gives n\_sims = 2C\_B²/S and n\_cal = 2C\_K²/S. Equal-variance
allocation minimizes total variance at fixed total sample count
(n\_sims + n\_cal), but not at fixed **runtime**. The per-unit costs
differ (benchmarked on NP/empirical, hot cache, 88 scenarios):

```
c_sim ≈ 0.589 s/unit    (29.45s at n_sims=50)
c_cal ≈ 0.113 s/unit    (34s at n_cal=300)
```

Calibration is ~5× cheaper per unit than simulation.

**Neyman allocation (minimum runtime per run).** Minimizing
T = c\_sim × n\_sims + c\_cal × n\_cal subject to the precision constraint,
the Lagrange optimality conditions give:

```
∂L/∂n_sims = c_sim − λ C_B²/n_sims² = 0   →   n_sims = C_B √(λ/c_sim)
∂L/∂n_cal  = c_cal − λ C_K²/n_cal² = 0   →   n_cal  = C_K √(λ/c_cal)

Ratio:  n_sims/n_cal = (C_B/C_K) × √(c_cal/c_sim)
                     = (0.17/0.112) × √(0.113/0.589)
                     = 1.518 × 0.438 = 0.665
```

The minimum-time allocation has n\_cal ≈ 1.50 × n\_sims — **more**
calibration than simulation — because calibration is cheaper per unit.
The minimum cost envelope is:

```
T_min = (C_B √c_sim + C_K √c_cal)² / S
T_bal = 2(C_B² c_sim + C_K² c_cal) / S

T_min / T_bal = (C_B √c_sim + C_K √c_cal)² / [2(C_B² c_sim + C_K² c_cal)]
              = (0.1305 + 0.0377)² / [2 × (0.01702 + 0.001417)]
              = 0.02827 / 0.03688
              = 0.767
```

Neyman allocation saves ~23% of total runtime vs the balanced design.
However, it **increases** calibration cost (higher n\_cal) to reduce
simulation cost. No reallocation simultaneously reduces both calibration
and total cost. Neyman allocation is therefore superseded by disk
persistence (below), which eliminates calibration from per-run cost
entirely.

### Disk-persistent calibration

Precomputing calibration once to disk and loading at runtime eliminates
calibration from per-run cost:

```
T_run = c_sim × n_sims    (calibration ≈ 0, just file loading)
```

The precomputed n\_cal (denoted n\_cal\_pre) can be set independently of
per-run cost since it is paid once.

**Option A (conservative):** keep current POWER\_TIERS and CI\_TIERS.
Precompute calibration at the existing n\_cal values. Per-run saving
equals the calibration warmup time: ~3 hrs sequential at ±0.001 (~8% of
total). No parameter changes required.

**Option B (aggressive):** precompute calibration at a larger
n\_cal\_pre, then use fewer n\_sims / n\_reps. With more precise
calibration, less of the variance budget is consumed by calibration
noise, freeing budget for bisection / inter-rep noise.

### Deriving reduced n\_sims with disk persistence (Option B)

From the precision constraint with n\_cal fixed at n\_cal\_pre:

```
C_B²/n_sims + C_K²/n_cal_pre = S

n_sims = C_B² / (S − C_K²/n_cal_pre)
```

Valid when n\_cal\_pre > C\_K²/S (calibration alone must not exceed the
total budget). At ±0.001: n\_cal\_pre > 48,200.

**Limiting case.** As n\_cal\_pre → ∞:

```
n_sims_min = C_B² / S = n_sims_balanced / 2
```

The balanced design allocates exactly half the variance budget to
calibration. Eliminating calibration noise frees that half for bisection,
halving the required n\_sims: a **50% reduction in per-run simulation
time** (theoretical maximum).

**For CI endpoints,** the same structure applies with σ\_rep replacing
C\_B and n\_reps replacing n\_sims:

```
n_reps = σ_rep² / (S − C_K²/n_cal_pre)
```

The same precomputed calibration cache serves both power and CI. A single
precompute at n\_cal\_pre = 500,000 enables reduced n\_sims (power)
**and** reduced n\_reps (CI) at all tiers.

**General recipe.** For arbitrary target half-width h and precomputed
n\_cal\_pre:

```
S        = (h / 1.96)²
n_sims   = ⌈C_B²    / (S − C_K² / n_cal_pre)⌉     (power)
n_reps   = ⌈σ_rep²  / (S − C_K² / n_cal_pre)⌉     (CI)
n_boot   = 500                                       (unchanged)

Requires: n_cal_pre > C_K² / S
```

### Practical savings vs n\_cal\_pre (±0.001 power tier)

Constants: C\_B = 0.17, C\_K = 0.112, c\_sim = 0.589, c\_cal = 0.113.

| n\_cal\_pre | Cal var (×10⁻⁷) | n\_sims | Per-run (hrs) | Δ vs balanced | Precompute seq (hrs) |
|---|---|---|---|---|---|
| 96,400 (balanced) | 1.301 | 222,050 | 36.3 | — | 3.0 |
| 200,000 | 0.627 | 146,250 | 23.9 | −34% | 6.3 |
| 500,000 | 0.251 | 122,900 | 20.1 | −45% | 15.7 |
| 1,000,000 | 0.125 | 116,600 | 19.1 | −47% | 31.4 |
| ∞ (limit) | 0 | 111,025 | 18.2 | −50% | — |

Per-run times: T\_run = c\_sim × n\_sims (88 scenarios, NP/empirical).
Diminishing returns: 68% of the possible savings are captured by
n\_cal\_pre = 200,000; 89% by 500,000.

### Optimized tier parameters (Option B, n\_cal\_pre = 500,000)

**Power tiers:**

| Half-width | n\_sims (balanced) | n\_sims (disk-opt) | Reduction | n\_cal\_pre |
|---|---|---|---|---|
| ±0.01 | 2,220 | 1,120 | 50% | 500,000 |
| ±0.002 | 55,520 | 28,450 | 49% | 500,000 |
| ±0.001 | 222,050 | 122,900 | 45% | 500,000 |

Verification (±0.001): HW = 1.96 × √(0.0289/122,900 + 0.01254/500,000)
= 1.96 × √(2.352×10⁻⁷ + 2.509×10⁻⁸) = 1.96 × 5.10×10⁻⁴ = 0.00100. ✓

**CI tiers** (n\_boot = 500):

| Half-width | n\_reps (balanced) | n\_reps (disk-opt) | Reduction | n\_cal\_pre |
|---|---|---|---|---|
| ±0.01 | 1,300 | 650 | 50% | 500,000 |
| ±0.002 | 32,500 | 16,650 | 49% | 500,000 |
| ±0.001 | 129,700 | 71,900 | 45% | 500,000 |

The reduction is larger at looser tiers because calibration variance
(C\_K²/500,000 = 2.51×10⁻⁸, a fixed quantity) is a smaller fraction of
their larger budgets: 0.1% of S at ±0.01, 2.4% at ±0.002, 9.6% at
±0.001.

### Precompute cost and break-even

One-time precompute cost: T\_pre = c\_cal × n\_cal\_pre / (P × η), where
P = logical cores and η = parallel efficiency. Calibration for each of
the 88 scenario types is independent — embarrassingly parallel. For
CPU-bound array operations on 2 physical cores + hyperthreading (4
logical), η ≈ 0.5–0.7 (HT adds ~20–30% throughput per physical core,
not 100%):

| n\_cal\_pre | Seq | 4-core η=0.7 | 4-core η=0.5 | 8-core η=0.75 |
|---|---|---|---|---|
| 200,000 | 6.3 hrs | 2.2 hrs | 3.2 hrs | 1.0 hrs |
| 500,000 | 15.7 hrs | 5.6 hrs | 7.8 hrs | 2.6 hrs |

Per-run savings (vs balanced + disk, ±0.001):

| n\_cal\_pre | ΔT per run | T\_pre (4c, η=0.7) | Break-even |
|---|---|---|---|
| 200,000 | 12.4 hrs | 2.2 hrs | < 1 run |
| 500,000 | 16.2 hrs | 5.6 hrs | < 1 run |

Break-even is always less than one run at ±0.001: the precompute cost is
recovered within a single run.

**Total time including precompute (±0.001, n\_cal\_pre = 500,000, 4-core
η = 0.7):**

| Runs | Balanced + disk | Disk-opt | Savings |
|---|---|---|---|
| 1 | 37.4 hrs | 25.7 hrs | 31% |
| 2 | 73.7 hrs | 45.8 hrs | 38% |
| 3 | 110.0 hrs | 65.9 hrs | 40% |
| Many | 36.3 hrs/run | 20.1 hrs/run | 45% |

### CI-specific runtime savings

The CI per-run cost depends on n\_reps and n\_boot (data generation +
bootstrap). From benchmarks (empirical, hot cache, 88 scenarios,
n\_boot = 500):

```
c_CI ≈ 0.587 s/rep    (117.44s at n_reps=200, sequential)
```

**Per-run CI time by tier (sequential, 88 scenarios):**

| Tier | n\_reps (bal) | CI time (bal) | n\_reps (opt) | CI time (opt) | Savings |
|---|---|---|---|---|---|
| ±0.01 | 1,300 | 12.7 min | 650 | 6.4 min | 50% |
| ±0.002 | 32,500 | 5.3 hrs | 16,650 | 2.7 hrs | 49% |
| ±0.001 | 129,700 | 21.1 hrs | 71,900 | 11.7 hrs | 45% |

CI savings are proportional to the n\_reps reduction and follow the same
pattern as power savings. CI endpoints are unaffected by permutation
noise — the CI pipeline uses bootstrap resampling, not permutation tests.
Reducing n\_reps has no interaction with n\_perm.

**Combined power + CI (±0.001, sequential, 88 scenarios):**

| | Power | CI | Combined | Precompute (once) |
|---|---|---|---|---|
| Balanced + disk | 36.3 hrs | 21.1 hrs | 57.5 hrs/run | 1.1 hrs (4c, η=0.7) |
| Disk-opt | 20.1 hrs | 11.7 hrs | 31.8 hrs/run | 5.6 hrs (4c, η=0.7) |
| Savings | 45% | 45% | 45% | — |

The same precomputed calibration cache serves both power and CI — the
precompute cost is paid once for both.

**Combined total including precompute (±0.001, n\_cal\_pre = 500,000,
4-core η = 0.7):**

| Runs | Balanced + disk | Disk-opt | Savings |
|---|---|---|---|
| 1 | 58.6 hrs | 37.4 hrs | 36% |
| 2 | 116.0 hrs | 69.2 hrs | 40% |
| Many | 57.5 hrs/run | 31.8 hrs/run | 45% |

### Interaction with permutation noise (MC fallback path)

With disk optimization, bisection accounts for ~90% of the power variance
budget (vs 50% in the balanced design). The multiplicative SE inflation
from per-sim permutation noise (Part 1, Source 5) therefore has slightly
more impact on the total HW. **This source is absent when using the
precomputed null (default configuration).** It affects only the MC fallback
path (`EMPIRICAL_USE_PRECOMPUTED_NULL = False`).

The SE inflation ε from finite n\_perm is a property of n\_perm alone,
independent of n\_sims (see README "Permutation p-value noise in power
estimates" for the derivation):

```
HW_inflated² = (1.96)² × [C_B²/n_sims × (1+ε)² + C_K²/n_cal_pre]
```

With the current config, n\_perm = 2,000 when n\_sims < 5,000 (ε ≈ 0.86%)
and n\_perm = 1,000 otherwise (ε ≈ 1.21%).

**Numerical verification (MC fallback path, worst case, all tiers):**

| Tier | Design | n\_sims | n\_perm | ε | HW (inflated) | Target | Over? |
|---|---|---|---|---|---|---|---|
| ±0.01 | Balanced | 2,220 | 2,000 | 0.86% | 0.00995 | 0.01 | no |
| ±0.01 | Disk-opt | 1,120 | 2,000 | 0.86% | 0.01005 | 0.01 | +0.5% |
| ±0.002 | Balanced | 55,520 | 1,000 | 1.21% | 0.00201 | 0.002 | +0.6% |
| ±0.002 | Disk-opt | 28,450 | 1,000 | 1.21% | 0.00202 | 0.002 | +1.2% |
| ±0.001 | Balanced | 222,050 | 1,000 | 1.21% | 0.00101 | 0.001 | +0.6% |
| ±0.001 | Disk-opt | 122,900 | 1,000 | 1.21% | 0.00101 | 0.001 | +1.1% |

The ε values are worst case (f\_p(α) = 1); under realistic conditions the
inflation is smaller (see README derivation).

**Observations:**

1. **Default (precomputed null): no interaction at any tier.** Source 5 is
   absent — the precomputed null provides a single fixed critical value
   with no per-sim noise. Disk-optimized tiers meet their precision
   targets exactly (for sources 1+2).

2. **MC fallback, ±0.01:** the balanced design stays within target
   (0.00995 < 0.01) thanks to slack from rounding n\_cal up (963 → 1,000).
   The disk-optimized design slightly exceeds (0.01005, +0.5%) because
   calibration variance is negligible at n\_cal\_pre = 500,000, leaving no
   slack. If using the MC fallback at ±0.01, increasing n\_sims from 1,120
   to 1,140 compensates (adds < 1s of runtime).

3. **MC fallback, ±0.002 and ±0.001:** both designs slightly exceed
   targets. This is not new — the existing balanced tiers already
   overshoot by ~0.6% when using the MC fallback. Disk optimization
   increases the overshoot to ~1.1–1.2%, still negligible.

4. **All overshoots ≤ 1.2%.** This is the same magnitude as Source 5 is
   documented to contribute in Part 1. No new precision risk is
   introduced by disk optimization.

**Systematic review of all other power error sources with lower n\_sims:**

| Source | Depends on n\_sims? | Effect of lower n\_sims |
|---|---|---|
| 1. Bisection MC | Yes (main term) | Accounted for in the formula |
| 2. Calibration MC | No (one calibration, shared) | Unchanged |
| 3. Null realization | No (one null, shared, correlated) | Unchanged (never averaged with n\_sims) |
| 4. Interpolation | No (fixed structural error) | Unchanged |
| 5. Permutation noise (MC only) | Multiplicative ε on SE | Analyzed above; ≤ 1.2% HW overshoot |
| 6. y-tie approx | No (systematic bias ~10⁻⁵) | Unchanged |

**CI endpoints:** no permutation tests are used. Source 5 does not exist
for CI. Reducing n\_reps has no interaction with any permutation-related
noise. The CI error budget (Part 2) depends only on n\_reps, n\_boot, and
n\_cal — all correctly accounted for in the disk-optimized CI tier
formulas.

### Caveats

1. **Cost linearity at scale.** The per-unit costs c\_sim and c\_cal are
   extrapolated from benchmarks at n\_sims = 50, n\_cal = 300. At
   n\_cal\_pre = 500,000, calibration arrays are ~320 MB per scenario
   (shape 500,000 × 80), exceeding L3 cache. Memory-bandwidth effects
   may increase effective c\_cal at large scale. This affects
   **precompute time estimates only** — the precision-based n\_sims
   reduction depends on C\_B and C\_K (precision constants), not cost
   estimates.

2. **Memory for parallel precompute.** Each worker holds ~640 MB–1 GB of
   arrays at n\_cal\_pre = 500,000. With 4 workers: ~2.5–4 GB. Verify
   available RAM before parallelizing at high n\_cal\_pre.

3. **Cache key compatibility.** The calibration cache key includes n\_cal:
   `(n, n_distinct, distribution_type, all_distinct, n_cal)`. Precomputed
   entries at n\_cal\_pre = 500,000 are found only when the runtime
   calibration call uses n\_cal = 500,000. If the disk cache is missing
   (not loaded), calibration at n\_cal = 500,000 will compute from
   scratch — extremely slow (~16 hrs sequential). Ensure the disk cache
   is always loaded when using the optimized tiers.

4. **Other error sources.** Reducing n\_sims / n\_reps affects sources 1
   and 2 only. Null realization (source 3) and calibration interpolation
   (source 4) are unchanged and must be managed independently. At
   ±0.001, the null still requires n\_pre ≥ 500,000 regardless of disk
   persistence (see Part 1). Permutation noise (source 5, MC fallback
   only) has a slightly larger impact with disk optimization because
   bisection accounts for a larger fraction of the variance budget — see
   "Interaction with permutation noise" above for the full analysis. The
   worst-case overshoot is ≤ 1.2% of the target HW and is absent in
   the default (precomputed null) configuration.

5. **Copula generator.** Copula calibration costs ~10× more per unit than
   NP/empirical before the fast-path optimization. Precompute times for
   copula are correspondingly higher. After the copula fast-path
   optimization (replacing generate\_y\_copula with rank-equivalent z\_y
   computation in the bisection loop), copula costs should approach
   NP/empirical levels and the tables above become applicable.

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
