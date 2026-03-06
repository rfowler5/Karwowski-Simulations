# Precision parameters when data changes — revisit and re-run benchmarks

**Reminder:** If you use **different data** or a **different tie structure** (e.g. heavier ties, different n, or a different study), you may need to **revisit precision parameters** so that target accuracies (±0.01, ±0.002, ±0.001) are still met. That typically means updating the bisection coefficient *c*, the inter-rep SD *σ_rep*, and (if needed) the calibration coefficient *k*; recomputing required `n_sims`/`n_cal`/`n_reps`; and **re-running the benchmark scripts** (quick runs at small params, then scale) to get updated runtime estimates.

---

## Three precision coefficients

| Coefficient | Controls | Formula | Current value | Script |
|-------------|----------|---------|---------------|--------|
| *c* (bisection) | n_sims | c = √(π(1−π)) / \|power′(ρ*)\| | **0.17** | `scripts/estimate_bisection_c.py` |
| *k* (calibration) | n_cal | k = (1 − ρ²) √(1.06/(n−3)) | **0.112** | `scripts/estimate_calibration_k.py` |
| *σ_rep* (inter-rep SD) | n_reps | σ = (1 − endpoint²) √(1.06/(n−3)) × FHP | **0.13** | `scripts/estimate_interrep_sd.py` |

All three use the Bonett-Wright framework (SE in Fisher z-space, delta method back to rho-space). The benchmark scripts (`benchmarks/benchmark_precision_params.py`) use the conservative analytical worst-case values.

---

## Current data: all three coefficients are appropriate

For the **current Karwowski-based setup** (n ≈ 73–82, tie structures in `config.CASES`, digitized B-Al/H-Al when using empirical):

### Bisection coefficient *c* = 0.17

- Min detectable rho across generators and scenarios falls in roughly **0.28–0.34**.
- The **asymptotic (no-tie)** formula gives *c* ≈ **0.17** at worst-case ρ* ≈ 0.33 for current data.
- **Empirical** *c* from `scripts/estimate_bisection_c.py` at converged n_sims (5000–10000) is ~**0.12–0.15** across all generators, consistently below the asymptotic value.
- **Competing factors with ties:** Ties alter the no-tie asymptotic in two ways. (1) Ties attenuate realised rho, so the 80% crossing moves to **higher ρ***; the asymptotic formula says slope is shallower there, so *c* would be **larger**. (2) The **actual** (tied, permutation-based) power curve is **steeper** at ρ* than the no-tie asymptotic predicts, so *c* is **smaller**. For current (moderate) ties, the second factor dominates and empirical *c* < 0.17; for heavier ties, ρ* can move far enough that the first factor wins and *c* can exceed 0.17, so re-estimate if use different data.
- The benchmark scripts use **c = 0.17** (asymptotic value), providing ~20% margin above the observed empirical maximum.
- **Note:** At n_sims=2000 (script default), the finite-difference slope has high seed-dependent variance — *c* can range from ~0.10 to ~0.18 for the same scenario. Use n_sims ≥ 5000 for stable estimates.

### Calibration coefficient *k* = 0.112

- Analytical: k = (1 − ρ²) √(1.06/(n−3)) at probe ρ = 0.30. Range: **0.105** (n=82) to **0.112** (n=73).
- Empirical k ≈ 0.10 (~10% below analytical). The benchmark default **0.112** is the analytical worst case.

### Inter-rep SD *σ_rep* = 0.13

- Analytical: SD(endpoint) = (1 − endpoint²) × √(1.06/(n−3)) × FHP_factor.
- The CI endpoint closest to zero has the highest SD (because (1 − endpoint²) is maximised near zero). Worst case is Case 3 (n=73, upper endpoint ≈ 0.11):
  - No ties: **0.122**
  - With worst-case FHP (k=4 heavy_center, +5.7%): **0.129**
- Empirical max from `results/confidence_intervals.csv` (n_reps=200): **0.130**, confirming the analytical prediction.
- The benchmark default **0.13** (rounded from 0.129) is conservative including tie correction.

No change to any coefficient is needed for the current study.

---

## Heavier ties (or different data): c and σ_rep can get larger

### Bisection coefficient *c*

The coefficient is

$$c = \frac{\sqrt{\pi(1-\pi)}}{|\text{power}'(\rho^*)|}$$

So *c* is **larger** when the power curve is **shallower** at the 80% crossing. Heavier ties attenuate the realised Spearman rho, so you need a **higher** target rho to reach 80% power. That moves the crossing to larger ρ*, where the slope is shallower and *c* increases.

Approximate *c* vs ρ* (asymptotic power curve, indicative only):

| ρ* (80% crossing) | slope | c   |
|-------------------|-------|-----|
| 0.30              | 2.86  | 0.14 |
| 0.33              | 2.46  | 0.16 |
| 0.35              | 2.02  | 0.20 |
| 0.38              | 1.38  | 0.29 |
| 0.40              | 1.00  | 0.40 |

So with **heavier ties**, the 80% crossing can move to ρ* ≈ 0.38–0.40, and *c* can reach roughly **0.3–0.4**.

**Impact:** If you keep using c = 0.17 but the true *c* is 0.30, then for the same `n_sims` the actual SE of min detectable rho is about (0.30/0.17) ≈ **1.8×** larger — e.g. a ±0.01 target becomes effectively ±0.018.

### Inter-rep SD *σ_rep*

With different data (different n, different observed rho), the worst-case endpoint and its SD change. The SD is highest when:
- **n is small** (amplifies SE_z = √(1.06/(n−3)))
- **The CI endpoint is near zero** (maximises (1 − endpoint²))
- **Ties are heavy** (FHP factor increases SE)

For the current data the worst case is σ_rep = 0.13. With smaller n or different rho (e.g. rho near 0 with both CI endpoints near 0), σ_rep could be larger. Run `scripts/estimate_interrep_sd.py --analytical --with-ties` to check.

**Impact:** σ_rep affects n_reps as (σ_rep)². If σ_rep grows from 0.13 to 0.15, n_reps scales by (0.15/0.13)² ≈ **1.33** (33% more reps needed).

---

## How parameters change when coefficients change

- **n_sims:** Proportional to *c*². Scale by (c_new/0.17)².  
  Example: if c = 0.20, n_sims multiplies by (0.20/0.17)² ≈ **1.39** (about 39% more).
- **n_cal:** Proportional to *k*². The analytical formula is k = (1 − ρ²) √(1.06/(n−3)). The benchmark default is **0.112** (analytical worst case, n=73). To re-estimate: `scripts/estimate_calibration_k.py --analytical` (instant) or without `--analytical` (empirical).
- **n_reps:** Proportional to *σ_rep*². Scale by (σ_new/0.13)². The analytical formula is σ_rep = (1 − endpoint²) √(1.06/(n−3)) × FHP_factor. The benchmark default is **0.13** (analytical worst case with ties, n=73). To re-estimate: `scripts/estimate_interrep_sd.py --analytical --with-ties` (instant) or without `--analytical` (empirical).
- **n_boot:** Unchanged. Bootstrap quantile noise is negligible for n_boot ≥ 500 regardless of σ_rep.

So when you update a coefficient in `benchmark_precision_params.py`, only the corresponding `n_*` values change.

---

## What to do when data or tie structure changes

1. **Estimate coefficients for the new setup**  
   - **Bisection *c*:** Run `scripts/estimate_bisection_c.py` for your scenario. Example:
     ```bash
     python scripts/estimate_bisection_c.py --case 3 --n-distinct 4 --dist-type heavy_center --generator nonparametric
     python scripts/estimate_bisection_c.py --case 3 --generator empirical --n-sims 5000
     ```
     If you already have min detectable rho, pass it with `--rho`:
     ```bash
     python scripts/estimate_bisection_c.py --case 3 --rho 0.33 --n-sims 5000
     ```
   - **Calibration *k*:** `scripts/estimate_calibration_k.py --analytical` (instant) or without `--analytical` (empirical).
   - **Inter-rep SD *σ_rep*:** `scripts/estimate_interrep_sd.py --analytical --with-ties` (instant) or without `--analytical` (empirical).

2. **Recompute precision parameters**  
   - Update `C_BISECTION`, `C_CAL`, and/or `SD_INTER_REP` in `benchmarks/benchmark_precision_params.py`, then run it to get the new `n_sims`, `n_cal`, `n_reps` for each tier.

3. **Re-run benchmarks**  
   - Run `benchmark_precision_params.py` (optional, for a clean summary).  
   - Run `benchmark_realistic_runtimes.py` at **small params** (quick), then use the scaling formulas so runtimes reflect the new params. No need to run at full precision; quick then scale is enough.

4. **Optional safety margin**  
   - If you know ties are heavy but don't want to estimate precisely, use conservative values (e.g. c = 0.25–0.30, σ_rep = 0.15) and scale the corresponding `n_*` values.

---

## Summary

| Situation                         | Action |
|----------------------------------|--------|
| Current Karwowski-based data    | c = 0.17, k = 0.112, σ_rep = 0.13 are all fine; no change. |
| New data / heavier ties / different study | Re-estimate *c*, *k*, *σ_rep* via the three scripts; update `benchmark_precision_params.py`; re-run benchmarks (quick then scale). |
