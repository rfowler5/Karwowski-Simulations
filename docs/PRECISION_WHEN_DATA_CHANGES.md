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

## Coefficient dependencies

| Coefficient | Depends on | Does NOT depend on |
|---|---|---|
| *k* (calibration) | n, probe ρ (fixed at 0.30) | n\_sims, n\_reps, observed ρ, ρ\*, generator, n\_boot |
| *c* (bisection) | n, ρ\* (min detectable rho) | n\_sims, n\_cal, observed ρ, probe ρ, generator (analytical) |
| *σ\_rep* (inter-rep) | n, observed ρ, tie structure (FHP) | n\_sims, n\_reps, n\_boot, ρ\*, probe ρ, generator |

None of the three depends on n\_sims, n\_reps, or n\_boot. These simulation
counts control how precisely the *outputs* are estimated, not the coefficients
themselves. Sufficient simulation counts are needed only when estimating the
coefficients *empirically* via the scripts; the analytical formulas bypass
this entirely.

### Only *c* requires a simulation output (ρ\*)

*k* and *σ\_rep* can be computed from scenario parameters known before any
simulation: n (sample size), observed ρ (from the data), and tie structure
(from `config.CASES`). These are "compute once" quantities — run the
analytical script and take the worst case.

*c* requires ρ\* — the minimum detectable rho at 80% power — which is itself
the main output of the power simulation. This creates a mild chicken-and-egg
situation: you need *c* to size n\_sims, but you need ρ\* to compute *c*, and
you need n\_sims to estimate ρ\*. The resolution is to start with a
conservative analytical estimate and refine after a quick ±0.01 run (see
"Step-by-step process" below).

### *k* and multipoint calibration

The calibration pipeline bisects independently at each of 5 probes
\[0.10, 0.20, 0.30, 0.40, 0.50\]. Each probe has its own calibration noise
level k\_probe = (1 − ρ\_probe²) √(1.06/(n−3)):

| Probe ρ | (1 − ρ²) | k (n=73) | k (n=82) |
|---|---|---|---|
| 0.10 | 0.990 | 0.122 | 0.115 |
| 0.20 | 0.960 | 0.118 | 0.111 |
| **0.30** | **0.910** | **0.112** | **0.105** |
| 0.40 | 0.840 | 0.103 | 0.097 |
| 0.50 | 0.750 | 0.092 | 0.087 |

For the **power** application (ρ\* ≈ 0.28–0.34), the relevant probes are 0.20
and 0.40, and k ≈ 0.112 (at probe 0.30) is appropriate. For **CI** at small
observed ρ (e.g. |ρ| ≈ 0.06), the relevant probe is 0.10 where k ≈ 0.122 —
about 9% higher. The absolute ceiling is k\_max = √(1.06/(n−3)), the ρ → 0
limit (0.123 for n=73).

The benchmark default k = 0.112 is conservative for power but slightly
underestimates calibration noise for CI at very small observed ρ. For a single
conservative value covering all applications, use k at the lowest probe (0.10)
or the ceiling √(1.06/(n\_min−3)).

### *σ\_rep* and observed ρ

*σ\_rep* depends on the observed ρ through the CI endpoint values: the endpoint
closest to zero has the highest SD. The observed ρ is a property of the study
data — it is read from `config.CASES[case_id]["observed_rho"]` and requires no
user judgment. Running `estimate_interrep_sd.py --analytical --with-ties`
reports *σ\_rep* for all cases; take the maximum.

*σ\_rep* does **not** depend on n\_boot. The analytical formula does not involve
n\_boot. When estimating *σ\_rep* empirically, the observed SD across
replications includes a small bootstrap quantile noise component that inflates
the estimate, but at n\_boot ≥ 500 this adds < 1.5% to the variance (see
UNCERTAINTY\_BUDGET.md Part 2, "Universal bound").

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

## Step-by-step process: selecting constants for new data

### Step 1: Compute *k* and *σ\_rep* analytically (instant)

These two coefficients depend only on scenario parameters known before any
simulation. Run:

```bash
python scripts/estimate_calibration_k.py --analytical
python scripts/estimate_interrep_sd.py --analytical --with-ties
```

Take the worst case across all cases:

- **k:** largest value (= smallest n, since k ∝ 1/√(n−3)).
- **σ\_rep:** largest SD across all cases and endpoints.

No user judgment is needed. The inputs (n, observed ρ, tie structure) come
from `config.CASES`. These values are final — they do not change after
simulation.

### Step 2: Estimate *c* analytically with a conservative ρ\* guess (instant)

The analytical formula for *c* requires ρ\* (the min detectable rho at 80%
power). Before running any simulation, use a rough guess — either the default
0.33 (current Karwowski data) or a rough estimate from the asymptotic power
formula. Run:

```bash
python scripts/estimate_bisection_c.py --case 3 --rho 0.33 --analytical
```

The analytical (no-tie) formula is conservative: at any given ρ\*, it
overestimates *c* because the actual tied power curve is steeper than the
no-tie asymptotic (see "Competing factors with ties" above). This holds for
all generators.

**Choosing the conservative direction for ρ\*.** *c* increases monotonically
with ρ\* in the relevant range (see table in "Heavier ties" above): a higher
ρ\* gives a shallower power curve slope, hence larger *c*. To be conservative,
round your ρ\* guess **up**, not down. If unsure, use 0.40 as a safe upper
bound for moderate-n studies; the analytical *c* at ρ\* = 0.40 is 0.40
(roughly 2× the current default), which would be very conservative but safe.

### Step 3: Run a quick simulation at ±0.01 (minutes)

Run the power analysis at the ±0.01 tier (n\_sims ≈ 2,220, using the *c* from
Step 2). This gives the actual min detectable rho to ±0.01 precision and takes
only minutes:

```bash
python benchmarks/benchmark_precision_params.py    # verify tier parameters
# then run your power analysis at the ±0.01 tier
```

Compare the actual ρ\* against the value assumed in Step 2.

- If the actual ρ\* is **within ~0.03** of your guess, the analytical *c* from
  Step 2 is fine.
- If the actual ρ\* is **substantially higher** (e.g. 0.05+ above your
  guess), re-evaluate *c* analytically at the upper end of the ±0.01
  confidence interval (ρ\* + 0.01), which is the conservative direction:

```bash
python scripts/estimate_bisection_c.py --case 3 --rho <rho_star_plus_0.01> --analytical
```

**Generator dependence.** The analytical *c* formula at a given ρ\* is
generator-independent (it uses the no-tie asymptotic). However, the ρ\*
output from the ±0.01 run **is** generator-dependent — different generators
give slightly different power curves and hence slightly different min
detectable rho. If you run multiple generators, use the **largest ρ\*** across
generators for the most conservative *c*.

### Step 4: Optional — empirical verification of *c*

If the analytical *c* is close to the current default (0.17) and you want
reassurance, run the empirical estimation at n\_sims ≥ 5,000:

```bash
python scripts/estimate_bisection_c.py --case 3 --n-sims 5000
```

The empirical *c* should be below the analytical value (the tied power curve
is steeper). If it exceeds the analytical — which would indicate an unusual
power curve shape — use the empirical value with a safety margin.

This step is usually unnecessary: the analytical *c* is a reliable upper bound
for moderate ties. It is most useful when tie structure is unusually heavy and
the power curve shape might differ qualitatively from the no-tie asymptotic.

### Step 5: Update constants and re-run benchmarks

Update `C_BISECTION`, `C_CAL`, and `SD_INTER_REP` in
`benchmarks/benchmark_precision_params.py` with the values from Steps 1–3
(round up for conservatism). Then:

```bash
python benchmarks/benchmark_precision_params.py       # updated tier parameters
python benchmarks/benchmark_realistic_runtimes.py     # updated runtime estimates (quick then scale)
```

### Quick-reference safety margins

If you know ties are heavy but don't want to estimate precisely, use these
conservative values and scale the corresponding n\_\* values:

| Coefficient | Default | Conservative | Very conservative |
|---|---|---|---|
| *c* | 0.17 | 0.25 | 0.40 |
| *k* | 0.112 | 0.122 | √(1.06/(n\_min−3)) |
| *σ\_rep* | 0.13 | 0.15 | 0.18 |

---

## Summary

| Situation | Action |
|---|---|
| Current Karwowski-based data | c = 0.17, k = 0.112, σ\_rep = 0.13 are all fine; no change needed. |
| New data, similar n and ties | Run Steps 1–2 (instant). If *k* and *σ\_rep* are similar, likely no change. Check *c* at ±0.01 (Step 3) to confirm ρ\* is in range. |
| New data, different n or heavier ties | Full Steps 1–5. *k* and *σ\_rep* update analytically (instant); *c* may need refinement after a quick ±0.01 run. |
| Unsure / want safety | Use "Very conservative" column above; accept the higher n\_sims / n\_reps cost. |
