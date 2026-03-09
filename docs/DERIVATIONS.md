# Derivations: Calibration Uncertainty from Tied X-Values

This document records the theoretical derivations developed for understanding why
`E[Spearman(rho_in)]` behaves as a step function when x contains ties (grouped data),
the precise structural cause of that staircase, the resonance formula predicting its
location, and the jitter fix that eliminates the staircase without biasing the
correlation.  All predictions were confirmed experimentally.

---

## 1. Setup and Notation

The nonparametric Spearman calibration is built around the rank-mixing formula:

```math
\text{mixed}_i = \rho_{\text{in}} \cdot s_{x,i} + \sqrt{1 - \rho_{\text{in}}^2} \cdot s_{n,i}
```

where

- `s_x` = standardized x-ranks (fixed for a given group template)
- `s_n` = standardized noise ranks (a random permutation of `{1,...,n}`, standardized by population constants)
- `rho_in` ∈ (−1, 1) is the input correlation

Because y is lognormal (no ties), `rank(y_final) = rank(mixed)` (the "skip-y identity"), so calibration depends only on `(n, template, seed, n_cal)`.

For the **k = 4, even** case with n = 80:

- Groups of size 20 each.  Average x-ranks: {10.5, 30.5, 50.5, 70.5}.
- Population mean of x-ranks: $\bar{x} = 40.5$
- Population variance: $\sigma_x^2 = 500$ (computed across all 80 obs), $\sigma_x = \sqrt{500} \approx 22.361$
- Standardized x values (same for all 20 obs in each group):

```math
s_x \in \left\{-\frac{30}{22.361},\; -\frac{10}{22.361},\; +\frac{10}{22.361},\; +\frac{30}{22.361}\right\}
  = \{-1.342,\; -0.447,\; +0.447,\; +1.342\}
```

- Adjacent-group gap in `s_x`: $\Delta = 20/\sqrt{500} = 0.8944$
- Noise standard deviation (population constant): $\sigma_n = \sqrt{(n^2-1)/12} = \sqrt{(6400-1)/12} = \sqrt{533.25} \approx 23.092$
- Noise rank spacing after standardization: $1/23.092 \approx 0.04330$

---

## 2. Main Theorem: E[Spearman(rho_in)] Is a Step Function

**Claim.** When the noise is a permutation of `{1,...,n}`, the function

```math
F(\rho_{\text{in}}) = E_{\text{permutation}}\!\left[\hat{\rho}_S\!\left(\text{mixed}, y\right)\right]
```

is a *step function* (piecewise constant) with jumps at a *deterministic, discrete* set of
values of `rho_in`.

**Mechanism.** The ordering of `mixed_i` relative to `mixed_j` changes when:

```math
\rho_{\text{in}} \cdot s_{x,i} + \sqrt{1-\rho_{\text{in}}^2} \cdot s_{n,i}
= \rho_{\text{in}} \cdot s_{x,j} + \sqrt{1-\rho_{\text{in}}^2} \cdot s_{n,j}
```

Rearranging:

```math
\frac{\rho_{\text{in}}}{\sqrt{1-\rho_{\text{in}}^2}} = -\frac{s_{n,i} - s_{n,j}}{s_{x,i} - s_{x,j}}
```

Since `s_x` differences between groups take only 2(k−1) distinct values
($\pm j\Delta$ for $j = 1, \ldots, k-1$) and `s_n` differences are always integer
multiples of $1/\sigma_n$ (because the raw noise ranks are integers), the right-hand side
ranges over a **finite discrete set** of values for any given permutation.  Each value maps
to at most one $\rho \in (0,1)$ via the strictly monotone function
$\rho \mapsto \rho/\sqrt{1-\rho^2}$.  The set of crossover rho values is therefore
**finite and deterministic** — it does not depend on which random permutation was drawn,
only on the discrete structure of the ranks `{1,...,n}`.

Because Spearman rho is a function of the rank ordering of `mixed`, and that ordering only
changes at the finite discrete crossover points, `F(rho_in)` is constant between
crossovers. **No amount of increased `n_cal` can smooth it.**  More Monte Carlo samples
make the *step heights* more precise, but the steps remain steps.

---

## 3. Crossover Condition for Grouped Data

For two observations in groups separated by j steps (j = 1 adjacent, j = 2 skip-one,
j = 3 skip-two), the s_x difference is:

```math
s_{x,a} - s_{x,b} = j \cdot \Delta = j \cdot \frac{20}{\sqrt{500}}
```

If the noise-rank difference is an integer $m$ (positive, so group-b noise rank is higher):

```math
s_{n,a} - s_{n,b} = \frac{m}{\sigma_n} = \frac{m}{23.092}
```

The crossover condition becomes:

```math
\frac{\rho}{\sqrt{1-\rho^2}} = \frac{m/\sigma_n}{j \cdot \Delta}
  = \frac{m}{j \cdot \sigma_n \cdot \Delta}
  = \frac{m}{j \cdot 23.092 \cdot 0.8944}
  = \frac{m}{j \cdot 20.654}
```

---

## 4. The Resonance Condition

For equal-sized, equally-spaced groups, the factor of $j$ **cancels**:

- Adjacent pairs (j = 1) with noise diff $m$ fire at $\rho/\sqrt{1-\rho^2} = m/20.654$
- Skip-1 pairs (j = 2) with noise diff $2m$ fire at $\rho/\sqrt{1-\rho^2} = 2m/(2 \cdot 20.654) = m/20.654$
- Skip-2 pairs (j = 3) with noise diff $3m$ fire at $\rho/\sqrt{1-\rho^2} = 3m/(3 \cdot 20.654) = m/20.654$

All three group-separation types produce crossovers at **exactly the same rho** for every
integer $m$.  This is the resonance: crossovers from all group-pair types pile up
simultaneously.

---

## 5. The Resonance Formula

Solving $\rho/\sqrt{1-\rho^2} = m/20.654$ for $\rho$:

```math
\rho_m = \frac{m}{\sqrt{m^2 + 20.654^2}}
```

The first several resonance rho values are:

| m | Predicted $\rho_m$ | Notes |
|---|---|---|
| 1 | 0.0484 | First resonance |
| 2 | 0.0965 | Second resonance |
| **3** | **0.14374** | **Dominant calibration obstacle** |
| 4 | 0.1902 | |
| 5 | 0.2355 | |

These are separated by approximately $\Delta\rho \approx 1/20.654 \approx 0.0484$ for small $m$.

---

## 6. Step Size Estimation

At the m = 3 resonance ($\rho \approx 0.14374$), the following crossovers all fire simultaneously.

For a given pair of groups (each of size $n/k = 20$), the expected number of cross-group
observation pairs with a specific noise-rank difference $m$ is approximately
$(n/k)^2 / n = 20^2/80 = 5$ (since the noise is a random permutation of $\{1,\ldots,n\}$
and for each observation in group A, approximately $20/80$ of the observations in group B
have noise rank differing by exactly $m$).

| Group-pair type | Pairs of groups | Noise diff | Obs pairs per group-pair | Total obs pairs |
|---|---|---|---|---|
| Adjacent (j=1) | 3 pairs: (1–2), (2–3), (3–4) | m = 3 | ≈ 5 | ≈ 15 |
| Skip-1 (j=2) | 2 pairs: (1–3), (2–4) | m = 6 | ≈ 5 | ≈ 10 |
| Skip-2 (j=3) | 1 pair: (1–4) | m = 9 | ≈ 5 | ≈ 5 |

Spearman $\hat\rho_S$ is the Pearson correlation of the ranks:

```math
\hat\rho_S = \frac{\sum_i (r_{x,i} - \bar{r}_x)(r_{y,i} - \bar{r}_y)}
  {\sqrt{SS_x \cdot SS_y}}
```

where $SS_x = \sum(r_{x,i} - \bar{r}_x)^2$ and $SS_y = \sum(r_{y,i} - \bar{r}_y)^2$.
For k = 4 even (n = 80):

- $SS_x = 20((-30)^2 + (-10)^2 + 10^2 + 30^2) = 20 \times 2000 = 40{,}000$
- $SS_y = n(n^2-1)/12 = 80 \times 6399 / 12 = 42{,}660$ (for untied y-ranks)

When two mixed values cross at an isolated crossover, their y-ranks swap by exactly
one position: $|r_{y,b} - r_{y,a}| = 1$.  At a resonance, multiple pairs may cross
simultaneously — but the Spearman numerator is **linear** in the y-ranks, so the total
change is the exact sum of per-pair contributions, each with $|\Delta r_y| = 1$.
(Multi-way ties at a resonance redistribute ranks by more than 1 per observation, but
the cross-terms cancel by linearity.)  The x-rank gap between adjacent groups (j = 1)
is $n/k = 20$.

Per-swap Spearman change for adjacent groups:

```math
\frac{20}{\sqrt{SS_x \cdot SS_y}} = \frac{20}{\sqrt{40{,}000 \times 42{,}660}} \approx 0.000484
```

Each swap at separation j has j times the x-rank gap, so j times the per-swap
impact.  With a weighted total of:

```math
15 \times 1 + 10 \times 2 + 5 \times 3 = 50 \text{ effective swap-units}
```

the predicted step size is:

```math
\Delta F \approx 50 \times 0.000484 \approx 0.024
```

**Observed step** (experimental): $0.13949 - 0.11681 = 0.02268$.  The 5% discrepancy
from the predicted 0.024 comes from the approximate observation-pair count ($\approx 5$
per group-pair per noise-diff); the per-swap formula itself is exact by the linearity
argument above.

---

## 7. Practical Consequences

1. **Inherent resolution limit.** For k = 4 even, n ≈ 80, the resonance steps of ≈ 0.024
   mean that the calibration error is bounded by ≈ ±0.012, regardless of `n_cal`.  This
   explains the observed plateau at max_dev ≈ 0.010–0.013.

2. **Unreachable target values.** F(rho_in) jumps from 0.117 to 0.140 at the m = 3
   resonance.  No rho_in produces F = 0.128 — that Spearman target is simply
   *inaccessible* with discrete noise.

3. **The hi/lo endpoint trick.** Bisection picks whichever endpoint (hi or lo) is closer
   to the target.  This halves the worst-case error (from one full step to half a step),
   reducing the bound from ≈ 0.024 to ≈ 0.012.  It cannot do better, because no
   intermediate rho_in exists between the two flat segments.

---

## 8. Scaling Law: Step Size is O(1/n)

Combining the quantities from §6 into a closed-form for $k$ equal groups of size
$g = n/k$:

- Per-swap impact (j = 1): $g / \sqrt{SS_x \cdot SS_y}$
- Obs pairs per group-pair per noise-diff: $g^2 / n$
- Weighted sum over all group-pair separations:
  $\sum_{j=1}^{k-1}(k - j) \cdot j = k(k^2-1)/6$

The resonance step size is:

```math
\Delta F^{\text{resonance}}
  = \frac{g^2}{n} \cdot \frac{k(k^2-1)}{6} \cdot \frac{g}{\sqrt{SS_x \cdot SS_y}}
  = \frac{2\sqrt{k^2 - 1}}{k \, n}
```

(The derivation uses $SS_x = n g^2 (k^2-1)/12$ and $SS_y = n(n^2-1)/12$.)

For fixed $k$ this is $O(1/n)$.  For $k \geq 3$ the factor $2\sqrt{k^2-1}/k$ is
approximately 2 and varies only slowly with $k$:

| k | $2\sqrt{k^2-1}/k$ | Predicted step (n = 80) |
|---|---|---|
| 2 | 1.732 | 0.022 |
| 3 | 1.886 | 0.024 |
| 4 | 1.936 | 0.024 |
| 9 | 1.988 | 0.025 |
| ∞ | → 2 | → 2/n |

So the resonance step size is approximately $2/n$ for all $k \geq 3$.  Numerical
estimates at fixed k = 4:

| n | Predicted step |
|---|---|
| 80 | 0.024 |
| 800 | 0.0024 |
| 8000 | 0.00024 |

For n ≈ 80 (the Karwowski study), the staircase is a real obstacle; it becomes
sub-0.001 naturally only for $n \gtrsim 2000$.

---

## 9. Non-Equal Group Distributions

The resonance is specific to **equally-spaced, equal-sized groups**.  It requires that
crossovers for separations j = 1, 2, 3 align at common rho values, which happens because
the gaps are integer multiples of a common base gap.

For non-even distributions (unequal group sizes), the alignment fails.  Crossover points
for different group-pair types land at different rho values — the resonance is broken.
Individual group-pair steps survive, with magnitude set by the two largest adjacent groups.

**Example (k = 4, n = 80):**

| Distribution | Group sizes | Largest adjacent product | Approx max single-pair step |
|---|---|---|---|
| even | 20, 20, 20, 20 | 400 (+ resonance) | ≈ 0.024 |
| heavy_tail | 23, 18, 18, 21 | 414 | ≈ 0.003 |
| heavy_center | 12, 30, 29, 9 | 870 | ≈ 0.008 |

For heavy_tail and heavy_center, the resonance is broken and individual steps are 3–8×
smaller, but still above ±0.001 for n ≈ 80.

**More severe ties (larger groups, smaller k):**

From the formula in §8 ($\Delta F \approx 2\sqrt{k^2-1}/(kn)$), the resonance step for
equal groups is approximately $2/n$ for all $k \geq 3$.  More severe ties do not
dramatically worsen the situation:

| k | Theoretical resonance step (n = 80, equal groups) |
|---|---|
| 2 | 0.022 |
| 3 | 0.024 |
| 4 | 0.024 |
| 9 | 0.025 |

**Caveat for non-integer $n/k$:** When $n/k$ is not an integer, groups cannot be
exactly equal, breaking the resonance.  For example, k = 9 with n = 80 uses groups
`[9,9,9,9,8,9,9,9,9]`; the one smaller group shifts some crossover rho values away from
the resonance, reducing the effective step.  The actual max step for such configurations
was not experimentally measured but is expected to be smaller than the equal-group
prediction.

For very unbalanced groups (e.g., {3, 50, 20, 7}), the dominant step is set by the largest
adjacent-group product (50 × 20 = 1000 pairs out of n = 80), giving an estimated step of
≈ 0.012.  The jitter fix (§11) eliminates all these cases uniformly.

---

## 10. Experimental Confirmation

### 10.1 Confirming the Staircase Is Structural

A fine-grid diagnostic was run: `F(rho_in)` evaluated at 200 points in [0.1430, 0.1445]
with spacing 0.0000075, at n_cal = 500,000, with two independent random seeds (99 and
12345).

Results:

| Segment | Grid points | F (seed 99) | F (seed 12345) |
|---|---|---|---|
| Below step | 1–32 (rho ≤ 0.143736) | 0.11681 (constant) | 0.11724 (constant) |
| Above step | 33–... (rho ≥ 0.143740) | 0.13949 (constant) | 0.13990 (constant) |

Three definitive properties:

1. **Single sharp step, not a ramp.** Both seeds show F constant to five decimal places
   across all grid points in each segment.  A smooth underlying function would show at
   least some slope.
2. **Both seeds jump at exactly the same rho_in.** The step occurs between the same
   adjacent pair of grid points for both seeds.  If the staircase were a finite-n_cal
   artifact, the two seeds would place the step at slightly different locations.
3. **Inter-seed difference is uniform MC noise** (≈ 0.00043) at all grid points, with
   no spike at the transition.  This equals the expected MC noise for n_cal = 500k.

### 10.2 Confirming the Predicted Step Location

With grid spacing 0.0000075 starting at 0.1436 (`np.arange(0.1436, 0.1445, 0.0000075)`):
- Last "low" point: rho_in = 0.1436 + 18 × 0.0000075 = **0.143735**
- First "high" point: rho_in = 0.1436 + 19 × 0.0000075 ≈ **0.143743**

The step falls between 0.143735 and 0.143743, i.e., at rho_in ≈ **0.14374**.

(The later jitter smoothing diagnostic confirms the same step location, with the last "low"
F_plain value at rho_in ≈ 0.143736 and the first "high" value at rho_in ≈ 0.143740.)

Theoretical prediction (m = 3 resonance):

```math
\rho_3 = \frac{3}{\sqrt{9 + 20.654^2}} = \frac{3}{\sqrt{435.6}} = \mathbf{0.14374}
```

This is an **exact match** to the precision of the grid spacing (0.0000075).

### 10.3 Confirming the Observed Step Size

Observed step: $0.13949 - 0.11681 = 0.02268$.
Theoretical prediction: $\approx 0.024$.
Discrepancy: 5%, consistent with the back-of-envelope approximation.

---

## 11. The Jitter Fix

### 11.1 The Fix

Replace the integer-permutation noise with jittered noise:

```python
# Before (causes staircase)
noise_ranks = rng.permutation(n) + 1.0

# After (smooths staircase)
noise_ranks = rng.permutation(n) + 1.0 + rng.uniform(-0.49, 0.49, size=n)
```

The jitter must be applied consistently in both calibration and data generation.

### 11.2 Why the Jitter Does Not Attenuate the Correlation

This was a key concern: the earlier copula jitter *did* attenuate the correlation
significantly.  The two cases are structurally different.

**The copula jitter (attenuated):** Applied to `z_x` (the signal).  In
$z_y = \rho \cdot z_x + \sqrt{1-\rho^2} \cdot z_\varepsilon$, adding noise to $z_x$ is
equivalent to *measurement error in the predictor*, which biases the apparent correlation
toward zero (regression dilution / attenuation bias).

**The noise jitter (does not attenuate):** Applied to `noise_ranks`, which becomes `s_n`
(the error term).  The signal `s_x` is completely untouched.  In classical regression
terms: adding measurement error to the *error term* $\varepsilon$ increases residual
variance but does not bias the regression coefficient $\rho$.

Three ways to see it:

1. **Signal is untouched.** Regression dilution requires adding noise to the predictor
   (x).  The noise jitter only perturbs the error term (s_n).
2. **Rank order preserved.** With jitter Uniform(−0.49, 0.49) added to integer noise
   values, adjacent integers (e.g., 37 and 38) cannot cross: their jittered values lie in
   [36.51, 37.49] and [37.51, 38.49] which do not overlap.  Therefore
   `rank(noise_jittered) = rank(noise_original)` always.  The noise provides exactly the
   same random ordering of observations.
3. **Only crossover timing is affected.** The jitter shifts the exact rho_in at which
   pairs of mixed values cross (the staircase steps) by a continuous amount, smearing
   each discrete crossover into a narrow band.  On average, the expected Spearman at any
   rho_in equals the midpoint between the adjacent flat segments — pure interpolation,
   zero bias.

**Quantitative argument.** The jitter adds variance $\approx 0.49^2/3 = 0.08$ per noise
value.  The original noise variance is $(n^2-1)/12 = 533.25$.  The relative perturbation
is $0.08/533.25 = 0.015\%$.  Even if the jitter interacted with the signal, the effect
would be $\sim 100\times$ smaller than the copula attenuation of 0.01–0.06.

### 11.3 Why Jitter Is Preferred over Full IID Replacement

An alternative fix would replace `rng.permutation(n) + 1.0` with `rng.uniform(0.5, n+0.5,
size=n)`.  This eliminates the staircase completely, but introduces a subtle issue:

- With a permutation, sample mean $= (n+1)/2$ **exactly** and sample std $= \sqrt{(n^2-1)/12}$
  **exactly**, every draw.  The standardization code uses these population constants
  (not per-sample statistics), which is exact.
- With IID Uniform noise, sample mean and std **fluctuate** across draws.  The standard
  deviation of n = 80 IID uniform draws fluctuates by roughly $1/\sqrt{2n} \approx 8\%$
  relative.  This changes the effective noise weight in the mixing formula by ∼8% across
  simulations, increasing Var[Spearman] (though not biasing E[Spearman]).

The jitter approach preserves the permutation structure:
- Sample mean remains within ≈ 0.01 of $(n+1)/2$ (jitter is zero-mean).
- Sample std is within ≈ 0.015% of the population constant.
- Population-constant standardization remains valid.
- The staircase is smoothed because each crossover gains a continuous perturbation
  from `jitter[i] − jitter[j]`.

---

## 12. Post-Jitter Calibration Accuracy

After implementing the jitter fix in calibration (n_cal = 500,000), per-target bisection
accuracy for k = 4 even:

| Target rho | Deviation (jittered) | Deviation (plain) |
|---|---|---|
| 0.05 – 0.48 | +0.00016 to +0.00020 | up to ±0.013 |
| Max absolute deviation | **0.00020** | **~0.013** |

The jitter fix achieves approximately a **50× improvement** in calibration accuracy.
The residual bias of ≈ +0.0002 is pure Monte Carlo noise (difference between the
calibration seed and the evaluation seed; decreases as $1/\sqrt{n_{\text{cal}}}$).

**Jitter smoothing diagnostic** (same fine grid, n_cal = 500,000):

| Metric | Plain | Jittered |
|---|---|---|
| F at rho ≈ 0.143600 | 0.11681 (flat) | 0.12801 |
| F at rho ≈ 0.143740 | 0.13949 (step) | 0.12815 |
| F at rho ≈ 0.143781 | 0.13949 (flat) | 0.12819 |
| Behavior across range | Flat → jump → flat | Smooth monotone ramp |
| Gradient per grid step | 0 (then infinite) | ≈ 0.00001 |

The jittered midpoint value at the step location (≈ 0.12815) matches the theoretical
midpoint: $(0.11681 + 0.13949)/2 = 0.12815$ exactly, confirming that the jitter
interpolates over the discrete crossover as predicted.

---

## 13. Multipoint Interpolation Accuracy

The production calibration pipeline does not bisect per-target.  Instead, it bisects at a
fixed set of **probe** Spearman values and uses linear interpolation (with linear
extrapolation outside the range) to map arbitrary targets to calibrated rho_in values.

### 13.1 The Interpolation Error Budget

Per-target bisection on the jittered curve achieves max deviation ≈ 0.0002 (§12).  Any
additional deviation in the multipoint pipeline comes from **interpolation error**: the
difference between the true (smooth, nonlinear) F(rho_in) and the piecewise-linear
approximation through the probe points.

This error is proportional to the second derivative of F(rho_in) times the square of
the probe spacing.  For bivariate normal data, the Pearson-to-Spearman relationship
$E[\hat\rho_S] \approx (6/\pi)\arcsin(\rho/2)$ is concave; the rank-mixing model here
(with tied x) differs in detail but shares the concavity.  Thus the interpolation
error grows with target rho, and extrapolation beyond the outermost probe diverges
rapidly.

### 13.2 Expansion from 3 to 5 Probes

The original probe set was `[0.10, 0.30, 0.50]` (3 probes).  With wide 0.20 spacing,
interpolation error between probes exceeded ±0.001 for some targets (particularly near the
midpoints of segments where curvature peaks).

Adding probes at 0.20 and 0.40 reduced the spacing to 0.10, yielding the current set:

```python
_MULTIPOINT_PROBES = [0.10, 0.20, 0.30, 0.40, 0.50]
```

### 13.3 Experimental Results (k = 4 even, jittered, n_cal = 100,000)

Probe map (bisection accuracy at each probe is effectively exact):

| Probe | Calibrated rho_in | F_cal |
|---|---|---|
| 0.10 | 0.113737 | 0.10000 |
| 0.20 | 0.218893 | 0.20000 |
| 0.30 | 0.324556 | 0.30000 |
| 0.40 | 0.427959 | 0.40000 |
| 0.50 | 0.526749 | 0.50000 |

Deviation by target region (evaluated on independent seed):

| Region | Target range | Max |deviation| | Behavior |
|---|---|---|---|
| Extrapolation below | 0.05 | 0.00110 | Linear extrapolation undershoots |
| Interpolation (first segment) | 0.06 – 0.10 | 0.00082 | Rapidly improving toward probe |
| Interpolation (inner segments) | 0.10 – 0.30 | 0.00021 | Tight; near-linear regime |
| Interpolation (upper segments) | 0.30 – 0.50 | 0.00062 | Growing curvature increases error |
| Extrapolation above | 0.51 – 0.60 | 0.00656 | Diverges rapidly |

Within the interpolation range [0.10, 0.50], max absolute deviation is **0.00062** — well
within the ±0.001 target for typical simulation rho values.  The mean absolute deviation
across all targets (0.05–0.60) is 0.00074.

### 13.4 The Extrapolation Problem

Linear extrapolation outside the probe range diverges because F(rho_in) is concave: the
true curve bends away from the linear projection.  At target 0.60 (0.10 beyond the last
probe), the deviation reaches +0.00656.

Mitigation options:

1. **Extend the probe range** by adding probes at lower (e.g., 0.05) and higher
   (e.g., 0.60, 0.70) target rho values to cover the full simulation range.
2. **Add more probes in the high-curvature region** (0.40–0.60) to reduce per-segment
   interpolation error.
3. **Use a nonlinear interpolant** (e.g., the $\arcsin$ relationship) instead of piecewise
   linear, which would capture the curvature analytically.

---

## 14. Summary of Theoretical Predictions vs. Experimental Observations

| Prediction | Predicted value | Observed value | Match? |
|---|---|---|---|
| Step location (m=3 resonance) | $\rho_3 = 0.14374$ | 0.14374 (between 0.143733 and 0.143740) | ✓ Exact |
| Step height | ≈ 0.024 | 0.02268 | ✓ Within 5% |
| Step is structural (not finite-n_cal noise) | Both seeds jump identically | Confirmed (inter-seed diff = 0.00043 everywhere, no spike) | ✓ |
| Jitter midpoint interpolation | $(0.117+0.140)/2 = 0.12815$ | 0.12815 | ✓ Exact |
| Post-jitter per-target bisection max dev | ≪ 0.001 | 0.00020 | ✓ |
| Post-jitter multipoint max dev (within probes) | < 0.001 | 0.00062 | ✓ |

---

## 15. Batch vs Sequential RNG: Statistical Equivalence

### 15.1 Context

The calibration bisection estimates $E[\rho_S \mid \rho_{\text{in}}]$ by the Monte Carlo average

$$\hat{\rho}(\rho_{\text{in}}) = \frac{1}{n_{\text{cal}}} \sum_{i=1}^{n_{\text{cal}}} \rho_S(x_i,\, y_i)$$

where each $(x_i, y_i)$ is drawn from the data-generating process (random shuffle of the x-template, independent noise permutation). Two implementations exist:

**Sequential path** (legacy single-point calibration, `calibrate_rho` with `calibration_mode="single"`): datasets are generated in a Python loop; each iteration calls `cal_rng.shuffle(x)` and `cal_rng.permutation(n)` sequentially. The RNG is reset to `seed` at the start of every call to `_mean_rho`, so all bisection steps see the same $n_\text{cal}$ datasets (common-random-numbers variance reduction).

**Batch path** (multipoint fast path, `_precompute_calibration_arrays_fast`): all $n_\text{cal}$ shuffles are drawn at once via `cal_rng.permuted(x_batch, axis=1)` and `cal_rng.permuted(noise_base, axis=1)`. The arrays are precomputed once before bisection and reused at every bisection step (same common-random-numbers property, implemented differently). The copula fast path (when implemented) follows the same batch pattern: `z_x` and noise are precomputed once, and per bisection step only `z_y = rho_p * z_x + sqrt(1 - rho_p^2) * noise` is recomputed.

### 15.2 Both paths draw from the correct distribution

For the batch path, `cal_rng.permuted(M, axis=1)` applies an independent, uniformly random permutation to each row of matrix $M$. This is equivalent to calling `cal_rng.shuffle` independently for each row. Therefore:

- Each row $i$ of `x_batch` is a uniformly random permutation of the x-template. ✓
- Each row $i$ of `noise_batch` is a uniformly random permutation of $\{1, \ldots, n\}$. ✓
- Rows are mutually independent (each permutation is drawn from fresh RNG state). ✓

The marginal distribution of each $(x_i, y_i)$ is identical between paths. The joint distribution across $i$ is also the same: independent across $i$ in both cases.

### 15.3 Estimator properties are identical

Since both paths produce $n_\text{cal}$ IID draws from the same marginal:

**Bias:**
$$E\!\left[\hat{\rho}(\rho_{\text{in}})\right] = E\!\left[\rho_S(x_i, y_i)\right] = E[\rho_S]$$
(unbiased, regardless of draw order or batch vs sequential generation).

**Variance:**
$$\mathrm{Var}\!\left(\hat{\rho}(\rho_{\text{in}})\right) = \frac{\mathrm{Var}(\rho_S)}{n_{\text{cal}}}$$
(by independence of draws across $i$; same for both paths).

**Convergence:** By the strong law of large numbers, $\hat{\rho}(\rho_{\text{in}}) \to E[\rho_S]$ almost surely as $n_\text{cal} \to \infty$, for both paths.

### 15.4 What differs: numerical values, not statistical properties

The two paths consume pseudo-random numbers in a different order. From the same `seed`, the batch path assigns different pseudo-random numbers to each sample than the sequential path does. This means:

- The calibrated ratios $\hat{\rho} / \text{probe}$ will differ numerically between paths for the same `seed`.
- The difference is not a bias — it is sampling variation. For large $n_\text{cal}$, both ratios are close to the true $E[\rho_S] / \text{probe}$, with the same standard error $\mathrm{SD}(\rho_S) / \sqrt{n_\text{cal}}$.

### 15.5 Common-random-numbers property is preserved in both paths

Within a single calibration run, the bisection loop evaluates $\hat{\rho}$ at multiple trial values of $\rho_{\text{in}}$ (the bisection steps). For the comparison $\hat{\rho}(\rho_1)$ vs $\hat{\rho}(\rho_2)$ to be low-variance (so bisection converges accurately), the same $n_\text{cal}$ datasets should be used at every bisection step.

- **Sequential path:** `_mean_rho` resets to `seed` at the start of each call, so every bisection step evaluates on the same $n_\text{cal}$ shuffles. ✓
- **Batch path:** Arrays are precomputed once before bisection begins, and `_eval_mean_rho_fast` re-uses the same `(s_x, s_n, x_ranks_batch)` at every bisection step. ✓

Both paths achieve common-random-numbers variance reduction for bisection comparisons.

### 15.6 Consequence for testing

Tests should check calibration **accuracy** (is the realised mean Spearman close to the target?) rather than exact numerical equality of calibration ratios. A switch from sequential to batch RNG will change the ratio numerically while leaving the accuracy unchanged. The existing `tests/test_calibration_accuracy.py` already tests accuracy (checking that realised mean Spearman is within tolerance of the target), not exact ratio values.
