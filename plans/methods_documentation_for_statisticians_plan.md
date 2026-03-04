---
name: Methods documentation for statisticians
overview: Add a clear, verification-oriented description of all implemented statistical methods to the README (and optionally code comments) so statisticians and simulation experts can check validity. This includes (1) exact test/power/CI formulas and the unverified sqrt(1/1.06) and t-based p-value caveat, (2) method-by-method descriptions of what the code does for linear, copula, empirical, nonparametric, and asymptotic—filling gaps where currently missing (e.g. linear, copula), and (3) empirical-generator decisions for p-value and calibration (new y per sim, single reference pool, per-dataset permutation for empirical) and the alternative "generate new y datasets each time" as a future generalization.
todos: []
isProject: false
---

# Methods documentation for statistician verification

## Goal

Report **exactly** what formulas and choices the code implements so statisticians can verify correctness. This is documentation-only; no change to whether methods are "valid" — just make them explicit. Key gaps from the past chat:

- **Asymptotic power:** An extra factor `sqrt(1/1.06)` is applied to the noncentrality parameter in [power_asymptotic.py](power_asymptotic.py) (line 135). Bonett–Wright (2000) only justify 1.06 for **Fisher z SE (CIs)**; no literature was found supporting this factor in the **power** (noncentral t) formula. This should be stated clearly so reviewers can confirm or reject it.
- **Monte Carlo power p-value:** The power simulation uses a **t-approximation** p-value ([spearman_helpers.spearman_rho_pvalue_2d](spearman_helpers.py)), not permutation. With heavy ties the t-approximation is not reliable (can be anticonservative). The README does not currently state this explicitly or give the exact formula.

---

## 1. Add a dedicated "Implemented methods (for verification)" section

**Location:** [README.md](README.md), after **Analysis Goals** (or after **Four Monte Carlo Methods**) and before **Method Choices and Rationale**. Alternatively, a new top-level section **"Statistical formulas (for statistician verification)"** that collects all formulas in one place.

**Purpose:** One place where every test/power/CI formula and caveat is written so a reviewer can check against the code and literature.

**Suggested structure:**

- **Hypothesis and test**
  - H0: ρ = 0 vs H1: ρ ≠ 0, two-sided, α = 0.05.
  - Test statistic used in both asymptotic power and in the Monte Carlo power loop (see below).
- **Test statistic and p-value (current implementation)**
  - **Formula:** t = \rho \sqrt{(n-2)/(1-\rho^2)} with df = n − 2; two-sided p-value = 2 × P(T ≥ |t|) for T ~ t_{n-2}.
  - **Where used:**
    - **Monte Carlo power:** Each simulated dataset (x, y) gets one Spearman ρ and one p-value from this t-approximation ([spearman_helpers.spearman_rho_pvalue_2d](spearman_helpers.py), lines 215–216). Power = proportion of sims with p < α.
    - **Asymptotic power:** Same t-statistic; power is computed via the non-central t distribution (below).
  - **Caveat:** This p-value is the **t-approximation** for testing ρ = 0. With **heavy ties** in x (e.g. few distinct values), the null distribution of ρ is not well approximated by the t-formula; the approximation can be anticonservative (p-values too small, type I error inflated). For tied scenarios, a permutation-based p-value would be appropriate; the code currently uses the t-approximation everywhere (permutation is planned elsewhere).
- **Asymptotic power (non-central t)**
  - **Formula (all-distinct):** Noncentrality nc = \rho_{\mathrm{true}} \sqrt{n-2} / \sqrt{1 - \rho_{\mathrm{true}}^2}. Power = P(|T| > t_crit) for T ~ noncentral t(df = n−2, nc).
  - **Tie adjustment (FHP):** When ties are present, nc is scaled by \sqrt{\mathrm{var}_0^{\mathrm{noties}} / \mathrm{var}_0^{\mathrm{ties}}} (see FHP below).
  - **Extra factor in code:** The implementation additionally multiplies nc by \sqrt{1/1.06} ([power_asymptotic.py](power_asymptotic.py) line 135). **Verification status:** The 1.06 factor is from Bonett & Wright (2000) for the **Fisher z standard error (confidence intervals)**. We are not aware of a source that justifies applying this factor to the **noncentrality** in the power formula. It is included in the code for consistency with the CI efficiency loss; statisticians may wish to verify or remove it.
- **Asymptotic confidence interval**
  - **Formula:** z = arctanh(ρ); SE in z-space = \sqrt{1.06/(n-3)} (Bonett–Wright). With ties: SE_z scaled by \sqrt{\mathrm{var}_0^{\mathrm{ties}} / \mathrm{var}_0^{\mathrm{noties}}}. CI in z-space then back-transformed via tanh.
- **Fieller–Hartley–Pearson (FHP) variance under H0**
  - **Formula:** Tx = Σ(t_i³ − t_i) over x tie groups (similarly Ty for y); Dx = (n³−n−Tx)/12, Dy = (n³−n−Ty)/12; Var(ρ) = (1/(n−1)) × (n³−n)²/(144·Dx·Dy). No ties ⇒ 1/(n−1).
  - **Tie correction for x only in asymptotic formulas:** In this codebase, the asymptotic power and CI use FHP **for x only**. Only `x_counts` is passed to the variance formula; `y_counts` is not used (y is effectively treated as having no ties for the asymptotic expressions). So the implemented var_0 uses Dx from x tie groups and Dy = (n³−n)/12 (no y ties). This matches the design: x is cumulative vaccine aluminum (heavy ties); y is aluminum level (typically few or no ties in our generators).
- **Ranks:** Average-rank tie handling (midranks) throughout; same as `scipy.stats.rankdata(..., method="average")`.

---

## 2. Cross-link and trim existing README

- **"Why non-central t for power?"** and **"Why Bonett–Wright SE?"**: Keep them; add a sentence in each pointing to the new "Implemented methods" section for the exact formulas and the note on 1.06 in power.
- **"Tie correction (Fieller–Hartley–Pearson)"**: Keep; ensure the FHP formula here matches the new section (or the new section points to this). **Add** that the asymptotic power and CI apply FHP **for x only** (not y): only the x tie structure is used; y is treated as no ties in the asymptotic formulas. This should be stated in the README so readers know we do not use y_counts there.
- **Asymptotic subsection** under "Four Monte Carlo Methods (plus Asymptotic)": State explicitly that asymptotic power uses the non-central t with FHP and the 1.06 factor as above, and refer to the verification section for details.
- **Bootstrap: separate RNG streams:** The README already has a subsection **"Separate RNG streams for data and bootstrap"** under Bootstrap CI that explains why data generation and bootstrap resampling use separate RNG streams (`np.random.SeedSequence.spawn(2)`): a shared RNG would advance by `n_boot * n` draws per rep, so the n_reps datasets would depend on `n_boot`, making results non-comparable across `n_boot` and invalidating "more bootstraps = more accurate." **Retain** this subsection and its reasoning. If a new "Statistical formulas" or verification section is added, it may briefly reference it (e.g. "Bootstrap uses separate streams so the same seed yields the same n_reps datasets regardless of n_boot; see Bootstrap CI.").

---

## 3. Optional: code comments

- **[power_asymptotic.py](power_asymptotic.py) line 135:** Add a one-line comment: e.g. "Scale nc by sqrt(1/1.06); 1.06 is from Bonett–Wright for Fisher z (CI). Not verified in literature for power; see README."
- **[spearman_helpers.py](spearman_helpers.py) spearman_rho_pvalue_2d:** In the docstring, add one sentence: "Uses the t-approximation for testing ρ=0; with heavy ties a permutation p-value is more appropriate (see README)."

---

## 4. Method-by-method: what the code does (data generation)

Ensure the README explains **what each method does** (steps and formulas) so statisticians can verify or reproduce. Current state and gaps:

- **Nonparametric:** README is already detailed (steps 1–4, Cholesky mixing, calibration, attenuation formula). No change beyond any cross-links from §1–2.
- **Gaussian copula:** README has one sentence plus a limitation paragraph. **Add:** (1) **Conversion:** ρ_p = 2 sin(π ρ_s/6) (exact Spearman–Pearson for the bivariate normal). (2) **Steps:** x → ranks (average) → u_x = (ranks−0.5)/n, small jitter to break ties → z_x = Φ^{-1}(u_x); draw z_y = ρ_p z_x + √(1−ρ_p²) Z; u_y = Φ(z_y); y = F_ln^{-1}(u_y) with log-normal fitted to median/IQR. (3) With ties, jittering collapses rank information so realised Spearman is attenuated; single-point calibration at ρ_s = 0.30 (cached per scenario) compensates. Reference [data_generator.generate_y_copula](data_generator.py) and `_spearman_to_pearson`.
- **Linear Monte Carlo:** README has one sentence. **Add:** (1) **Model:** log(y) = μ_ln + b·x_std + σ_noise·Z, with x_std = (x − mean(x))/std(x), and (μ_ln, σ_ln) from fitting log-normal to median/IQR. (2) **Target:** ρ_p = 2 sin(π ρ_s/6); set b = ρ_p·σ_ln and noise variance σ_ln²(1−ρ_p²) so that theoretical Pearson(x_std, log y) = ρ_p. (3) **Caveat:** Spearman(x, y) is only approximate to ρ_s because Spearman uses ranks of y, which are a nonlinear transform of log(y); no calibration step. Useful as a parametric complement; see [data_generator.generate_y_linear](data_generator.py).
- **Empirical:** README describes pools and calibration. **Add:** Same rank-mixing core as nonparametric (standardised midranks of x, mix with noise ranks, reorder); y-values come from **resampling** the digitized Karwowski pool (no log-normal). Calibration uses the same multipoint/single-point logic as nonparametric, with a **separate cache** so empirical and nonparametric do not share calibration state.
- **Asymptotic:** No data generation; formulas are in §1. The short "Asymptotic" subsection under "Four Monte Carlo Methods" can point to the new "Statistical formulas" section for full detail.

**Placement:** Expand the existing subsections under "Four Monte Carlo Methods (plus Asymptotic)" (copula, linear, empirical) with the above; keep asymptotic short and cross-linked. No new top-level section is required—fill the existing method blocks.

---

## 5. Empirical generator: p-value and calibration decisions (for README)

Document in the README the **decisions made for the empirical distribution** when generating y and when calculating p-values by Monte Carlo (e.g. permutation), so simulation experts can understand what we did, why it is legitimate, and what the alternative would mean as a future generalization. Source: decisions from implementation and discussion (e.g. empirical y-distribution and calibration; see agent transcript 90b716b3).

### 5.1 What we do (current design)

- **Pool and y per sim/rep:** The empirical generator uses a **pool** of y-values (e.g. 71 digitized + 2 or 10 “fill” for n=73 or 81; or 71 fixed + remainder resampled per sim if the “fixed 71” fix is applied). Each **n_sim** or **n_rep** produces a **new** y dataset: either by resampling **n** values from the pool with replacement (current), or by using exactly the 71 digitized values plus a new random draw for the remainder (fixed-71 design). So we do **not** reuse the same y across sims; we intentionally generate new y-data each sim/rep.
- **Why new y each time is correct:** We are averaging over many (x, y) realizations. The pool (or the 71) is fixed; the **draws** from it (or the remainder) are random. So power and CI estimates average over the randomness in y, giving unbiased expectations and valid assessment of variability. This is the same principle as other generators (nonparametric, copula, linear), where y is newly generated every sim; here the “population” is the finite pool (or 71 + random remainder) instead of a parametric distribution.
- **Calibration:** Calibration for empirical runs **once per scenario** (cached by n, n_distinct, distribution_type, etc.). It uses a **single reference pool** from `get_pool(n)` (fixed seed), not the per-sim pool. So the same `cal_rho` is used for all n_sims/n_reps in that scenario. Any small mismatch between that reference pool’s tie structure and a given sim’s pool (e.g. different 2 or 10 fill values) is **averaged over** by the randomness of the different y-data across sims: over many n_sims we average over (a) different x, (b) different y-pools/realizations, and (c) rank-mix assignment. The Monte Carlo mean remains close to the target; per-rep variation contributes to the variance we are estimating. So one reference pool and one `cal_rho` per scenario is a reasonable and consistent setup.
- **P-value by Monte Carlo (permutation):** For the **empirical** generator, y can have **ties** (finite pool → repeated values). The null distribution of Spearman ρ under permutation depends on **both** the x and y tie structures. Because the y tie pattern **varies per dataset** (each sim has a different y from the pool), there is no single precomputed null per scenario. So for empirical we **always use per-dataset Monte Carlo permutation** (no precomputed null cache). This is the only option for empirical; for other generators we can use a precomputed null keyed by (n, x tie structure) when y has no ties.

### 5.2 Why this is legitimate

- **New y per sim:** Intentional and correct for operating characteristics (power, CI). We are not reusing the same dataset; we are drawing new (x, y) each time and averaging. No analogue to the earlier n_boot bug (where data depended on n_boot); here data and bootstrap/RNG streams are separated as intended.
- **Single reference pool for calibration:** Cost and simplicity (one calibration per scenario). The approximation (one `cal_rho` for all sims) is valid because any bias from pool-to-pool variation is averaged out over n_sims. No need to run calibration per n_sim unless we want a more expensive, per-rep calibration (see alternative below).

### 5.3 Alternative: “generate new y datasets each time” as population sampling (future generalization)

A different design would be to treat the **pool as the population** and, each sim, **sample n values with replacement** from that pool (current “resampling from whole pool” is already in this spirit). That answers: **“If the pool were the true population and we ran many experiments, each drawing n from it, what would we see?”** — i.e. **what might happen if multiple such experiments were run**, with the 71 not special but part of the population (so in a given simulated sample, the 71 can appear 0, 1, 2, … times).

- **Contrast with current (fixed-71) target:** Our target is **“For this Karwowski study (these 71), what would power/CI look like under uncertainty in the missing/extra values?”** So the 71 are fixed in every sim; only the remainder varies. That is a **conditional** question given this study’s digitized values.
- **Generalization:** The “pool as population, new sample of n each time” approach generalizes naturally to **other scenarios**: different studies, different n, or different pools. Scripts could be extended to accept an arbitrary pool (or parametric model) and simulate “many replications from a representative population.” Document this in the README as a **potential future generalization** so readers understand the design choice (conditional on this study vs. unconditional over replications) and how one might extend the code.

**README placement:** Add a subsection under **Empirical** (or under a new “Empirical generator and Monte Carlo p-value” subsection) that states: (1) what we do (new y per sim, single reference pool for calibration, per-dataset permutation p-value for empirical); (2) why it is legitimate (averaging over (x,y), calibration mismatch averaged out); (3) the alternative (pool as population, new sample of n each time) and that it answers “many experiments” and is a natural future generalization for other scenarios.

---

## 6. What this plan does not do

- Does **not** implement permutation p-values or change the power/CI logic.
- Does **not** remove or add the sqrt(1/1.06) factor; it only **documents** it and its verification status.
- Does **not** duplicate the full permutation p-value implementation plan; that remains in [plans/permutation_pvalue_implementation_plan.md](plans/permutation_pvalue_implementation_plan.md).

---

## 7. Deliverables

1. **README — Statistical formulas:** New section "Statistical formulas (for statistician verification)" (or equivalent) with the content in §1 (including FHP and **tie correction for x only** in the asymptotic formulas), and short cross-links in §2.
2. **README — Method descriptions:** Ensure each of the five methods (nonparametric, copula, linear, empirical, asymptotic) has a clear "what we do" description. Expand **copula** and **linear** (and **empirical** if needed) with the steps and formulas in §4; nonparametric and asymptotic only need cross-links or minor tweaks.
3. **README — Empirical generator and p-value/calibration:** Add a subsection (under Empirical or a dedicated "Empirical generator and Monte Carlo p-value" subsection) documenting: (a) what we do (new y per sim, single reference pool for calibration, per-dataset permutation p-value for empirical when permutation is implemented); (b) why it is legitimate (averaging over (x,y), calibration mismatch averaged over n_sims); (c) the alternative (treating pool as population and generating new y datasets each time by sampling n from the pool) as answering "what if many such experiments were run?" and as a potential future generalization for other scenarios. Content as in §5.
4. **README — Bootstrap RNG reasoning:** Ensure the reasoning for **separate RNG streams for data and bootstrap** is present (the README already has "Separate RNG streams for data and bootstrap" under Bootstrap CI; retain it and do not remove or shorten the explanation when editing other sections).
5. **Optional:** Two code comments (power_asymptotic.py, spearman_helpers.py) as in §3.

After this, statisticians and simulation experts have (a) exact test/power/CI formulas and caveats in one place, (b) a clear account of what each data-generation method does (linear, copula, etc.), (c) explicit notes on what is unverified (1.06 in power) or approximate (t-based p-value with ties), and (d) a clear record of empirical-generator and p-value/calibration decisions and the alternative design for future generalization.
