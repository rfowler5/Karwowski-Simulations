# Spearman Power Simulation -- Vaccine Aluminum Association

Statistical power and confidence-interval framework for Spearman rank correlations, designed to evaluate the **non-significant associations** reported in Karwowski et al. (2018) between blood/hair aluminum levels and cumulative vaccine aluminum exposure.

## Background

Karwowski et al. (2018) found no significant Spearman correlations between cumulative vaccine aluminum and either blood aluminum (B-Al) or hair aluminum (H-Al) in a cohort of 9--13-month-old infants. This simulation framework asks: **what correlations *could* the study have detected?**

Because most children followed a similar vaccination schedule, cumulative aluminum values cluster around a narrow range (median 2.9 mg, IQR 0.11 mg), producing **ties in the x-variable**. The framework explicitly models these ties to assess their impact on statistical power.

## Four Cases

| Case | Analyte | Outliers | N  | Observed rho | P    |
|------|---------|----------|----|-------------|------|
| 1    | B-Al    | Included | 80 | -0.13       | 0.26 |
| 2    | H-Al    | Included | 82 | +0.06       | 0.56 |
| 3    | B-Al    | Excluded | 73 | -0.13       | 0.26 |
| 4    | H-Al    | Excluded | 81 | +0.06       | 0.56 |

## Analysis Goals

1. **Minimum detectable correlation** -- the smallest |rho| detectable with 80% power at alpha = 0.05, for each case x tie structure x distribution shape.
2. **Confidence intervals for observed rho** -- bootstrap and asymptotic CIs, showing the plausible range of true correlations consistent with the data.

## Tie Handling

- X-values are generated with 4--10 distinct values using three frequency distributions: `even`, `heavy_tail`, and `heavy_center`.
- An additional all-distinct baseline (no ties) is computed for each case.
- Total: 84 tied scenarios + 4 all-distinct = 88 scenarios per method.

## Three Analysis Methods

### Non-parametric rank-mixing (recommended)

The non-parametric method is the recommended Monte Carlo approach, especially when x has ties. It works by:

1. Computing the standardised midranks of x (handling ties naturally).
2. Generating independent noise ranks via random permutation.
3. Mixing the two at weight rho: `mixed = rho_cal * s_x + sqrt(1 - rho_cal^2) * s_noise`.
4. Mapping the resulting ordering onto random draws from the target log-normal y-marginal.

**Calibration**: Because ties attenuate the realised Spearman rho relative to the mixing weight, the method includes an automatic calibration step. For each tie structure (N, k, distribution type), a bisection search over 300 calibration samples computes a rho-independent attenuation ratio. This ratio is then applied linearly to any target rho, meaning calibration runs only once per unique tie structure and is reused across all rho values tested during bisection -- a major performance improvement over per-rho calibration.

**Why this method?** Unlike the Gaussian copula, it does not rely on the continuous-marginals assumption. Unlike the linear model, it does not assume a parametric relationship between x and y. It handles tied x-values naturally because the mixing operates directly in rank space.

### Gaussian copula

Uses a Gaussian copula with non-parametric marginals. Converts the target Spearman rho to a Pearson correlation via `rho_p = 2 * sin(pi * rho_s / 6)`, then draws from a conditional bivariate normal. Y-values are mapped through the inverse CDF of a fitted log-normal.

**Limitation**: When x has heavy ties, the jittering step that breaks tied ranks collapses rank information and attenuates the realised Spearman rho, leading to underestimated power. Alternatives tested -- distributional transform (random uniform within each tie group's CDF band) and adaptive jitter (scaling jitter by tie group size) -- improved the mild cases (k=10) but still failed the 0.01 accuracy threshold for heavy ties (k=4). This is a fundamental limitation of the continuous-marginals assumption. For this reason, the copula is no longer the default; it is retained as an option for comparison and for all-distinct scenarios where it performs well.

### Linear Monte Carlo

Generates `y = a + b*x + noise` on a log scale, calibrated so that the Pearson correlation of ranks approximates the target Spearman rho. This is a parametric model -- useful as a complement to the non-parametric method, but approximate because it assumes a linear log-scale relationship.

### Asymptotic

Closed-form formulas (no simulation). Power uses the non-central t-distribution with df = n-2; CIs use the Fisher z-transform with Bonett-Wright SE = sqrt(1.06/(n-3)). Ties are handled via Fieller-Hartley-Pearson correction throughout. Fast and stable, serves as a cross-check against the Monte Carlo methods.

## Installation

```bash
pip install -r requirements.txt
```

Required packages: numpy, scipy, pandas, joblib. **Numba** (recommended) is included in `requirements.txt` for significant speedups. If Numba cannot be installed on your platform, the code falls back to pure NumPy automatically.

### Numba warm-up (optional, recommended)

On the first run after installation, Numba compiles JIT functions (5-15 seconds). To pre-warm the cache:

```bash
python warm_up_numba.py
```

To copy the cache to a cloud VM for zero compile delay:

```bash
rsync -avz ~/.cache/numba/ user@YOUR-IP:~/.cache/numba/
```

## Usage

### Programmatic usage (no command line)

All scripts expose a `main()` function that can be called from Python:

```python
# Full simulation
from run_simulation import main as run_sim
power_df, ci_df, all_distinct_df = run_sim(n_sims=500, skip_linear=True)

# Single scenario
from run_single_scenario import main as run_single
result = run_single(case=3, n_distinct=4, dist_type="heavy_center", n_sims=500)
result = run_single(case=3, freq=[19, 18, 18, 18], power_only=True, verbose=False)

# Accuracy testing
from test_simulation_accuracy import main as test_accuracy
df = test_accuracy(n_sims=50, cases=[3], custom_freq=[(3, [19, 18, 18, 18])])
```

### Full simulation (all methods, all scenarios)

```bash
python run_simulation.py
```

### Quick test run

```bash
python run_simulation.py --n-sims 500 --seed 42
```

### Disable Numba (force pure NumPy fallback)

```bash
python run_simulation.py --n-sims 500 --seed 42 --no-numba
python run_single_scenario.py --case 3 --n-distinct 4 --dist-type heavy_center --n-sims 500 --no-numba
```

### Skip specific methods

```bash
python run_simulation.py --skip-linear --skip-copula   # nonparametric + asymptotic only
python run_simulation.py --skip-nonparametric           # copula + linear + asymptotic
```

### Filter scenarios

```bash
python run_simulation.py --cases 1,3 --n-distinct 4,10 --dist-types even,heavy_center
```

### Single-scenario quick script

Run power and/or CI for a specific (case, k, distribution) combination:

```bash
# Power + CI for Case 3, k=4, heavy_center
python run_single_scenario.py --case 3 --n-distinct 4 --dist-type heavy_center --n-sims 500

# Power only, skip copula and linear
python run_single_scenario.py --case 3 --n-distinct 4 --dist-type even --power-only --skip-copula --skip-linear

# CI only for all-distinct baseline
python run_single_scenario.py --case 1 --all-distinct --ci-only --n-reps 20 --n-boot 500

# Custom frequency distribution (counts must sum to case's n)
python run_single_scenario.py --case 3 --freq 19,18,18,18 --n-sims 500
```

### Asymptotic tie-correction modes

```bash
python run_simulation.py --tie-correction both          # report both corrected and uncorrected
python run_simulation.py --tie-correction without_tie_correction
```

## Validation

Test whether generators achieve the target Spearman rho:

```bash
# Quick check on worst-case scenario
python test_simulation_accuracy.py --n-sims 50 --case 3 --n-distinct 4

# Full sweep, all generators
python test_simulation_accuracy.py --n-sims 200

# Specific generators only
python test_simulation_accuracy.py --generators nonparametric,copula --n-sims 100

# Custom frequency distribution (requires --case; counts must sum to case's n)
python test_simulation_accuracy.py --case 3 --freq 19,18,18,18 --n-sims 50

# Save results to CSV
python test_simulation_accuracy.py --n-sims 200 --outfile accuracy_report.csv
```

The script flags scenarios where |mean_simulated_rho - target_rho| > 0.01 (configurable via `--threshold`).

## Output

Results are saved to the `results/` directory:

- `min_detectable_rho.csv` -- minimum detectable rho for all scenarios and methods
- `confidence_intervals.csv` -- bootstrap and asymptotic CIs for observed rho
- `all_distinct_summary.csv` -- combined power + CI table for the 4 all-distinct baselines

## Project Structure

```
config.py                        -- Case parameters, frequency dictionary, settings
data_generator.py                -- X generation (ties) and Y generation (all methods)
power_simulation.py              -- Unified Monte Carlo power (all generators)
power_simulation_copula.py       -- Legacy copula-specific power module
power_simulation_linear.py       -- Legacy linear-specific power module
power_asymptotic.py              -- Asymptotic power and CI formulas
confidence_interval_calculator.py -- Bootstrap CIs (averaged over multiple datasets)
table_outputs.py                 -- Summary table construction and CSV export
run_simulation.py                -- Main orchestrator / CLI entry point
run_single_scenario.py           -- Quick single-scenario testing script
test_simulation_accuracy.py      -- Validation: generator accuracy testing
```

## Method Choices and Rationale

### Why non-parametric rank-mixing over copula?

The Gaussian copula assumes continuous marginals for the rank-to-normal-to-rank transformation to preserve Spearman rho. When x has heavy ties (e.g., k=4 with N=73--82), the jittering step that breaks ties is too small to restore the lost rank information, causing systematic attenuation of 0.01--0.06 in the realised rho. Distributional transform and adaptive jitter alternatives were tested but do not fix this for heavy ties. The non-parametric rank-mixing method avoids this by operating directly in rank space and using empirical calibration to compensate for any residual attenuation.

### Why calibration?

With tied x-values, the midrank representation has lower variance than distinct ranks. This means the rank-mixing formula `mixed = rho * s_x + sqrt(1-rho^2) * s_noise` produces a Spearman rho slightly below the target rho. The calibration step computes a rho-independent attenuation ratio by probing at a fixed rho (0.30) and using bisection over 300 samples. This ratio is then multiplied by any target rho to compensate for the attenuation. The ratio is cached per (n, k, dist_type), so the cost is ~3s per unique tie structure.

**If calibration fails for a custom tie structure:** Run `test_simulation_accuracy` on the new structure. If mean realised rho deviates from target by >0.01, try increasing `n_cal` (e.g. 300 → 500 or 1000). If it still fails, the attenuation may be nonlinear (ratio varies with rho)—consider implementing **multi-point calibration**: probe at several targets (e.g. 0.10, 0.30, 0.50), fit a curve (rho_in vs rho_target), and use that to map any target to the required input. If even rho_input=0.999 cannot reach the probe, the tie structure has hit a structural ceiling (maximum achievable |rho|) and the method cannot reach that target.

### Why Bonett-Wright SE (1.06 factor)?

The standard Fisher z-transform SE for Pearson r is `sqrt(1/(n-3))`. For Spearman rho, Bonett and Wright (2000) showed that the variance is approximately 6% larger, giving SE = `sqrt(1.06/(n-3))`. The theoretical asymptotic variance factor for Spearman rho (no ties) is π²/9 ≈ 1.0966; Bonett & Wright (2000) recommended the simpler 1.06 approximation (~6% efficiency loss), which is commonly used in practice.

### Why non-central t for power?

The Spearman test statistic `t = rho * sqrt(n-2) / sqrt(1 - rho^2)` follows a non-central t-distribution under the alternative. This is more accurate than the normal approximation (which uses constant H0 variance) because the `sqrt(1 - rho^2)` denominator captures the variance reduction as |rho| grows toward 1.

### Why Fisher z-transform for CIs?

The arctanh transform stabilises the variance of the correlation coefficient and improves normality of the sampling distribution. The back-transform via tanh guarantees the CI stays within [-1, 1] and is properly asymmetric around the point estimate.

### Tie correction (Fieller-Hartley-Pearson)

The FHP formula adjusts the variance of Spearman rho under H0 for tied observations. The impact on SE ranges from negligible (k=10, even: +0.5%) to moderate (k=4, heavy_center: +5.7%). This correction is applied to both the asymptotic power calculation (reduces the noncentrality parameter) and the asymptotic CI (widens the interval).

## Bootstrap CI

The bootstrap CI averages endpoints over many simulated datasets (n_reps) to estimate the expected bootstrap CI under the model. Two design choices affect correctness and precision:

### Separate RNG streams for data and bootstrap

Data generation and bootstrap resampling use **separate** RNG streams derived from the same seed via `np.random.SeedSequence.spawn(2)`. A shared RNG would advance by `n_boot * n` extra draws per rep inside the bootstrap loop, so the datasets for reps 1..n_reps would depend on `n_boot`. That would make results non-comparable across different `n_boot` values and invalidate the interpretation that "more bootstraps = more accurate." With separate streams, the same seed always produces the same n_reps datasets regardless of `n_boot`.

### n_reps, SE, and reliability of the second decimal

The CI endpoints vary across reps (inter-rep SD ≈ 0.10–0.11 for N=73, slightly lower for N=80+). The SE of the mean endpoint is SD/√n_reps. The 95% CI for the true mean is approximately ±1.96×SE.

| Target | SE | 95% CI half-width | n_reps | When it matters |
|--------|-----|-------------------|--------|-----------------|
| Borderline | 0.005 | ±0.01 | ≈400 | Second decimal can still be off by one unit |
| Strong (README "reliable") | 0.0025 | ±0.005 | ≈1600 | Useful precision; fine when *not* near a rounding boundary |
| Rounding guarantee | 0.00128 | ±0.0025 | ≈7400 | Needed only when the value is *near* a boundary (e.g. 0.345, 0.355) |

**Rounding boundaries:** The boundary between rounding to 0.34 vs 0.35 is 0.345. With n_reps=1600 (95% CI ±0.005), if you observe 0.345 the CI spans [0.34, 0.35] and crosses the boundary—you cannot confidently round. When the value is not near a boundary (e.g. 0.32, 0.38), n_reps=1600 is adequate. The n_reps≈7400 "rounding guarantee" is only needed when you happen to land near a boundary and want confidence in which way to round.

With n_reps=200, SE ≈ 0.007 (worst case N=73), so the third decimal is uncertain and the second decimal is borderline. The default n_reps=200 is a practical trade-off.

### n_boot choice (with high n_reps)

Bootstrap quantile noise scales as 1/√n_boot and is negligible compared with inter-rep variability (σ_inter ≈ 0.11) when n_reps is high. With n_reps ≥ 1600:

- **n_boot=200–400** suffices: bootstrap variance adds under 0.5% to total; 2.5th percentile estimated from 5–10 order statistics.
- **n_boot=500** is comfortable; going higher gives no meaningful improvement.
- **n_boot=1000** (default in `config.py`) is more than needed for high n_reps; use 200–500 to save time.

When n_reps=200, **n_boot=500** is sufficient; the ~0.001–0.002 difference from n_boot=1000 is swamped by the ~0.007 SE from inter-rep variability.

### Verifying n_boot

To check whether your chosen n_boot is sufficient, run the same scenario with n_boot=500 and n_boot=2000 (same seed). Compare the printed bootstrap CI endpoints:

**Quick check** (n_reps=50, ~1–2 min total): If the difference in CI endpoints is under ~0.003, n_boot=500 is fine for 2-decimal precision. Example:

```bash
python run_single_scenario.py --case 3 --n-distinct 4 --dist-type heavy_center --ci-only --n-reps 50 --n-boot 500 --seed 42 --skip-copula --skip-linear
python run_single_scenario.py --case 3 --n-distinct 4 --dist-type heavy_center --ci-only --n-reps 50 --n-boot 2000 --seed 42 --skip-copula --skip-linear
```

**Thorough check** (n_reps=200, ~5–10 min total): Use n_reps=200 for a more stable comparison and to confirm the averaged CI is well converged.

## Performance and Runtime Estimates

### Optimisation summary

**Vectorization and parallelization:** The original implementation used a scalar loop over bootstrap replicates and power simulations. Vectorized Spearman (argsort-based ranking, O(n log n) per row) plus scenario-level parallelization (`--n-jobs`) yielded roughly **3.5× speedup** for CI (e.g. single scenario 200 reps × 1000 boot: ~42s → ~12s) and **~2×** from parallelization on 4 logical cores (full grid CI: ~16 min sequential → ~8 min with `n_jobs=4`). Combined, the full CI grid is ~7× faster than the original sequential code.

**Calibration:** Uses a rho-independent attenuation ratio cached per (n, k, dist_type). The calibration cost is paid **once per tie structure**, not once per rho value tested during bisection -- eliminating the dominant bottleneck of the original implementation (~60× speedup for power estimation).

**Other optimisations:** `_fit_lognormal` results are LRU-cached; x-value templates are cached and only shuffled per call; a fast inline Spearman (Pearson-of-ranks + t-distribution p-value) replaces the full `scipy.stats.spearmanr` in the power estimation loop.

### Typical runtimes (nonparametric generator, 4-core Windows machine)

| Task | Approximate time |
|------|-----------------|
| Single scenario CI (200 reps x 1000 boot) | ~12s |
| Single scenario power (500 sims, n_cal=300) | ~5s (includes calibration) |
| Single scenario power (10,000 sims, n_cal=300) | ~45s |
| Full grid CI (88 scenarios), n_jobs=1, 200 reps x 1000 boot | ~16 min |
| Full grid CI (88 scenarios), n_jobs=4, 200 reps x 1000 boot | ~8 min |
| Full grid power (88 scenarios, 500 sims, n_cal=300), n_jobs=1 | ~6 min |
| Full grid power (88 scenarios, 500 sims, n_cal=300), n_jobs=4 | ~3 min |

### With Numba JIT (recommended)

Numba JIT compilation adds inner thread parallelism to the ranking and bootstrap loops. Combined with scenario-level `--n-jobs`, expected speedups on a 4-logical-core machine:

| Task | Without Numba | With Numba | Speedup |
|------|--------------|------------|---------|
| Single scenario CI (200 reps x 500 boot) | ~12s | ~3-5s | ~3x |
| Single scenario power (500 sims, n_cal=300) | ~5s | ~2-3s | ~2x |
| Full grid CI (88 scenarios, n_reps=7400, n_boot=500, n_jobs=4) | ~2.5 h | ~20-45 min | ~4-8x |
| Full grid power (88 scenarios, n_sims=10k, n_cal=300, n_jobs=4) | ~30-60 min | ~5-12 min | ~4-6x |

On a 16-vCPU cloud machine (e.g. Hetzner CPX51), full CI grid per generator (88 scenarios, n_reps=7400, n_boot=500, n_jobs=-1) completes in under 12 minutes with Numba and pre-warmed cache.

Copula and linear generators have similar per-sim cost but no calibration overhead. The asymptotic method is instantaneous.

### Tips

- **Benchmarking:** Run sequential and parallel benchmarks separately, one at a time. Concurrent runs cause CPU contention and invalidate timing results.
- Use `--n-sims 500` for exploratory runs (seconds to minutes) vs `10000` for production.
- Use `python run_simulation.py --n-jobs 4` to parallelize across 4 cores and (if this is actually 2 cores with hyperthreading) roughly halve full-grid runtimes. Use `--n-jobs -1` to use all available cores.
- The calibration step adds ~3s per unique (N, k, distribution) tie structure on first run; cached thereafter and reused across all rho values.
- Bootstrap CIs dominate total runtime when using many reps and resamples. With `--n-reps 200`, use `--n-boot 500` (bootstrap noise is negligible). With `--n-reps 1600` or higher, `--n-boot 200`–`400` suffices; `--n-boot 500` is comfortable. Use `--n-reps 20 --n-boot 500` for quick checks.
- Use `run_single_scenario.py` to test individual scenarios quickly before committing to a full run.
- Filter scenarios with `--cases`, `--n-distinct`, `--dist-types` to reduce the grid.
- Use `--skip-copula --skip-linear` to run only the recommended nonparametric + asymptotic methods.

## Cloud Deployment

### Running on a cloud VM (e.g. Hetzner CPX51, 16 vCPU)

**With pre-copied Numba cache (recommended):**

```bash
# On local machine (after running warm_up_numba.py):
rsync -avz ~/.cache/numba/ root@YOUR-IP:~/.cache/numba/

# On cloud VM:
python run_simulation.py --n-sims 10000 --skip-copula --skip-linear --n-jobs -1 --seed 42
```

**Full CI grid (all generators):**

```bash
python -c '
import time, pickle
from confidence_interval_calculator import run_all_ci_scenarios
for gen in ["nonparametric", "copula", "linear"]:
    t0 = time.time()
    print(f"Starting {gen}...")
    res = run_all_ci_scenarios(generator=gen, n_reps=7400, n_boot=500, n_jobs=-1, seed=42)
    with open(f"ci_{gen}.pkl", "wb") as f: pickle.dump(res, f)
    print(f"{gen} done in {time.time()-t0:.1f} s")
'
```

## Reference

Karwowski MP, et al. (2018). Blood and Hair Aluminum Levels, Vaccine History, and Early Infant Development: A Cross-Sectional Study. *Academic Pediatrics*, 18(2), 161--165.
