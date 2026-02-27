# Karwowski Spearman Power Simulation — Project Summary

Use this document as context when resuming work on the codebase.

---

## Purpose

Statistical power and confidence-interval framework for Spearman rank correlations, evaluating what correlations the Karwowski et al. (2018) study *could* have detected between cumulative vaccine aluminum and blood/hair aluminum in infants. The key challenge is **ties in the x-variable** (cumulative aluminum clusters around 2.9 mg, IQR 0.11 mg), modeled as 4–10 distinct x-values with three frequency distributions: `even`, `heavy_tail`, `heavy_center`.

**Four cases**: B-Al and H-Al, each with/without outlier exclusion (N = 73, 80, 81, 82). **88 scenarios** per method: 84 tied (4 cases × 7 k-values × 3 dist types) + 4 all-distinct.

---

## Analysis Methods

| Method | Description | When to use |
|--------|-------------|-------------|
| **Nonparametric** (recommended) | Rank-mixing in rank space + calibration | Default for power/CI when x has ties |
| **Copula** | Gaussian copula, non-parametric marginals + calibration | All-distinct or comparison; calibration compensates attenuation |
| **Linear** | y = a + b·x + noise on log scale | Parametric complement; approximate |
| **Asymptotic** | Closed-form (non-central t, Fisher z) | Fast cross-check; no simulation |

---

## Key Implementation Details

### Calibration (nonparametric and copula)

- Ties attenuate realised Spearman rho vs mixing weight (or input rho for copula).
- **Multipoint (default)**: Probes at rho = 0.10, 0.30, 0.50; interpolates the calibration curve. Fixes nonlinear attenuation (0.01–0.03 bias that single-point can have). ~3× calibration cost (~9s vs ~3s per tie structure on first run); cached thereafter.
- **Single-point** (`calibration_mode="single"`): Probes only at 0.30, applies constant ratio. Faster for exploratory runs.
- Ratio/curve cached per `(n, k, dist_type)` and reused for all rho values during bisection — avoids per-rho calibration (major speedup).
- **Nonparametric**: calibrates input mixing weight for `_raw_rank_mix`.
- **Copula**: calibrates input rho_s for `generate_y_copula`; same rho-independent ratio approach (single-point only).
- Uses **symmetry** for negative rho_target: probes only at positive values, applies sign to result.

**Reference configuration**: seed=99, n_sims=4000, n_cal=700 achieves the floor (mean\|diff\| ≈ 0.002, max\|diff\| ≈ 0.003). Other seeds may not reach this accuracy.

### Other implementation notes

- **Spearman p-value**: t-approximation (`t = rho * sqrt((n-2)/(1-rho²))`) implemented inline.
- **Caching**: `_fit_lognormal`, x-value templates, and calibration ratios cached per tie structure.
- **Copula limitation**: Continuous-marginals assumption causes attenuation with heavy ties (k=4); documented in README.

---

## Performance Optimizations

| Change | Impact |
|--------|--------|
| Rho-independent calibration ratio | ~20× (calibration once per tie structure, not per rho). Multipoint: ~3× slower than single-point (~9s vs ~3s per structure) but more accurate. |
| Vectorized Spearman (argsort-based ranking) | ~3.5× for CI (single scenario: ~42s → ~12s) |
| Scenario-level parallelization (`n_jobs`) | ~2× additional (full grid CI: ~16 min → ~8 min with 4 logical cores) |
| **Numba JIT** (ranking + bootstrap loops) | ~3× for CI, ~2× for power (with inner-thread parallelism via `prange`) |
| Fast inline Spearman (t-approximation) | ~2× vs scipy.spearmanr |
| LRU-cached `_fit_lognormal` | ~1.5× |
| Cached x-value templates | ~1.2× |

**Combined result**: Full CI grid ~7× faster than original sequential code. With Numba: ~20–28× total speedup.

**Implementation**: Vectorized Spearman (`spearman_helpers.py`), scenario-level parallelization (`joblib.Parallel`), **Numba JIT** (`_rank_rows_numba`, `_bootstrap_rhos_jit`) when `config.USE_NUMBA=True`. CLI: `--no-numba` / `--numba` flags; warm-up via `warm_up_numba.py`. Default `n_jobs=1`; `n_jobs=-1` uses all cores.

---

## File Roles

| File | Role |
|------|------|
| `config.py` | CASES, FREQ_DICT, N_SIMS, ALPHA, TARGET_POWER, `CALIBRATION_MODE` (multipoint/single), `USE_NUMBA=True` (Numba control) |
| `data_generator.py` | X generation, Y generators (copula, linear, nonparametric), `calibrate_rho`, `calibrate_rho_copula`, `_raw_rank_mix`, `_fit_lognormal` |
| `power_simulation.py` | `estimate_power`, `min_detectable_rho`, `run_all_scenarios`; uses `spearman_rho_pvalue_2d` (vectorized) |
| `power_asymptotic.py` | Non-central t power, Fisher z CI, tie correction (FHP), `get_x_counts` |
| `confidence_interval_calculator.py` | `bootstrap_ci_single`, `bootstrap_ci_averaged`, `run_all_ci_scenarios`; uses `spearman_rho_2d` (vectorized) and `_bootstrap_rhos_jit` (Numba) when available |
| `spearman_helpers.py` | `spearman_rho_2d`, `_fast_spearman_rho`, `spearman_rho_pvalue_2d` (vectorized helpers). **Numba JIT**: `_rank_rows_numba`, `_bootstrap_rhos_jit` (optional, controlled by `config.USE_NUMBA`) |
| `run_simulation.py` | Main orchestrator; runs MC + asymptotic + CI; CLI with `--no-numba`/`--numba` flags |
| `run_single_scenario.py` | Single-scenario power/CI; supports `--freq` for custom distributions; CLI with `--no-numba`/`--numba` flags |
| `warm_up_numba.py` | Pre-compiles Numba JIT functions (~5–15s first run, cached thereafter) |
| `validation_test_spearman2d.py` | Validates Numba vs NumPy fallback produce identical results |
| `test_simulation_accuracy.py` | Validates generators achieve target rho; flags |diff| > 0.01 |
| `power_simulation_copula.py` | Legacy; delegates to `power_simulation` |
| `power_simulation_linear.py` | Legacy; delegates to `power_simulation` |
| `table_outputs.py` | Builds/saves CSV tables |

---

## Bootstrap CI Precision

Inter-rep SD ≈ 0.10–0.11 (N=73). SE of mean = SD/√n_reps. **n_reps guidance**: ≈400 (borderline), ≈1600 (strong), ≈7400 (rounding guarantee). **n_boot**: With n_reps ≥ 1600, n_boot=200–400 suffices; n_boot=500 is comfortable. Runtime scales linearly with n_reps × n_boot. See README for details.

---

## Typical Runtimes (nonparametric, vectorized + parallelized)

**Full grid (88 scenarios, n_jobs=4, 4-core Windows):**
| Task | Parameters | Without Numba | With Numba |
|------|------------|--------------|------------|
| CI | 200 reps × 1000 boot | ~8 min | ~8 min |
| CI | 7400 reps × 500 boot | ~2.5 h | ~20–45 min |
| Power | 500 sims, n_cal=300 | ~3 min | ~3 min |
| Power | 10,000 sims, n_cal=300 | ~30–60 min | ~5–12 min |

**Cloud (16 vCPU, Numba, pre-warmed):** CI (7400×500) ~12 min per generator.

**Note**: Power gains limited (calibration/data gen not Numba-accelerated). CI bootstrap loop benefits most. See README for detailed benchmarks.

---

## CLI Quick Reference

```bash
# Full run (Numba enabled by default if installed)
python run_simulation.py --n-sims 500 --skip-copula --skip-linear

# Disable Numba (force pure NumPy fallback)
python run_simulation.py --n-sims 500 --no-numba

# Single scenario
python run_single_scenario.py --case 3 --n-distinct 4 --dist-type heavy_center --n-sims 500 --power-only

# Custom frequency
python run_single_scenario.py --case 3 --freq 19,18,18,18 --n-sims 500

# Accuracy test
python test_simulation_accuracy.py --n-sims 50 --case 3 --n-distinct 4
# Reference config (calibration seed 99, floor): --n-sims 4000 --n-cal 700 --generators nonparametric
# Calibration mode: --calibration-mode multipoint (default) or --calibration-mode single

# Warm up Numba cache (recommended before large runs)
python warm_up_numba.py
```

---

---

*Last updated: 2026-02-27*
