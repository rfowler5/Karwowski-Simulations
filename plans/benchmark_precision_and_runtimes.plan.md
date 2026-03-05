---
name: Benchmark precision and runtimes
overview: "Create two benchmark scripts: Script 1 computes and prints the (n_sims, n_cal, n_reps, n_boot) needed for three accuracy tiers. Script 2 measures realistic runtimes per generator at small params and scales to all three tiers, reporting sequential, parallel, and estimated high-core-count runtimes."
todos:
  - id: script1-precision
    content: "Create benchmarks/benchmark_precision_params.py: pure-math script that computes and prints (n_sims, n_cal) and (n_reps, n_boot) for +/-0.01, +/-0.002, +/-0.001 accuracy tiers, using the README formulas"
    status: pending
  - id: script2-runtimes
    content: "Create benchmarks/benchmark_realistic_runtimes.py: measure per-generator runtimes (power + CI) at small params (n_sims=500, n_reps=200, n_boot=500), both sequential and parallel; scale to 3 precision tiers; print tables with measured, scaled, and high-core estimates"
    status: pending
  - id: verify-output
    content: Run both scripts (one at a time per benchmarking rule) and verify output is correct and tables are clear
    status: pending
  - id: oneoff-ci-high-load
    content: "One-off: run CI batch bootstrap at high load (n_reps=2000, n_boot=1000 or n_reps=7400, n_boot=500), sequential then parallel; record T_seq and T_par to scale parallel time to other cores"
    status: pending
isProject: false
---

# Benchmark Precision Params and Realistic Runtimes

## Verified claims from prior discussion

**Joblib overhead vs thread competition (confirmed):** Per [BENCHMARKING_FINDINGS.md](docs/BENCHMARKING_FINDINGS.md) lines 90-104, the gap between batch CI sequential (159.74s) and parallel (265.61s) decomposes as:

- **~89% joblib overhead** (sequential 159.74s vs parallel-with-Numba=1 253.59s = 94s gap)
- **~11% Numba thread oversubscription** (parallel default 265.61s vs parallel Numba=1 253.59s = 12s gap)

So joblib overhead is the dominant factor. Thread competition exists but is minor (~11%). On higher-core machines with longer per-scenario work, joblib overhead becomes a smaller fraction, so `n_jobs=-1` can help.

**CI generators (confirmed):** From [run_simulation.py](scripts/run_simulation.py) line 176: `ci_generators = [g for g in mc_methods if g != "linear"]`. So CI uses 3 generators: nonparametric, copula, empirical (not linear).

**Power generators:** All 4 Monte Carlo generators: nonparametric, copula, linear, empirical.

## Corrections / improvements to earlier design

1. **Earlier design said "efficiency 0.4-0.6" for multi-core scaling.** This is fine for power (where joblib can parallelize bisection scenarios), but for **CI with batch bootstrap, parallel is often slower even on the test machine** (per BENCHMARKING_FINDINGS.md). The scaling formula should be applied only to **power** and to **CI with per-rep bootstrap (or future implementations)**. For CI with batch bootstrap, sequential remains recommended unless per-scenario time is long enough (e.g. very high n_reps) to amortize joblib overhead. The script should note this distinction.
2. **SE_cal coefficient:** README says "SE ~0.005-0.006 at n_cal=300." Using 0.006 gives coefficient k = 0.006 x sqrt(300) = 0.104. So SE_cal = 0.104/sqrt(n_cal). This is used below.
3. **Bisection steps:** Search range [0.25, 0.42], tolerance 0.49e-4. Steps = ceil(log2(0.17 / 0.49e-4)) = 12. So each power scenario calls `estimate_power` ~12 times.

## Computed precision parameters (three tiers)

### Power (min detectable rho)

Formulas: SE_bisection = 0.16/sqrt(n_sims), SE_cal = 0.104/sqrt(n_cal), SE_total = sqrt(SE_bisection^2 + SE_cal^2), half-width = 1.96 x SE_total.


| Tier     | Half-width | n_sims  | n_cal  | SE_total | Actual half-width |
| -------- | ---------- | ------- | ------ | -------- | ----------------- |
| +/-0.01  | 0.01       | 2,000   | 1,000  | 0.00486  | 0.00953           |
| +/-0.002 | 0.002      | 50,000  | 21,000 | 0.00101  | 0.00199           |
| +/-0.001 | 0.001      | 200,000 | 83,000 | 0.000508 | 0.000996          |


### CI (bootstrap endpoint)

Formula: inter-rep SD = 0.11 (worst case, N=73). SE = 0.11/sqrt(n_reps). Half-width = 1.96 x SE.


| Tier     | Half-width | n_reps | n_boot | SE       | Actual half-width |
| -------- | ---------- | ------ | ------ | -------- | ----------------- |
| +/-0.01  | 0.01       | 500    | 500    | 0.00492  | 0.00964           |
| +/-0.002 | 0.002      | 12,000 | 500    | 0.00100  | 0.00197           |
| +/-0.001 | 0.001      | 47,000 | 500    | 0.000507 | 0.000994          |


## Script 1: `benchmarks/benchmark_precision_params.py`

**Purpose:** Compute and print the tables above (no long runs). Pure arithmetic from the README formulas.

**What it does:**

- Define the three target half-widths: 0.01, 0.002, 0.001.
- For power: solve for (n_sims, n_cal) that achieve SE_total <= w/1.96, balancing the two SE components (SE_bisection = SE_cal).
- For CI: solve for n_reps from SD=0.11 and SE <= w/1.96. n_boot = 500 for all (justified by README).
- Print both tables in a clear format suitable for pasting into the README.
- Print the formulas used.
- No imports from the project needed (pure math with `math.sqrt`, `math.log2`, `math.ceil`).

## Script 2: `benchmarks/benchmark_realistic_runtimes.py`

**Purpose:** Measure runtimes at small params, then scale to all three precision tiers.

### CLI

- `--generators {all|nonparametric|copula|linear|empirical}` (default: `all` = all available based on `digitized_available()`).
- `--quick` flag: single-scenario only (skip full grid).
- `--skip-parallel` flag: skip parallel benchmarks (saves time).

### Benchmark params (small, for measurement)

- Power: n_sims=500, n_cal=300 (default in code).
- CI: n_reps=200, n_boot=500.

### Structure

1. **Warmup:** One small run per generator to trigger Numba JIT.
2. **Per-generator power benchmarks (sequential, n_jobs=1):**
  - For each generator in [nonparametric, copula, linear, empirical] (filtered by availability and CLI):
    - Single scenario: time `min_detectable_rho(...)` (Case 3, k=4, heavy_center).
    - Full grid: time `run_all_scenarios(generator=gen, n_sims=500, n_jobs=1, calibration_mode="multipoint")`.
3. **Per-generator CI benchmarks (sequential, n_jobs=1):**
  - For each generator in [nonparametric, copula, empirical] (filtered):
    - Single scenario: time `bootstrap_ci_averaged(...)` (Case 3, k=4, heavy_center, batch_bootstrap=True, calibration_mode="multipoint").
    - Full grid: time `run_all_ci_scenarios(generator=gen, n_reps=200, n_boot=500, n_jobs=1, calibration_mode="multipoint", batch_bootstrap=True)`.
4. **Parallel benchmarks (n_jobs=-1):** Unless `--skip-parallel`:
  - Repeat full-grid power per generator with n_jobs=-1.
  - Repeat full-grid CI per generator with n_jobs=-1 (even though sequential is faster on this machine, the parallel timings are needed for scaling estimates to higher-core machines).
5. **Scaling and output:**
  - **Power scaling:** `time_scaled = time_measured * (n_sims_target / n_sims_bench)`. (Calibration cost is constant and already included once; at higher n_sims it becomes negligible fraction, so linear scaling is slightly conservative = good.)
  - **CI scaling:** `time_scaled = time_measured * (n_reps_target * n_boot_target) / (n_reps_bench * n_boot_bench)`.
  - Print tables:
    - **Per-generator power** (single + full grid, seq): measured + scaled for 3 tiers.
    - **Per-generator CI** (single + full grid, seq): measured + scaled for 3 tiers.
    - **All-generators total** (sum of per-generator times) for power and CI, plus combined (power + CI) for each tier.
    - **Parallel timings** (full grid only): measured + scaled.
  - **High-core-count estimate:** For power full grid parallel: `estimated_time = (sequential_time / N_cores) / efficiency`, with efficiency = 0.5 (midpoint of 0.4-0.6 range). Print for 8-core and 16-core. For CI: note that batch bootstrap parallel is not recommended; print the sequential time and note that on high-core machines with very high n_reps, per-rep bootstrap with n_jobs=-1 may be faster (but this is not benchmarked here).

### Output format

Per-generator tables show **both sequential and parallel** full-grid times so you can see which is faster for each generator and task. The COMBINED table then uses the **best config per component** (whichever of seq/par is faster for power, and whichever is faster for CI) to give realistic total runtimes.

```
=== POWER: Per-Generator Runtimes (full grid, all generators) ===

Generator       | Single (seq) | Grid seq | Grid par | Grid +/-0.01 | Grid +/-0.002 | Grid +/-0.001
nonparametric   |        X.Xs  |    X.Xs  |    X.Xs  |        X min |         X min |        X min
copula          |        X.Xs  |    X.Xs  |    X.Xs  |        X min |         X min |        X min
linear          |        X.Xs  |    X.Xs  |    X.Xs  |        X min |         X min |        X min
empirical       |        X.Xs  |    X.Xs  |    X.Xs  |        X min |         X min |        X min
ALL GENERATORS  |           -  |    X.Xs  |    X.Xs  |        X min |         X min |        X min

    Single = single scenario, n_jobs=1. Grid seq = full grid, n_jobs=1. Grid par = full grid, n_jobs=-1.
    Scaled columns use whichever of seq/par is faster on this machine (noted below table).

=== CI: Per-Generator Runtimes (full grid, all generators, batch bootstrap) ===

Generator       | Single (seq) | Grid seq | Grid par | Grid +/-0.01 | Grid +/-0.002 | Grid +/-0.001
nonparametric   |        X.Xs  |    X.Xs  |    X.Xs  |        X min |         X min |        X min
copula          |        X.Xs  |    X.Xs  |    X.Xs  |        X min |         X min |        X min
empirical       |        X.Xs  |    X.Xs  |    X.Xs  |        X min |         X min |        X min
ALL GENERATORS  |           -  |    X.Xs  |    X.Xs  |        X min |         X min |        X min

    Scaled columns use whichever of seq/par is faster on this machine.

=== COMBINED (Power + CI) Full Grid, All Generators ===
    "Best config" = for each of power and CI, use whichever of seq/par was faster on this machine, then sum.

Tier      | Best config (this machine) | Est. 8-core | Est. 16-core
Measured  |                      X min |           - |            -
+/-0.01   |                      X min |       X min |        X min
+/-0.002  |                      X hrs |       X min |        X min
+/-0.001  |                      X hrs |       X hrs |        X hrs

    Power best: seq / par (X.Xs). CI best: seq / par (X.Xs).
    Higher-core scaling (power only, if par wins): est_time = power_par_time * (this_cores / target_cores) / efficiency.
    CI with batch bootstrap: if seq is faster at small params, parallel may still win at high n_reps (see one-off benchmark).
```

### Notes in script output

- "Measured on: X-logical-core machine (Y physical), Windows, Numba enabled."
- "Power parallel scaling to N cores: (sequential_time / N) / 0.5. This assumes joblib overhead is amortized on larger machines where per-scenario work >> joblib spawn cost."
- "CI with batch bootstrap: parallel is slower than sequential on this machine (joblib overhead ~89% of gap, thread oversubscription ~11%). On higher-core machines with very high n_reps, the per-scenario time may be large enough for parallel to help; benchmark on your machine."

**Caveat (power and n_jobs=-1):** Whether power full-grid benefits from `n_jobs=-1` on this machine is currently undocumented (BENCHMARKING_FINDINGS.md covers only CI). Running this benchmark will reveal it: we measure both sequential and parallel for full-grid power and report both in the COMBINED table. If parallel is slower than sequential for power (as it is for CI batch), use sequential for power on this machine; the script output should then recommend sequential for both power and CI when that is the case.

## One-off benchmark: CI batch bootstrap at high load (for parallel scaling)

**Purpose:** Run CI with batch bootstrap at parameters large enough that per-scenario time is ~20–40s, so joblib overhead is amortized and parallel (n_jobs=-1) can beat sequential. That gives a measured parallel baseline to scale to other core counts (e.g. 8, 16).

**No script changes:** Use the parameters below and call [confidence_interval_calculator.run_all_ci_scenarios](confidence_interval_calculator.py) from a one-off Python invocation (REPL or a single-run script).

### Parameters to use

Pick one:


| Option              | n_reps | n_boot | Work multiple vs 200×1000 | Expected sequential (4-core) | Use case                                   |
| ------------------- | ------ | ------ | ------------------------- | ---------------------------- | ------------------------------------------ |
| **A (shorter)**     | 2,000  | 1,000  | 10×                       | ~25–35 min                   | Get parallel-vs-sequential data quickly    |
| **B (±0.002 tier)** | 7,400  | 500    | ~18.5×                    | ~45–60 min                   | Same plus direct timing for ±0.002 CI tier |


Other kwargs: `generator="nonparametric"`, `seed=42`, `calibration_mode="multipoint"`, `batch_bootstrap=True`.

### How to run

From project root. Run **one at a time** (sequential first, then parallel), per [benchmarking rule](.cursor/rules/benchmarking.mdc).

1. **Warmup** (optional but recommended):
  `run_all_ci_scenarios(generator="nonparametric", n_reps=50, n_boot=200, n_jobs=1, seed=42, calibration_mode="multipoint", batch_bootstrap=True)`
2. **Sequential:** Time a single call:
  `run_all_ci_scenarios(generator="nonparametric", n_reps=2000, n_boot=1000, n_jobs=1, seed=42, calibration_mode="multipoint", batch_bootstrap=True)`  
   (Use `n_reps=7400`, `n_boot=500` for Option B.)
3. **Parallel:** Time a single call:
  Same as step 2 but `n_jobs=-1`.

Example one-off snippet (Option A). From project root, run with project root in `sys.path` (e.g. paste into REPL after adding project root to `sys.path`):

```python
import time
from confidence_interval_calculator import run_all_ci_scenarios

# Warmup
run_all_ci_scenarios(generator="nonparametric", n_reps=50, n_boot=200, n_jobs=1, seed=42, calibration_mode="multipoint", batch_bootstrap=True)
# Sequential then parallel (Option A: n_reps=2000, n_boot=1000)
kwargs = dict(generator="nonparametric", n_reps=2000, n_boot=1000, seed=42, calibration_mode="multipoint", batch_bootstrap=True)
t0 = time.perf_counter(); run_all_ci_scenarios(n_jobs=1, **kwargs); t_seq = time.perf_counter() - t0
t0 = time.perf_counter(); run_all_ci_scenarios(n_jobs=-1, **kwargs); t_par = time.perf_counter() - t0
print(f"T_seq={t_seq:.1f}s  T_par={t_par:.1f}s  speedup={t_seq/t_par:.2f}x")
```

### What to record and how to scale

- **Record:** T_seq (s), T_par (s), and number of logical cores on the machine (e.g. 4).
- If **T_par < T_seq** (parallel wins): parallel speedup = T_seq / T_par. Estimated time on a machine with C cores ≈ T_par × (N_measured_cores / C) × efficiency; use efficiency 0.5, or estimate from speedup as efficiency ≈ (T_seq/T_par) / N_measured_cores.
- **Example scaling:** On 4 logical cores, if T_par = 600s (10 min). Est. 16-core ≈ 600 × (4/16) × 2 = 300s (5 min), or simply T_par/2 if assuming linear scaling with cores at efficiency 0.5.
- If T_par ≥ T_seq, parallel still doesn't help at this load on this machine; try Option B (larger n_reps) or a machine with more cores.

---

## Key files to reference

- [power_simulation.py](power_simulation.py): `estimate_power`, `min_detectable_rho`, `run_all_scenarios`
- [confidence_interval_calculator.py](confidence_interval_calculator.py): `bootstrap_ci_averaged`, `run_all_ci_scenarios`
- [config.py](config.py): `CASES`, `N_SIMS`, `N_BOOTSTRAP`, `CALIBRATION_MODE`, etc.
- [data_generator.py](data_generator.py): `digitized_available()`, calibration functions
- [benchmarks/benchmark_power.py](benchmarks/benchmark_power.py): existing pattern to follow
- [docs/BENCHMARKING_FINDINGS.md](docs/BENCHMARKING_FINDINGS.md): joblib decomposition data
