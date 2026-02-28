---
name: Batch CI Bootstrap
overview: Restructure bootstrap_ci_averaged in confidence_interval_calculator.py to loop over n_boot instead of n_reps, eliminating the 200-iteration Python rep loop and replacing it with vectorized Spearman calls on (n_reps, n) arrays.
todos:
  - id: add-batch-config
    content: Add BATCH_CI_BOOTSTRAP = True to config.py; thread batch_bootstrap param through bootstrap_ci_averaged and run_all_ci_scenarios
    status: completed
  - id: restructure-ci-averaged
    content: Restructure bootstrap_ci_averaged to pre-generate (n_reps, n) datasets and loop over n_boot
    status: completed
  - id: validate-ci-batch
    content: Validate output matches old code within Monte Carlo noise using same seed
    status: completed
  - id: benchmark-ci-batch
    content: "Benchmark old vs new path (sequential then parallel) -- STOP here and wait for user prompt"
    status: completed
isProject: false
---

# Batch CI Bootstrap

**Prerequisite: Plan 1 (Vectorize Data Generation) must be completed first.** This plan requires `generate_cumulative_aluminum_batch` and the batch y-generators from that plan to pre-generate all `n_reps` datasets as `(n_reps, n)` arrays.

## Goal

Replace the Python `for rep in range(n_reps)` loop in [confidence_interval_calculator.py](../confidence_interval_calculator.py) `bootstrap_ci_averaged` with a structure that loops over `n_boot` instead, calling `spearman_rho_2d` on `(n_reps, n)` arrays each iteration. **Optional but default:** batch bootstrap is enabled by default; a config flag allows falling back to the original per-rep loop.

The current code has two `for rep in range(n_reps)` loops inside `bootstrap_ci_averaged`: one when `vectorize=True` (pre-generates batch data but still loops per-rep for bootstrap) and one when `vectorize=False` (generates per-rep). This plan replaces both with a single batch path when `batch_bootstrap=True`.

## Optional / Default

- Add `BATCH_CI_BOOTSTRAP = True` (default) to [config.py](../config.py).
- When `True`: use the restructured batch bootstrap (faster). This path requires `VECTORIZE_DATA_GENERATION=True` (the batch generators from Plan 1). If `VECTORIZE_DATA_GENERATION=False` and `batch_bootstrap=True`, fall back to the original per-rep loop and emit a warning.
- When `False`: use the original per-rep loop calling `bootstrap_ci_single` (slower, but identical behavior for debugging or compatibility).
- Thread through `bootstrap_ci_averaged` and `run_all_ci_scenarios` via a `batch_bootstrap` parameter (default from config).

## Why

Currently, `bootstrap_ci_averaged` runs 200 Python-level reps, each calling `bootstrap_ci_single` which itself runs `n_boot` bootstrap resamples. This means the hot path is driven by a Python loop of 200 iterations, even though the inner Numba path (`_bootstrap_rhos_jit`) is fast. The outer loop pays Python function-call overhead 200 times.

Restructuring to loop over `n_boot` (1000 iterations of cheap vectorized calls on small arrays) is faster because each call does meaningful NumPy/Numba work on `(n_reps, n)` = `(200, ~80)` arrays rather than one per rep.

## Restructured bootstrap_ci_averaged (when batch_bootstrap=True)

RNG streams: `data_rng` is used exclusively for steps 1-2 (data generation). `boot_rng` is used exclusively for step 4 (bootstrap index generation). This preserves the existing separation so datasets are independent of `n_boot`.

```
1. Pre-generate x_all (n_reps, n) and y_all (n_reps, n) using batch generators (Plan 1)
   - Uses data_rng for both x and y generation
2. Compute rho_hats = spearman_rho_2d(x_all, y_all)  -- (n_reps,) in one call
3. Allocate boot_rho_matrix (n_reps, n_boot)
4. for b in range(n_boot):                             -- uses boot_rng
       boot_idx = boot_rng.integers(0, n, size=(n_reps, n))
       rows = np.arange(n_reps)[:, None]
       x_boot = x_all[rows, boot_idx]   # (n_reps, n)
       y_boot = y_all[rows, boot_idx]   # (n_reps, n)
       boot_rho_matrix[:, b] = spearman_rho_2d(x_boot, y_boot)
5. lowers = np.nanpercentile(boot_rho_matrix, 100*alpha/2, axis=1)    -- (n_reps,)
   uppers = np.nanpercentile(boot_rho_matrix, 100*(1-alpha/2), axis=1) -- (n_reps,)
6. Return means/SDs of lowers, uppers, rho_hats as before
```

Parameters `n_boot` and `alpha` come from function arguments (defaulting to `config.N_BOOTSTRAP` and `config.ALPHA`), same as the current code.

## Memory

- `x_all`, `y_all`: `(200, 80) * 8 bytes` = ~128 KB each (float64)
- `boot_rho_matrix`: `(200, 1000) * 8 bytes` = 1.6 MB
- Peak per-iteration: `boot_idx (int32)` + `x_boot (float64)` + `y_boot (float64)` = `(200*80*4) + 2*(200*80*8)` = ~320 KB
- Total peak per scenario: ~2.5 MB — safe with `n_jobs=-1` on 4 cores (~10 MB total)

## Fallback (when batch_bootstrap=False)

Use the existing `vectorize=True/False` branching logic unchanged: `for rep in range(n_reps):` call `bootstrap_ci_single`, accumulate lowers/uppers. This path is unchanged from current code.

## `bootstrap_ci_single` and `bootstrap_ci_simulated`

These functions are unchanged. `bootstrap_ci_single` is retained for diagnostic use and for the fallback path when `batch_bootstrap=False`. `bootstrap_ci_simulated` is unchanged. The restructuring only affects `bootstrap_ci_averaged` when `batch_bootstrap=True`.

## Files changed

- [config.py](../config.py) — add `BATCH_CI_BOOTSTRAP = True`
- [confidence_interval_calculator.py](../confidence_interval_calculator.py) — restructure `bootstrap_ci_averaged` with `batch_bootstrap` branch (default True); add `batch_bootstrap` parameter to `run_all_ci_scenarios` and `_ci_one_scenario` to thread through; no changes to other functions

## Validation

- Run one scenario with old vs new code using same seed; confirm `ci_lower`, `ci_upper`, `mean_rho_hat` match to within Monte Carlo noise (~0.01 for n_reps=200)
- Run `run_all_ci_scenarios` on a small subset (e.g. `cases=[3]`, 1 generator) and confirm output dict structure is unchanged

## Benchmarking (do NOT run -- stop after validation)

After validation passes, **stop and wait for the user to prompt you** to run benchmarks. Do not run benchmarks in the same session as implementation/validation. When benchmarking is requested later:

- Follow the benchmarking rule: run each benchmark **one at a time**, never concurrently. Run sequential first, wait for completion, then run parallel.
- Compare old (`batch_bootstrap=False`) vs new (`batch_bootstrap=True`) on the same scenario (e.g. Case 3, k=4, even, nonparametric, n_reps=200, n_boot=1000, seed=42).
- Report wall-clock time for both paths.

---

## Implementation Context

This section provides the detailed context an implementer needs.

### What exists today

**`config.py`** has `VECTORIZE_DATA_GENERATION = True` (line 253). Add `BATCH_CI_BOOTSTRAP = True` on the line after it.

**`confidence_interval_calculator.py`** contains `bootstrap_ci_averaged` (lines 149-270). The function currently:
1. Splits RNG into `data_rng` and `boot_rng` (lines 201-202)
2. Runs calibration for nonparametric/copula (lines 208-218)
3. Has a `vectorize=True` branch (lines 220-241) that pre-generates `x_reps`, `y_reps` as `(n_reps, n)` arrays, then loops `for rep in range(n_reps)` calling `bootstrap_ci_single` per rep
4. Has a `vectorize=False` branch (lines 242-262) that generates and bootstraps per rep
5. Returns a dict with `ci_lower`, `ci_upper`, `ci_lower_sd`, `ci_upper_sd`, `mean_rho_hat` (lines 264-270)

The function signature is:
```python
def bootstrap_ci_averaged(n, n_distinct, distribution_type, rho_s, y_params,
                           generator="copula", n_reps=200, n_boot=None,
                           alpha=None, all_distinct=False, seed=None,
                           freq_dict=None, calibration_mode=None, vectorize=None):
```

**`_ci_one_scenario`** (lines 277-311) calls `bootstrap_ci_averaged`. Its current signature is:
```python
def _ci_one_scenario(case_id, case, k, dt, all_distinct, generator,
                     n_reps, n_boot, alpha, tie_correction_mode, seed,
                     calibration_mode=None):
```

**`run_all_ci_scenarios`** (lines 314-363) builds scenario tuples and unpacks them into `_ci_one_scenario(*args)`. The current tuples are:
```python
scenarios.append((case_id, case, k, dt, False,
                  generator, n_reps, n_boot, alpha,
                  tie_correction_mode, sc_seed,
                  calibration_mode))
```
The last element (`calibration_mode`) is passed as a keyword arg via positional unpacking.

**`run_simulation.py`** (line 160) calls `run_all_ci_scenarios`. Since the new `batch_bootstrap` parameter defaults to `None` (which reads from config), **no changes are needed in `run_simulation.py`** or any other caller.

### Step-by-step implementation

#### 1. `config.py`

Add after line 253 (`VECTORIZE_DATA_GENERATION = True`):

```python
BATCH_CI_BOOTSTRAP = True
```

#### 2. `confidence_interval_calculator.py` — imports

Add `BATCH_CI_BOOTSTRAP` to the import from config (line 46):

```python
from config import (CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES,
                    N_BOOTSTRAP, ALPHA, ASYMPTOTIC_TIE_CORRECTION_MODE,
                    CALIBRATION_MODE, VECTORIZE_DATA_GENERATION,
                    BATCH_CI_BOOTSTRAP)
```

#### 3. `confidence_interval_calculator.py` — `bootstrap_ci_averaged`

Add `batch_bootstrap=None` parameter to the function signature. At the top of the function body, add:

```python
if batch_bootstrap is None:
    batch_bootstrap = BATCH_CI_BOOTSTRAP
```

The new logic is: if `batch_bootstrap=True` and `vectorize=True`, use the new batch path. Otherwise, fall back to the existing code. Structure:

```python
if batch_bootstrap and vectorize:
    # --- NEW BATCH PATH ---
    # Step 1: Pre-generate all datasets using data_rng
    x_all = generate_cumulative_aluminum_batch(
        n_reps, n, n_distinct, distribution_type=distribution_type,
        all_distinct=all_distinct, freq_dict=freq_dict, rng=data_rng)

    if generator == "nonparametric":
        y_all = generate_y_nonparametric_batch(
            x_all, rho_s, y_params, rng=data_rng,
            _calibrated_rho=cal_rho, _ln_params=ln_params)
    elif generator == "copula":
        rho_in = cal_rho if cal_rho is not None else rho_s
        y_all = generate_y_copula_batch(x_all, rho_in, y_params, rng=data_rng)
    else:
        y_all = generate_y_linear_batch(x_all, rho_s, y_params, rng=data_rng)

    # Step 2: Compute rho_hats in one vectorized call
    rho_hats = spearman_rho_2d(x_all, y_all)         # (n_reps,)

    # Step 3-4: Bootstrap loop over n_boot using boot_rng
    boot_rho_matrix = np.empty((n_reps, n_boot))
    rows = np.arange(n_reps)[:, None]
    for b in range(n_boot):
        boot_idx = boot_rng.integers(0, n, size=(n_reps, n))
        x_boot = x_all[rows, boot_idx]
        y_boot = y_all[rows, boot_idx]
        boot_rho_matrix[:, b] = spearman_rho_2d(x_boot, y_boot)

    # Step 5: Percentiles per rep
    lowers = np.nanpercentile(boot_rho_matrix, 100 * alpha / 2, axis=1)
    uppers = np.nanpercentile(boot_rho_matrix, 100 * (1 - alpha / 2), axis=1)

elif vectorize:
    # --- EXISTING VECTORIZE PATH (unchanged) ---
    <current lines 220-241, no changes>

else:
    # --- EXISTING SCALAR PATH (unchanged) ---
    <current lines 242-262, no changes>
```

The data generation calls above are copied from the existing `vectorize=True` block (lines 221-232) with identical arguments. The only difference is what happens after: instead of looping per-rep and calling `bootstrap_ci_single`, we do the batch bootstrap loop.

The existing fallback branches already use `lowers`, `uppers`, `rho_hats` arrays, so the return block at the bottom works for all three paths without any renaming.

**Important**: In the new batch path, `boot_rng.integers` dtype should be `np.intp` (default), not `np.int32`. The current `bootstrap_ci_single` uses `np.int32` for the Numba path, but fancy indexing with `x_all[rows, boot_idx]` works with any integer dtype. Using the default (`np.intp`) is fine.

#### 4. Thread `batch_bootstrap` through callers

**`_ci_one_scenario`** (line 277): Add `batch_bootstrap=None` after the existing `calibration_mode=None` in the signature. Pass it through to `bootstrap_ci_averaged(..., batch_bootstrap=batch_bootstrap)`.

**`run_all_ci_scenarios`** (line 314): Add `batch_bootstrap=None` to its signature. Append `batch_bootstrap` after `calibration_mode` in every `scenarios.append(...)` tuple. Example -- the current tuple:
```python
scenarios.append((case_id, case, k, dt, False,
                  generator, n_reps, n_boot, alpha,
                  tie_correction_mode, sc_seed,
                  calibration_mode))
```
becomes:
```python
scenarios.append((case_id, case, k, dt, False,
                  generator, n_reps, n_boot, alpha,
                  tie_correction_mode, sc_seed,
                  calibration_mode, batch_bootstrap))
```
Do the same for the all-distinct tuple below it.

#### 5. Validation

Run a quick comparison:

```python
from confidence_interval_calculator import bootstrap_ci_averaged
from config import CASES

case = CASES[3]
y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}

# New batch path
result_new = bootstrap_ci_averaged(
    73, 4, "even", -0.13, y_params,
    generator="nonparametric", n_reps=200, n_boot=1000,
    seed=42, batch_bootstrap=True)

# Old per-rep path
result_old = bootstrap_ci_averaged(
    73, 4, "even", -0.13, y_params,
    generator="nonparametric", n_reps=200, n_boot=1000,
    seed=42, batch_bootstrap=False)

for key in result_new:
    diff = abs(result_new[key] - result_old[key])
    print(f"{key}: new={result_new[key]:.4f}  old={result_old[key]:.4f}  diff={diff:.4f}")
```

The `ci_lower`/`ci_upper`/`mean_rho_hat` values will NOT be identical (different resampling order), but should agree within Monte Carlo noise (~0.01-0.02 for n_reps=200). The `ci_lower_sd` and `ci_upper_sd` should be similar (~0.10-0.11).

Also run `run_all_ci_scenarios` on a small subset to confirm the dict structure is unchanged:

```python
from confidence_interval_calculator import run_all_ci_scenarios
results = run_all_ci_scenarios(generator="nonparametric", n_reps=20, n_boot=100, seed=42)
print(results[0].keys())  # should have same keys as before
```

#### 6. Benchmarking — STOP HERE

After validation passes, **stop and tell the user** that validation is complete and benchmarking is ready. Do **not** run benchmarks until the user explicitly asks. When benchmarking is requested, follow the benchmarking rule in `.cursor/rules/benchmarking.mdc`: run each benchmark one at a time, sequential first, then parallel.
