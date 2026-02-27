---
name: Batch CI Bootstrap
overview: Restructure bootstrap_ci_averaged in confidence_interval_calculator.py to loop over n_boot instead of n_reps, eliminating the 200-iteration Python rep loop and replacing it with vectorized Spearman calls on (n_reps, n) arrays.
todos:
  - id: add-batch-config
    content: Add BATCH_CI_BOOTSTRAP = True to config.py; thread batch_bootstrap param through bootstrap_ci_averaged and run_all_ci_scenarios
    status: pending
  - id: restructure-ci-averaged
    content: Restructure bootstrap_ci_averaged to pre-generate (n_reps, n) datasets and loop over n_boot
    status: pending
  - id: validate-ci-batch
    content: Validate output matches old code within Monte Carlo noise using same seed
    status: pending
isProject: false
---

# Batch CI Bootstrap

**Prerequisite: Plan 1 (Vectorize Data Generation) must be completed first.** This plan requires `generate_cumulative_aluminum_batch` and the batch y-generators from that plan to pre-generate all `n_reps` datasets as `(n_reps, n)` arrays.

## Goal

Replace the Python `for rep in range(n_reps)` loop in [confidence_interval_calculator.py](../confidence_interval_calculator.py) `bootstrap_ci_averaged` (lines 211-229) with a structure that loops over `n_boot` instead, calling `spearman_rho_2d` on `(n_reps, n)` arrays each iteration. **Optional but default:** batch bootstrap is enabled by default; a config flag allows falling back to the original per-rep loop.

## Optional / Default

- Add `BATCH_CI_BOOTSTRAP = True` (default) to [config.py](../config.py).
- When `True`: use the restructured batch bootstrap (faster).
- When `False`: use the original per-rep loop calling `bootstrap_ci_single` (slower, but identical behavior for debugging or compatibility).
- Thread through `bootstrap_ci_averaged` and `run_all_ci_scenarios` via a `batch_bootstrap` parameter (default from config).

## Why

Currently, `bootstrap_ci_averaged` runs 200 Python-level reps, each calling `bootstrap_ci_single` which itself runs `n_boot` bootstrap resamples. This means the hot path is driven by a Python loop of 200 iterations, even though the inner Numba path (`_bootstrap_rhos_jit`) is fast. The outer loop pays Python function-call overhead 200 times.

Restructuring to loop over `n_boot` (1000 iterations of cheap vectorized calls on small arrays) is faster because each call does meaningful NumPy/Numba work on `(n_reps, n)` = `(200, ~80)` arrays rather than one per rep.

## Restructured bootstrap_ci_averaged (when batch_bootstrap=True)

```
1. Pre-generate x_all (n_reps, n) and y_all (n_reps, n) using batch generators (Plan 1)
2. Compute rho_hats = spearman_rho_2d(x_all, y_all)  -- (n_reps,) in one call
3. Allocate boot_rho_matrix (n_reps, n_boot)
4. for b in range(n_boot):
       boot_idx = boot_rng.integers(0, n, size=(n_reps, n))
       rows = np.arange(n_reps)[:, None]
       x_boot = x_all[rows, boot_idx]   # (n_reps, n)
       y_boot = y_all[rows, boot_idx]   # (n_reps, n)
       boot_rho_matrix[:, b] = spearman_rho_2d(x_boot, y_boot)
5. lowers = np.nanpercentile(boot_rho_matrix, 100*alpha/2, axis=1)    -- (n_reps,)
   uppers = np.nanpercentile(boot_rho_matrix, 100*(1-alpha/2), axis=1) -- (n_reps,)
6. Return means/SDs of lowers, uppers, rho_hats as before
```

## Memory

- `x_all`, `y_all`: `(200, 80) * 8 bytes` = ~256 KB each
- `boot_rho_matrix`: `(200, 1000) * 8 bytes` = 1.6 MB
- Peak per-iteration: `boot_idx` + `x_boot` + `y_boot` = `3 * (200, 80) * 4/8 bytes` = ~384 KB
- Total peak per scenario: ~3–4 MB — safe with `n_jobs=-1` on 4 cores (~12–16 MB total)

## Fallback (when batch_bootstrap=False)

Use the original per-rep loop: `for rep in range(n_reps):` generate (x, y), call `bootstrap_ci_single`, accumulate lowers/uppers. This path is unchanged from current code.

## `bootstrap_ci_single` and `bootstrap_ci_simulated`

These functions are unchanged. `bootstrap_ci_single` is retained for diagnostic use and for the fallback path when `batch_bootstrap=False`. `bootstrap_ci_simulated` is unchanged. The restructuring only affects `bootstrap_ci_averaged` when `batch_bootstrap=True`.

## Files changed

- [config.py](../config.py) — add `BATCH_CI_BOOTSTRAP = True`
- [confidence_interval_calculator.py](../confidence_interval_calculator.py) — restructure `bootstrap_ci_averaged` with `batch_bootstrap` branch (default True); no changes to other functions

## Validation

- Run one scenario with old vs new code using same seed; confirm `ci_lower`, `ci_upper`, `mean_rho_hat` match to within Monte Carlo noise
- Run `run_all_ci_scenarios` on a small subset and confirm output dict structure is unchanged
