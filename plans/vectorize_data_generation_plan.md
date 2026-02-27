---
name: Vectorize Data Generation
overview: Add batch data generation functions to data_generator.py that produce (n_sims, n) arrays in a single NumPy call, replacing the Python for-loops in estimate_power and bootstrap_ci_averaged.
todos:
  - id: add-vectorize-config
    content: Add VECTORIZE_DATA_GENERATION = True to config.py; thread vectorize param through estimate_power and bootstrap_ci_averaged
    status: pending
  - id: batch-x-gen
    content: Add generate_cumulative_aluminum_batch to data_generator.py
    status: pending
  - id: batch-nonparametric
    content: Add _raw_rank_mix_batch and generate_y_nonparametric_batch to data_generator.py
    status: pending
  - id: batch-copula
    content: Add generate_y_copula_batch to data_generator.py
    status: pending
  - id: batch-linear
    content: Add generate_y_linear_batch to data_generator.py
    status: pending
  - id: update-power-sim
    content: Update estimate_power loop in power_simulation.py to use batch generators
    status: pending
  - id: update-ci-data-gen
    content: Update data generation portion of bootstrap_ci_averaged in confidence_interval_calculator.py
    status: pending
  - id: validate-batch
    content: Validate with validation_test_spearman2d.py and test_simulation_accuracy.py
    status: pending
isProject: false
---

# Vectorize Data Generation

## Goal

Replace the Python for-loops in [power_simulation.py](../power_simulation.py) and [confidence_interval_calculator.py](../confidence_interval_calculator.py) with batch NumPy calls that generate all `n_sims` (or `n_reps`) samples at once. **Optional but default:** vectorization is enabled by default; a config flag allows falling back to the original loop-based code.

## Optional / Default

- Add `VECTORIZE_DATA_GENERATION = True` (default) to [config.py](../config.py).
- When `True`: use batch generators (faster).
- When `False`: use the original per-sample loop (slower, but identical behavior for debugging or compatibility).
- Thread through `estimate_power` and `bootstrap_ci_averaged` via a `vectorize` parameter (default from config).

## Why

The dominant bottleneck is not ranking or Spearman computation (already Numba-accelerated), but the Python loops calling `generate_cumulative_aluminum` and `generate_y_nonparametric` one row at a time. For example, `estimate_power` (lines 75-91 of [power_simulation.py](../power_simulation.py)):

```python
for i in range(n_sims):
    xi = generate_cumulative_aluminum(...)   # template.copy() + shuffle
    yi = generate_y_nonparametric(xi, ...)   # rankdata + rank-mixing
    x_all[i] = xi
    y_all[i] = yi
```

`spearman_rho_pvalue_2d` is already called once on the full `(n_sims, n)` matrix — the loop is the only remaining Python overhead.

## New functions in data_generator.py

### `generate_cumulative_aluminum_batch`

Trivial: tile cached template `n_sims` times, permute each row independently.

```python
def generate_cumulative_aluminum_batch(n_sims, n, n_distinct,
                                        distribution_type=None, freq_dict=None,
                                        all_distinct=False, rng=None):
    template = _get_x_template(n, n_distinct, distribution_type, freq_dict, all_distinct)
    x_batch = np.tile(template, (n_sims, 1))
    return rng.permuted(x_batch, axis=1)   # (n_sims, n)
```

### `_raw_rank_mix_batch`

Core rank-mixing vectorized over rows. Key steps:

- `noise_ranks`: `rng.permuted` on tiled `arange` — `(n_sims, n)`
- Standardize `x_ranks_batch` and `noise_ranks` with `axis=1` mean/std
- Compute `mixed = rho_c * s_x + sqrt(1-rho_c^2) * s_n` — `(n_sims, n)`
- `y_values = rng.lognormal(size=(n_sims, n))`, sorted per row
- Assign: `y_final[rows, np.argsort(mixed, axis=1)] = y_values`

### `generate_y_nonparametric_batch`

Calls `_rank_rows` (already in [spearman_helpers.py](../spearman_helpers.py), Numba-accelerated) on `x_batch`, then `_raw_rank_mix_batch`.

### `generate_y_copula_batch`

All steps in `generate_y_copula` are vectorizable: `_rank_rows` replaces per-row `rankdata`, `norm.ppf`/`norm.cdf` work on 2D arrays, `_lognormal_quantile` works on 2D arrays.

### `generate_y_linear_batch`

All NumPy ops — straightforward extension of current 1D logic to 2D.

## Callers to update

### `power_simulation.py` — `estimate_power` (lines 75-91)

When `vectorize=True` (default), replace loop with batch calls. When `vectorize=False`, keep the original loop.

```python
if vectorize:
    x_all = generate_cumulative_aluminum_batch(n_sims, n, n_distinct, ...)
    if generator == "nonparametric":
        y_all = generate_y_nonparametric_batch(x_all, rho_s, y_params, ...)
    elif generator == "copula":
        y_all = generate_y_copula_batch(x_all, rho_in, y_params, ...)
    else:
        y_all = generate_y_linear_batch(x_all, rho_s, y_params, ...)
else:
    # Original loop (unchanged)
    for i in range(n_sims):
        xi = generate_cumulative_aluminum(...)
        yi = generate_y_nonparametric(xi, ...)  # or copula/linear
        x_all[i] = xi
        y_all[i] = yi
```

### `confidence_interval_calculator.py` — `bootstrap_ci_averaged` (lines 211-229)

When `vectorize=True` (default), the data generation sub-loop becomes a single batch call producing `(n_reps, n)`. When `vectorize=False`, keep the original per-rep loop. The per-rep `bootstrap_ci_single` calls remain for now when not using batch bootstrap (see batch CI bootstrap plan).

## Files changed

- [config.py](../config.py) — add `VECTORIZE_DATA_GENERATION = True`
- [data_generator.py](../data_generator.py) — add 5 new batch functions; existing 1D functions unchanged
- [power_simulation.py](../power_simulation.py) — update `estimate_power` with `vectorize` branch (default True)
- [confidence_interval_calculator.py](../confidence_interval_calculator.py) — update data generation in `bootstrap_ci_averaged` with `vectorize` branch (default True)

## Validation

- Run `validation_test_spearman2d.py` to confirm rho values are unchanged
- Run `test_simulation_accuracy.py` to confirm mean realized rho matches target within tolerance
- Spot-check one scenario: compare `estimate_power` output with old vs new code using same seed
