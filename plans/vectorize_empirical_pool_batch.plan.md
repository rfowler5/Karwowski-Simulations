---
name: Vectorize empirical pool batch
overview: Replace the Python loop that builds one empirical pool per rep in generate_y_empirical_batch with a vectorized implementation that produces (n_reps, n) in a few NumPy/RNG calls. Optional future optimization; no commitment to implement.
todos: []
isProject: false
---

# Vectorize empirical pool batch construction

**Status:** Optional / deferred. Documented so the approach is not forgotten.

**Current code:** [data_generator.py](data_generator.py) — `generate_y_empirical_batch` (lines 282–297) builds the pool batch with:

```python
pool_batch = np.array([build_empirical_pool(n, rng) for _ in range(n_reps)])
```

Each call to `build_empirical_pool(n, rng)` (lines 236–264) does small RNG draws and concatenates; the loop has Python overhead and a final copy into the array.

---

## Goal

Add a vectorized path that produces the same `(n_reps, n)` array using bulk `rng.choice(..., size=(n_reps, k))` and array ops, so we avoid the per-rep function call and list construction.

---

## Design

**New helper:** `_build_empirical_pool_batch(n, n_reps, rng)` → `(n_reps, n)` float array. Same semantics as building `n_reps` pools with `build_empirical_pool(n, rng)` in order (RNG stream must advance identically for reproducibility).

**Per-n logic:**

| n  | Current per-rep | Vectorized approach |
|----|------------------|----------------------|
| 73 | B_AL71 (71) + choice(B_AL71, 2) | Tile B_AL71 to (n_reps, 71); `rng.choice(B_AL71, size=(n_reps, 2), replace=True)`; hstack → (n_reps, 73). |
| 81 | H_AL71 (71) + choice(H_AL71, 10) | Tile H_AL71 to (n_reps, 71); `rng.choice(H_AL71, size=(n_reps, 10), replace=True)`; hstack → (n_reps, 81). |
| 82 | pool_81 + H_AL_OUTLIER | Build (n_reps, 81) as above; `np.column_stack([pool_81, np.full(n_reps, H_AL_OUTLIER)])` → (n_reps, 82). |
| 80 | pool_73 + outliers (7) | Build (n_reps, 73) as above. Add **batch outlier** helper: 2 known + 5 log-uniform per row → (n_reps, 7). Hstack → (n_reps, 80). |

**Batch outlier helper:** Either extend `generate_b_al_outliers` with a `size=(n_reps, 5)`-style draw (and broadcast the 2 known values), or add `_generate_b_al_outliers_batch(n_reps, rng)` that returns `(n_reps, 7)` using `rng.uniform(log_low, log_high, (n_reps, 5))` and concatenation with `known_b_al_outliers`.

**Integration:** In `generate_y_empirical_batch`, replace the list comprehension with:

```python
pool_batch = _build_empirical_pool_batch(n, n_reps, rng)
```

Keep `build_empirical_pool` unchanged for single-rep callers (`generate_y_empirical`).

---

## Testing

- **Equivalence:** For a fixed seed, compare `np.array([build_empirical_pool(n, rng) for _ in range(n_reps)])` (with a fresh rng per test) to `_build_empirical_pool_batch(n, n_reps, rng)`. RNG advancement will differ (one call vs many), so exact array equality is not required; instead check **distribution** (e.g. same shape, same unique-value sets per row for n=73/81, or a short Monte Carlo check that rank statistics match).
- **Regression:** Run existing empirical tests (e.g. `test_empirical_generator.py`, `test_empirical_invalid_n.py`, and any calibration/power tests that use empirical) to ensure no behavioral change beyond possible RNG stream differences. If the test suite pins exact values, consider running with vectorized path and updating expected values only if intentional.
- **Performance (optional):** Time `generate_y_empirical_batch(x_batch, ...)` for n_reps in [200, 10_000] with vectorized vs loop; expect roughly 5–15× faster for the pool step and a modest percent gain on the full batch path.

---

## Scope and notes

- **Files to touch:** [data_generator.py](data_generator.py) only (add `_build_empirical_pool_batch`, optionally `_generate_b_al_outliers_batch` or extended `generate_b_al_outliers`, and the one-line change in `generate_y_empirical_batch`).
- **Memory:** Safe for n_reps up to 50k+ (pool batch and temporaries stay in the low hundreds of MB). No Numba: Generator API and branchy logic make Numba a poor fit; vectorization alone is sufficient.
- **When to do it:** If profiling shows the empirical batch pool loop as a noticeable cost (e.g. large n_sims power runs), or for code clarity. Otherwise the current loop is acceptable as in [plans/empirical_y_fixed_71_fix.plan.md](empirical_y_fixed_71_fix.plan.md).
