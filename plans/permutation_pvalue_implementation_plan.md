---
name: Permutation p-value implementation
overview: "Implement permutation-based (Monte Carlo or precomputed null) p-values for the power simulation: precomputed null cache for non-empirical generators, adaptive n_perm (1k/2k), Monte Carlo fallback with full batched Numba parallelism and adaptive chunking by n_sims, plus README and benchmark/test updates."
todos: []
isProject: false
---

# Permutation p-value implementation plan

## Scope and behavior summary

- **Non-empirical (nonparametric, copula, linear):** Use precomputed null when available (cache keyed by scenario); else run per-dataset Monte Carlo with adaptive n_perm (1k when n_sims high, 2k when lower). No precomputed null for empirical (y-ties vary per dataset).
- **Empirical:** Always use per-dataset Monte Carlo (adaptive n_perm).
- **Precomputed null:** One null distribution per (n, tie structure); stored null is a large fixed sample (e.g. n_pre = 50_000). Cache does **not** key by n_perm; n_perm only affects the **Monte Carlo** path. So "check cache for which p-value" is simply: do we have a cached null for this (n, tie structure)? If yes, use it (p from count against that null). If no (or empirical), use Monte Carlo with chosen n_perm.
- **Adaptive n_perm:** e.g. n_perm = 1000 if n_sims >= 5000, else n_perm = 2000. Config constants for threshold and both values.
- **Monte Carlo path:** Full batched parallelism (one Numba kernel over all (sim, perm) pairs). No early stopping. When n_sims exceeds a threshold (e.g. 15_000), process sims in chunks to cap memory; below the threshold, one full batch.

---

## 1. Config and constants

**File:** [config.py](config.py)

- Add constants, e.g.:
  - `N_PERM_DEFAULT = 1000`, `N_PERM_LOW_SIMS = 2000`
  - `N_SIMS_THRESHOLD_FOR_N_PERM = 5000` (use 1k when n_sims >= this, else 2k)
  - `PVALUE_PRECOMPUTED_N_PRE = 50_000` (number of null rhos per cached null)
  - `N_SIMS_BATCH_THRESHOLD = 15_000` (when n_sims > this, process sims in chunks to cap memory)
  - `N_SIMS_CHUNK_SIZE = 10_000` (chunk size when batching; keeps peak memory ~3 GB for 1k n_perm, n≈80)
- Optional: `USE_PERMUTATION_PVALUE = True` to gate the new behavior (default True for power path).

---

## 2. Precomputed null: cache and build

**Location:** New module or extend [spearman_helpers.py](spearman_helpers.py). Prefer a dedicated module `permutation_pvalue.py` (or `pvalue_null.py`) to keep spearman_helpers focused on correlation/ranking and avoid circular imports (power_simulation → spearman_helpers; if null cache needs get_x_counts, it may need to live next to code that already imports power_asymptotic/data_generator).

**Cache key:** Uniquely identify tie structure so the same null is reused.

- Standard scenarios: `(n, n_distinct, distribution_type, all_distinct)`.
- Custom freq_dict: `(n, "custom", tuple(x_counts))` (all_distinct=False) or `(n, all_distinct=True)`.
- Use `get_x_counts(n, n_distinct, distribution_type, all_distinct, freq_dict=freq_dict)` from [power_asymptotic.get_x_counts](power_asymptotic.py) to get `x_counts`; then key by `(n, all_distinct, tuple(x_counts))` so custom and standard are consistent. So cache key = `(n, all_distinct, tuple(x_counts))`.

**Build step (precompute if not in cache):**

1. From `x_counts` build the **x-rank vector** of length n (midranks): for group sizes c_1, c_2, ..., assign midranks (1-based) per group, e.g. group 1: (1 + c_1)/2, group 2: (c_1 + 1 + c_1 + c_2)/2, etc.
2. Draw `n_pre` (e.g. 50_000) random permutations of `np.arange(1, n+1, dtype=float)` (or integer ranks then cast). For each permutation, compute Pearson correlation between the fixed x_rank vector and the permuted ranks (same as Spearman null).
3. Store the array of null rhos (length n_pre). For fast lookup, store **sorted** absolute values and use binary search to compute p = (1 + count(|null_rho| >= |rho_obs|)) / (1 + n_pre) per sim.
4. Cache: global dict keyed by `(n, all_distinct, tuple(x_counts))`, value = 1d array of null rhos (or sorted |null_rho|). Thread-safety: same as calibration (single process; if later parallel, document or add a lock).

**API:** e.g. `get_precomputed_null(n, all_distinct, x_counts, n_pre, rng)` → returns 1d array (or a small wrapper that exposes `compute_pvalues(rho_obs_array)`). If cache hit, return cached; else build, store, return.

**Where to place:** Put cache and build logic in [spearman_helpers.py](spearman_helpers.py) or a new `permutation_pvalue.py`. If new module: it can import `spearman_helpers` (spearman_rho_2d, _tied_rank, _pearson_on_ranks_1d), and `power_asymptotic.get_x_counts`; `power_simulation` then imports this module and calls a single entry point that chooses precomputed vs Monte Carlo. Recommendation: **new module `permutation_pvalue.py`** to keep spearman_helpers free of cache/get_x_counts and to group all permutation-pvalue logic (precompute + batched Monte Carlo).

---

## 3. Adaptive n_perm and when to use precomputed vs Monte Carlo

**Rule:**

- **Empirical:** Always Monte Carlo (no precomputed null). Choose n_perm = 1000 if n_sims >= threshold, else 2000.
- **Non-empirical:** Try precomputed: compute cache key from (n, all_distinct, x_counts). If cache hit, use precomputed null to get p-values (no n_perm). If cache miss (e.g. custom freq not yet cached), fall back to Monte Carlo with same adaptive n_perm.

So the "cache check" is: key = (n, all_distinct, tuple(x_counts)); if key in _NULL_CACHE then use cached null; else run Monte Carlo with n_perm = f(n_sims). No need to store "which n_perm" in the cache because the cached object is the null sample (n_pre), not a per-dataset permutation count.

**User notification when precomputed null is not available:** When the code falls back to Monte Carlo because of a cache miss (non-empirical scenario but no cached null for this tie structure), the user must be informed so they can precompute later or stop and precompute now. Implement:

- **Warning on cache miss:** When using Monte Carlo fallback for a non-empirical generator (i.e. cache miss), emit a `warnings.warn` (or log) stating that the precomputed null was not available for this scenario, so per-dataset Monte Carlo is being used; suggest precomputing the null for faster future runs (and optionally point to how: see below).
- **Precompute API / script:** Provide a way to warm the precomputed null cache (e.g. `warm_precomputed_null_cache(cases=None, n_distinct_values=None, dist_types=None, freq_dict=None)` in `permutation_pvalue.py` that builds and caches nulls for all requested (n, tie structure) combinations). Document in README: users can call this (or run a small script) before a long run so that cache misses do not occur. Optionally document that users may stop the current run, call the warm function, then re-run.

---

## 4. Monte Carlo path: full batched parallelism and adaptive chunking

**Requirements:** One Numba kernel over all (sim, perm) pairs for maximum parallelism. No early stopping. When n_sims is large, process sims in chunks to cap memory.

**Design:**

- **Full batch:** Pre-generate permutation indices for all (sim, perm) pairs in the current batch. One Numba function computes Spearman rho for each (sim, perm): same pattern as [spearman_helpers._batch_bootstrap_rhos_jit](spearman_helpers.py) but with permutation indices (reorder y by perm) instead of bootstrap indices. Inputs: `x_all` (n_sims, n), `y_all` (n_sims, n), `perm_idx_all` (n_perm, n_sims, n) int32. Output: `perm_rhos` (n_sims, n_perm). Then in Python: rhos_obs = spearman_rho_2d(x_all, y_all); for each sim i, count = sum(|perm_rhos[i]| >= |rhos_obs[i]|); p[i] = (1 + count) / (1 + n_perm); reject[i] = (p[i] < alpha).
- **Adaptive chunking:** If n_sims <= N_SIMS_BATCH_THRESHOLD (e.g. 15_000), do one batch: perm_idx_all shape (n_perm, n_sims, n), one JIT call, compute all p-values. If n_sims > N_SIMS_BATCH_THRESHOLD, split sims into chunks of N_SIMS_CHUNK_SIZE (e.g. 10_000). For each chunk: extract (x_chunk, y_chunk), generate perm_idx for (n_perm, chunk_size, n), run JIT, compute p/reject for that chunk. Concatenate results. Same RNG seeding per chunk (e.g. seed + chunk_start_index) so results are reproducible.
- **Numba:** Add `_batch_permutation_rhos_jit(x_all, y_all, perm_idx_all)` in [spearman_helpers.py](spearman_helpers.py): same structure as _batch_bootstrap_rhos_jit — parallel over flat index (n_sims * n_perm), each iteration computes one (sim, perm) rho by applying perm to y row, ranking, Pearson on ranks. Reuse _tied_rank and _pearson_on_ranks_1d.
- **Reproducibility:** When chunking, use deterministic seeds per chunk (e.g. main seed + chunk index) so that full run reproduces.

**Implementation sketch:**

- **spearman_helpers:** `_batch_permutation_rhos_jit(x_all, y_all, perm_idx_all)` → (n_sims, n_perm) float64. perm_idx_all has shape (n_perm, n_sims, n); for each (b, rep) we permute y_all[rep] by perm_idx_all[b, rep, :] and compute rho with x_all[rep]. Same prange pattern as bootstrap.
- **permutation_pvalue.py:** `pvalues_mc(x_all, y_all, n_perm, alpha, rng, n_sims_batch_threshold, n_sims_chunk_size)`:  
  - rhos_obs = spearman_rho_2d(x_all, y_all).  
  - If n_sims <= n_sims_batch_threshold: generate perm_idx_all (n_perm, n_sims, n), call _batch_permutation_rhos_jit, then p = (1 + count) / (1 + n_perm) per sim.  
  - Else: loop over chunks of n_sims_chunk_size; for each chunk generate perm_idx for (n_perm, chunk_size, n), call JIT, compute p for chunk; concatenate.  
  - Return (rhos_obs, pvals) or (reject,) for power.
- **Power use:** estimate_power calls pvalues_mc and uses power = mean(reject).

---

## 5. Wiring in power_simulation.estimate_power

**File:** [power_simulation.py](power_simulation.py)

- After generating `x_all`, `y_all` (unchanged):
  - Get scenario key: `x_counts = get_x_counts(n, n_distinct, distribution_type, all_distinct, freq_dict)` (from power_asymptotic).
  - If generator != "empirical": try precomputed null: call `get_precomputed_null(n, all_distinct, x_counts, n_pre, rng)`. If returned null array is not None (cache hit or just built), compute rhos_obs = spearman_rho_2d(x_all, y_all), then for each row i p[i] = (1 + count(|null_rho| >= |rhos_obs[i]|)) / (1 + n_pre). Power = mean(p < alpha).
  - Else (empirical or cache miss): if generator != "empirical", emit a warning that precomputed null was not available and per-dataset Monte Carlo is being used; suggest precomputing (e.g. `warm_precomputed_null_cache`) for faster future runs. Choose n_perm from n_sims (if n_sims >= N_SIMS_THRESHOLD_FOR_N_PERM then n_perm = N_PERM_DEFAULT else N_PERM_LOW_SIMS). Call `pvalues_mc(x_all, y_all, n_perm, alpha, rng)` (or equivalent that returns reject 0/1). Power = mean(reject).
- Remove or gate the current `spearman_rho_pvalue_2d` call for the power path when the new permutation path is enabled (e.g. always use permutation path for tied scenarios; for all_distinct could keep t-based or use same permutation for consistency—specify in plan: use permutation for all when USE_PERMUTATION_PVALUE, so one code path).

---

## 6. Backward compatibility and t-based p-value

- Keep [spearman_helpers.spearman_rho_pvalue_2d](spearman_helpers.py) for callers that still want t-based (e.g. tests, or if someone disables permutation). The power simulation will call the new permutation/precomputed path only; no change to the function signature of spearman_rho_pvalue_2d.
- Optional: add a small note in docstring of spearman_rho_pvalue_2d that for tied data permutation-based p-value is preferred (and used in power_simulation).

---

## 7. README updates

**File:** [README.md](README.md)

Add or extend a subsection (e.g. under "Monte Carlo precision and runtime" or new "P-value and power") that covers:

- **P-value method:** Power simulation uses permutation-based p-values (precomputed null for non-empirical when available; else per-dataset Monte Carlo with adaptive n_perm). Empirical always uses per-dataset Monte Carlo. Rationale: t-approximation is unreliable with heavy ties. When precomputed null is not available (cache miss), a warning is emitted and the user can precompute later (e.g. `warm_precomputed_null_cache(...)`) or stop and precompute now.
- **Precision vs n_sims and n_cal:** Summary as below, plus **formulas** so users can compute their own errors for any n_sims, n_cal, or target half-width.
- **Batching:** Monte Carlo path uses full batched Numba over all (sim, perm) pairs; when n_sims > threshold (e.g. 15k), sims are processed in chunks to cap memory.

**Memory (Monte Carlo p-value batching) — include in README:**

- **What counts:** When using the Monte Carlo p-value path (no precomputed null), peak memory is dominated by: (1) permutation indices `perm_idx_all` of shape `(n_perm, n_sims, n)` int32 → 4 × n_perm × n_sims × n bytes; (2) permutation rhos `(n_sims, n_perm)` float64 → 8 × n_sims × n_perm bytes. Data `x_all`, `y_all` are 2 × n_sims × n × 8 bytes and small by comparison.
- **Formula:** Extra memory (on top of data) ≈ 4 × n_perm × n_sims × n + 8 × n_sims × n_perm. For n ≈ 80: ≈ n_sims × n_perm × (320 + 8) bytes ≈ 328 × n_sims × n_perm bytes (indices dominate).
- **Chunking:** When n_sims > config threshold (e.g. 15_000), sims are processed in chunks (e.g. 10_000). Peak memory is then for one chunk: e.g. 10k sims × 1k n_perm × n ≈ 80 gives ~3.2 GB for indices + ~80 MB for rhos, so ~3.3 GB per chunk instead of a single ~16 GB batch for 50k × 1k.
- **Example table (n ≈ 80):** Include a small table so users can plan:
  - n_sims=10k, n_perm=1k: extra ~400 MB (full batch).
  - n_sims=10k, n_perm=2k: extra ~800 MB.
  - n_sims=20k, n_perm=1k: extra ~800 MB; n_perm=2k: extra ~1.6 GB.
  - n_sims=50k, n_perm=1k: full batch ~16 GB → use chunking (e.g. chunk 10k) so peak ~3–4 GB.
- State the config constants (N_SIMS_BATCH_THRESHOLD, N_SIMS_CHUNK_SIZE) so users know when chunking kicks in and can adjust if needed.

**Formulas to include in README (for user-computed errors):**

- **Power estimate (binomial):**  
  \(\hat{p}\) = proportion of sims with p < α.  
  \(\mathrm{SE}(\hat{p}) = \sqrt{\hat{p}(1-\hat{p}) / n_{\mathrm{sims}}}\).  
  For power near 0.80, \(\mathrm{SE}(\hat{p}) \approx 0.4 / \sqrt{n_{\mathrm{sims}}}\).
- **Bisection contribution to SE(min ρ):**  
  The estimated minimum detectable ρ has variance from the bisection step. Approximate:  
  \(\mathrm{SE}_{\mathrm{bisection}}(\rho) \approx c / \sqrt{n_{\mathrm{sims}}}\)  
  where c depends on the slope of the power curve (often c ≈ 0.15–0.2). So \(\mathrm{SE}_{\mathrm{bisection}} \approx 0.16 / \sqrt{n_{\mathrm{sims}}}\) as a rule of thumb.
- **Calibration contribution:**  
  Calibration uncertainty in the mean realised ρ scales as \(1/\sqrt{n_{\mathrm{cal}}}\); this propagates to the min detectable ρ. Approximate:  
  \(\mathrm{SE}_{\mathrm{cal}}(\rho) \propto 1 / \sqrt{n_{\mathrm{cal}}}\)  
  (coefficient depends on scenario; users can treat it as a constant for a given tie structure).
- **Combined uncertainty (independent):**  
  \(\mathrm{SE}_{\mathrm{total}}(\rho) \approx \sqrt{ \mathrm{SE}_{\mathrm{bisection}}^2 + \mathrm{SE}_{\mathrm{cal}}^2 }\).
- **95% CI half-width:**  
  Half-width ≈ 1.96 × \(\mathrm{SE}_{\mathrm{total}}(\rho)\). So for target half-width w, aim for \(\mathrm{SE}_{\mathrm{total}} \leq w / 1.96\).
- **Rounding:**  
  - For the reported value to round "safely" to a given band (e.g. 0.35), the whole 95% CI should lie inside that band. So e.g. for rounding to 0.35, need estimate − (1.96×SE) ≥ 0.345, i.e. half-width ≤ 0.001 at the boundary.  
  - Wrong-rounding probability at a boundary (e.g. true 0.346): with half-width w, the estimate is in [0.346−w, 0.346+w]. The fraction of that interval that rounds the "wrong" way (e.g. to 0.34) gives an approximate wrong-rounding probability (or use normality: P(estimate < 0.345) with mean 0.346 and SE = w/1.96).  
  - Overall probability that rounding is correct: average over the distribution of true min ρ of P(round(\(\hat{\rho}\)) = round(\(\rho_{\mathrm{true}}\))); typically ~90–95% when half-width is ±0.002 and true values are not concentrated at boundaries.

Include the **±0.01** and **±0.002** numerical guidance as before (n_sims ≈ 3k, n_cal ≈ 1k for ±0.01; and rounding safety only for 0.347–0.353 etc., wrong-rounding ~15–25% at 0.346, overall ~90–95%), and add a short "Formulas" sub-subsection with the equations above so users can plug in their own n_sims, n_cal, or desired half-width.

---

## 8. Benchmarks (write only, do not run)

**New or extended file:** e.g. [benchmarks/benchmark_permutation_pvalue.py](benchmarks/benchmark_permutation_pvalue.py)

- **Benchmark 1 – Precomputed null build and lookup:** For one scenario (e.g. n=73, k=4, even), time: (a) building precomputed null (first time, cache miss), (b) lookup + p-value for 10k rhos (cache hit). Report times and memory if easy (e.g. size of cached null).
- **Benchmark 2 – Monte Carlo path (batched):** Time `estimate_power(..., n_sims=1000, generator="nonparametric")` with permutation p-value (no precomputed null), and with generator="empirical" for same n_sims. Optionally time with n_sims=20k to measure chunked path. Compare to current t-based timing if a flag exists.
- **Benchmark 3 – Full grid:** Time run_all_scenarios with permutation p-value (precomputed used where applicable), n_sims=500, report total time and per-scenario breakdown if feasible.

Ensure benchmark scripts follow the pattern of existing [benchmarks/benchmark_power.py](benchmarks/benchmark_power.py) (warmup, print timings, no assertions). Do not run them in the plan.

---

## 9. Tests and smoke tests

- **Smoke test:** Existing smoke test or a small test that runs `estimate_power` for one scenario with small n_sims (e.g. 50) and permutation p-value enabled, and checks that power is in [0, 1] and that no exception is raised. No need to assert exact power value.
- **test_calibration_accuracy:** User notes it may fail with small n_sims (e.g. 50) because of Monte Carlo noise; that is acceptable. Ensure the test still runs (no crash) with the new p-value path; user will verify manually if it passes. Do not relax the test's n_sims or threshold in the plan unless we add a separate "quick" mode that skips accuracy assertions.
- **Unit test for precomputed null:** Add a test that builds a null for (n, all_distinct=True) and (n, k, even), checks shape and that mean of null rhos is near 0 and SD is in a plausible range. Optionally check that p-value for a fixed rho_obs matches a rough expectation (e.g. rho_obs=0 gives p near 1).
- **Unit test for Monte Carlo p-value:** For one (x, y) with known rho, run pvalues_mc with small n_perm (e.g. 200) and check that returned p is in (0, 1] and that reject agrees with (p < alpha).

---

## 10. File and dependency summary

| File                                            | Changes                                                                                                                                                     |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [config.py](config.py)                          | Add N_PERM_*, N_SIMS_THRESHOLD_*, PVALUE_PRECOMPUTED_N_PRE, N_SIMS_BATCH_THRESHOLD, N_SIMS_CHUNK_SIZE, optionally USE_PERMUTATION_PVALUE.                   |
| New: permutation_pvalue.py                      | get_precomputed_null (cache + build), pvalues_mc (full batch or chunked by n_sims), warm_precomputed_null_cache. Caller emits warning on cache miss.        |
| [spearman_helpers.py](spearman_helpers.py)      | Add _batch_permutation_rhos_jit(x_all, y_all, perm_idx_all) — same pattern as _batch_bootstrap_rhos_jit, parallel over (sim, perm).                        |
| [power_simulation.py](power_simulation.py)      | In estimate_power: get x_counts, branch precomputed vs Monte Carlo, call get_precomputed_null / pvalues_mc, compute power from p or reject.                  |
| [power_asymptotic.py](power_asymptotic.py)      | No change; get_x_counts already exists and is used.                                                                                                        |
| [README.md](README.md)                          | Add subsection on p-value method, ±0.01 / ±0.002 precision, n_sims/n_cal, rounding confidence, memory/batching.                                           |
| New: benchmarks/benchmark_permutation_pvalue.py | Benchmarks for precomputed build/lookup, MC path (batched/chunked), full grid (write only).                                                                 |
| tests/                                          | Smoke test for estimate_power with permutation; unit tests for null build and MC p-value; test_calibration_accuracy left as-is (may fail with small n_sims). |

---

## 11. Order of implementation (suggested)

1. Config constants (including N_SIMS_BATCH_THRESHOLD, N_SIMS_CHUNK_SIZE).
2. spearman_helpers: _batch_permutation_rhos_jit(x_all, y_all, perm_idx_all) — same pattern as _batch_bootstrap_rhos_jit.
3. permutation_pvalue.py: cache key helper, get_precomputed_null (build + cache), pvalues_mc (full batch when n_sims <= threshold, else chunked by n_sims), warm_precomputed_null_cache. Caller (power_simulation) emits warning on cache miss when using Monte Carlo fallback for non-empirical.
4. power_simulation.estimate_power: integrate precomputed/MC branch and remove direct spearman_rho_pvalue_2d for power.
5. Unit tests (null build, MC p-value, smoke).
6. Benchmark script (write).
7. README updates.
