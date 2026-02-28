# Benchmarking Findings — CI Bootstrap Paths

**Date:** February 2026  
**Purpose:** Summary of benchmark results and issues discovered for the batch CI bootstrap optimization. Use this as context when updating the README or running future benchmarks.

---

## What Was Benchmarked

### Two CI Paths (both use vectorized data generation)

| Path | Config | Description |
|------|--------|-------------|
| **Old path** | `batch_bootstrap=False` | Per-rep loop: 200 calls to `bootstrap_ci_single`, each using `_bootstrap_rhos_jit` (Numba, 1000 tasks per call) |
| **New path** | `batch_bootstrap=True` | Batch path: 1 call to `_batch_bootstrap_rhos_jit` (Numba, 200×1000 = 200k tasks per scenario) |

Both paths use:
- `generate_cumulative_aluminum_batch`, `generate_y_nonparametric_batch` (or copula/linear batch)
- Same calibration (single-point or multipoint)
- Same `n_reps`, `n_boot`, seed

**The only difference:** Bootstrap loop structure — per-rep vs batch.

---

## Benchmark Results (4-core machine, n_reps=200, n_boot=1000, single-point calibration)

### Single scenario (Case 3, k=4, even)

| Path | Time | Speedup |
|------|------|---------|
| Old (per-rep) | ~4.5s | — |
| New (batch) | ~1.2s | **3.78×** |

### Full grid (88 scenarios)

| Config | Time |
|--------|------|
| Old path, sequential (n_jobs=1) | ~285–323s |
| New path, sequential (n_jobs=1) | ~112–138s |
| Old path, parallel (n_jobs=-1) | ~189–224s |
| New path, parallel (n_jobs=-1) | ~172–268s |

**Sequential:** New path is **~2.3–2.6× faster** than old path.

**Parallel:** Results varied by run. In some runs, new path parallel beat old path parallel (~172s vs ~197s). In earlier runs (before Numba warmup?), new path parallel was slower than old path parallel (~268s vs ~224s).

---

## Key Finding: Nested Parallelism Hurts the New Path

**Problem:** When running with `n_jobs=-1` (joblib uses all cores), the new path can be **slower than new path sequential**.

Example: New path sequential 112s vs new path parallel 172s — parallel is 1.5× slower.

**Cause:** Nested parallelism.
- joblib spawns N workers (e.g. 4)
- Each worker runs `_batch_bootstrap_rhos_jit` with Numba's `prange` (default = 4 threads)
- Total: 4 workers × 4 Numba threads = 16 threads on 4 cores → oversubscription, contention

**Why the old path didn't have this:** The old path has 200 short Numba bursts per scenario (each `prange(1000)`). Brief parallel regions with Python work in between. Less sustained overlap between workers. The new path has one long Numba region (200k tasks) per scenario — all workers are fully busy simultaneously.

---

## Mitigations Tested

| Mitigation | Result |
|------------|--------|
| `NUMBA_NUM_THREADS=1` | Slight improvement in some runs (175s vs 172s with default). Reduces oversubscription but joblib spawn overhead remains. |
| `n_jobs=2` | Slower than n_jobs=-1 (201s vs 175s) — fewer workers, less parallelism. |
| Warmup (`warm_up_numba.py` or inline) | Recommended. First run pays Numba JIT compile; subsequent runs use cache. |

**Recommendation for CI runs:** Use **`n_jobs=1`** (sequential) for the batch path — it is the fastest configuration (~112s for full grid). The new path's efficiency comes from eliminating Python loop overhead; adding joblib parallelism introduces overhead that can outweigh gains.

---

## Memory Issues

**Symptom:** `TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault or by an excessive memory usage causing the Operating System to kill the worker.`

**Cause:** The batch path allocates `boot_idx_all` of shape `(n_boot, n_reps, n)` = (1000, 200, ~80) ≈ **64 MB per scenario**. With `n_jobs=-1`, multiple workers run concurrently; total memory can spike.

**Mitigations:**
1. Use `n_jobs=4` or `n_jobs=2` instead of `-1` to limit concurrent workers
2. Run parallel benchmarks one at a time (don't run old and new path parallel back-to-back if memory is tight)
3. Reduce `n_reps` or `n_boot` for benchmarking (e.g. n_reps=100, n_boot=500)
4. Use sequential (`n_jobs=1`) for the batch path — avoids worker memory multiplication

**Killing stuck workers:**
```powershell
Get-Process python* | Stop-Process -Force
```

---

## Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `benchmark_batch_vs_no_batch.py` | Single-scenario comparison: old vs new path |
| `benchmark_full_grid.py` | Full 88-scenario grid: sequential and parallel for both paths |
| `benchmark_ci_bootstrap.py` | Multipoint vs single-point calibration (uses batch path) |

---

## README Update Suggestions

1. **Performance section:** Add that the batch path (`BATCH_CI_BOOTSTRAP=True`) gives ~2.5× sequential speedup over the per-rep path. Single scenario: ~1.2s vs ~4.5s (200×1000).

2. **Parallelization caveat:** Note that for CI with the batch path, `n_jobs=1` (sequential) is often faster than `n_jobs=-1` due to nested parallelism (joblib + Numba). Recommend sequential for batch path; use parallel for the old path if needed.

3. **Memory:** If running parallel CI with batch path, consider `n_jobs=4` or lower to avoid worker OOM. Full grid with n_jobs=-1 can trigger `TerminatedWorkerError` on memory-constrained systems.

4. **Typical runtimes (update):** With batch path, sequential:
   - Single scenario (200×1000): ~1.2s
   - Full grid (88 scenarios): ~2 min (vs ~5 min old path, ~8 min previously documented)

5. **Config:** Document `BATCH_CI_BOOTSTRAP` in config (default True). `VECTORIZE_DATA_GENERATION` must be True for batch path.

---

## Config Reference

```python
# config.py
VECTORIZE_DATA_GENERATION = True   # Required for batch path
BATCH_CI_BOOTSTRAP = True         # Use batch bootstrap (default)
USE_NUMBA = True                  # Numba JIT for bootstrap loops
```

---

## File Roles (relevant to benchmarking)

| File | Role |
|------|------|
| `spearman_helpers.py` | `_bootstrap_rhos_jit` (old path), `_batch_bootstrap_rhos_jit` (new path) |
| `confidence_interval_calculator.py` | `bootstrap_ci_averaged` — branches on `batch_bootstrap` and `vectorize` |
| `config.py` | `BATCH_CI_BOOTSTRAP`, `VECTORIZE_DATA_GENERATION`, `USE_NUMBA` |
