# Benchmarking Findings — CI Bootstrap Paths

**Date:** February 2026 (updated with validation and decomposition)  
**Purpose:** Summary of benchmark results and issues discovered for the batch CI bootstrap optimization. Use this as context when updating the README or running future benchmarks.

---

## Summary of Findings

- **Batch path is correct:** Bit-identical to old path when given same data and bootstrap indices (see Validation below).
- **Batch path is much faster sequential:** ~2.5× faster than old path when both run with `n_jobs=1`.
- **Large gap (seq vs parallel for batch):** When using `n_jobs=-1` with the batch path, parallel is slower than sequential (~160s vs ~265s). **~89% of that gap is joblib overhead** (parallel slower even with `NUMBA_NUM_THREADS=1`). **~11% is nested parallelism** (default Numba threads add ~12s on top).
- **Joblib helps the old path, hurts the batch path:** Old path parallel (263s) beats old path sequential (424s). Batch path parallel (265s) loses to batch path sequential (160s). Reason: batch path is so fast per scenario (~1.8s) that joblib’s fixed overhead (spawn, serialization) dominates; old path is slower per scenario (~4.8s) so overhead is amortized.
- **Recommendation:** For the batch path, use **`n_jobs=1`** (sequential). For large runs (e.g. 10k n_reps, 1k n_boot), sequential with default Numba is fine on memory (~3–4 GB peak per scenario on 32 GB machine).

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

**Old benchmarking data** (ranges from earlier runs, run-to-run variance, 4-core, n_reps=200, n_boot=1000):

| Config | Time |
|--------|------|
| Old path, sequential (n_jobs=1) | ~285–323s |
| New path, sequential (n_jobs=1) | ~112–138s |
| Old path, parallel (n_jobs=-1) | ~189–224s |
| New path, parallel (n_jobs=-1) | ~172–268s |

**New benchmarking data** (single measured run, 4-core, n_reps=200, n_boot=1000):

| Config | Time |
|--------|------|
| Old path, sequential (n_jobs=1) | 423.60s |
| New path, sequential (n_jobs=1) | 159.74s |
| Old path, parallel (n_jobs=-1) | 263.07s |
| New path, parallel (n_jobs=-1) | 265.61s |
| New path, parallel Numba=1 (n_jobs=-1) | 253.59s |
| New path, parallel Numba=1 (n_jobs=2) | 297.61s |

**Sequential:** New path is **~2.3–2.7× faster** than old path (ranges: 112–138s vs 285–323s; single run: 160s vs 424s).

**Parallel:** Results varied by run. Old path parallel typically beats old path sequential (e.g. 263s vs 424s, or 189–224s vs 285–323s). New path parallel can be **slower than new path sequential** (e.g. 266s vs 160s); in earlier runs, new path parallel ranged ~172–268s (sometimes slower than new path parallel old path, e.g. 268s vs 224s, possibly before Numba warmup). Use the ranges above to estimate runtimes and variance.

---

## Key Finding: Nested Parallelism Hurts the New Path

**Problem:** When running with `n_jobs=-1` (joblib uses all cores), the new path can be **slower than new path sequential**.

Example: New path sequential 112s vs new path parallel 172s — parallel is 1.5× slower.

**Cause:** Nested parallelism.
- joblib spawns N workers (e.g. 4)
- Each worker runs `_batch_bootstrap_rhos_jit` with Numba's `prange` (default = 4 threads)
- Total: 4 workers × 4 Numba threads = 16 threads on 4 cores → oversubscription, contention

**Why the old path didn't have this:** The old path has 200 short Numba bursts per scenario (each `prange(1000)`). Brief parallel regions with Python work in between. Less sustained overlap between workers. The new path has one long Numba region (200k tasks) per scenario — all workers are fully busy simultaneously.

**Verified:** Running the same workload with default Numba vs `NUMBA_NUM_THREADS=1` (both with `n_jobs=-1`) showed default consistently 4–11% slower across three runs. So nested parallelism is a real, reproducible cost.

---

## Decomposition of the Large Gap (New Path Sequential vs Parallel)

Using the measured full-grid run above:

- **New path sequential:** 159.74s  
- **New path parallel (default Numba):** 265.61s  
- **Gap:** ~106s (parallel is ~66% slower)

**Split:**

1. **Nested parallelism:** Parallel with default Numba (265.61s) vs parallel with `NUMBA_NUM_THREADS=1` (253.59s) → **~12s (~11%)**. So only a small fraction of the gap is from Numba oversubscription.

2. **Joblib overhead / no benefit:** Sequential (159.74s) vs parallel with Numba=1 (253.59s) → **~94s (~89%)**. Even with no nested parallelism, parallel is much slower than sequential. So **most of the large gap is joblib** (process spawn, serialization, scheduling), not Numba.

**Why joblib doesn’t play well with batching:** Per-scenario time is much shorter for the batch path (~1.8s) than for the old path (~4.8s). Joblib has roughly fixed overhead per run (spawn, serializing tasks/results). That overhead is a small fraction of the old path’s work (so parallel helps) and a large fraction of the batch path’s work (so parallel hurts).

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

**Cause:** The batch path allocates `boot_idx_all` of shape `(n_boot, n_reps, n)` = (1000, 200, ~80) ≈ **64 MB per scenario**. At 10k n_reps, 1k n_boot: (1000, 10_000, ~80) ≈ **3.2 GB per scenario**. With `n_jobs=-1`, multiple workers run concurrently; total memory can spike.

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

**TODO:** Run `benchmark_ci_bootstrap.py` and record single vs multipoint calibration timings (single-scenario and 22-scenario); add a summary to this doc and optionally to the README. The README currently has only the per-tie-structure calibration cost (~3s single vs ~9s multipoint); we do not yet have documented end-to-end benchmark estimates for CI runs.

---

## Verification Scripts (validation and isolation)

| Script | Purpose |
|--------|---------|
| `validate_batch_ci_three_steps.py` | (1) Bit-identical test: batch vs old path, same data & bootstrap indices. (2) Sequential Numba=1: batch faster than old. (3) Batch path parallel vs sequential; (3b) old path parallel vs sequential. Optional: power sim vectorized vs scalar. Uses small n_reps/n_boot. |
| `verify_nested_parallelism.py` | Isolates nested parallelism: runs same workload (batch, n_jobs=-1) in two processes — default Numba threads vs `NUMBA_NUM_THREADS=1`. Use `--compare`. Confirmed default 4–11% slower across three runs. |
| `verify_big_gap.py` | Compares batch sequential (default Numba) vs batch parallel (Numba=1) in separate processes. Use `--compare`. Shows whether the “large gap” is joblib vs nested parallelism. |

**Validation result:** Bit-identical test (Step 1 of `validate_batch_ci_three_steps.py`) passes: `max |batch - old| = 0` over bootstrap rho matrix when both paths get the same data and bootstrap index order.

---

## Scaling: Large Runs (e.g. 10k n_reps, 1k n_boot)

**Sequential batch path with default Numba:**

- **Memory:** Dominant allocation is `boot_idx_all` shape `(n_boot, n_reps, n)` → (1000, 10_000, ~80) ≈ **3.2 GB** per scenario. Sequential runs one scenario at a time, so peak ~3–4 GB. Fine on a 32 GB machine.
- **Time:** Work scales with n_reps × n_boot; 10k×1k is 50× more work than 200×1k, so full grid runtime scales accordingly (~2+ hours for 88 scenarios). No efficiency cliff.

**Recommendation:** For large runs, use **sequential (`n_jobs=1`)** with the batch path; no need to worry about memory or slowdown from the algorithm.

---

## Reducing Joblib Overhead (if parallel is desired)

- **Batched tasks:** Run multiple scenarios per joblib task (e.g. 22 scenarios per task → 4 tasks instead of 88). Cuts serialization and scheduling; requires changing how work is passed (e.g. a function that runs a list of scenarios in one worker).
- **Threading backend:** `Parallel(..., backend="threading")` avoids process spawn and pickle. Use `NUMBA_NUM_THREADS=1` to avoid nested parallelism (4 threads × 1 Numba thread). Worth trying; may or may not beat sequential.
- **Pool reuse:** joblib’s loky backend can reuse the executor across runs; helps repeated runs in one session, not a single huge run.

---

## README Update Suggestions

1. **Performance section:** Add that the batch path (`BATCH_CI_BOOTSTRAP=True`) gives ~2.5–2.7× sequential speedup over the per-rep path. Single scenario: ~1.2s vs ~4.5s (200×1000). Full grid (88 scenarios, 200×1000): ~2.5–2.7 min (batch) vs ~7 min (old path).

2. **Parallelization caveat:** For CI with the batch path, **`n_jobs=1` (sequential) is faster than `n_jobs=-1`**. Most of the gap is joblib overhead (parallel slower even with `NUMBA_NUM_THREADS=1`); a smaller part is nested parallelism (joblib + Numba). Recommend sequential for the batch path. Use parallel for the old path if needed (joblib helps the old path).

3. **Memory:** Batch path allocates ~64 MB per scenario at 200×1000; at 10k×1k, ~3.2 GB per scenario. Sequential runs one scenario at a time; 32 GB RAM is sufficient. If running parallel with batch path, consider `n_jobs=4` or lower to avoid worker OOM; full grid with n_jobs=-1 can trigger `TerminatedWorkerError` on memory-constrained systems.

4. **Typical runtimes (4-core, 200×1000):** With batch path, sequential: single scenario ~1.2s; full grid (88 scenarios) ~2.5–3 min. Old path sequential ~7 min; old path parallel ~4.5 min.

5. **Config:** Document `BATCH_CI_BOOTSTRAP` in config (default True). `VECTORIZE_DATA_GENERATION` must be True for batch path.

6. **Validation/verification:** Mention `validate_batch_ci_three_steps.py`, `verify_nested_parallelism.py`, and `verify_big_gap.py` for reproducing findings (see this doc).

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
