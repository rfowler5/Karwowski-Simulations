# Benchmarking Findings — CI Bootstrap Paths and Parallelism

**Date:** February–March 2026 (updated March 2026 with warm-cache-injection findings)  
**Purpose:** Summary of benchmark results and issues discovered for the batch CI bootstrap optimization and parallel worker cache injection. Use this as context when updating the README or running future benchmarks.

---

## Summary of Findings

- **Batch path is correct:** Bit-identical to old path when given same data and bootstrap indices (see Validation below).
- **Batch path is much faster sequential:** ~2.5× faster than old path when both run with `n_jobs=1`.
- **Cache injection reverses the parallel recommendation:** When workers had cold caches, parallel was slower than sequential for the batch path. After adding warm-cache injection (`_init_worker_caches` / `_init_ci_worker_caches`), parallel with default Numba beats sequential at production params (~1.22× for CI, ~1.39× for power at n_sims=1000).
- **Old "89% joblib overhead" finding was mostly cold-cache rebuild:** The prior decomposition showed parallel losing by ~106s even with `NUMBA_NUM_THREADS=1`. That gap was dominated by workers rebuilding calibration and null caches from scratch, not by true joblib spawn overhead. With warm caches injected, the gap closes and parallel wins.
- **Default Numba threads wins over Numba=1 for parallel CI:** With warm workers, letting Numba use all threads beats throttling to 1. The old "Numba=1 wins" finding was an artifact of cold-cache overhead masking the nested-parallelism cost.
- **Parallel crossover depends on per-scenario work:** Power parallel is slower than sequential at n_sims=50 (per-scenario ~32ms, joblib overhead dominates), roughly tied at n_sims=500, and wins at n_sims=1000. CI parallel wins at n_reps=200, n_boot=1000. At higher production tiers, the joblib overhead fraction continues to shrink and speedups of 1.7–2× are plausible.
- **Updated recommendation:** Use **`n_jobs=-1`** (parallel, default Numba) for both power and CI at production params. Sequential remains better only at very small params (benchmark-scale n_sims=50).

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

## Benchmark Results

### Single scenario (Case 3, k=4, even)

| Path | Time | Speedup |
|------|------|---------|
| Old (per-rep) | ~4.5s | — |
| New (batch) | ~1.2s | **3.78×** |

### Full grid (88 scenarios) — Historical data (cold workers, single-point calibration)

These were measured before warm-cache injection. Workers rebuilt caches for each scenario independently.

**Old benchmarking ranges** (run-to-run variance, 4-core, n_reps=200, n_boot=1000):

| Config | Time |
|--------|------|
| Old path, sequential (n_jobs=1) | ~285–323s |
| New path, sequential (n_jobs=1) | ~112–138s |
| Old path, parallel (n_jobs=-1) | ~189–224s |
| New path, parallel (n_jobs=-1) | ~172–268s |

**Single measured run** (4-core, n_reps=200, n_boot=1000, cold workers):

| Config | Time |
|--------|------|
| Old path, sequential (n_jobs=1) | 423.60s |
| New path, sequential (n_jobs=1) | 159.74s |
| Old path, parallel (n_jobs=-1) | 263.07s |
| New path, parallel (n_jobs=-1) | 265.61s |
| New path, parallel Numba=1 (n_jobs=-1) | 253.59s |
| New path, parallel Numba=1 (n_jobs=2) | 297.61s |

**Observation (cold workers):** New path parallel ≈ sequential or slower. The apparent "89% joblib overhead" was actually dominated by cold-cache rebuild in workers (each worker rebuilt calibration + null for its share of scenarios before computing anything).

### Full grid (88 scenarios) — Warm-cache injection, multipoint calibration

After adding `_init_worker_caches` / `_init_ci_worker_caches`, the main process warms caches and injects them into each worker via `initializer` / `initargs`. Workers start with full calibration and null caches.

**CI results (n_reps=200, n_boot=1000):**

| Config | Time | vs seq |
|--------|------|--------|
| New path, sequential (n_jobs=1) | 119.11s | — |
| New path, parallel, default Numba (n_jobs=-1) | 97.66s | **1.22× faster** |
| New path, parallel, Numba=1 (n_jobs=-1) | 106.50s | 1.12× faster |
| New path, parallel, Numba=1 (n_jobs=2) | 120.40s | ~equal |

**Key reversal:** Default Numba (4 threads per worker) now beats Numba=1 for CI. With warm caches, actual Numba compute is the bottleneck — Numba threads help rather than hurt. n_jobs=2 with Numba=1 is the worst config because it underutilizes the available cores.

**Power results (nonparametric generator, n_sims=50/500/1000, n_cal=300):**

| n_sims | Seq | Par | Par vs seq |
|--------|-----|-----|------------|
| 50 | 2.80s | 13.56s | 4.8× **slower** |
| 500 | 18.43s | 18.21s | ~tied |
| 1000 | ~38–41s | ~27–29s | **~1.39× faster** |

Per-scenario time at n_sims=50 is ~32ms — too small for joblib overhead to be amortized. Crossover is around n_sims=500. At n_sims=1000, parallel wins clearly. At production tier n_sims values (2,220+), the joblib fraction is even smaller and speedups of 1.7–2× are plausible; benchmark at your actual tier to confirm.

---

## Estimating True Joblib Overhead from the Power Crossover Data

The power benchmark gives three (seq, par) pairs at different n_sims, which allows fitting a simple two-parameter model:

```
par_time = seq_time / n_eff + T_overhead
```

where `n_eff` is the effective worker count (accounting for hyperthreading inefficiency) and `T_overhead` is the fixed per-run joblib cost (spawn, pickle, scheduling).

**Two-point fit using n_sims = 50 and n_sims = 500:**

```
13.56 = 2.80 / n_eff + T_o    ... (1)
18.21 = 18.43 / n_eff + T_o   ... (2)

Subtracting (1) from (2):
4.65 = 15.63 / n_eff  →  n_eff ≈ 3.36

T_o = 13.56 − 2.80 / 3.36 ≈ 12.7s
```

**Check against n_sims = 1000:**

```
predicted: 39.5 / 3.36 + 12.7 ≈ 24.5s
actual:    ~27–29s  (mid ~28s, ~14% off)
```

The n_sims=1000 point overshoots the prediction by ~3.5s, likely due to larger result serialization at higher n_sims, so 12.7s is a lower bound for overhead at very large params.

**Conclusions:**

| Quantity | Estimate |
|----------|----------|
| Fixed joblib overhead per full-grid run | **~12–13s** |
| Effective parallel workers (4 logical / 2 physical HT) | **~3.4×** (close to 4, HT helps for this workload) |

The effective worker count of ~3.4 is notably close to 4, suggesting HT is more useful here than the usual CPU-bound ~1.3–1.5× ceiling. This may reflect memory-latency overlap in the Numba kernels. With n_eff ≈ 3.4, the theoretical maximum parallel speedup on this machine for the power workload is ~3.4×; the overhead of ~12.7s is what prevents achieving it in practice until seq_time is large enough.

**Note:** The CI workload does not fit this model. CI parallel (97.66s) is only 1.22× faster than sequential (119.11s), whereas the power model would predict ~1.7× at these seq times. CI likely has higher per-run overhead due to larger data volumes being serialized (bootstrap rho matrices) and different Numba call structure per scenario.

---

## Revised Understanding: What the Old "89% Joblib Overhead" Was

The prior decomposition (Sequential 159s vs Parallel Numba=1 253s → 94s gap attributed to joblib) was measured with cold workers. Workers spent most of their time rebuilding calibration curves and null distributions for their assigned scenarios before doing any bootstrap work. That rebuild cost appeared as "joblib overhead" when comparing to hot-cache sequential. With injection, that 94s disappears and parallel genuinely wins.

True joblib overhead (spawn, serialization, scheduling) is real but much smaller. Fitting a two-point model to the power crossover data (see section above) gives ~12–13s of fixed overhead per full-grid run, amortized across all 88 scenarios — roughly **0.14s per scenario** or **~3s per worker**. This is the overhead that must be paid regardless of n_sims; once per-scenario compute time is large relative to it, parallel wins.

---

## Nested Parallelism: Old Finding vs New

**Old finding (cold workers):** Numba=1 (253s) beat default Numba (265s) for parallel CI — nested parallelism cost ~12s (~11% of the old gap).

**New finding (warm workers):** Default Numba (97.66s) beats Numba=1 (106.50s) — Numba threads are now net beneficial. The machine has 4 logical cores (2 physical, hyperthreaded); with 4 workers × 4 Numba threads = 16 threads on 4 logical cores there is oversubscription, but the Numba bootstrap kernel is sufficiently memory-latency-bound that extra threads help by overlapping I/O. Net: leave Numba at its default for parallel CI.

The old verification (`test_nested_parallelism.py`, "default 4–11% slower across three runs") was conducted with cold workers where the cache-rebuild cost swamped everything. Those results no longer apply.

---

## Parallel Crossover Summary

| Workload | Sequential wins | Roughly tied | Parallel wins |
|----------|----------------|-------------|---------------|
| Power (n_sims) | < ~200 | ~500 | ≥ 1000 |
| CI (n_reps × n_boot) | n_reps≈50, n_boot=500 | — | n_reps=200, n_boot=1000 |

At higher production tiers (much larger n_sims/n_reps), the joblib overhead is an increasingly small fraction of total work and parallel speedup is expected to approach the theoretical limit for this machine. Prior to warm-cache injection, speedups of 1.7–1.9× were observed before the batch optimization, suggesting the hardware is capable of more than the current 1.2–1.4× — the remaining gap is likely the Numba parallelism interaction and joblib scheduling, not a hard ceiling.

---

## Mitigations Tested

| Mitigation | Old result (cold workers) | New result (warm workers) |
|------------|--------------------------|--------------------------|
| `NUMBA_NUM_THREADS=1` | Slight improvement (~4–11%) | Slightly worse than default |
| `n_jobs=2` with Numba=1 | Slower than n_jobs=-1 | Worst config — underutilizes cores |
| Warm-cache injection | N/A | **Primary fix** — parallel now wins |
| Warmup (Numba JIT) | Recommended | Still recommended |

**Current recommendation:** Use **`n_jobs=-1`** (parallel) with **default Numba threads** for both power and CI at production params. Sequential is only preferable at benchmark-scale n_sims (≤ 50 for power) where per-scenario time is too small for parallel to pay off.

---

## Memory Issues

**Fixed:** The batch path previously allocated `boot_idx_all` of shape `(n_boot, n_reps, n)` in a single call, which would require:

| Tier | n_reps | Allocation size |
|------|--------|----------------|
| Benchmark | 200 | ~64 MB |
| ±0.01 | 1,300 | ~417 MB |
| ±0.002 | 32,500 | ~10.4 GB — would crash |
| ±0.001 | 129,700 | ~41.5 GB — impossible |

Bootstrap indices are now generated in chunks of `_BATCH_BOOTSTRAP_CHUNK_SIZE` (default 2000) reps at a time. Peak memory per chunk is bounded at `n_boot × 2000 × n × 4` bytes ≈ **328 MB** regardless of total `n_reps`. The values are statistically equivalent and reproducible for a fixed seed and chunk size.

**Parallel memory note:** With `n_jobs=-1`, multiple joblib workers run concurrently, each holding its own x_all/y_all/boot_rho_matrix arrays. For very large `n_reps` tiers on memory-constrained machines, reduce `n_jobs` if workers are killed.

**Killing stuck workers:**
```powershell
Get-Process python* | Stop-Process -Force
```

---

## Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `benchmark_batch_vs_no_batch.py` | Single-scenario comparison: old vs new path |
| `benchmark_full_grid.py` | Full 88-scenario grid: sequential and parallel (new path, multipoint cal) |
| `benchmark_ci_bootstrap.py` | Multipoint vs single-point calibration (uses batch path) |
| `benchmark_realistic_runtimes.py` | Power + CI runtimes at small params scaled to precision tiers |

**TODO:** Run `benchmark_ci_bootstrap.py` and record single vs multipoint calibration timings (single-scenario and 22-scenario); add a summary to this doc and optionally to the README. The README currently has only the per-tie-structure calibration cost (~3s single vs ~9s multipoint); we do not yet have documented end-to-end benchmark estimates for CI runs.

---

## Verification Scripts (validation and isolation)

| Script | Purpose |
|--------|---------|
| `test_batch_bootstrap_ci.py` | (1) Bit-identical test: batch vs old path, same data & bootstrap indices. (2) Sequential Numba=1: batch faster than old. (3) Batch path parallel vs sequential; (3b) old path parallel vs sequential. (4) Chunked bootstrap: JIT slice bit-identical test + reproducibility under forced `_boot_chunk_size=2`. Optional: power sim vectorized vs scalar. Uses small n_reps/n_boot. |
| `test_nested_parallelism.py` | Isolates nested parallelism: runs same workload (batch, n_jobs=-1) in two processes — default Numba threads vs `NUMBA_NUM_THREADS=1`. Results were measured with cold workers; re-run with warm-cache injection to get updated numbers. |
| `test_batch_sequential_vs_parallel.py` | Compares batch sequential (default Numba) vs batch parallel (Numba=1) in separate processes. Original results were cold-worker; re-run for updated warm-cache comparison. |
| `test_parallel_worker_cache.py` | Smoke test: parallel worker cache injection via joblib initializer/initargs. |

**Validation result:** Bit-identical test (Step 1 of `test_batch_bootstrap_ci.py`) passes: `max |batch - old| = 0` over bootstrap rho matrix when both paths get the same data and bootstrap index order.

---

## Scaling: Large Runs (e.g. high-tier n_reps)

**Parallel batch path (recommended at production params):**

- **Memory:** Bootstrap index memory is bounded at ~328 MB per chunk. At large n_reps, the dominant allocations are `x_all` and `y_all` of shape `(n_reps, n)` float64 — e.g. at n_reps=129,700 and n=82 these are ~85 MB each. With n_jobs=-1 and 4 workers, peak RAM ≈ 4 × per-scenario peak.
- **Time:** Work scales with n_reps × n_boot. Parallel speedup grows as per-scenario time increases relative to fixed joblib overhead.

**Sequential batch path (use if memory-constrained):**

Sequential processes one scenario at a time; peak RAM is bounded by chunking and a single scenario's arrays.

---

## Reducing Joblib Overhead (if further tuning is desired)

- **Batched tasks:** Run multiple scenarios per joblib task (e.g. 22 scenarios per task → 4 tasks instead of 88). Cuts serialization and scheduling; requires changing how work is passed.
- **Threading backend:** `Parallel(..., backend="threading")` avoids process spawn and pickle. Untested with warm-cache injection; may improve or worsen results.
- **Pool reuse:** joblib's loky backend can reuse the executor across runs; helps repeated runs in one session.

---

## README Update Suggestions

1. **Performance section:** Add that the batch path (`BATCH_CI_BOOTSTRAP=True`) gives ~2.5–2.7× sequential speedup over the per-rep path. Single scenario: ~1.2s vs ~4.5s (200×1000). Full grid (88 scenarios, 200×1000): ~2 min (batch, parallel) vs ~7 min (old path, sequential).

2. **Parallelization:** With warm-cache injection, **`n_jobs=-1`** (parallel, default Numba) is the recommended config for both power and CI at production params. Sequential only wins at benchmark-scale n_sims (≤ ~200 for power, very small n_reps for CI).

3. **Memory:** Batch path peak memory per scenario is bounded by chunking (~328 MB for bootstrap indices + ~170 MB for data arrays at n_reps=129k). With n_jobs=-1, multiply by worker count; reduce n_jobs on memory-constrained machines.

4. **Typical runtimes (4-core HT machine, 200×1000, warm caches):** CI full grid: ~98s parallel, ~119s sequential. Power full grid: ~27–29s at n_sims=1000 parallel, ~38–41s sequential.

5. **Config:** Document `BATCH_CI_BOOTSTRAP` in config (default True). `VECTORIZE_DATA_GENERATION` must be True for batch path.

6. **Validation/verification:** Mention `test_batch_bootstrap_ci.py` and `test_parallel_worker_cache.py` for reproducing findings (see this doc).

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
| `confidence_interval_calculator.py` | `bootstrap_ci_averaged` — branches on `batch_bootstrap` and `vectorize`; `_init_ci_worker_caches` for warm-cache injection |
| `power_simulation.py` | `run_all_scenarios` — `_init_worker_caches` for warm-cache injection |
| `config.py` | `BATCH_CI_BOOTSTRAP`, `VECTORIZE_DATA_GENERATION`, `USE_NUMBA` |
