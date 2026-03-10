---
name: Backlog and follow-ups
overview: A reminder checklist for README improvements, code hardening, benchmarks, simulation/calibration tuning, copula calibration path, asymptotic comparison, linear-generator calibration bias, and CI bootstrap optimizations (counting sort, skip-y).
---

# Backlog and follow-ups

Reminder list (minimal detail; you will flesh out when doing each).

**Context for CI bootstrap items (below):** Full issue analysis, verdicts, and step-by-step implementation instructions (Phases 1–4) are in Cursor agent transcript **5a2c26f6-2ac9-4dad-9d52-ce0efc39a5c9** — search or open that chat when ready to implement. Issue 1 (memory chunking for `boot_idx_all`) is already implemented.

---

1. **README: TOC, quick start, and math reorganization**
   - Add table of contents and quick start section to [README.md](../README.md).
   - Consider reorganizing and moving heavy math to other docs; reference those from the README.

2. **Harden code per Claude suggestion**
   - Apply the hardening suggestions from the other conversation (you will know which).

3. **Run benchmarks and record results**
   - Execute benchmarks and document results (e.g. per [.cursor/rules/benchmarking.mdc](../.cursor/rules/benchmarking.mdc) if applicable).

4. **Small +/- 0.01-style simulation for parameter tuning**
   - Run a small simulation (e.g. rho step ±0.01) to inspect output rhos and re-estimate bisection c and interrep SD so parameters are well understood.

5. ~~**Copula: multipoint calibration path**~~ *(done)*
   - ~~Consider giving the copula the multipoint calibration path (design/implementation to be decided).~~

6. **Asymptotic vs empirical and y-ties correction**
   - After running simulations, compare asymptotic to empirical.
   - Depending on results, add the y-ties correction to the asymptotic path.

7. **Linear generator calibration (currently biased upward)**
   - Linear generator does not use calibration (only nonparametric/copula/empirical do). Consider adding calibration for linear; it is currently biased, seemingly upward. Details: linear model calibration chat (agent-transcripts 971ab5af-9437-404a-a1b1-47f465024665).

8. **CI bootstrap: O(n) counting sort for ranking (Issue 2)**
   - Replace argsort-based `_tied_rank` in the batch bootstrap JIT with counting sort for ~1.5–2× speedup on the JIT step (not 3×; see transcript). Applies to all generators; empirical needs rank-group mapping (Phase 3). **Recommended order:** Phase 2 first (non-empirical: [spearman_helpers.py](../spearman_helpers.py) + [confidence_interval_calculator.py](../confidence_interval_calculator.py) — add `_counting_midranks`, `_batch_bootstrap_rhos_preranked_jit`, prerank x/y to integer rank indices, handle tied x via group indices). Phase 3 optional: extend to empirical (dense rank-group IDs for y, e.g. `scipy.stats.rankdata(..., method='dense')` per row). **Do not** implement "Issue 3: preranked x" as a separate optimization — the bootstrap resamples (x,y) pairs so x changes per bootstrap sample; the valid idea is precomputed rank arrays + counting sort, which is Issue 2.

9. ~~**CI bootstrap: skip-y for non-empirical generators (Issue 4, optional)**~~ *(done) and expanded to skip-y for non-empirical generators in power simulation.*
   - ~~For nonparametric, copula, linear: return rank arrays instead of float y in batch CI data generation; skip lognormal/copula/exp. Saves ~15–20% of data-generation fraction of total CI time. Does not apply to empirical (pool ties). Phase 4 in same transcript; lower priority than Issue 2.~~
