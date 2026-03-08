---
name: Backlog and follow-ups
overview: A reminder checklist for README improvements, code hardening, benchmarks, simulation/calibration tuning, copula calibration path, asymptotic comparison, and linear-generator calibration bias.
---

# Backlog and follow-ups

Reminder list (minimal detail; you will flesh out when doing each).

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

5. **Copula: multipoint calibration path**
   - Consider giving the copula the multipoint calibration path (design/implementation to be decided).

6. **Asymptotic vs empirical and y-ties correction**
   - After running simulations, compare asymptotic to empirical.
   - Depending on results, add the y-ties correction to the asymptotic path.

7. **Linear generator calibration (currently biased upward)**
   - Linear generator does not use calibration (only nonparametric/copula/empirical do). Consider adding calibration for linear; it is currently biased, seemingly upward. Details: linear model calibration chat (agent-transcripts 971ab5af-9437-404a-a1b1-47f465024665).
