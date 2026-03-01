"""
Validate batch CI path: (1) bit-identical with old path, (2) batch faster when sequential,
(3) slowdown only when parallelism is used. Uses small n_reps/n_boot for speed.
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import os
import numpy as np
import time

# Use single Numba thread for timing comparisons (avoid nested parallelism)
os.environ["NUMBA_NUM_THREADS"] = "1"

from config import CASES
from confidence_interval_calculator import bootstrap_ci_averaged, run_all_ci_scenarios
from spearman_helpers import (
    _bootstrap_rhos_jit,
    _batch_bootstrap_rhos_jit,
    spearman_rho_2d,
)
from data_generator import (
    generate_cumulative_aluminum_batch,
    generate_y_nonparametric_batch,
    calibrate_rho,
    _fit_lognormal,
)

# Small sizes for quick run (Steps 1–2); slightly larger for Step 3/3b so parallel has work
N_REPS, N_BOOT, SEED = 5, 20, 42
N_REPS_SCEN, N_BOOT_SCEN = 30, 80  # used only in Step 3/3b
case = CASES[3]
n, k, dt = case["n"], 4, "even"
y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
rho_s = -0.13

# ---------------------------------------------------------------------------
# Step 1: Bit-identical test (same data + same bootstrap indices as old path)
# ---------------------------------------------------------------------------
print("Step 1: Bit-identical test (batch vs old path, same data & bootstrap order)")
ss = np.random.SeedSequence(SEED)
data_rng, boot_rng = [np.random.default_rng(s) for s in ss.spawn(2)]

ln_params = _fit_lognormal(y_params["median"], y_params["iqr"])
cal_rho = calibrate_rho(
    n, k, dt, rho_s, y_params,
    all_distinct=False, calibration_mode="single")

x_all = generate_cumulative_aluminum_batch(
    N_REPS, n, k, distribution_type=dt, all_distinct=False, rng=data_rng)
y_all = generate_y_nonparametric_batch(
    x_all, rho_s, y_params, rng=data_rng,
    _calibrated_rho=cal_rho, _ln_params=ln_params)

# Bootstrap indices in OLD path order: one (n_boot, n) per rep
boot_idx_all = np.empty((N_BOOT, N_REPS, n), dtype=np.int32)
for rep in range(N_REPS):
    boot_idx_all[:, rep, :] = boot_rng.integers(0, n, size=(N_BOOT, n), dtype=np.int32)

# Batch path result
boot_rho_batch = _batch_bootstrap_rhos_jit(
    x_all.astype(np.float64), y_all.astype(np.float64), boot_idx_all)

# Old path: per-rep bootstrap
boot_rho_old_per_rep = []
for rep in range(N_REPS):
    rhos = _bootstrap_rhos_jit(
        x_all[rep].astype(np.float64), y_all[rep].astype(np.float64),
        boot_idx_all[:, rep, :])
    boot_rho_old_per_rep.append(rhos)
boot_rho_old = np.array(boot_rho_old_per_rep)  # (n_reps, n_boot)

# Compare
diff = np.abs(boot_rho_batch - boot_rho_old)
max_diff = np.max(diff)
ok1 = max_diff < 1e-10
print(f"  max |batch - old| over (rep, boot): {max_diff:.2e}  ->  {'PASS' if ok1 else 'FAIL'}")

# ---------------------------------------------------------------------------
# Step 2: Sequential + single-thread Numba: batch should be faster than old
# ---------------------------------------------------------------------------
print("\nStep 2: Sequential, Numba=1 — batch path should be faster than old path")

t0 = time.perf_counter()
bootstrap_ci_averaged(
    n, k, dt, rho_s, y_params,
    generator="nonparametric", n_reps=N_REPS, n_boot=N_BOOT,
    seed=SEED + 1, batch_bootstrap=False)
t_old = time.perf_counter() - t0

t0 = time.perf_counter()
bootstrap_ci_averaged(
    n, k, dt, rho_s, y_params,
    generator="nonparametric", n_reps=N_REPS, n_boot=N_BOOT,
    seed=SEED + 1, batch_bootstrap=True)
t_batch = time.perf_counter() - t0

print(f"  Old path:   {t_old:.3f}s")
print(f"  Batch path: {t_batch:.3f}s  ->  batch faster: {t_batch < t_old} (expect True)")

# ---------------------------------------------------------------------------
# Step 3: Batch path — with Numba=1, parallel often slower than sequential
# Step 3b: Old path — joblib parallel should be faster than sequential
# Use a small subset of scenarios so this step finishes quickly.
# ---------------------------------------------------------------------------
from joblib import Parallel, delayed
from confidence_interval_calculator import _ci_one_scenario
from config import N_DISTINCT_VALUES, DISTRIBUTION_TYPES, ALPHA, ASYMPTOTIC_TIE_CORRECTION_MODE

def build_scenarios(batch_bootstrap, n_cases=4, n_k=2, n_dt=2):
    """Build scenario list: n_cases × n_k × n_dt scenarios (so parallel has enough work)."""
    out = []
    scenario_idx = 0
    for case_id, case in list(CASES.items())[:n_cases]:
        for k_sc in N_DISTINCT_VALUES[:n_k]:
            for dt_sc in DISTRIBUTION_TYPES[:n_dt]:
                sc_seed = SEED + 2 + scenario_idx
                out.append(
                    (case_id, case, k_sc, dt_sc, False, "nonparametric",
                     N_REPS_SCEN, N_BOOT_SCEN,
                     ALPHA, ASYMPTOTIC_TIE_CORRECTION_MODE, sc_seed, "single", batch_bootstrap)
                )
                scenario_idx += 1
    return out

def run_subset(scenarios, n_jobs):
    if n_jobs == 1:
        return [_ci_one_scenario(*args) for args in scenarios]
    return Parallel(n_jobs=n_jobs)(delayed(_ci_one_scenario)(*args) for args in scenarios)

n_scenarios = len(build_scenarios(True))

print("\nStep 3: Batch path — Numba=1, parallel vs sequential")
scenarios_batch = build_scenarios(batch_bootstrap=True)
t0 = time.perf_counter()
run_subset(scenarios_batch, 1)
t_batch_seq = time.perf_counter() - t0
t0 = time.perf_counter()
run_subset(scenarios_batch, -1)
t_batch_par = time.perf_counter() - t0
print(f"  Sequential (n_jobs=1): {t_batch_seq:.2f}s  ({n_scenarios} scenarios)")
print(f"  Parallel   (n_jobs=-1): {t_batch_par:.2f}s")
batch_par_hurts = t_batch_par > t_batch_seq
print(f"  Parallel slower than sequential: {batch_par_hurts} (often True on 4-core due to nested parallelism)")
if not batch_par_hurts:
    print("  (Run-to-run variance; full-grid benchmarks in BENCHMARKING_FINDINGS show batch path sequential ~112s vs parallel ~172s.)")

print("\nStep 3b: Old path (no batching) — joblib parallel should help")
scenarios_old = build_scenarios(batch_bootstrap=False)
t0 = time.perf_counter()
run_subset(scenarios_old, 1)
t_old_seq = time.perf_counter() - t0
t0 = time.perf_counter()
run_subset(scenarios_old, -1)
t_old_par = time.perf_counter() - t0
print(f"  Sequential (n_jobs=1): {t_old_seq:.2f}s  ({n_scenarios} scenarios)")
print(f"  Parallel   (n_jobs=-1): {t_old_par:.2f}s")
old_par_helps = t_old_par < t_old_seq
print(f"  Parallel faster than sequential: {old_par_helps} (expect True when scenario work >> joblib overhead)")
if not old_par_helps:
    print("  (On Windows with spawn, small runs can show parallel slower; full grid 88×200×1000 shows old path ~2x faster with n_jobs=-1 per BENCHMARKING_FINDINGS.)")

# ---------------------------------------------------------------------------
# Optional: Power sim — vectorized vs scalar, same seed (no bootstrap)
# ---------------------------------------------------------------------------
print("\nOptional: Power sim — vectorized vs scalar, same seed (small n_sims)")
from power_simulation import estimate_power

n_sims = 100
pw_vec = estimate_power(
    n, k, dt, rho_s, y_params, n_sims=n_sims, seed=SEED, vectorize=True
)
pw_scalar = estimate_power(
    n, k, dt, rho_s, y_params, n_sims=n_sims, seed=SEED, vectorize=False
)
print(f"  vectorize=True:  power = {pw_vec:.4f}")
print(f"  vectorize=False: power = {pw_scalar:.4f}")
print(f"  difference: {abs(pw_vec - pw_scalar):.4f} (expect small for same seed)")

print("\nDone.")
