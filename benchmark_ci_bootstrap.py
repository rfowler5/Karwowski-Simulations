"""
Benchmark CI bootstrap: all optimizations ON, multipoint vs single calibration.

Run ONE benchmark at a time. Do not run concurrently with other benchmarks.
Run sequential first, then parallel.

Usage: python benchmark_ci_bootstrap.py [--quick]
  --quick  Run only single-scenario benchmarks (A, B) for fast feedback (~1 min).

Configurations benchmarked (all use Numba=True, vectorize=True, batch_bootstrap=True):
  A. Single scenario, multipoint calibration   (the "everything" config)
  B. Single scenario, single calibration        (everything minus multipoint)
  C. 22 scenarios (1 case), multipoint, sequential (n_jobs=1)
  D. 22 scenarios (1 case), multipoint, parallel   (n_jobs=-1)
  E. 22 scenarios (1 case), single, sequential
  F. 22 scenarios (1 case), single, parallel
"""

import sys
import time
from joblib import Parallel, delayed
from config import CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES, ASYMPTOTIC_TIE_CORRECTION_MODE
from confidence_interval_calculator import bootstrap_ci_averaged, _ci_one_scenario

CASE_ID = 3
CASE = CASES[CASE_ID]
Y_PARAMS = {"median": CASE["median"], "iqr": CASE["iqr"], "range": CASE["range"]}
# Use lighter params so benchmark completes in ~3-5 min. For production-like
# timings, set N_REPS=200, N_BOOT=1000 and use run_all_ci_scenarios (88 scenarios).
N_REPS = 50
N_BOOT = 200
SEED = 42
# Run 22 scenarios (1 case) for multi-scenario benchmarks to keep runtime reasonable
SUBSET_CASE_IDS = [3]
GENERATOR = "nonparametric"


def warmup():
    """One cheap call to trigger any Numba JIT compilation."""
    bootstrap_ci_averaged(
        CASE["n"], 4, "even", CASE["observed_rho"], Y_PARAMS,
        generator=GENERATOR, n_reps=5, n_boot=10, seed=0,
        batch_bootstrap=True, calibration_mode="single")


def bench_single_scenario(calibration_mode, label):
    t0 = time.perf_counter()
    result = bootstrap_ci_averaged(
        CASE["n"], 4, "even", CASE["observed_rho"], Y_PARAMS,
        generator=GENERATOR, n_reps=N_REPS, n_boot=N_BOOT, seed=SEED,
        batch_bootstrap=True, calibration_mode=calibration_mode)
    elapsed = time.perf_counter() - t0
    print(f"[{label}] {elapsed:.2f}s  "
          f"ci=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  "
          f"rho_hat={result['mean_rho_hat']:.4f}")
    return elapsed


def _build_subset_scenarios():
    """Build scenario tuples for SUBSET_CASE_IDS only (22 per case)."""
    scenarios = []
    scenario_idx = 0
    for case_id in SUBSET_CASE_IDS:
        case = CASES[case_id]
        n = case["n"]
        for k in N_DISTINCT_VALUES:
            for dt in DISTRIBUTION_TYPES:
                sc_seed = (SEED + scenario_idx) if SEED is not None else None
                scenarios.append((case_id, case, k, dt, False,
                                  GENERATOR, N_REPS, N_BOOT, 0.05,
                                  ASYMPTOTIC_TIE_CORRECTION_MODE, sc_seed,
                                  None, True))  # calibration_mode, batch_bootstrap
                scenario_idx += 1
        sc_seed = (SEED + scenario_idx) if SEED is not None else None
        scenarios.append((case_id, case, n, None, True,
                          GENERATOR, N_REPS, N_BOOT, 0.05,
                          ASYMPTOTIC_TIE_CORRECTION_MODE, sc_seed,
                          None, True))
        scenario_idx += 1
    return scenarios


def bench_multi_scenario(calibration_mode, n_jobs, label):
    scenarios = _build_subset_scenarios()
    # Override calibration_mode in each scenario tuple (index 11)
    scenarios = [
        (*s[:11], calibration_mode, s[12]) for s in scenarios
    ]
    t0 = time.perf_counter()
    if n_jobs == 1:
        results = [_ci_one_scenario(*args) for args in scenarios]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_ci_one_scenario)(*args) for args in scenarios)
    elapsed = time.perf_counter() - t0
    n_scenarios = len(results)
    print(f"[{label}] {elapsed:.2f}s  ({n_scenarios} scenarios, n_jobs={n_jobs})")
    return elapsed


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if quick:
        print("QUICK MODE: single-scenario benchmarks only\n")

    print(f"Case {CASE_ID}: n={CASE['n']}, rho={CASE['observed_rho']}")
    print(f"n_reps={N_REPS}, n_boot={N_BOOT}, generator={GENERATOR}")
    print(f"batch_bootstrap=True, vectorize=True, numba=True")
    print("-" * 70)

    print("\nWarming up Numba...")
    warmup()

    print("\n--- Single scenario (Case 3, k=4, even) ---")
    t_mp = bench_single_scenario("multipoint", "A: multipoint")
    t_sp = bench_single_scenario("single", "B: single    ")
    print(f"   multipoint overhead: {t_mp - t_sp:+.2f}s ({t_mp/t_sp:.2f}x)")

    if quick:
        print("\n--- Summary (quick mode) ---")
        print(f"A: single, multipoint  {t_mp:>7.2f}s")
        print(f"B: single, single-cal  {t_sp:>7.2f}s")
        print(f"Multipoint overhead:    +{t_mp - t_sp:.2f}s ({(t_mp/t_sp - 1)*100:.0f}%)")
        sys.exit(0)

    print(f"\n--- {22 * len(SUBSET_CASE_IDS)} scenarios (case(s) {SUBSET_CASE_IDS}), sequential (n_jobs=1) ---")
    t_seq_mp = bench_multi_scenario("multipoint", 1, "C: seq multipoint")
    t_seq_sp = bench_multi_scenario("single", 1, "E: seq single    ")
    print(f"   multipoint overhead: {t_seq_mp - t_seq_sp:+.2f}s ({t_seq_mp/t_seq_sp:.2f}x)")

    print(f"\n--- {22 * len(SUBSET_CASE_IDS)} scenarios (case(s) {SUBSET_CASE_IDS}), parallel (n_jobs=-1) ---")
    t_par_mp = bench_multi_scenario("multipoint", -1, "D: par multipoint")
    t_par_sp = bench_multi_scenario("single", -1, "F: par single    ")
    print(f"   multipoint overhead: {t_par_mp - t_par_sp:+.2f}s ({t_par_mp/t_par_sp:.2f}x)")

    print("\n--- Summary ---")
    print(f"{'Config':<30} {'Time':>8}")
    print("-" * 40)
    n_scen = 22 * len(SUBSET_CASE_IDS)
    for label, t in [
        ("A: single, multipoint", t_mp),
        ("B: single, single-cal", t_sp),
        (f"C: {n_scen} seq, multipoint", t_seq_mp),
        (f"D: {n_scen} par, multipoint", t_par_mp),
        (f"E: {n_scen} seq, single-cal", t_seq_sp),
        (f"F: {n_scen} par, single-cal", t_par_sp),
    ]:
        print(f"{label:<30} {t:>7.2f}s")
    print(f"\nParallel speedup (multipoint): {t_seq_mp/t_par_mp:.2f}x")
    print(f"Parallel speedup (single):     {t_seq_sp/t_par_sp:.2f}x")
    print(f"Multipoint cost (sequential):  +{t_seq_mp - t_seq_sp:.1f}s ({(t_seq_mp/t_seq_sp - 1)*100:.0f}%)")
    print(f"Multipoint cost (parallel):    +{t_par_mp - t_par_sp:.1f}s ({(t_par_mp/t_par_sp - 1)*100:.0f}%)")
