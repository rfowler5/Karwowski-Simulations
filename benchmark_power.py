"""
Benchmark power simulation: vectorize vs non-vectorize, calibration modes,
sequential vs parallel. Run one at a time per benchmarking rule.
"""
import sys
import time
from config import CASES
from power_simulation import estimate_power, min_detectable_rho, run_all_scenarios

# Lighter params for reasonable runtime. Use n_sims=5000+ for production-like.
N_SIMS = 500
SEED = 42
CASE_ID = 3
CASE = CASES[CASE_ID]
Y_PARAMS = {"median": CASE["median"], "iqr": CASE["iqr"], "range": CASE["range"]}


def warmup():
    """Warm up Numba and calibration cache."""
    print("Warming up (single scenario, vectorized)...")
    estimate_power(
        CASE["n"], 4, "even", 0.30, Y_PARAMS,
        generator="nonparametric", n_sims=50, seed=0,
        calibration_mode="single", vectorize=True)
    print("Warmup done.\n")


def bench_single_estimate_power(vectorize, calibration_mode, label):
    """Time estimate_power for one scenario."""
    t0 = time.perf_counter()
    pw = estimate_power(
        CASE["n"], 4, "even", 0.30, Y_PARAMS,
        generator="nonparametric", n_sims=N_SIMS, seed=SEED,
        calibration_mode=calibration_mode, vectorize=vectorize)
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.2f}s  power={pw:.4f}")
    return elapsed


def bench_single_min_detectable(calibration_mode, label):
    """Time min_detectable_rho for one scenario (bisection)."""
    t0 = time.perf_counter()
    md = min_detectable_rho(
        CASE["n"], 4, "even", Y_PARAMS,
        generator="nonparametric", n_sims=N_SIMS, seed=SEED,
        direction="positive", calibration_mode=calibration_mode)
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.2f}s  min_rho={md:.4f}")
    return elapsed


def bench_full_grid(generator, n_jobs, calibration_mode, label):
    """Time run_all_scenarios for full 88-scenario grid."""
    t0 = time.perf_counter()
    results = run_all_scenarios(
        generator=generator, n_sims=N_SIMS, seed=SEED,
        n_jobs=n_jobs, calibration_mode=calibration_mode)
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.2f}s  ({len(results)} scenarios, n_jobs={n_jobs})")
    return elapsed


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    print(f"Power benchmark: n_sims={N_SIMS}, Case {CASE_ID} (n={CASE['n']})")
    print("=" * 60)
    warmup()

    print("--- Single scenario: estimate_power (vectorize vs non-vectorize) ---")
    t_vec = bench_single_estimate_power(True, "single", "vectorize=True")
    t_scalar = bench_single_estimate_power(False, "single", "vectorize=False")
    print(f"  Vectorize speedup: {t_scalar/t_vec:.2f}x\n")

    print("--- Single scenario: estimate_power (multipoint vs single calibration) ---")
    t_mp = bench_single_estimate_power(True, "multipoint", "multipoint")
    t_sp = bench_single_estimate_power(True, "single", "single")
    print(f"  Multipoint overhead: {t_mp - t_sp:+.2f}s ({t_mp/t_sp:.2f}x)\n")

    if quick:
        print("(Quick mode: skipping full grid. Run without --quick for full benchmark.)")
        sys.exit(0)

    print("--- Single scenario: min_detectable_rho (bisection) ---")
    t_md_mp = bench_single_min_detectable("multipoint", "multipoint")
    t_md_sp = bench_single_min_detectable("single", "single")
    print(f"  Multipoint overhead: {t_md_mp - t_md_sp:+.2f}s ({t_md_mp/t_md_sp:.2f}x)\n")

    print("--- Full grid (88 scenarios): sequential (n_jobs=1) ---")
    t_seq = bench_full_grid("nonparametric", 1, "single",
                            "nonparametric, single-cal")
    print()

    print("--- Full grid (88 scenarios): parallel (n_jobs=-1) ---")
    t_par = bench_full_grid("nonparametric", -1, "single",
                            "nonparametric, single-cal")
    print()

    print("=" * 60)
    print("Summary")
    print("-" * 50)
    print(f"{'estimate_power vectorize':<35} {t_vec:>7.2f}s")
    print(f"{'estimate_power scalar':<35} {t_scalar:>7.2f}s")
    print(f"{'estimate_power multipoint':<35} {t_mp:>7.2f}s")
    print(f"{'estimate_power single-cal':<35} {t_sp:>7.2f}s")
    print(f"{'min_detectable_rho multipoint':<35} {t_md_mp:>7.2f}s")
    print(f"{'min_detectable_rho single-cal':<35} {t_md_sp:>7.2f}s")
    print(f"{'Full grid sequential':<35} {t_seq:>7.2f}s")
    print(f"{'Full grid parallel':<35} {t_par:>7.2f}s")
    print("-" * 50)
    print(f"Parallel speedup (full grid): {t_seq/t_par:.2f}x")
