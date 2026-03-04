"""
Benchmark permutation p-value: precomputed null build/lookup, MC path,
and full grid. Run one at a time per benchmarking rule.
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import time
import numpy as np
from config import CASES
from power_simulation import estimate_power
from permutation_pvalue import get_precomputed_null, pvalues_from_precomputed_null, _NULL_CACHE
from power_asymptotic import get_x_counts
from spearman_helpers import spearman_rho_2d

CASE_ID = 3
CASE = CASES[CASE_ID]
Y_PARAMS = {"median": CASE["median"], "iqr": CASE["iqr"], "range": CASE["range"]}


def warmup():
    """Warm up Numba and calibration cache."""
    print("Warming up (estimate_power, n_sims=50)...")
    estimate_power(
        CASE["n"], 4, "even", 0.30, Y_PARAMS,
        generator="nonparametric", n_sims=50, seed=0,
        calibration_mode="single")
    print("Warmup done.\n")


def bench_precomputed_null_build():
    """Time building precomputed null (cache miss)."""
    n = CASE["n"]
    x_counts = get_x_counts(n, 4, distribution_type="even")
    # Clear cache for this key so we time a real build
    key = (n, False, tuple(int(c) for c in x_counts))
    _NULL_CACHE.pop(key, None)

    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    null = get_precomputed_null(n, False, x_counts, n_pre=50_000, rng=rng)
    elapsed = time.perf_counter() - t0
    size_mb = null.nbytes / (1024 * 1024)
    print(f"  Precomputed null build (n_pre=50k, n={n}): {elapsed:.2f}s  array size: {size_mb:.2f} MB")
    return elapsed


def bench_precomputed_null_lookup():
    """Time p-value lookup for 10k observed rhos (cache hit)."""
    n = CASE["n"]
    x_counts = get_x_counts(n, 4, distribution_type="even")
    rng = np.random.default_rng(42)
    null = get_precomputed_null(n, False, x_counts, n_pre=50_000, rng=rng)

    n_rhos = 10_000
    rhos_obs = np.random.default_rng(123).uniform(-0.5, 0.5, size=n_rhos)

    t0 = time.perf_counter()
    for _ in range(10):
        pvalues_from_precomputed_null(rhos_obs, null)
    elapsed = (time.perf_counter() - t0) / 10
    print(f"  Precomputed null lookup (10k rhos): {elapsed*1000:.2f} ms")
    return elapsed


def bench_mc_path():
    """Time estimate_power with permutation (nonparametric, n_sims=1000)."""
    t0 = time.perf_counter()
    pw = estimate_power(
        CASE["n"], 4, "even", 0.35, Y_PARAMS,
        generator="nonparametric", n_sims=1000, seed=42,
        calibration_mode="single")
    elapsed = time.perf_counter() - t0
    print(f"  estimate_power (permutation, n_sims=1000): {elapsed:.2f}s  power={pw:.4f}")
    return elapsed


def bench_tbased_path():
    """Time estimate_power with t-based p-value (USE_PERMUTATION_PVALUE=False)."""
    import power_simulation as ps
    orig = ps.USE_PERMUTATION_PVALUE
    ps.USE_PERMUTATION_PVALUE = False
    try:
        t0 = time.perf_counter()
        pw = estimate_power(
            CASE["n"], 4, "even", 0.35, Y_PARAMS,
            generator="nonparametric", n_sims=1000, seed=42,
            calibration_mode="single")
        elapsed = time.perf_counter() - t0
        print(f"  estimate_power (t-based, n_sims=1000): {elapsed:.2f}s  power={pw:.4f}")
        return elapsed
    finally:
        ps.USE_PERMUTATION_PVALUE = orig


if __name__ == "__main__":
    print("Permutation p-value benchmark")
    print("=" * 60)
    warmup()

    print("--- Precomputed null: build (cache miss) ---")
    t_build = bench_precomputed_null_build()
    print()

    print("--- Precomputed null: lookup (10k rhos) ---")
    t_lookup = bench_precomputed_null_lookup()
    print()

    print("--- MC path: estimate_power (permutation, n_sims=1000) ---")
    t_mc = bench_mc_path()
    print()

    print("--- Comparison: t-based p-value (n_sims=1000) ---")
    t_tbased = bench_tbased_path()
    print()

    print("=" * 60)
    print("Summary")
    print("-" * 50)
    print(f"{'Precomputed null build (50k)':<35} {t_build:>7.2f}s")
    print(f"{'Precomputed null lookup (10k rhos)':<35} {t_lookup*1000:>6.2f} ms")
    print(f"{'estimate_power permutation (1k sims)':<35} {t_mc:>7.2f}s")
    print(f"{'estimate_power t-based (1k sims)':<35} {t_tbased:>7.2f}s")
    print("-" * 50)
