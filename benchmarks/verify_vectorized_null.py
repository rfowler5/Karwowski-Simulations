"""
Verify that the vectorised get_precomputed_null produces a statistically
equivalent null distribution to the original loop-based implementation,
and measure the speedup.

Usage:  python benchmarks/verify_vectorized_null.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
from permutation_pvalue import _build_x_midranks, _NULL_CACHE
from config import PVALUE_PRECOMPUTED_N_PRE


# --------------------------------------------------------------------------
# Original (loop-based) implementation for reference
# --------------------------------------------------------------------------
def _build_null_loop(n, x_counts, n_pre, rng):
    """Original loop implementation (copied verbatim from pre-vectorisation)."""
    x_midranks = _build_x_midranks(x_counts)
    x_std = x_midranks - np.mean(x_midranks)
    sx = np.std(x_midranks, ddof=0)
    if sx > 0:
        x_std = x_std / sx
    base_ranks = np.arange(1.0, n + 1.0)
    null_rhos = np.empty(n_pre, dtype=np.float64)
    for i in range(n_pre):
        perm_y = rng.permutation(base_ranks)
        y_std = perm_y - np.mean(perm_y)
        sy = np.std(perm_y, ddof=0)
        if sy > 0:
            y_std = y_std / sy
        null_rhos[i] = np.dot(x_std, y_std) / n
    return np.sort(np.abs(null_rhos))


# --------------------------------------------------------------------------
# New (vectorised) implementation — mirrors the current code
# --------------------------------------------------------------------------
def _build_null_vectorised(n, x_counts, n_pre, rng):
    """Vectorised implementation (same logic as updated get_precomputed_null)."""
    x_midranks = _build_x_midranks(x_counts)
    x_std = x_midranks - np.mean(x_midranks)
    sx = np.std(x_midranks, ddof=0)
    if sx > 0:
        x_std = x_std / sx
    base_ranks = np.arange(1.0, n + 1.0)
    std_y = np.std(base_ranks, ddof=0)
    all_perm_y = np.argsort(rng.random((n_pre, n)), axis=1) + 1.0
    if std_y > 0:
        null_rhos = (all_perm_y @ x_std) / (std_y * n)
    else:
        null_rhos = np.zeros(n_pre, dtype=np.float64)
    return np.sort(np.abs(null_rhos))


def main():
    n_pre = PVALUE_PRECOMPUTED_N_PRE
    test_cases = [
        ("all-distinct n=82", 82, np.ones(82, dtype=int)),
        ("single-group n=82 (sx=0 edge)", 82, np.array([82])),
        ("tied k=4 even n=80", 80, np.array([20, 20, 20, 20])),
        ("tied k=7 heavy_center n=73", 73, np.array([5, 9, 14, 17, 14, 9, 5])),
    ]

    print(f"Verification: loop vs vectorised null build (n_pre={n_pre:,})")
    print("=" * 72)

    for label, n, x_counts in test_cases:
        print(f"\n--- {label} ---")

        # Time the loop version
        rng_loop = np.random.default_rng(42)
        t0 = time.perf_counter()
        null_loop = _build_null_loop(n, x_counts, n_pre, rng_loop)
        t_loop = time.perf_counter() - t0

        # Time the vectorised version
        rng_vec = np.random.default_rng(42)
        t0 = time.perf_counter()
        null_vec = _build_null_vectorised(n, x_counts, n_pre, rng_vec)
        t_vec = time.perf_counter() - t0

        # The two implementations use different RNG algorithms (rng.permutation vs
        # argsort of rng.random), so even with the same seed they produce independent
        # samples from the same null distribution. At n_pre=50,000 per sample the KS
        # test has high power; p-values in the 0.01–0.05 range are expected by chance.
        ks_stat, ks_p = stats.ks_2samp(null_loop, null_vec)

        # Moment comparison
        mean_loop, std_loop = np.mean(null_loop), np.std(null_loop)
        mean_vec, std_vec = np.mean(null_vec), np.std(null_vec)

        # Quantile comparison
        quantiles = [0.5, 0.9, 0.95, 0.99]
        q_loop = np.quantile(null_loop, quantiles)
        q_vec = np.quantile(null_vec, quantiles)

        print(f"  Loop:       {t_loop:.3f}s")
        print(f"  Vectorised: {t_vec:.3f}s")
        print(f"  Speedup:    {t_loop / t_vec:.1f}x")
        print(f"  KS stat:    {ks_stat:.6f}  p={ks_p:.4f}  "
              f"{'PASS (p>0.01)' if ks_p > 0.01 else 'FAIL (p<=0.01)'}")
        print(f"  Mean:       loop={mean_loop:.6f}  vec={mean_vec:.6f}  "
              f"diff={abs(mean_loop - mean_vec):.2e}")
        print(f"  Std:        loop={std_loop:.6f}  vec={std_vec:.6f}  "
              f"diff={abs(std_loop - std_vec):.2e}")
        for i, q in enumerate(quantiles):
            print(f"  Q{q:.2f}:      loop={q_loop[i]:.6f}  vec={q_vec[i]:.6f}  "
                  f"diff={abs(q_loop[i] - q_vec[i]):.2e}")

    # Also time the actual get_precomputed_null through the public API
    print("\n--- Public API (get_precomputed_null) with cold cache ---")
    _NULL_CACHE.clear()
    from permutation_pvalue import get_precomputed_null
    rng_api = np.random.default_rng(42)
    x_counts_distinct = np.ones(82, dtype=int)
    t0 = time.perf_counter()
    result = get_precomputed_null(82, True, x_counts_distinct, rng=rng_api)
    t_api = time.perf_counter() - t0
    print(f"  Cold build (n=82, all-distinct): {t_api:.3f}s")
    print(f"  Shape: {result.shape}, min={result[0]:.6f}, max={result[-1]:.6f}")

    # Second call should be cached (instant)
    t0 = time.perf_counter()
    result2 = get_precomputed_null(82, True, x_counts_distinct, rng=rng_api)
    t_cached = time.perf_counter() - t0
    print(f"  Cached lookup:                   {t_cached:.6f}s")
    assert result is result2, "Cache should return same object"
    print("  Cache hit: same object? YES")

    # Full warmup timing (87 builds)
    print("\n--- Full warm_precomputed_null_cache (87 builds) ---")
    _NULL_CACHE.clear()
    from permutation_pvalue import warm_precomputed_null_cache
    t0 = time.perf_counter()
    n_built = warm_precomputed_null_cache()
    t_warm = time.perf_counter() - t0
    print(f"  Built {n_built} nulls in {t_warm:.2f}s "
          f"({t_warm / n_built:.3f}s per build)")


if __name__ == "__main__":
    main()
