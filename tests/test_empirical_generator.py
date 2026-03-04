"""Verify empirical generator fixes the 71 digitized values and resamples only the remainder."""

import sys
from pathlib import Path
import numpy as np
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data_generator import digitized_available


def main():
    if not digitized_available():
        print("SKIP: Digitized data not available.")
        return

    from data_generator import build_empirical_pool
    from data.digitized import B_AL71, H_AL71

    ok = True
    seed = 12345

    # --- Test 1: 71 digitized values appear exactly once in each pool ---
    for n, base_arr, label in [(73, B_AL71, "B-Al"), (81, H_AL71, "H-Al")]:
        rng = np.random.default_rng(seed)
        pool = build_empirical_pool(n, rng)
        if len(pool) != n:
            print(f"FAIL: {label} pool length {len(pool)} != {n}")
            ok = False
            continue
        sorted_pool = np.sort(pool)
        sorted_base = np.sort(base_arr)
        # The first 71 of the sorted pool should be >= the 71 base values
        # More precisely: every base value must appear at least once in pool
        for val in sorted_base:
            count = np.sum(pool == val)
            if count < 1:
                print(f"FAIL: {label} n={n} missing digitized value {val}")
                ok = False
                break
        else:
            print(f"PASS: {label} n={n} contains all 71 digitized values")

    # --- Test 2: 73 subset 80 ---
    rng73 = np.random.default_rng(seed)
    rng80 = np.random.default_rng(seed)
    pool_73 = build_empirical_pool(73, rng73)
    pool_80 = build_empirical_pool(80, rng80)
    if np.array_equal(pool_80[:73], pool_73):
        print("PASS: pool_80[:73] == pool_73 (73 subset 80)")
    else:
        print("FAIL: pool_80[:73] != pool_73")
        ok = False

    # --- Test 3: 81 subset 82 ---
    rng81 = np.random.default_rng(seed)
    rng82 = np.random.default_rng(seed)
    pool_81 = build_empirical_pool(81, rng81)
    pool_82 = build_empirical_pool(82, rng82)
    if np.array_equal(pool_82[:81], pool_81):
        print("PASS: pool_82[:81] == pool_81 (81 subset 82)")
    else:
        print("FAIL: pool_82[:81] != pool_81")
        ok = False

    # --- Test 4: Reproducibility ---
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed)
    pool_a = build_empirical_pool(73, rng_a)
    pool_b = build_empirical_pool(73, rng_b)
    if np.array_equal(pool_a, pool_b):
        print("PASS: Same seed produces identical pools")
    else:
        print("FAIL: Same seed produced different pools")
        ok = False

    if not ok:
        sys.exit(1)
    print("PASS: All empirical generator checks passed.")


if __name__ == "__main__":
    main()
