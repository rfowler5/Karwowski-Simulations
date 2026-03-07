"""Verify same seed produces identical results for estimate_power and bootstrap_ci_averaged.

If this test fails, especially after making code changes to spearman_helpers or
other Numba-JIT'd modules, try clearing the Numba cache first: remove
__pycache__/*.nbc and __pycache__/*.nbi (or at least spearman_helpers.*.nbc/.nbi).
Stale cached bytecode can cause spurious reproducibility failures.
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import CASES
from power_simulation import estimate_power
from confidence_interval_calculator import bootstrap_ci_averaged

SEED = 42
# With fastmath=True in spearman_helpers Numba JIT, FP reassociation can cause
# tiny run-to-run jitter; use a tolerance instead of exact equality.
REPRODUCIBILITY_TOL = 1e-9


def main():
    case = CASES[3]
    n, k, dt = case["n"], 4, "even"
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
    rho_s = case["observed_rho"]
    ok = True

    # --- Power reproducibility ---
    p1 = estimate_power(n, k, dt, rho_s, y_params, n_sims=200, seed=SEED,
                        calibration_mode="single")
    p2 = estimate_power(n, k, dt, rho_s, y_params, n_sims=200, seed=SEED,
                        calibration_mode="single")
    if p1 != p2:
        print(f"FAIL: estimate_power not reproducible: {p1} != {p2}")
        print("Tip: If you recently changed code, clear Numba cache (__pycache__/*.nbc, __pycache__/*.nbi) and re-run.")
        ok = False
    else:
        print(f"PASS: estimate_power reproducible (seed={SEED}): power={p1:.4f}")

    # --- CI reproducibility ---
    r1 = bootstrap_ci_averaged(n, k, dt, rho_s, y_params,
                                n_reps=10, n_boot=50, seed=SEED,
                                calibration_mode="single")
    r2 = bootstrap_ci_averaged(n, k, dt, rho_s, y_params,
                                n_reps=10, n_boot=50, seed=SEED,
                                calibration_mode="single")
    ci1_lo, ci1_hi = r1["ci_lower"], r1["ci_upper"]
    ci2_lo, ci2_hi = r2["ci_lower"], r2["ci_upper"]
    # Use tolerance: with fastmath=True + parallel, bit-exact reproducibility
    # is not guaranteed. For bit-exact, set fastmath=False in all @njit in
    # spearman_helpers.py and clear Numba cache, then use: if ci1_lo != ci2_lo or ci1_hi != ci2_hi
    if abs(ci1_lo - ci2_lo) > REPRODUCIBILITY_TOL or abs(ci1_hi - ci2_hi) > REPRODUCIBILITY_TOL:
        print(f"FAIL: bootstrap_ci_averaged not reproducible: "
              f"[{ci1_lo}, {ci1_hi}] != [{ci2_lo}, {ci2_hi}]")
        print("Tip: If you recently changed code, clear Numba cache (__pycache__/*.nbc, __pycache__/*.nbi) and re-run.")
        ok = False
    else:
        print(f"PASS: bootstrap_ci_averaged reproducible (seed={SEED}): "
              f"CI=[{ci1_lo:.4f}, {ci1_hi:.4f}]")

    if not ok:
        sys.exit(1)
    print("PASS: All reproducibility checks passed.")


if __name__ == "__main__":
    main()
