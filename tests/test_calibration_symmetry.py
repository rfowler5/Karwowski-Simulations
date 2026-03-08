"""TEST-3 (AUDIT): calibrate_rho(+rho) == -calibrate_rho(-rho).

The calibration functions work on abs(rho_target) and apply sign at the end.
If the sign logic has a bug this test catches it.  The assertion is exact
(pos + neg == 0.0) because both calls share the same cached calibration table
and the only difference is the `sign * result` multiply at the final step.

Covers both calibration modes (multipoint, single) and both nonparametric
and copula generators.  Does NOT test calibrate_rho_empirical (same sign
logic applies, analogous test can be added if empirical becomes primary).
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import CASES
from data_generator import calibrate_rho, calibrate_rho_copula


# Tolerance: should be exactly 0 (same cache hit, sign multiply only).
# 1e-9 guards against any future FP change while remaining far stricter
# than any calibration error.
_TOL = 1e-9

# Use small n_cal to keep the test fast.  Symmetry is a pure arithmetic
# property of the sign logic, not a function of calibration accuracy.
_N_CAL = 100


def _check_symmetry(fn_pos, fn_neg, label):
    """Return True if fn_pos and fn_neg are exact negatives of each other."""
    diff = abs(fn_pos + fn_neg)
    if diff > _TOL:
        print(f"  FAIL [{label}]: pos={fn_pos:.8f}, neg={fn_neg:.8f}, "
              f"|pos+neg|={diff:.2e} > tol={_TOL:.0e}")
        return False
    print(f"  PASS [{label}]: pos={fn_pos:.8f}, neg={fn_neg:.8f}")
    return True


def test_calibrate_rho_symmetry():
    """calibrate_rho(n, k, dt, +rho, ...) == -calibrate_rho(n, k, dt, -rho, ...)."""
    ok = True

    scenarios = [
        # (case_id, k, dist_type) — covers both analytes and distribution shapes
        (3, 4, "even"),
        (3, 4, "heavy_center"),
        (3, 7, "even"),
        (2, 4, "even"),
        (2, 10, "heavy_tail"),
    ]
    rho_targets = [0.10, 0.30, 0.50]

    for mode in ("multipoint", "single"):
        print(f"\n  calibration_mode={mode!r}")
        for case_id, k, dt in scenarios:
            case = CASES[case_id]
            n = case["n"]
            y_params = {"median": case["median"], "iqr": case["iqr"],
                        "range": case["range"]}
            for rho in rho_targets:
                pos = calibrate_rho(n, k, dt, +rho, y_params,
                                    calibration_mode=mode, n_cal=_N_CAL)
                neg = calibrate_rho(n, k, dt, -rho, y_params,
                                    calibration_mode=mode, n_cal=_N_CAL)
                label = f"case={case_id} k={k} dt={dt} rho={rho:+.2f} mode={mode}"
                if not _check_symmetry(pos, neg, label):
                    ok = False

    return ok


def test_calibrate_rho_copula_symmetry():
    """calibrate_rho_copula(n, k, dt, +rho, ...) == -calibrate_rho_copula(n, k, dt, -rho, ...)."""
    ok = True

    scenarios = [
        (3, 4, "even"),
        (3, 4, "heavy_center"),
        (2, 4, "even"),
    ]
    rho_targets = [0.10, 0.30, 0.50]

    print("\n  generator=copula")
    for case_id, k, dt in scenarios:
        case = CASES[case_id]
        n = case["n"]
        y_params = {"median": case["median"], "iqr": case["iqr"],
                    "range": case["range"]}
        for rho in rho_targets:
            pos = calibrate_rho_copula(n, k, dt, +rho, y_params, n_cal=_N_CAL)
            neg = calibrate_rho_copula(n, k, dt, -rho, y_params, n_cal=_N_CAL)
            label = f"copula case={case_id} k={k} dt={dt} rho={rho:+.2f}"
            if not _check_symmetry(pos, neg, label):
                ok = False

    return ok


def main():
    ok = True

    print("--- TEST-3: calibrate_rho sign symmetry ---")
    if not test_calibrate_rho_symmetry():
        ok = False

    print("\n--- TEST-3 (copula): calibrate_rho_copula sign symmetry ---")
    if not test_calibrate_rho_copula_symmetry():
        ok = False

    if ok:
        print("\nPASS: All calibration symmetry checks passed.")
    else:
        print("\nFAIL: One or more symmetry checks failed (see above).")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
