"""Unit tests for data_generator helper functions.

TEST-4 (AUDIT): _fit_lognormal(median, iqr) — np.exp(mu) recovers median exactly,
                sigma is positive.

TEST-5 (AUDIT): _interp_with_extrapolation(x, xp, fp) edge cases —
                left/right linear extrapolation beyond endpoints,
                single-point table returns fp[0] without IndexError.
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from data_generator import _fit_lognormal, _interp_with_extrapolation


# ---------------------------------------------------------------------------
# TEST-4 — _fit_lognormal median recovery
# ---------------------------------------------------------------------------

def test_fit_lognormal():
    """TEST-4: _fit_lognormal(median, iqr) must satisfy np.exp(mu) == median exactly.

    mu = log(median) is computed analytically with no approximation,
    so np.exp(mu) should equal median to floating-point precision (< 1e-10).
    sigma must be positive for any iqr > 0.

    Uses real-data medians and IQRs from all four study cases plus generic values.
    """
    # (median, iqr, description)
    test_inputs = [
        (2.5,      1.2,     "generic"),
        (14.3,    13.9,     "Case 3 B-Al"),
        (15.4,    19.3,     "Case 1 B-Al"),
        (42_485.0, 47_830.0, "Case 4 H-Al"),
        (42_542.0, 51_408.0, "Case 2 H-Al"),
        (1.0,      0.5,     "small"),
        (100.0,    1.0,     "tight IQR"),
        (0.001,    0.0001,  "very small"),
    ]

    ok = True
    for median, iqr, desc in test_inputs:
        mu, sigma = _fit_lognormal(median, iqr)

        # exp(mu) must equal median exactly (log is the analytical inverse)
        recovered = np.exp(mu)
        err = abs(recovered - median)
        if err > 1e-10:
            print(f"  FAIL [{desc}]: exp(mu)={recovered:.6e} != median={median:.6e}, "
                  f"err={err:.2e}")
            ok = False
        else:
            print(f"  PASS [{desc}]: median={median}, exp(mu)={recovered:.6e}, "
                  f"sigma={sigma:.4f}")

        # sigma must be positive
        if sigma <= 0:
            print(f"  FAIL [{desc}]: sigma={sigma} <= 0")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# TEST-5 — _interp_with_extrapolation edge cases
# ---------------------------------------------------------------------------

def test_interp_with_extrapolation():
    """TEST-5: Linear extrapolation beyond endpoints; single-point no IndexError.

    Uses a representative 3-point calibration table to verify:
      Case 1  x < xp[0]       : left linear extrapolation (slope from first two)
      Case 2  x > xp[-1]      : right linear extrapolation (slope from last two)
      Case 3  len(xp) == 1    : returns fp[0] for any x, no IndexError
      Case 4  x at interior   : standard np.interp result
      Case 5  x at endpoint   : returns fp endpoint exactly
    """
    # Representative 3-point table (calibration probe -> rho_in mapping)
    xp = [0.10, 0.30, 0.50]
    fp = [0.12, 0.33, 0.52]

    ok = True

    # --- Case 1: x < xp[0] — left extrapolation ---
    # slope = (fp[1]-fp[0]) / (xp[1]-xp[0]) = 0.21/0.20 = 1.05
    x = 0.05
    left_slope = (fp[1] - fp[0]) / (xp[1] - xp[0])   # 1.05
    expected = fp[0] + left_slope * (x - xp[0])        # 0.0675
    result = _interp_with_extrapolation(x, xp, fp)
    err = abs(result - expected)
    if err > 1e-12:
        print(f"  FAIL Case1 (x={x} < xp[0]={xp[0]}): "
              f"got {result:.8f}, expected {expected:.8f}, err={err:.2e}")
        ok = False
    else:
        print(f"  PASS Case1 (x={x} < xp[0]): result={result:.6f} "
              f"(slope={left_slope:.4f})")

    # --- Case 2: x > xp[-1] — right extrapolation ---
    # slope = (fp[-1]-fp[-2]) / (xp[-1]-xp[-2]) = 0.19/0.20 = 0.95
    x = 0.70
    right_slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])   # 0.95
    expected = fp[-1] + right_slope * (x - xp[-1])          # 0.71
    result = _interp_with_extrapolation(x, xp, fp)
    err = abs(result - expected)
    if err > 1e-12:
        print(f"  FAIL Case2 (x={x} > xp[-1]={xp[-1]}): "
              f"got {result:.8f}, expected {expected:.8f}, err={err:.2e}")
        ok = False
    else:
        print(f"  PASS Case2 (x={x} > xp[-1]): result={result:.6f} "
              f"(slope={right_slope:.4f})")

    # --- Case 3: single-point table — returns fp[0] for any x, no IndexError ---
    xp_1 = [0.30]
    fp_1 = [0.35]
    for x_test in [0.05, 0.30, 0.90]:
        try:
            result = _interp_with_extrapolation(x_test, xp_1, fp_1)
            if abs(result - fp_1[0]) > 1e-12:
                print(f"  FAIL Case3 (single-point, x={x_test}): "
                      f"got {result}, expected {fp_1[0]}")
                ok = False
            else:
                print(f"  PASS Case3 (single-point, x={x_test}): "
                      f"result={result:.6f}")
        except (IndexError, Exception) as exc:
            print(f"  FAIL Case3 (single-point, x={x_test}): "
                  f"raised {type(exc).__name__}: {exc}")
            ok = False

    # --- Case 4: x at interior point — standard interpolation ---
    x = 0.30   # exactly xp[1]; np.interp should return fp[1] exactly
    expected = 0.33
    result = _interp_with_extrapolation(x, xp, fp)
    if abs(result - expected) > 1e-12:
        print(f"  FAIL Case4 (x={x} at interior point): "
              f"got {result:.8f}, expected {expected:.8f}")
        ok = False
    else:
        print(f"  PASS Case4 (x={x} at interior point): result={result:.6f}")

    # --- Case 5: x exactly at endpoints — no extrapolation, exact values ---
    for x_ep, fp_ep in [(xp[0], fp[0]), (xp[-1], fp[-1])]:
        result = _interp_with_extrapolation(x_ep, xp, fp)
        if abs(result - fp_ep) > 1e-12:
            print(f"  FAIL Case5 (x={x_ep} at endpoint): "
                  f"got {result:.8f}, expected {fp_ep:.8f}")
            ok = False
        else:
            print(f"  PASS Case5 (x={x_ep} at endpoint): result={result:.6f}")

    return ok


def main():
    ok = True

    print("--- TEST-4: _fit_lognormal median recovery ---")
    if not test_fit_lognormal():
        ok = False

    print("\n--- TEST-5: _interp_with_extrapolation edge cases ---")
    if not test_interp_with_extrapolation():
        ok = False

    if ok:
        print("\nPASS: All data_generator unit tests passed.")
    else:
        print("\nFAIL: One or more checks failed (see above).")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
