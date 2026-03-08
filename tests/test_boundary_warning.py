"""TEST-1 (AUDIT): boundary warning fires when bisection hits search bounds.

Tests the extracted _check_and_warn_boundary helper directly — no MC
simulation needed.  Deterministic, runs in < 1 ms.
"""

import sys
import warnings
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from power_simulation import _check_and_warn_boundary


def test_boundary_warning():
    ok = True
    lo_bound, hi_bound = 0.25, 0.42

    # --- near hi_bound: should warn ---
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_and_warn_boundary(0.42005, lo_bound, hi_bound)
    if len(w) != 1 or not issubclass(w[0].category, UserWarning):
        print(f"  FAIL near hi_bound: expected 1 UserWarning, got {len(w)}")
        ok = False
    else:
        print(f"  PASS near hi_bound: warning emitted")

    # --- near lo_bound: should warn ---
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_and_warn_boundary(0.24995, lo_bound, hi_bound)
    if len(w) != 1 or not issubclass(w[0].category, UserWarning):
        print(f"  FAIL near lo_bound: expected 1 UserWarning, got {len(w)}")
        ok = False
    else:
        print(f"  PASS near lo_bound: warning emitted")

    # --- safely inside: should NOT warn ---
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_and_warn_boundary(0.3500, lo_bound, hi_bound)
    if len(w) != 0:
        print(f"  FAIL middle: expected 0 warnings, got {len(w)}")
        ok = False
    else:
        print(f"  PASS middle: no warning (correct)")

    # --- just outside tolerance: result = hi_bound - 1.001e-4 so distance > 1e-4 ---
    # (Exact 1e-4 is float-ambiguous; use clearly outside to avoid flakiness.)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_and_warn_boundary(0.42 - 1.001e-4, lo_bound, hi_bound)
    if len(w) != 0:
        print(f"  FAIL just outside tolerance: expected 0 warnings, got {len(w)}")
        ok = False
    else:
        print(f"  PASS just outside tolerance: no warning (correct)")

    # --- negative bounds ---
    neg_lo, neg_hi = -0.42, -0.25
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_and_warn_boundary(-0.41995, neg_lo, neg_hi)
    if len(w) != 1 or not issubclass(w[0].category, UserWarning):
        print(f"  FAIL near negative lo_bound: expected 1 UserWarning, got {len(w)}")
        ok = False
    else:
        print(f"  PASS near negative lo_bound: warning emitted")

    return ok


def main():
    print("\n--- TEST-1: boundary warning (deterministic helper) ---")
    if test_boundary_warning():
        print("\nPASS: All boundary warning checks passed.")
    else:
        print("\nFAIL: One or more boundary warning checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
