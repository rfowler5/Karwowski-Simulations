"""Sanity/regression test for asymptotic power, CI, and min-detectable-rho formulas.

Tests
-----
Original smoke test  : Case 3 only; checks power in [0,1], CI in [-1,1], MDR in (0,1).
TEST-2 (AUDIT)       : spearman_var_h0 monotonicity — ties must inflate variance.
TEST-6 (AUDIT)       : All four cases — power, CI, MDR checks for each.
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import CASES
from power_asymptotic import (
    asymptotic_power,
    asymptotic_ci,
    min_detectable_rho_asymptotic,
    spearman_var_h0,
    get_x_counts,
)


# ---------------------------------------------------------------------------
# TEST-2 — spearman_var_h0 monotonicity
# ---------------------------------------------------------------------------

def test_spearman_var_h0_monotonicity():
    """TEST-2: Var(rho | ties) >= Var(rho | no ties) for every tie structure.

    Ties reduce the effective rank information, inflating the null variance.
    The FHP formula must preserve this direction.  A sign error would flip
    the relationship and produce anticonservative CIs.

    Also verifies the no-tie baseline equals exactly 1/(n-1).
    """
    ok = True

    # --- baseline: no-tie variance equals 1/(n-1) exactly ---
    for n in [73, 80, 81, 82]:
        var = spearman_var_h0(n, None)
        expected = 1.0 / (n - 1)
        if abs(var - expected) > 1e-12:
            print(f"  FAIL baseline n={n}: got {var:.6e}, expected {expected:.6e}")
            ok = False
        else:
            print(f"  PASS baseline n={n}: var_no_ties = {var:.6e} = 1/(n-1)")

    # --- ties inflate variance: var_ties >= var_no_ties ---
    # Representative tie structures covering both analytes and distribution shapes.
    tie_scenarios = [
        # (n, k, dist_type, description)
        (73,  4, "even",         "n=73  k=4  even"),
        (73,  4, "heavy_center", "n=73  k=4  heavy_center"),
        (73,  7, "heavy_tail",   "n=73  k=7  heavy_tail"),
        (80,  4, "even",         "n=80  k=4  even"),
        (80,  4, "heavy_center", "n=80  k=4  heavy_center"),
        (82, 10, "even",         "n=82  k=10 even"),
        (81,  4, "heavy_center", "n=81  k=4  heavy_center"),
    ]

    for n, k, dt, desc in tie_scenarios:
        x_counts = get_x_counts(n, k, distribution_type=dt)
        var_ties    = spearman_var_h0(n, x_counts)
        var_no_ties = spearman_var_h0(n, None)

        if var_ties < var_no_ties - 1e-15:
            print(f"  FAIL [{desc}]: var_ties={var_ties:.6e} < var_no_ties={var_no_ties:.6e}")
            ok = False
        else:
            ratio = var_ties / var_no_ties
            print(f"  PASS [{desc}]: var_ties/var_no_ties = {ratio:.6f}")

    return ok


# ---------------------------------------------------------------------------
# TEST-6 — asymptotic formulas across all four cases
# ---------------------------------------------------------------------------

def test_all_four_cases():
    """TEST-6: asymptotic_power, asymptotic_ci, min_detectable_rho_asymptotic
    must produce valid output for all four study cases.

    Each case uses k=4 even distribution (representative tie structure).
    Direction for MDR bisection matches the sign of observed_rho so that
    the result is in the expected half of [-1, 0) or (0, 1].

    Cases 1, 2, 4 have different n and observed rho from Case 3; edge cases
    in the FHP correction (especially for Cases 2 and 4 with small observed
    rho = 0.06) could manifest differently.
    """
    ok = True
    alpha = 0.05
    k = 4
    dt = "even"

    for case_id, case in CASES.items():
        n = case["n"]
        rho_obs = case["observed_rho"]
        direction = "negative" if rho_obs < 0 else "positive"
        x_counts = get_x_counts(n, k, distribution_type=dt)

        label = f"Case {case_id} (n={n}, rho_obs={rho_obs})"

        # --- asymptotic_power ---
        power = asymptotic_power(n, rho_obs, alpha=alpha, x_counts=x_counts)
        if not (0.0 <= power <= 1.0):
            print(f"  FAIL [{label}] asymptotic_power={power:.4f} out of [0,1]")
            ok = False
        else:
            print(f"  PASS [{label}] asymptotic_power={power:.4f}")

        # --- asymptotic_ci ---
        ci_lo, ci_hi = asymptotic_ci(rho_obs, n, alpha=alpha, x_counts=x_counts)
        ci_ok = (-1.0 <= ci_lo <= 1.0
                 and -1.0 <= ci_hi <= 1.0
                 and ci_lo < ci_hi
                 and ci_lo < rho_obs < ci_hi)
        if not ci_ok:
            print(f"  FAIL [{label}] CI=[{ci_lo:.4f}, {ci_hi:.4f}], "
                  f"rho_obs={rho_obs} — invalid")
            ok = False
        else:
            print(f"  PASS [{label}] CI=[{ci_lo:.4f}, {ci_hi:.4f}] "
                  f"width={ci_hi-ci_lo:.4f}")

        # --- min_detectable_rho_asymptotic ---
        mdr = min_detectable_rho_asymptotic(
            n, target_power=0.80, alpha=alpha,
            x_counts=x_counts, direction=direction)
        if direction == "positive":
            mdr_ok = 0.0 < mdr < 1.0
        else:
            mdr_ok = -1.0 < mdr < 0.0
        if not mdr_ok:
            print(f"  FAIL [{label}] MDR={mdr:.4f} not in expected range "
                  f"for direction={direction!r}")
            ok = False
        else:
            print(f"  PASS [{label}] MDR={mdr:.4f} (direction={direction})")

    return ok


# ---------------------------------------------------------------------------
# Original smoke test (Case 3 only) — kept for historical reference
# ---------------------------------------------------------------------------

def _smoke_test_case3():
    case = CASES[3]
    n = case["n"]
    rho_true = case["observed_rho"]
    alpha = 0.05
    x_counts = get_x_counts(n, 4, distribution_type="even")

    power = asymptotic_power(n, rho_true, alpha=alpha, x_counts=x_counts)
    assert 0.0 <= power <= 1.0, f"asymptotic_power out of range: {power}"
    print(f"  asymptotic_power(n={n}, rho={rho_true}): {power:.4f}")

    ci_lower, ci_upper = asymptotic_ci(rho_true, n, alpha=alpha, x_counts=x_counts)
    assert -1.0 <= ci_lower <= 1.0, f"CI lower out of range: {ci_lower}"
    assert -1.0 <= ci_upper <= 1.0, f"CI upper out of range: {ci_upper}"
    assert ci_lower < ci_upper, f"CI lower >= upper: [{ci_lower}, {ci_upper}]"
    assert ci_lower < rho_true < ci_upper, (
        f"Observed rho {rho_true} not inside CI [{ci_lower}, {ci_upper}]")
    print(f"  asymptotic_ci(rho_obs={rho_true}, n={n}): [{ci_lower:.4f}, {ci_upper:.4f}]")

    min_rho = min_detectable_rho_asymptotic(
        n, target_power=0.80, alpha=alpha, x_counts=x_counts, direction="positive")
    assert 0.0 < min_rho < 1.0, f"min_detectable_rho out of range: {min_rho}"
    print(f"  min_detectable_rho_asymptotic(n={n}, power=0.80): {min_rho:.4f}")


def main():
    ok = True

    print("\n--- Smoke test: Case 3 ---")
    try:
        _smoke_test_case3()
        print("  PASS: Case 3 smoke test passed.")
    except AssertionError as exc:
        print(f"  FAIL: {exc}")
        ok = False

    print("\n--- TEST-2: spearman_var_h0 monotonicity ---")
    if not test_spearman_var_h0_monotonicity():
        ok = False

    print("\n--- TEST-6: asymptotic formulas, all four cases ---")
    if not test_all_four_cases():
        ok = False

    if ok:
        print("\nPASS: All asymptotic formula checks passed.")
    else:
        print("\nFAIL: One or more checks failed (see above).")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
