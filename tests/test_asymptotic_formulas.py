"""Sanity/regression test for asymptotic power, CI, and min-detectable-rho formulas."""

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
    get_x_counts,
)


def main():
    case = CASES[3]
    n = case["n"]
    rho_true = case["observed_rho"]
    alpha = 0.05
    x_counts = get_x_counts(n, 4, distribution_type="even")

    power = asymptotic_power(n, rho_true, alpha=alpha, x_counts=x_counts)
    assert 0.0 <= power <= 1.0, f"asymptotic_power out of range: {power}"
    print(f"asymptotic_power(n={n}, rho={rho_true}): {power:.4f}")

    ci_lower, ci_upper = asymptotic_ci(rho_true, n, alpha=alpha, x_counts=x_counts)
    assert -1.0 <= ci_lower <= 1.0, f"CI lower out of range: {ci_lower}"
    assert -1.0 <= ci_upper <= 1.0, f"CI upper out of range: {ci_upper}"
    assert ci_lower < ci_upper, f"CI lower >= upper: [{ci_lower}, {ci_upper}]"
    assert ci_lower < rho_true < ci_upper, (
        f"Observed rho {rho_true} not inside CI [{ci_lower}, {ci_upper}]")
    print(f"asymptotic_ci(rho_obs={rho_true}, n={n}): [{ci_lower:.4f}, {ci_upper:.4f}]")

    min_rho = min_detectable_rho_asymptotic(
        n, target_power=0.80, alpha=alpha, x_counts=x_counts, direction="positive")
    assert 0.0 < min_rho < 1.0, f"min_detectable_rho out of range: {min_rho}"
    print(f"min_detectable_rho_asymptotic(n={n}, power=0.80): {min_rho:.4f}")

    print("PASS: All asymptotic formula checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, Exception) as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
