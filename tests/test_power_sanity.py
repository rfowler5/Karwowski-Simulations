"""Sanity check: under H0 (rho=0), power should be near alpha (type I error rate)."""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import CASES
from power_simulation import estimate_power


def main():
    case = CASES[3]
    n, k, dt = case["n"], 4, "even"
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}

    alpha = 0.05
    n_sims = 2000
    power = estimate_power(n, k, dt, rho_s=0.0, y_params=y_params,
                           n_sims=n_sims, alpha=alpha, seed=123,
                           calibration_mode="single")

    lo, hi = 0.02, 0.10
    print(f"Power at rho=0 (n_sims={n_sims}): {power:.4f}  (expected in [{lo}, {hi}])")
    if not (lo <= power <= hi):
        print(f"FAIL: Power {power:.4f} outside plausible range [{lo}, {hi}] for alpha={alpha}")
        sys.exit(1)
    print("PASS: Power under H0 is near alpha.")


if __name__ == "__main__":
    main()
