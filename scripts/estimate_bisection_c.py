"""
Estimate the bisection coefficient c for a given scenario (new data or tie structure).

The coefficient c appears in SE_bisection(rho*) = c / sqrt(n_sims), where rho* is the
min detectable rho at target power (default 80%). It is used to plan n_sims for target
half-widths (see README "Equations to compute Precision" and docs/PRECISION_WHEN_DATA_CHANGES.md).

Two modes:
  - Default (no --analytical): empirical c from the actual power curve via simulation.
    Runs estimate_power at rho* ± delta and uses the finite-difference slope. Use when
    you want c for the real (tied, permutation-based) power curve.
  - --analytical: c from the asymptotic (no-tie) power curve formula. Uses only n and rho*;
    ties and generator are ignored. You must have rho* first: either pass --rho (e.g. from
    a previous min_detectable_rho run or from results/min_detectable_rho.csv) or omit --rho
    and the script will find rho* via bisection (one simulation run), then compute c
    analytically. Use to compare with the README derivation or when you have no simulation
    data for the slope.

Usage:
  python scripts/estimate_bisection_c.py --case 3 --n-distinct 4 --dist-type heavy_center
  python scripts/estimate_bisection_c.py --case 3 --generator empirical --n-sims 3000
  python scripts/estimate_bisection_c.py --case 3 --rho 0.33  # skip min_detectable_rho, use this rho*
  python scripts/estimate_bisection_c.py --case 3 --rho 0.33 --analytical  # analytical c (need rho* first)

With different data: use the case that matches your n and Y summary (or add a custom case in config),
then run this script. Use the reported c in benchmarks/benchmark_precision_params.py (C_BISECTION)
and re-run benchmark_precision_params.py and benchmark_realistic_runtimes.py (quick then scale).

Note on n_sims and slope stability: The finite-difference slope estimate has high variance at low
n_sims. At n_sims=2000 the empirical generator gives c ranging from ~0.10 to ~0.18 depending on
seed (e.g. seed 42 -> 0.18, seed 50 -> 0.10, seed 55 -> 0.12). At n_sims=5000-10000 the estimates
converge to ~0.14 regardless of seed. Use --n-sims 5000 or higher for stable c estimates. The
default --n-sims 2000 is adequate for quick checks but should not be used to set benchmark constants.

Dependencies: c depends on n and rho* (min detectable rho at 80% power). It does
NOT depend on n_sims (c is a property of the power curve, not the simulation count;
n_sims >= 5000 is needed only for stable *estimation* of c via finite-difference
slope). The analytical formula is generator-independent, but rho* itself is
generator-dependent. c increases with rho*, so use the upper bound on rho* for
conservatism. See docs/PRECISION_WHEN_DATA_CHANGES.md "Coefficient dependencies"
and "Step-by-step process" for the recommended workflow.
"""

import math
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
from scipy.stats import norm

from config import CASES, TARGET_POWER, ALPHA, CALIBRATION_MODE, N_CAL
from power_simulation import estimate_power, min_detectable_rho

# sqrt(pi * (1-pi)) for pi = 0.80
SQRT_PI_1MINUS_PI = (TARGET_POWER * (1 - TARGET_POWER)) ** 0.5  # 0.4


def c_analytical(n, rho_star, alpha=ALPHA):
    """
    Bisection coefficient c from the asymptotic (no-tie) power curve.

    power(rho) = Phi(ncp(rho) - z_{alpha/2}),  ncp(rho) = rho * sqrt(n-2) / sqrt(1 - rho^2).
    slope = phi(ncp - z) * d(ncp)/d(rho),  d(ncp)/d(rho) = sqrt(n-2) / (1 - rho^2)^{3/2}.
    c = sqrt(pi(1-pi)) / |slope|.
    """
    z_alpha2 = norm.ppf(1 - alpha / 2)
    ncp = rho_star * math.sqrt(n - 2) / math.sqrt(1 - rho_star ** 2)
    dncp_drho = math.sqrt(n - 2) / (1 - rho_star ** 2) ** 1.5
    slope = norm.pdf(ncp - z_alpha2) * dncp_drho
    if slope < 1e-10:
        return float("inf"), slope
    return SQRT_PI_1MINUS_PI / slope, slope


def main():
    parser = argparse.ArgumentParser(
        description="Estimate bisection coefficient c for planning n_sims (see docs/PRECISION_WHEN_DATA_CHANGES.md)."
    )
    parser.add_argument("--case", type=int, default=3, help="Case id (config.CASES). Default 3.")
    parser.add_argument("--n-distinct", type=int, default=4, help="Number of distinct x-values. Default 4.")
    parser.add_argument(
        "--dist-type",
        type=str,
        default="heavy_center",
        choices=["even", "heavy_tail", "heavy_center"],
        help="Tie distribution type. Default heavy_center.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="nonparametric",
        choices=["nonparametric", "copula", "linear", "empirical"],
        help="Generator for power simulation. Default nonparametric.",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=2000,
        help="n_sims for each power estimate (higher = more stable slope). Default 2000. Use 5000+ for converged estimates (see docstring).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="Finite-difference step for slope (rho* ± delta). Default 0.01.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        metavar="RHO",
        help="Use this as rho* (skip min_detectable_rho). Useful if you already have min detectable rho.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default 42.")
    parser.add_argument(
        "--direction",
        type=str,
        default="positive",
        choices=["positive", "negative"],
        help="Search direction for min_detectable_rho (only if --rho not set). Default positive.",
    )
    parser.add_argument(
        "--n-cal",
        type=int,
        default=None,
        help="Calibration samples per bisection step. Default from config.",
    )
    parser.add_argument(
        "--analytical",
        action="store_true",
        help="Use asymptotic (no-tie) formula for c. Requires rho* first: pass --rho or let script find it. Ignores ties/generator.",
    )
    args = parser.parse_args()

    case = CASES[args.case]
    n = case["n"]
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
    n_cal = args.n_cal if args.n_cal is not None else N_CAL

    if args.rho is not None:
        rho_star = abs(args.rho)
        print(f"Using supplied rho* = {rho_star:.4f} (skipping min_detectable_rho).")
    else:
        print(f"Finding min detectable rho (direction={args.direction}, generator={args.generator})...")
        md = min_detectable_rho(
            n,
            args.n_distinct,
            args.dist_type,
            y_params,
            generator=args.generator,
            n_sims=args.n_sims,
            direction=args.direction,
            seed=args.seed,
            calibration_mode=CALIBRATION_MODE,
            n_cal=n_cal,
        )
        rho_star = abs(md)
        print(f"  min_detectable_rho = {md:.4f}  =>  |rho*| = {rho_star:.4f}")

    if args.analytical:
        c, slope = c_analytical(n, rho_star)
        print()
        print("Results (analytical, asymptotic no-tie power curve)")
        print("----------------------------------------------------")
        print(f"  Scenario: case={args.case}, n={n}  (generator/tie structure not used in formula)")
        print(f"  |rho*| = {rho_star:.4f}")
        print(f"  |slope| = {slope:.4f}  (from d(power)/d(rho) at rho*)")
        print(f"  c = sqrt(pi(1-pi)) / |slope| = {SQRT_PI_1MINUS_PI:.4f} / {slope:.4f} = {c:.4f}")
        print()
        print("Note: Analytical c uses the same formula as the README derivation. For c from the")
        print("actual (tied) power curve, omit --analytical and run without --rho to use simulation.")
        print("Next steps: If c > 0.25, update C_BISECTION in benchmarks/benchmark_precision_params.py;")
        print("  see docs/PRECISION_WHEN_DATA_CHANGES.md.")
        return 0

    # Finite-difference slope at rho* (use positive rho; |slope| is same by symmetry)
    rho_lo = max(0.05, rho_star - args.delta)
    rho_hi = min(0.49, rho_star + args.delta)

    pw_lo = estimate_power(
        n,
        args.n_distinct,
        args.dist_type,
        rho_lo,
        y_params,
        generator=args.generator,
        n_sims=args.n_sims,
        seed=args.seed + 1,
        calibration_mode=CALIBRATION_MODE,
        n_cal=n_cal,
    )
    pw_hi = estimate_power(
        n,
        args.n_distinct,
        args.dist_type,
        rho_hi,
        y_params,
        generator=args.generator,
        n_sims=args.n_sims,
        seed=args.seed + 2,
        calibration_mode=CALIBRATION_MODE,
        n_cal=n_cal,
    )

    slope = (pw_hi - pw_lo) / (2 * args.delta)
    slope = abs(slope)
    if slope < 1e-6:
        print("Warning: slope nearly zero; c would be very large. Check scenario and delta.")
    c = SQRT_PI_1MINUS_PI / slope if slope >= 1e-6 else float("inf")

    print()
    print("Results")
    print("-------")
    print(f"  Scenario: case={args.case}, n={n}, k={args.n_distinct}, dist_type={args.dist_type}, generator={args.generator}")
    print(f"  |rho*| = {rho_star:.4f}")
    print(f"  power(rho={rho_lo:.4f}) = {pw_lo:.4f},  power(rho={rho_hi:.4f}) = {pw_hi:.4f}")
    print(f"  |slope| = {slope:.4f}")
    print(f"  c = sqrt(pi(1-pi)) / |slope| = {SQRT_PI_1MINUS_PI:.4f} / {slope:.4f} = {c:.4f}")
    print()
    print("Next steps:")
    print("  - If c is much larger than 0.17 (e.g. > 0.25), update C_BISECTION in benchmarks/benchmark_precision_params.py")
    print("  - Re-run benchmark_precision_params.py and benchmark_realistic_runtimes.py (quick then scale)")
    print("  - See docs/PRECISION_WHEN_DATA_CHANGES.md for full guidance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
