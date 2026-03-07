"""
Estimate the inter-rep SD of bootstrap CI endpoints for planning n_reps.

The coefficient sigma_rep appears in SE = sigma_rep / sqrt(n_reps), where sigma_rep
is the SD of a single bootstrap CI endpoint (e.g. 2.5th percentile) across independent
simulated datasets. It is used to plan n_reps for target half-widths (see README
"CI endpoint precision" and docs/PRECISION_WHEN_DATA_CHANGES.md).

Two modes:
  - --analytical (default): sigma_rep from the Bonett-Wright asymptotic formula.
    Uses case parameters (n, observed_rho) and optionally the FHP tie correction.
    Instant, no simulation needed. The derivation:

        z_hat = arctanh(rho_hat) has SD = sqrt(1.06 / (n - 3))  (Bonett-Wright).
        CI endpoint in z-space: z_endpoint = z_hat +/- 1.96 * SE_z,
          so SD(z_endpoint) = SD(z_hat) = sqrt(1.06 / (n - 3)).
        Back-transform via delta method (d(tanh z)/dz = 1 - tanh^2(z)):
          SD(endpoint) = (1 - endpoint^2) * sqrt(1.06 / (n - 3))
        With ties: multiply by FHP factor sqrt(var_ties / var_no_ties).

    The worst-case endpoint is the one closest to zero, where (1 - endpoint^2) is
    maximised. The script computes both endpoints and reports the max SD.

  - Without --analytical: empirical sigma_rep from actual bootstrap CI runs.
    Runs bootstrap_ci_averaged at the observed rho for many seeds and computes
    the SD of the CI endpoints across runs. Use when you want sigma_rep for the
    real (tied, bootstrap-based) CI process.

Usage:
  python scripts/estimate_interrep_sd.py --analytical               # all 4 cases, no ties
  python scripts/estimate_interrep_sd.py --analytical --case 3      # single case
  python scripts/estimate_interrep_sd.py --analytical --with-ties   # include worst-case FHP
  python scripts/estimate_interrep_sd.py --case 3 --n-distinct 4 --dist-type heavy_center
  python scripts/estimate_interrep_sd.py --case 3 --n-reps 200 --n-boot 500

With different data: use the case that matches your n (or pass --n and --rho directly
with --analytical), then run this script. Use the reported sigma_rep in
benchmarks/benchmark_precision_params.py (SD_INTER_REP) and re-run benchmark scripts.
"""

import math
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import numpy as np

from config import CASES, FREQ_DICT, N_DISTINCT_VALUES, DISTRIBUTION_TYPES


def sd_analytical(n, rho, fhp_factor=1.0):
    """Inter-rep SD of CI endpoint from Bonett-Wright + delta method.

    Returns (sd_lower, sd_upper, endpoint_lower, endpoint_upper).

    Derivation:
        z = arctanh(rho), SE_z = sqrt(1.06/(n-3)) * fhp_factor
        z_lo = z - 1.96*SE_z, z_hi = z + 1.96*SE_z
        endpoint = tanh(z_endpoint)
        SD(endpoint) = (1 - endpoint^2) * SE_z   (delta method on tanh)
    """
    se_z = math.sqrt(1.06 / (n - 3)) * fhp_factor
    z = math.atanh(rho)
    z_lo = z - 1.96 * se_z
    z_hi = z + 1.96 * se_z
    lo = math.tanh(z_lo)
    hi = math.tanh(z_hi)
    sd_lo = (1.0 - lo ** 2) * se_z
    sd_hi = (1.0 - hi ** 2) * se_z
    return sd_lo, sd_hi, lo, hi


def fhp_factor_worst_case(n):
    """Worst-case FHP SE inflation factor across all tie structures for sample size n.

    Computes sqrt(var_ties / var_no_ties) for every (n_distinct, dist_type)
    in FREQ_DICT[n] and returns the maximum.
    """
    from power_asymptotic import spearman_var_h0

    var_no_ties = spearman_var_h0(n, tie_correction=False)
    worst = 1.0
    if n not in FREQ_DICT:
        return worst
    for k in FREQ_DICT[n]:
        for dt in FREQ_DICT[n][k]:
            counts = FREQ_DICT[n][k][dt]
            var_ties = spearman_var_h0(n, x_counts=counts, tie_correction=True)
            factor = math.sqrt(var_ties / var_no_ties)
            worst = max(worst, factor)
    return worst


def main():
    ap = argparse.ArgumentParser(
        description="Estimate inter-rep SD of bootstrap CI endpoints for planning "
                    "n_reps (see docs/PRECISION_WHEN_DATA_CHANGES.md)."
    )
    ap.add_argument("--case", type=int, default=None, choices=(1, 2, 3, 4),
                    help="Case id (config.CASES). Default: all cases for --analytical, "
                         "case 3 for empirical.")
    ap.add_argument("--n", type=int, default=None,
                    help="Sample size (overrides case n). For --analytical only.")
    ap.add_argument("--rho", type=float, default=None,
                    help="Observed rho (overrides case rho). For --analytical only.")
    ap.add_argument("--analytical", action="store_true",
                    help="Use Bonett-Wright + delta method formula. Instant, no "
                         "simulation. Default mode.")
    ap.add_argument("--with-ties", action="store_true",
                    help="Include worst-case FHP tie correction (--analytical only).")
    ap.add_argument("--n-distinct", type=int, default=4,
                    help="Number of distinct x-values (empirical mode). Default 4.")
    ap.add_argument("--dist-type", type=str, default="heavy_center",
                    choices=("even", "heavy_tail", "heavy_center"),
                    help="x distribution type (empirical mode). Default heavy_center.")
    ap.add_argument("--generator", type=str, default="nonparametric",
                    choices=("nonparametric", "copula", "empirical"),
                    help="Generator for bootstrap CI (empirical mode). Default nonparametric.")
    ap.add_argument("--n-reps", type=int, default=200,
                    help="Bootstrap reps per run (empirical mode). Default 200.")
    ap.add_argument("--n-boot", type=int, default=500,
                    help="Bootstrap resamples (empirical mode). Default 500.")
    ap.add_argument("--replications", type=int, default=30,
                    help="Number of independent runs to estimate SD (empirical mode). "
                         "Default 30.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base seed (empirical mode). Default 42.")
    args = ap.parse_args()

    # --analytical mode (default when no empirical args used)
    if args.analytical or (args.case is None and args.n is None
                           and not _has_empirical_args(args)):
        return _run_analytical(args)
    return _run_empirical(args)


def _has_empirical_args(args):
    """Check if user passed empirical-specific arguments."""
    defaults = {"n_distinct": 4, "dist_type": "heavy_center",
                "generator": "nonparametric", "n_reps": 200, "n_boot": 500,
                "replications": 30, "seed": 42}
    for k, v in defaults.items():
        if getattr(args, k) != v:
            return True
    return False


def _run_analytical(args):
    if args.n is not None and args.rho is not None:
        cases_to_show = [("custom", args.n, args.rho)]
    elif args.case is not None:
        c = CASES[args.case]
        cases_to_show = [(str(args.case), c["n"], c["observed_rho"])]
    else:
        cases_to_show = [(str(cid), CASES[cid]["n"], CASES[cid]["observed_rho"])
                         for cid in sorted(CASES)]

    print("Inter-rep SD of bootstrap CI endpoints (analytical, Bonett-Wright)")
    print("=" * 72)
    print()
    print("  Formula: SD(endpoint) = (1 - endpoint^2) * sqrt(1.06/(n-3))")
    if args.with_ties:
        print("           * sqrt(var_ties / var_no_ties)  [FHP worst-case]")
    print("  (Bonett-Wright SE in z-space, delta method back to rho-space)")
    print()

    header = (f"  {'Case':<8} {'n':<5} {'rho':<7} {'FHP':<6} "
              f"{'Lower':<8} {'SD(Lo)':<8} {'Upper':<8} {'SD(Up)':<8} {'Max SD':<8}")
    print(header)
    print(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    overall_max = 0.0
    for label, n, rho in cases_to_show:
        fhp = fhp_factor_worst_case(n) if args.with_ties else 1.0
        sd_lo, sd_hi, lo, hi = sd_analytical(n, rho, fhp_factor=fhp)
        mx = max(sd_lo, sd_hi)
        overall_max = max(overall_max, mx)
        fhp_str = f"{fhp:.3f}" if args.with_ties else "1.000"
        print(f"  {label:<8} {n:<5} {rho:<7.2f} {fhp_str:<6} "
              f"{lo:<8.4f} {sd_lo:<8.4f} {hi:<8.4f} {sd_hi:<8.4f} {mx:<8.4f}")

    print()
    print(f"  Overall worst case: {overall_max:.4f}")
    if not args.with_ties:
        # Also show what the tie-corrected value would be
        worst_fhp = max(fhp_factor_worst_case(n) for _, n, _ in cases_to_show)
        print(f"  With worst-case FHP ({worst_fhp:.3f}): {overall_max * worst_fhp:.4f}")
    print()

    print("  The endpoint closest to zero has the highest SD, because")
    print("  (1 - endpoint^2) is maximised near zero. Smallest n amplifies SE_z.")
    print()
    print("Note: Analytical SD is conservative for bootstrap CI endpoints (the BW")
    print("asymptotic slightly overestimates variance). For empirical SD, omit --analytical.")
    print()
    print("Next steps: Use max SD (rounded up) in benchmarks/benchmark_precision_params.py")
    print("  (SD_INTER_REP); see docs/PRECISION_WHEN_DATA_CHANGES.md.")
    return 0


def _run_empirical(args):
    from confidence_interval_calculator import bootstrap_ci_averaged

    case_id = args.case if args.case is not None else 3
    case = CASES[case_id]
    n = args.n if args.n is not None else case["n"]
    rho = args.rho if args.rho is not None else case["observed_rho"]
    y_params = {"median": case["median"], "iqr": case["iqr"],
                "range": case["range"]}

    print(f"Scenario: case {case_id} (n={n}), rho={rho}, k={args.n_distinct}, "
          f"{args.dist_type}, generator={args.generator}")
    print(f"n_reps={args.n_reps}, n_boot={args.n_boot}, replications={args.replications}")

    # Analytical reference
    sd_lo_a, sd_hi_a, _, _ = sd_analytical(n, rho)
    print(f"Analytical SD (no ties): lower={sd_lo_a:.4f}, upper={sd_hi_a:.4f}, "
          f"max={max(sd_lo_a, sd_hi_a):.4f}")
    print()

    lower_sds = []
    upper_sds = []
    for i in range(args.replications):
        seed = args.seed + i * 1000
        result = bootstrap_ci_averaged(
            n=n,
            n_distinct=args.n_distinct,
            distribution_type=args.dist_type,
            rho_s=rho,
            y_params=y_params,
            n_reps=args.n_reps,
            n_boot=args.n_boot,
            seed=seed,
            generator=args.generator,
        )
        lo_ci = result["ci_lower"]
        hi_ci = result["ci_upper"]
        lower_sds.append(result["ci_lower_sd"])
        upper_sds.append(result["ci_upper_sd"])
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Run {i+1}/{args.replications}: CI = [{lo_ci:.4f}, {hi_ci:.4f}]")

    sd_lo_emp = float(np.mean(lower_sds))
    sd_hi_emp = float(np.mean(upper_sds))
    se_lo = np.std(lower_sds, ddof=1) / math.sqrt(args.replications)
    se_hi = np.std(upper_sds, ddof=1) / math.sqrt(args.replications)

    print()
    print("Results")
    print("-" * 50)
    print(f"  SD(lower endpoint): {sd_lo_emp:.4f}  (analytical: {sd_lo_a:.4f})")
    print(f"  SD(upper endpoint): {sd_hi_emp:.4f}  (analytical: {sd_hi_a:.4f})")
    print(f"  Max empirical SD:   {max(sd_lo_emp, sd_hi_emp):.4f}  "
          f"(analytical: {max(sd_lo_a, sd_hi_a):.4f})")
    print()
    print("  Note: Empirical SD has sampling noise from finite replications.")
    print(f"  SE of the σ_rep estimate ~ std(per-run SDs)/sqrt(replications) "
          f"~ {max(se_lo, se_hi):.4f}")
    print()
    print("Next steps: Use max SD (rounded up) in benchmarks/benchmark_precision_params.py")
    print("  (SD_INTER_REP); see docs/PRECISION_WHEN_DATA_CHANGES.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
