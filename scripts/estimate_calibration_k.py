"""
Estimate the calibration coefficient k for a given scenario (new data or tie structure).

The coefficient k appears in SE_cal = k / sqrt(n_cal), where k is the SD of a single
realised Spearman rho in the calibration process. It is used to plan n_cal for target
half-widths (see README "Equations to compute Precision" and docs/PRECISION_WHEN_DATA_CHANGES.md).

Two modes:
  - Default (no --analytical): empirical k from repeated calibration-mean runs with
    different seeds. Runs _mean_rho at the probe rho for many seeds and computes
    k_est = SD(means) * sqrt(n_cal). Use when you want k for the real (tied, rank-mixing)
    calibration process.
  - --analytical: k from the Bonett-Wright asymptotic formula. Uses only n and rho (the
    probe); ties and generator are ignored. This is the same Bonett-Wright SE used for
    CIs (see README "Why Bonett-Wright SE"), evaluated at the probe rho via the delta
    method: k = (1 - rho^2) * sqrt(1.06 / (n - 3)). The formula is conservative relative
    to empirical k (typically ~10% higher) because the BW asymptotic overestimates
    variance slightly for the rank-mixing mechanism, especially with ties.

Usage:
  python scripts/estimate_calibration_k.py --analytical                   # all 4 cases
  python scripts/estimate_calibration_k.py --analytical --case 3          # single case
  python scripts/estimate_calibration_k.py --analytical --rho 0.35        # custom probe
  python scripts/estimate_calibration_k.py --case 3 --n-distinct 4 --dist-type heavy_center
  python scripts/estimate_calibration_k.py --case 3 --replications 200

With different data: use the case that matches your n (or pass --n directly with
--analytical), then run this script. Use the reported k in
benchmarks/benchmark_precision_params.py (C_CAL) and re-run benchmark scripts.

Dependencies: k depends on n and probe rho (fixed at 0.30 by default). It does
NOT depend on n_sims, n_reps, observed rho, rho*, generator, or n_boot. The
analytical formula is instant and stable; the empirical mode needs sufficient
n_cal/replications for estimation accuracy, but the true k is unchanged. With
multipoint calibration, k varies by probe (lowest probe = largest k); see
docs/PRECISION_WHEN_DATA_CHANGES.md "Coefficient dependencies" for details.
"""

import math
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import numpy as np

from config import CASES, FREQ_DICT

PROBE_DEFAULT = 0.30


def k_analytical(n, rho):
    """Calibration coefficient k from the Bonett-Wright asymptotic formula.

    Derivation: The Bonett-Wright SE for Spearman's rho in Fisher z-space is
    SE(z) = sqrt(1.06 / (n - 3)). By the delta method (d(tanh z)/dz = 1 - rho^2),
    the SD of the sample Spearman rho at true rho is:

        k = (1 - rho^2) * sqrt(1.06 / (n - 3))

    This is the same BW SE used in asymptotic_ci (see power_asymptotic.py),
    evaluated at the calibration probe rho rather than the observed rho.

    Conservative for tied scenarios: empirical k with ties is typically ~10%
    lower than this no-tie formula, so this serves as an upper bound for
    planning n_cal.
    """
    return (1.0 - rho ** 2) * math.sqrt(1.06 / (n - 3))


def main():
    ap = argparse.ArgumentParser(
        description="Estimate calibration coefficient k for planning n_cal "
                    "(see docs/PRECISION_WHEN_DATA_CHANGES.md)."
    )
    ap.add_argument("--case", type=int, default=None, choices=(1, 2, 3, 4),
                    help="Case id (config.CASES). Default: all cases for --analytical, "
                         "case 3 for empirical.")
    ap.add_argument("--n", type=int, default=None,
                    help="Sample size (overrides case n). For --analytical only.")
    ap.add_argument("--rho", type=float, default=PROBE_DEFAULT,
                    help="Probe rho for analytical formula or empirical estimation. "
                         "Default 0.30 (single-point calibration probe).")
    ap.add_argument("--analytical", action="store_true",
                    help="Use Bonett-Wright formula. Instant, no simulation needed. "
                         "Ignores ties/generator.")
    ap.add_argument("--n-distinct", type=int, default=4,
                    help="Number of distinct x-values (empirical mode). Default 4.")
    ap.add_argument("--dist-type", type=str, default="heavy_center",
                    choices=("even", "heavy_tail", "heavy_center"),
                    help="x distribution type (empirical mode). Default heavy_center.")
    ap.add_argument("--replications", type=int, default=80,
                    help="Number of calibration-mean replications per n_cal "
                         "(empirical mode). Default 80.")
    ap.add_argument("--n-cals", type=int, nargs="+", default=[300, 500, 1000],
                    help="n_cal values to test (empirical mode). Default: 300 500 1000.")
    args = ap.parse_args()

    rho = args.rho

    # --analytical mode
    if args.analytical:
        if args.n is not None:
            cases_to_show = [("custom", args.n)]
        elif args.case is not None:
            c = CASES[args.case]
            cases_to_show = [(str(args.case), c["n"])]
        else:
            cases_to_show = [(str(cid), CASES[cid]["n"]) for cid in sorted(CASES)]

        print("Calibration coefficient k (analytical, Bonett-Wright)")
        print("=" * 58)
        print()
        print("  Formula: k = (1 - rho^2) * sqrt(1.06 / (n - 3))")
        print("  (Bonett-Wright asymptotic SD of sample Spearman rho)")
        print()
        print(f"  Probe rho = {rho}")
        print()
        print(f"  {'Case':<8} {'n':<6} {'k':<10} {'SE at n_cal=300':<18} {'SE at n_cal=1000'}")
        print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*18} {'-'*17}")
        for label, n in cases_to_show:
            k = k_analytical(n, rho)
            se300 = k / math.sqrt(300)
            se1000 = k / math.sqrt(1000)
            print(f"  {label:<8} {n:<6} {k:<10.4f} {se300:<18.6f} {se1000:.6f}")

        if len(cases_to_show) > 1:
            ks = [k_analytical(n, rho) for _, n in cases_to_show]
            print()
            print(f"  Range: {min(ks):.4f} - {max(ks):.4f}")
            print(f"  Worst case (smallest n): {max(ks):.4f}")

        print()
        print("Note: Analytical k is conservative (~10% above empirical) because the BW")
        print("asymptotic slightly overestimates variance for the rank-mixing mechanism,")
        print("especially with ties. For empirical k, omit --analytical.")
        print()
        print("Next steps: Use k in benchmarks/benchmark_precision_params.py (C_CAL);")
        print("  see docs/PRECISION_WHEN_DATA_CHANGES.md.")
        return 0

    # Empirical mode
    case_id = args.case if args.case is not None else 3
    case = CASES[case_id]
    n = args.n if args.n is not None else case["n"]
    y_params = {"median": case["median"], "iqr": case["iqr"]}

    from data_generator import _mean_rho, _get_x_template  # noqa: E402

    template = _get_x_template(n, args.n_distinct, args.dist_type,
                               FREQ_DICT, all_distinct=False)

    k_ana = k_analytical(n, rho)

    print(f"Scenario: case {case_id} (n={n}), n_distinct={args.n_distinct}, {args.dist_type}")
    print(f"Probe rho = {rho}, replications = {args.replications}")
    print(f"Analytical k (Bonett-Wright, no ties): {k_ana:.4f}")
    print()

    results = []
    for n_cal in args.n_cals:
        means = np.array([
            _mean_rho(rho, template, y_params, n_cal, seed)
            for seed in range(args.replications)
        ])
        se_est = np.std(means, ddof=1)
        k_est = se_est * math.sqrt(n_cal)
        results.append((n_cal, se_est, k_est))
        print(f"  n_cal = {n_cal:4d}  ->  SE(mean) = {se_est:.6f}  "
              f"k_est = {k_est:.4f}  (analytical: {k_ana:.4f})")

    print()
    k_mean = np.mean([r[2] for r in results])
    k_std = np.std([r[2] for r in results])
    print(f"  Empirical k across n_cal:  mean = {k_mean:.4f}, SD = {k_std:.4f}")
    print(f"  Analytical k (BW):         {k_ana:.4f}")
    ratio = k_mean / k_ana if k_ana > 0 else float("nan")
    print(f"  Ratio empirical/analytical: {ratio:.3f}")
    print()
    print("  Analytical k is conservative (upper bound); empirical k is the")
    print("  direct estimate. For planning, use analytical or a value between")
    print("  the two (e.g. the benchmark default 0.112).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
