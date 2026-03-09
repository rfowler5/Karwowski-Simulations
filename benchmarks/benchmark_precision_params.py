"""
Compute (n_sims, n_cal) and (n_reps, n_boot, n_cal) for target accuracy tiers.

Pure arithmetic from README formulas. Run from project root so config is importable.
Output is suitable for pasting into README or for use by benchmark_realistic_runtimes.py.

Power budget:   SE_bisection = C_BISECTION/sqrt(n_sims),  SE_cal = C_CAL/sqrt(n_cal)
                SE_total = sqrt(SE_bisection^2 + SE_cal^2), balanced allocation.
                See docs/UNCERTAINTY_BUDGET.md Part 1.

CI budget:      SE_inter_rep = SD_INTER_REP/sqrt(n_reps)
                SE_boot_quantile = sqrt(p*(1-p) / (n_reps * n_boot * f^2))
                    with p=0.025, f=1 (simplification; actual f ~ 0.5 at the 2.5th
                    percentile for bell-shaped distributions with SD ~ 0.11, making
                    real noise ~4x larger — still negligible at ~5.4% of inter-rep SE)
                SE_cal = C_CAL/sqrt(n_cal)   [shifts CI center, not width]
                SE_total = sqrt(SE_inter_rep^2 + SE_boot_quantile^2 + SE_cal^2), balanced.
                See docs/UNCERTAINTY_BUDGET.md Part 2.
"""
import math
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import TIERS

# --- Power coefficients ---
# Bisection: SE = C_BISECTION / sqrt(n_sims).
# Depends on: n and rho* (min detectable rho). Does NOT depend on n_sims.
# Asymptotic (no-tie) c at worst-case rho* (~0.33) for current data. Empirical (tied) c
# converges to ~0.12-0.15 at n_sims >= 5000; 0.17 provides ~20% margin. c increases
# monotonically with rho*, so use upper bound on rho* for conservatism.
# See docs/PRECISION_WHEN_DATA_CHANGES.md "Coefficient dependencies" and "Step-by-step
# process" for when and how to re-estimate for new data.
C_BISECTION = 0.17
# Calibration: SE = C_CAL / sqrt(n_cal).
# Depends on: n and probe rho (fixed at 0.30). Does NOT depend on n_sims, observed rho,
# rho*, or generator. Varies by probe in multipoint calibration (0.092-0.122 for n=73);
# 0.112 at probe 0.30 is appropriate for power. For CI at very small observed rho, the
# relevant probe is 0.10 where k ≈ 0.122. See docs/PRECISION_WHEN_DATA_CHANGES.md.
C_CAL = 0.112

# --- CI coefficients ---
# Inter-rep SD: SD(endpoint) = (1 - endpoint^2) * sqrt(1.06/(n-3)) * FHP_factor.
# Depends on: n, observed rho (from case data), and tie structure (FHP).
# Does NOT depend on n_sims, n_reps, n_boot, rho*, or generator.
# Worst case: Case 3 (n=73), upper endpoint near 0 -> (1 - 0.11^2) * sqrt(1.06/70)
# = 0.1216 (no ties), * 1.057 (worst-case FHP, k=4 heavy_center) = 0.129.
# Use 0.13 (rounded) as conservative estimate. Empirical max (n_reps=200): 0.130.
# See docs/PRECISION_WHEN_DATA_CHANGES.md for re-estimation guidance.
SD_INTER_REP = 0.13
N_BOOT_DEFAULT = 500
# Bootstrap quantile noise: p*(1-p) for p=0.025 (2.5th percentile endpoint).
_P_BOOT = 0.025
_P1MP = _P_BOOT * (1 - _P_BOOT)   # 0.024375; used with f=1 simplification


def power_params_for_halfwidth(w):
    """Return (n_sims, n_cal) so that 1.96*SE_total <= w (balanced SE components)."""
    se_target = w / 1.96
    se_each = se_target / math.sqrt(2)
    n_sims = (C_BISECTION / se_each) ** 2
    n_cal = (C_CAL / se_each) ** 2
    return (max(1, int(round(n_sims))), max(1, int(round(n_cal))))


def ci_params_for_halfwidth(w, n_boot=N_BOOT_DEFAULT):
    """Return (n_reps, n_boot, n_cal) so that 1.96*SE_total <= w.

    Balances three independent SE components: inter-rep noise, bootstrap quantile
    noise, and calibration MC noise.  Bootstrap quantile noise is negligible
    (< 4% of inter-rep SE at all tiers), so the allocation is effectively
    balanced between inter-rep and calibration.

    Parameters
    ----------
    w : float
        Target 95% CI half-width.
    n_boot : int
        Bootstrap resamples per rep (default 500).

    Returns
    -------
    (n_reps, n_boot, n_cal) : tuple of int
    """
    se_target = w / 1.96
    # Balance inter-rep and calibration noise (bootstrap quantile is negligible).
    se_each = se_target / math.sqrt(2)
    n_reps = (SD_INTER_REP / se_each) ** 2
    n_cal = (C_CAL / se_each) ** 2
    return (max(1, int(round(n_reps))), n_boot, max(1, int(round(n_cal))))


def ci_n_reps_for_halfwidth(w):
    """Return n_reps ignoring calibration noise (legacy; use ci_params_for_halfwidth).

    Returns the n_reps needed when calibration noise is not accounted for
    (i.e. n_cal is assumed infinite / negligible).  This gives approximately
    half the n_reps returned by ci_params_for_halfwidth.  Kept for backwards
    compatibility with scripts that only need n_reps.
    """
    se_target = w / 1.96
    n_reps = (SD_INTER_REP / se_target) ** 2
    return max(1, int(round(n_reps)))


def main():
    print("Precision parameters (from README formulas)")
    print("=" * 70)
    print()
    print("Power (min detectable rho):")
    print(f"  SE_bisection = {C_BISECTION}/sqrt(n_sims),  SE_cal = {C_CAL}/sqrt(n_cal)")
    print("  SE_total = sqrt(SE_bisection^2 + SE_cal^2),  half_width = 1.96 * SE_total")
    print("  Note: SE = c/sqrt(n_sims) is valid at all n_sims. Re-estimating c via")
    print("  scripts/estimate_bisection_c.py requires n_sims >= 5000 for stable results.")
    print()
    print("CI (bootstrap endpoint, calibration included):")
    print(f"  SE_inter_rep = {SD_INTER_REP}/sqrt(n_reps)")
    print(f"  SE_boot = sqrt({_P1MP}/(n_reps*n_boot*f^2)),  f=1 (simplification; actual f~0.5)")
    print(f"  SE_cal = {C_CAL}/sqrt(n_cal)  [shifts CI center; does not affect CI width]")
    print("  SE_total = sqrt(SE_inter_rep^2 + SE_boot^2 + SE_cal^2),  half_width = 1.96 * SE_total")
    print(f"  n_boot = {N_BOOT_DEFAULT} for all tiers (bootstrap quantile noise is negligible; ~5.4% of inter-rep SE with f=1)")
    print()

    print("Power (n_sims, n_cal) for each tier")
    print("-" * 70)
    print(f"{'Tier':<12} {'Half-width':<10} {'n_sims':<10} {'n_cal':<10} {'SE_total':<10} {'Actual half-width':<16}")
    print("-" * 70)
    for w, label in TIERS:
        n_sims, n_cal = power_params_for_halfwidth(w)
        se_bis = C_BISECTION / math.sqrt(n_sims)
        se_cal = C_CAL / math.sqrt(n_cal)
        se_total = math.sqrt(se_bis**2 + se_cal**2)
        actual_hw = 1.96 * se_total
        print(f"{label:<12} {w:<10.3f} {n_sims:<10} {n_cal:<10} {se_total:<10.6f} {actual_hw:<16.6f}")
    print()

    print("CI (n_reps, n_boot, n_cal) for each tier — includes calibration noise")
    print("-" * 90)
    print(f"{'Tier':<12} {'Half-width':<10} {'n_reps':<10} {'n_boot':<8} {'n_cal':<10} "
          f"{'SE_inter_rep':<14} {'SE_boot':<12} {'SE_cal':<10} {'Actual half-width':<16}")
    print("-" * 90)
    for w, label in TIERS:
        n_reps, n_boot, n_cal = ci_params_for_halfwidth(w)
        se_rep = SD_INTER_REP / math.sqrt(n_reps)
        se_boot = math.sqrt(_P1MP / (n_reps * n_boot))
        se_cal = C_CAL / math.sqrt(n_cal)
        se_total = math.sqrt(se_rep**2 + se_boot**2 + se_cal**2)
        actual_hw = 1.96 * se_total
        print(f"{label:<12} {w:<10.3f} {n_reps:<10} {n_boot:<8} {n_cal:<10} "
              f"{se_rep:<14.6f} {se_boot:<12.6f} {se_cal:<10.6f} {actual_hw:<16.6f}")
    print()

    print("Bootstrap quantile noise as % of inter-rep SE (f=1, n_boot=500):")
    print("-" * 50)
    for w, label in TIERS:
        n_reps, n_boot, _ = ci_params_for_halfwidth(w)
        se_rep = SD_INTER_REP / math.sqrt(n_reps)
        se_boot = math.sqrt(_P1MP / (n_reps * n_boot))
        pct = 100 * se_boot / se_rep
        print(f"  {label}: SE_boot = {se_boot:.6f},  {pct:.1f}% of inter-rep SE")
    print()

    # Machine-readable summary for script 2 or README
    print("Summary (for README or benchmark_realistic_runtimes.py)")
    print("-" * 50)
    print("Power:")
    for w, label in TIERS:
        n_sims, n_cal = power_params_for_halfwidth(w)
        print(f"  {label}: n_sims={n_sims}, n_cal={n_cal}")
    print("CI (calibration-corrected):")
    for w, label in TIERS:
        n_reps, n_boot, n_cal = ci_params_for_halfwidth(w)
        print(f"  {label}: n_reps={n_reps}, n_boot={n_boot}, n_cal={n_cal}")


if __name__ == "__main__":
    main()
