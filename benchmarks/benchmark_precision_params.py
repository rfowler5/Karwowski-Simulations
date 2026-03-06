"""
Compute (n_sims, n_cal) and (n_reps, n_boot) for target accuracy tiers.

Pure arithmetic from README formulas. No project imports. Run from any directory.
Output is suitable for pasting into README or for use by benchmark_realistic_runtimes.py.
"""
import math

# Target half-widths (95% CI half-width for min detectable rho or CI endpoint)
TIERS = [
    (0.01, "+/-0.01"),
    (0.002, "+/-0.002"),
    (0.001, "+/-0.001"),
]

# Power: SE_bisection = c/sqrt(n_sims), SE_cal = k/sqrt(n_cal)
# SE_total = sqrt(SE_bisection^2 + SE_cal^2), half_width = 1.96 * SE_total
# For target w: SE_total <= w/1.96. Balance: SE_bisection = SE_cal = (w/1.96)/sqrt(2)
# Asymptotic (no-tie) c at worst-case rho* (~0.33) for current data. Empirical (tied) c
# converges to ~0.12-0.15 at n_sims >= 5000; 0.17 provides ~20% margin. See README and estimate_bisection_c.py.
C_BISECTION = 0.17
# Analytical: k = (1 - rho^2) * sqrt(1.06/(n-3)) via Bonett-Wright delta method.
# For rho=0.30 (probe), range 0.105 (n=82) to 0.112 (n=73). Empirical ~0.10.
# 0.104 was earlier used as a practical default between analytical and empirical. Use 
# 0.112 as worst case scenario/more conservative estimate.
C_CAL = 0.112

# CI: SE = SD_INTER_REP/sqrt(n_reps), half_width = 1.96 * SE
# For target w: n_reps >= (SD_INTER_REP * 1.96 / w)^2
# Analytical: SD(endpoint) = (1 - endpoint^2) * sqrt(1.06/(n-3)) * FHP_factor.
# Worst case: Case 3 (n=73), upper endpoint near 0 -> (1 - 0.11^2) * sqrt(1.06/70)
# = 0.1216 (no ties), * 1.057 (worst-case FHP, k=4 heavy_center) = 0.129.
# Use 0.13 (rounded) as conservative estimate including tie correction.
# Empirical max from results/confidence_intervals.csv (n_reps=200): 0.130.
SD_INTER_REP = 0.13
N_BOOT_DEFAULT = 500


def power_params_for_halfwidth(w):
    """Return (n_sims, n_cal) so that 1.96*SE_total <= w (balanced SE components)."""
    se_target = w / 1.96
    se_each = se_target / math.sqrt(2)
    n_sims = (C_BISECTION / se_each) ** 2
    n_cal = (C_CAL / se_each) ** 2
    return (max(1, int(round(n_sims))), max(1, int(round(n_cal))))


def ci_n_reps_for_halfwidth(w):
    """Return n_reps so that 1.96*SE <= w (SE = SD_INTER_REP/sqrt(n_reps))."""
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
    print()
    print("CI (bootstrap endpoint):")
    print(f"  SE = {SD_INTER_REP}/sqrt(n_reps),  half_width = 1.96 * SE")
    print("  n_boot = 500 for all tiers (README: sufficient when n_reps is high)")
    print()

    print("Power (n_sims, n_cal) for each tier")
    print("-" * 50)
    print(f"{'Tier':<12} {'Half-width':<10} {'n_sims':<10} {'n_cal':<10} {'SE_total':<10} {'Actual half-width':<16}")
    print("-" * 50)
    for w, label in TIERS:
        n_sims, n_cal = power_params_for_halfwidth(w)
        se_bis = C_BISECTION / math.sqrt(n_sims)
        se_cal = C_CAL / math.sqrt(n_cal)
        se_total = math.sqrt(se_bis**2 + se_cal**2)
        actual_hw = 1.96 * se_total
        print(f"{label:<12} {w:<10.3f} {n_sims:<10} {n_cal:<10} {se_total:<10.6f} {actual_hw:<16.6f}")
    print()

    print("CI (n_reps, n_boot) for each tier")
    print("-" * 50)
    print(f"{'Tier':<12} {'Half-width':<10} {'n_reps':<10} {'n_boot':<10} {'SE':<10} {'Actual half-width':<16}")
    print("-" * 50)
    for w, label in TIERS:
        n_reps = ci_n_reps_for_halfwidth(w)
        se = SD_INTER_REP / math.sqrt(n_reps)
        actual_hw = 1.96 * se
        print(f"{label:<12} {w:<10.3f} {n_reps:<10} {N_BOOT_DEFAULT:<10} {se:<10.6f} {actual_hw:<16.6f}")
    print()

    # Machine-readable summary for script 2 or README
    print("Summary (for README or benchmark_realistic_runtimes.py)")
    print("-" * 50)
    print("Power:")
    for w, label in TIERS:
        n_sims, n_cal = power_params_for_halfwidth(w)
        print(f"  {label}: n_sims={n_sims}, n_cal={n_cal}")
    print("CI:")
    for w, label in TIERS:
        n_reps = ci_n_reps_for_halfwidth(w)
        print(f"  {label}: n_reps={n_reps}, n_boot={N_BOOT_DEFAULT}")


if __name__ == "__main__":
    main()
