"""
Asymptotic power and confidence-interval formulas for Spearman correlation.

Methods
-------
Power uses the non-central t-distribution with df = n-2, which is the
exact framework for the t-test statistic t = rho*sqrt(n-2)/sqrt(1-rho^2).
This is more accurate than the normal approximation because the
sqrt(1-rho^2) denominator captures the variance reduction as |rho| grows.

CIs use the Fisher z-transform (arctanh / tanh) with the Bonett-Wright
(2000) SE = sqrt(1.06/(n-3)).  The 1.06 factor accounts for the ~6%
efficiency loss of Spearman ranks vs Pearson.  The back-transform via
tanh guarantees the CI stays within [-1, 1] and is properly asymmetric.

Ties are handled via Fieller-Hartley-Pearson correction throughout:
  - Power: nc parameter scaled by sqrt(var_no_ties / var_ties)
  - CIs: SE in z-space scaled by sqrt(var_ties / var_no_ties)

Supports three tie-correction modes:
  - "with_tie_correction"    : variance adjusted for tied ranks
  - "without_tie_correction" : classical 1/(n-1) variance
  - "both"                   : returns results under both modes

Tie-correction impact for planned tie structures (n=80 example)
---------------------------------------------------------------
The tie correction inflates SE(rho) relative to the no-ties baseline
(SE = 1/sqrt(n-1) ≈ 0.1125 for n=80).  For the planned frequency
distributions in this study, the inflation is modest:

  Distribution                   SE(tc)   SE ratio
  all distinct                   0.1125   1.000
  even k=10 [8x10]              0.1131   1.005
  even k=4  [20,20,20,20]       0.1162   1.033
  heavy_center k=4 [12,30,29,9] 0.1189   1.057

More extreme heavy-centered concentrations (not used in this study but
mentioned for context) increase the correction further:

  extreme [4,35,34,7]            0.1228   1.092
  extreme [2,39,38,1]            0.1276   1.134

Note: numerical CI widths and min-detectable-rho values from the old
normal-approximation method have been removed.  Run the smoke test to
obtain updated values under the new non-central t / Fisher z methods.
"""

import numpy as np
from scipy.stats import norm, t as t_dist, nct

from config import FREQ_DICT, X_PARAMS, ALPHA


# ---------------------------------------------------------------------------
# Variance of Spearman rho under H0
# ---------------------------------------------------------------------------

def _tie_correction_factor(counts):
    """Compute the tie-correction term Σ(t_i³ - t_i) for a set of group sizes."""
    counts = np.asarray(counts, dtype=float)
    return np.sum(counts ** 3 - counts)


def spearman_var_h0(n, x_counts=None, y_counts=None, tie_correction=True):
    """Variance of Spearman rho under H0 (rho=0).

    Parameters
    ----------
    n : int
        Sample size.
    x_counts : array-like or None
        Group sizes for tied x-values.  If None, x is treated as all-distinct.
    y_counts : array-like or None
        Group sizes for tied y-values.  Typically None (no y ties).
    tie_correction : bool
        Whether to apply the Fieller-Hartley-Pearson tie correction.

    Returns
    -------
    float
        Var(rho_s) under H0.
    """
    if not tie_correction or (x_counts is None and y_counts is None):
        return 1.0 / (n - 1)

    Tx = _tie_correction_factor(x_counts) if x_counts is not None else 0.0
    Ty = _tie_correction_factor(y_counts) if y_counts is not None else 0.0
    n3n = n ** 3 - n

    Dx = (n3n - Tx) / 12.0
    Dy = (n3n - Ty) / 12.0

    if Dx <= 0 or Dy <= 0:
        return 1.0 / (n - 1)

    return (1.0 / (n - 1)) * (n3n / 12.0) ** 2 / (Dx * Dy)


# ---------------------------------------------------------------------------
# Asymptotic power (non-central t)
# ---------------------------------------------------------------------------

def asymptotic_power(n, rho_true, alpha=None, x_counts=None,
                     tie_correction=True, two_sided=True):
    """Power of a two-sided (or one-sided) test for Spearman rho != 0.

    Uses the non-central t-distribution.  The standard Spearman test
    statistic is:
        t = rho_hat * sqrt(n-2) / sqrt(1 - rho_hat^2)
    Under H0 this follows t(n-2); under the alternative it follows a
    non-central t(df=n-2, nc) with noncentrality parameter:
        nc = rho_true * sqrt(n-2) / sqrt(1 - rho_true^2)

    The sqrt(1 - rho^2) denominator captures variance reduction as |rho|
    grows toward 1, making this more accurate than the normal approximation
    which uses constant H0 variance.

    When ties are present, the Fieller-Hartley-Pearson correction inflates
    var(rho_hat) under H0, reducing the effective information.  This is
    incorporated by scaling nc by sqrt(var0_no_ties / var0_ties).
    """
    if alpha is None:
        alpha = ALPHA

    df = n - 2
    t_crit = t_dist.ppf(1 - alpha / (2 if two_sided else 1), df)

    # NC parameter: standard formula for correlation t-test
    nc = rho_true * np.sqrt(n - 2) / np.sqrt(max(1.0 - rho_true ** 2, 1e-12))

    # Adjust for ties: scale by sqrt(var_no_ties / var_ties) ≤ 1
    if tie_correction and x_counts is not None:
        var0_ties = spearman_var_h0(n, x_counts=x_counts, tie_correction=True)
        var0_no_ties = 1.0 / (n - 1)
        nc *= np.sqrt(var0_no_ties / var0_ties)*np.sqrt(1/1.06)


    if two_sided:
        power = (1.0 - nct.cdf(t_crit, df, nc)
                 + nct.cdf(-t_crit, df, nc))
    else:
        if rho_true >= 0:
            power = 1.0 - nct.cdf(t_crit, df, nc)
        else:
            power = nct.cdf(-t_crit, df, nc)

    return float(np.clip(power, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Minimum detectable rho (asymptotic)
# ---------------------------------------------------------------------------

def min_detectable_rho_asymptotic(n, target_power=0.80, alpha=None,
                                   x_counts=None, tie_correction=True,
                                   direction="positive"):
    """Find minimum |rho| detectable at *target_power* using bisection.

    Parameters
    ----------
    direction : str
        'positive' searches [0, 0.6]; 'negative' searches [-0.6, 0].

    Returns
    -------
    float
        Minimum detectable rho (signed).
    """
    if alpha is None:
        alpha = ALPHA

    if direction == "positive":
        lo, hi = 0.0, 0.6
        for _ in range(60):
            mid = (lo + hi) / 2.0
            pw = asymptotic_power(n, mid, alpha=alpha, x_counts=x_counts,
                                  tie_correction=tie_correction)
            if pw < target_power:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0
    else:
        lo, hi = -0.6, 0.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            pw = asymptotic_power(n, mid, alpha=alpha, x_counts=x_counts,
                                  tie_correction=tie_correction)
            if pw < target_power:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Asymptotic confidence interval (Fisher z-transform)
# ---------------------------------------------------------------------------

def asymptotic_ci(rho_obs, n, alpha=None, x_counts=None, tie_correction=True):
    """Two-sided CI for Spearman rho using the Fisher z-transform.

    The Fisher z-transform z = arctanh(rho) stabilises variance and
    improves normality.  For Spearman correlation, the Bonett-Wright
    (2000) SE in z-space is sqrt(1.06 / (n-3)), which accounts for
    the ~6% efficiency loss of ranks versus raw data.

    When ties are present, the Fieller-Hartley-Pearson correction
    inflates the variance: se_z is scaled by sqrt(var0_ties / var0_no_ties)
    so that the CI widens appropriately.

    The CI is constructed in z-space and back-transformed via tanh,
    guaranteeing the interval stays within [-1, 1] and is properly
    asymmetric around the point estimate.

    Returns
    -------
    (lower, upper) : tuple of float
    """
    if alpha is None:
        alpha = ALPHA

    z_obs = np.arctanh(np.clip(rho_obs, -0.9999, 0.9999))

    # Bonett-Wright SE in z-space (1.06 factor for Spearman efficiency)
    se_z = np.sqrt(1.06 / max(n - 3, 1))

    # Inflate SE for ties
    if tie_correction and x_counts is not None:
        var0_ties = spearman_var_h0(n, x_counts=x_counts, tie_correction=True)
        var0_no_ties = 1.0 / (n - 1)
        se_z *= np.sqrt(var0_ties / var0_no_ties)

    za = norm.ppf(1 - alpha / 2.0)
    z_lo = z_obs - za * se_z
    z_hi = z_obs + za * se_z

    return (float(np.tanh(z_lo)), float(np.tanh(z_hi)))


# ---------------------------------------------------------------------------
# Convenience: run both tie-correction modes
# ---------------------------------------------------------------------------

def asymptotic_results(n, rho_obs, target_power, alpha, x_counts,
                       direction, tie_correction_mode="with_tie_correction"):
    """Return dict with min-detectable rho and CI under the requested mode(s)."""
    modes = (["with_tie_correction", "without_tie_correction"]
             if tie_correction_mode == "both"
             else [tie_correction_mode])

    results = {}
    for mode in modes:
        tc = mode == "with_tie_correction"
        label = "tie_corrected" if tc else "no_tie_correction"
        md_rho = min_detectable_rho_asymptotic(
            n, target_power=target_power, alpha=alpha,
            x_counts=x_counts, tie_correction=tc, direction=direction)
        ci = asymptotic_ci(rho_obs, n, alpha=alpha,
                           x_counts=x_counts, tie_correction=tc)
        results[label] = {
            "min_detectable_rho": md_rho,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
        }
    return results


def get_x_counts(n, n_distinct, distribution_type=None, all_distinct=False,
                 x_counts=None, freq_dict=None):
    """Return the frequency counts for x-values (used for tie correction).

    Parameters
    ----------
    x_counts : array-like or None
        When provided, returned directly (for custom distributions).
    freq_dict : dict or None
        When provided with distribution_type "custom", use
        freq_dict[n][n_distinct]["custom"] instead of FREQ_DICT.
    """
    if x_counts is not None:
        return np.asarray(x_counts, dtype=int)
    if all_distinct:
        return np.ones(n, dtype=int)
    if freq_dict is not None and distribution_type == "custom":
        return np.array(freq_dict[n][n_distinct]["custom"])
    return np.array(FREQ_DICT[n][n_distinct][distribution_type])
