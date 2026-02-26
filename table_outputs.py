"""
Summary table generation (no plots).

Produces three table types:
  1. Minimum detectable rho – tied scenarios by method (copula, linear, asymptotic)
  2. Confidence interval table – tied scenarios by method and tie-correction mode
  3. All-distinct combined table (4 rows) with both min-detectable rho and CIs
"""

import pandas as pd

from config import CASES


# ---------------------------------------------------------------------------
# Table 1: Minimum detectable rho (tied scenarios)
# ---------------------------------------------------------------------------

def build_min_detectable_table(power_results):
    """Build a DataFrame from the list of power-result dicts.

    *power_results* should be a combined list from copula, linear, and
    asymptotic analyses (each dict must include 'method', 'case',
    'n_distinct', 'dist_type', 'direction', 'min_detectable_rho',
    'all_distinct').
    """
    df = pd.DataFrame(power_results)
    df["case_label"] = df["case"].map(lambda c: CASES[c]["label"])
    col_order = ["case", "case_label", "n", "n_distinct", "dist_type",
                 "direction", "method", "min_detectable_rho", "all_distinct"]
    present = [c for c in col_order if c in df.columns]
    return df[present].sort_values(
        ["case", "all_distinct", "n_distinct", "dist_type", "method"]
    ).reset_index(drop=True)


def save_min_detectable_table(df, path="results/min_detectable_rho.csv"):
    """Write min-detectable rho table to CSV."""
    df.to_csv(path, index=False, float_format="%.4f")
    return path


# ---------------------------------------------------------------------------
# Table 2: Confidence intervals (tied scenarios)
# ---------------------------------------------------------------------------

def build_ci_table(ci_results):
    """Build a CI summary DataFrame.

    *ci_results* should be the list from confidence_interval_calculator.
    """
    df = pd.DataFrame(ci_results)
    df["case_label"] = df["case"].map(lambda c: CASES[c]["label"])
    return df.sort_values(
        ["case", "all_distinct", "n_distinct", "dist_type"]
    ).reset_index(drop=True)


def save_ci_table(df, path="results/confidence_intervals.csv"):
    df.to_csv(path, index=False, float_format="%.4f")
    return path


# ---------------------------------------------------------------------------
# Table 3: All-distinct combined table
# ---------------------------------------------------------------------------

def build_all_distinct_table(power_results, ci_results):
    """Single table for all-distinct scenarios (4 rows) with power + CI info."""
    pow_df = pd.DataFrame([r for r in power_results if r.get("all_distinct")])
    ci_df = pd.DataFrame([r for r in ci_results if r.get("all_distinct")])

    if pow_df.empty and ci_df.empty:
        return pd.DataFrame()

    if not pow_df.empty:
        pow_pivot = pow_df.pivot_table(
            index=["case", "n", "direction"],
            columns="method",
            values="min_detectable_rho",
            aggfunc="first"
        ).reset_index()
        pow_pivot.columns = [
            f"md_rho_{c}" if c not in ("case", "n", "direction") else c
            for c in pow_pivot.columns
        ]
    else:
        pow_pivot = pd.DataFrame()

    if not ci_df.empty:
        ci_sub = ci_df[["case", "n"]].copy()
        for col in ci_df.columns:
            if col.startswith(("boot_ci", "asym_")):
                ci_sub[col] = ci_df[col]
        ci_sub["observed_rho"] = ci_df["observed_rho"]
        ci_sub = ci_sub.drop_duplicates(subset=["case", "n"])
    else:
        ci_sub = pd.DataFrame()

    if not pow_pivot.empty and not ci_sub.empty:
        merged = pow_pivot.merge(ci_sub, on=["case", "n"], how="outer")
    elif not pow_pivot.empty:
        merged = pow_pivot
    else:
        merged = ci_sub

    merged["case_label"] = merged["case"].map(lambda c: CASES[c]["label"])
    return merged.sort_values("case").reset_index(drop=True)


def save_all_distinct_table(df, path="results/all_distinct_summary.csv"):
    df.to_csv(path, index=False, float_format="%.4f")
    return path


# ---------------------------------------------------------------------------
# Console display
# ---------------------------------------------------------------------------

def print_summary(power_df, ci_df, all_distinct_df):
    """Print key results to console."""
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)

    print("\n" + "=" * 80)
    print("MINIMUM DETECTABLE RHO (tied scenarios)")
    print("=" * 80)
    tied_pow = power_df[~power_df["all_distinct"]]
    if not tied_pow.empty:
        print(tied_pow.to_string(index=False))

    print("\n" + "=" * 80)
    print("CONFIDENCE INTERVALS (tied scenarios)")
    print("=" * 80)
    tied_ci = ci_df[~ci_df["all_distinct"]]
    if not tied_ci.empty:
        print(tied_ci.to_string(index=False))

    print("\n" + "=" * 80)
    print("ALL-DISTINCT SUMMARY")
    print("=" * 80)
    if not all_distinct_df.empty:
        print(all_distinct_df.to_string(index=False))
    print()
