"""
Configuration for Spearman power/CI simulations based on Karwowski et al. 2018.

Four cases from Table 3 of the paper: blood aluminum (B-Al) and hair aluminum
(H-Al), each with and without outlier exclusion, correlated against cumulative
vaccine aluminum exposure.
"""

# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------
CASES = {
    1: {
        "label": "B-Al, no outliers excluded",
        "analyte": "B-Al",
        "n": 80,
        "median": 15.4,
        "iqr": 19.3,
        "range": (12.7, 19.5),
        "observed_rho": -0.13,
        "observed_p": 0.26,
    },
    2: {
        "label": "H-Al, no outliers excluded",
        "analyte": "H-Al",
        "n": 82,
        "median": 42_542,
        "iqr": 51_408,
        "range": (32_527, 52_957),
        "observed_rho": 0.06,
        "observed_p": 0.56,
    },
    3: {
        "label": "B-Al, outliers excluded",
        "analyte": "B-Al",
        "n": 73,
        "median": 14.3,
        "iqr": 13.9,
        "range": (11.6, 18.2),
        "observed_rho": -0.13,
        "observed_p": 0.26,
    },
    4: {
        "label": "H-Al, outliers excluded",
        "analyte": "H-Al",
        "n": 81,
        "median": 42_485,
        "iqr": 47_830,
        "range": (32_524, 52_655),
        "observed_rho": 0.06,
        "observed_p": 0.56,
    },
}

# ---------------------------------------------------------------------------
# Cumulative vaccine aluminum (x-variable) parameters
# ---------------------------------------------------------------------------
X_PARAMS = {
    "median": 2.9,
    "iqr": 0.11,
    "range": (1.43, 3.55),
}

# ---------------------------------------------------------------------------
# Distinct x-values to test and distribution types
# ---------------------------------------------------------------------------
N_DISTINCT_VALUES = [4, 5, 6, 7, 8, 9, 10]
DISTRIBUTION_TYPES = ["even", "heavy_tail", "heavy_center"]

# ---------------------------------------------------------------------------
# Frequency dictionary
# Keys: sample_size -> n_distinct -> distribution_type -> list of counts
# Each list sums to the corresponding sample size.
# ---------------------------------------------------------------------------
FREQ_DICT = {
    80: {
        4: {
            "even": [20, 20, 20, 20],
            "heavy_tail": [23, 18, 18, 21],
            "heavy_center": [12, 30, 29, 9],
        },
        5: {
            "even": [16, 16, 16, 16, 16],
            "heavy_tail": [20, 13, 15, 13, 19],
            "heavy_center": [11, 15, 28, 15, 11],
        },
        6: {
            "even": [14, 13, 13, 13, 13, 14],
            "heavy_tail": [17, 11, 12, 12, 11, 17],
            "heavy_center": [8, 13, 19, 19, 13, 8],
        },
        7: {
            "even": [12, 11, 11, 12, 11, 11, 12],
            "heavy_tail": [15, 10, 9, 12, 9, 10, 15],
            "heavy_center": [6, 10, 15, 18, 15, 10, 6],
        },
        8: {
            "even": [10, 10, 10, 10, 10, 10, 10, 10],
            "heavy_tail": [14, 9, 8, 9, 9, 8, 9, 14],
            "heavy_center": [5, 8, 12, 15, 15, 12, 8, 5],
        },
        9: {
            "even": [9, 9, 9, 9, 8, 9, 9, 9, 9],
            "heavy_tail": [13, 8, 7, 8, 8, 8, 7, 8, 13],
            "heavy_center": [4, 7, 10, 13, 12, 13, 10, 7, 4],
        },
        10: {
            "even": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            "heavy_tail": [12, 7, 7, 7, 7, 7, 7, 7, 7, 12],
            "heavy_center": [4, 6, 8, 11, 11, 11, 11, 8, 6, 4],
        },
    },
    82: {
        4: {
            "even": [21, 20, 20, 21],
            "heavy_tail": [24, 18, 18, 22],
            "heavy_center": [12, 30, 30, 10],
        },
        5: {
            "even": [17, 16, 16, 16, 17],
            "heavy_tail": [21, 13, 15, 13, 20],
            "heavy_center": [11, 16, 28, 16, 11],
        },
        6: {
            "even": [14, 14, 13, 13, 14, 14],
            "heavy_tail": [18, 11, 12, 12, 11, 18],
            "heavy_center": [8, 14, 19, 19, 14, 8],
        },
        7: {
            "even": [12, 12, 12, 11, 12, 11, 12],
            "heavy_tail": [16, 10, 9, 12, 9, 10, 16],
            "heavy_center": [6, 10, 16, 18, 16, 10, 6],
        },
        8: {
            "even": [11, 10, 10, 10, 10, 10, 10, 11],
            "heavy_tail": [14, 9, 9, 9, 9, 9, 9, 14],
            "heavy_center": [5, 8, 12, 16, 16, 12, 8, 5],
        },
        9: {
            "even": [10, 9, 9, 9, 9, 9, 9, 9, 9],
            "heavy_tail": [13, 8, 8, 8, 8, 8, 8, 8, 13],
            "heavy_center": [4, 7, 10, 13, 14, 13, 10, 7, 4],
        },
        10: {
            "even": [9, 8, 8, 8, 8, 8, 8, 8, 8, 9],
            "heavy_tail": [12, 7, 7, 7, 8, 8, 7, 7, 7, 12],
            "heavy_center": [4, 6, 8, 11, 12, 12, 11, 8, 6, 4],
        },
    },
    73: {
        4: {
            "even": [19, 18, 18, 18],
            "heavy_tail": [21, 16, 16, 20],
            "heavy_center": [11, 27, 26, 9],
        },
        5: {
            "even": [15, 15, 14, 14, 15],
            "heavy_tail": [19, 12, 13, 12, 17],
            "heavy_center": [10, 14, 25, 14, 10],
        },
        6: {
            "even": [13, 12, 12, 12, 12, 12],
            "heavy_tail": [16, 10, 11, 11, 10, 15],
            "heavy_center": [7, 12, 17, 17, 12, 8],
        },
        7: {
            "even": [11, 10, 10, 11, 10, 11, 10],
            "heavy_tail": [14, 9, 8, 11, 8, 9, 14],
            "heavy_center": [5, 9, 14, 17, 14, 9, 5],
        },
        8: {
            "even": [10, 9, 9, 9, 9, 9, 9, 9],
            "heavy_tail": [13, 8, 7, 8, 8, 7, 8, 14],
            "heavy_center": [4, 7, 11, 14, 15, 11, 7, 4],
        },
        9: {
            "even": [9, 8, 8, 8, 8, 8, 8, 8, 8],
            "heavy_tail": [12, 7, 7, 7, 7, 7, 7, 7, 12],
            "heavy_center": [4, 6, 9, 12, 13, 12, 9, 5, 3],
        },
        10: {
            "even": [8, 7, 7, 8, 7, 7, 8, 7, 7, 7],
            "heavy_tail": [11, 6, 6, 7, 7, 7, 6, 6, 6, 11],
            "heavy_center": [3, 5, 8, 10, 11, 10, 10, 8, 5, 3],
        },
    },
    81: {
        4: {
            "even": [21, 20, 20, 20],
            "heavy_tail": [23, 18, 18, 22],
            "heavy_center": [12, 30, 29, 10],
        },
        5: {
            "even": [17, 16, 16, 16, 16],
            "heavy_tail": [20, 13, 15, 13, 20],
            "heavy_center": [11, 16, 27, 16, 11],
        },
        6: {
            "even": [14, 14, 13, 13, 13, 14],
            "heavy_tail": [17, 11, 12, 12, 11, 18],
            "heavy_center": [8, 13, 19, 19, 14, 8],
        },
        7: {
            "even": [12, 12, 11, 12, 11, 11, 12],
            "heavy_tail": [15, 10, 9, 12, 9, 10, 16],
            "heavy_center": [6, 10, 15, 18, 16, 10, 6],
        },
        8: {
            "even": [11, 10, 10, 10, 10, 10, 10, 10],
            "heavy_tail": [14, 9, 8, 9, 9, 8, 9, 15],
            "heavy_center": [5, 8, 12, 15, 16, 12, 8, 5],
        },
        9: {
            "even": [9, 9, 9, 9, 9, 9, 9, 9, 9],
            "heavy_tail": [13, 8, 7, 7, 7, 7, 7, 7, 18],
            "heavy_center": [4, 7, 10, 13, 13, 13, 10, 7, 4],
        },
        10: {
            "even": [9, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            "heavy_tail": [12, 7, 7, 7, 7, 7, 7, 7, 7, 13],
            "heavy_center": [4, 6, 8, 11, 11, 12, 11, 8, 6, 4],
        },
    },
}

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
N_SIMS = 10_000
# n_boot choice: For 2-decimal CI precision (third decimal trustworthy for
# rounding, e.g., knowing if can trust 0.345 and so know that rounds to 0.35), n_boot=1000 is recommended. The SE of the bootstrap quantile
# estimate scales as ~1/sqrt(n_boot); with n_reps averaging, the combined
# SE on the mean endpoint is ~0.001-0.002 for n_boot=1000. n_boot=500
# can differ by ~0.002-0.003 from higher values; n_boot=2000+ adds
# little beyond 1000. See README "Bootstrap CI" for verification commands.
N_BOOTSTRAP = 1000 # Don't need 10,000 for CI, 1000 is enough
ALPHA = 0.05
TARGET_POWER = 0.80

RHO_SEARCH_POSITIVE = (0.0, 0.6)
RHO_SEARCH_NEGATIVE = (-0.6, 0.0)

# "needed_direction_only" (default) or "both_directions"
POWER_SEARCH_DIRECTION = "needed_direction_only"

# "with_tie_correction" (default), "without_tie_correction", or "both"
ASYMPTOTIC_TIE_CORRECTION_MODE = "with_tie_correction"

# "multipoint" (default, more accurate) or "single" (faster, ~3x less calibration cost)
CALIBRATION_MODE = "multipoint"

USE_NUMBA = True
VECTORIZE_DATA_GENERATION = True
BATCH_CI_BOOTSTRAP = True
