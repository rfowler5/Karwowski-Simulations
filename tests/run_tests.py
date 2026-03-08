"""
Test runner.

Quick (default): runs core tests (fast, foundational first).
Full (--full):   adds slower parallelism/benchmarking tests.

Calibration test: by default runs without --strict (reports flags only).
Use --strict with quick/full to make calibration exit 1 if any scenario is flagged.

Usage:
    python tests/run_tests.py              # quick, calibration not strict
    python tests/run_tests.py --strict     # quick, calibration strict
    python tests/run_tests.py --full       # full, calibration not strict
    python tests/run_tests.py --full --strict
"""

import argparse
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# Base argv for calibration in quick mode (no --strict by default)
CALIBRATION_QUICK_ARGV = [
    "--n-sims", "50", "--case", "3", "--n-distinct", "4",
    "--generators", "nonparametric",
]

QUICK_TESTS = [
    # --- Pure unit tests: no simulation, no calibration ---
    # Run these first so a low-level failure is diagnosed before any MC test obscures it.
    ("test_freq_dict_sums_to_n.py", []),         # freq_dict arithmetic; no dependencies
    ("test_data_generator.py", []),              # TEST-4/5: _fit_lognormal, _interp_with_extrapolation; pure math, no MC; before calibration because calibration calls these helpers
    ("test_spearman_2d.py", []),                 # rank/correlation core math; before anything that calls spearman_rho_2d

    # --- Bootstrap / CI ---
    # Depends on data generation working (caught above) but not on calibration.
    ("test_batch_bootstrap_ci.py", []),          # batch vs single bootstrap CI agreement; no calibration dependency

    # --- Calibration stack (ordered: unit → accuracy → symmetry) ---
    # Precompute first: if the precompute/eval split is broken, later calibration tests
    # would all fail with the same root cause, making diagnosis harder.
    ("test_calibration_precompute.py", []),      # precompute/eval split unit tests; run before calibration_accuracy so failures pinpoint the refactored path
    ("test_calibration_accuracy.py", None),      # filled in main() with CALIBRATION_QUICK_ARGV ± --strict; after precompute so a precompute bug is caught first
    ("test_calibration_symmetry.py", []),        # TEST-3: calibrate_rho(+rho) == -calibrate_rho(-rho); after accuracy so a broken table is caught before testing sign logic on top of it

    # --- Asymptotic formulas ---
    # No MC, no calibration; independent of the calibration stack above.
    ("test_asymptotic_formulas.py", []),         # TEST-2/6: spearman_var_h0 monotonicity + all four cases; purely analytic

    # --- Reproducibility / RNG ---
    # Depends on data generation and simulation being correct (validated above).
    ("test_reproducibility.py", []),             # seed-reproducibility of spearman_rho_2d and power estimates; after core correctness tests

    # --- Empirical generator ---
    # invalid_n first: if the generator crashes on bad input it will also crash the
    # longer empirical_generator test, so invalid_n gives the cleaner failure message.
    ("test_empirical_invalid_n.py", []),         # edge-case guard: empirical generator raises on n outside digitized data range
    ("test_empirical_generator.py", []),         # integration test for empirical generator output; after invalid_n

    # --- Permutation p-value ---
    # Independent of calibration and empirical generator; tests permutation_pvalue.py in isolation.
    ("test_permutation_pvalue.py", []),          # Davison-Hinkley p-value formula and null construction

    # --- Power simulation ---
    # boundary_warning first: it tests _check_and_warn_boundary, a helper called inside
    # min_detectable_rho.  If the helper is broken, power_sanity's MC run is wasted.
    ("test_boundary_warning.py", []),            # TEST-1: _check_and_warn_boundary deterministic helper; no MC, < 1 ms; before power_sanity which calls min_detectable_rho end-to-end
    ("test_power_sanity.py", []),                # end-to-end min_detectable_rho with MC; last substantial MC test

    # --- RNG independence ---
    # Last: integration-level check that data_rng and boot_rng streams don't cross-contaminate.
    # Placed after all functional tests so a contamination failure is not confused with a correctness bug.
    ("test_rng_data_independence.py", []),       # verifies SeedSequence.spawn(2) produces independent streams
]

FULL_ONLY_TESTS = [
    # These two calibration accuracy tests do a full sweep over all the cases to spot potential areas of structural bias.
    # These n-sims and n-cal were chosen because that's where it is known that they all currently pass--nothing flagged.
    ("test_calibration_accuracy.py", ["--n-sims 1000", "--n-cal 2000" "--generators copula,empirical", "--strict"]),
    ("test_calibration_accuracy.py", ["--n-sims 1000", "--n-cal 2000" "--generators parametric", "--strict"]),
    # These spawn subprocesses or run long MC loops; too slow for quick.
    ("test_nested_parallelism.py", ["--compare"]),           # nested multiprocessing deadlock / correctness check
    ("test_batch_sequential_vs_parallel.py", ["--compare"]), # batch vs sequential output agreement under parallelism
]


def run_tests(test_list):
    """Run each test in order. Return 0 if all pass, 1 on first failure."""
    for script_name, argv in test_list:
        if argv is None:
            continue  # skip placeholder
        script_path = TESTS_DIR / script_name
        if not script_path.exists():
            print(f"  SKIP (not found): {script_name}")
            continue

        print(f"  RUN: {script_name}")
        result = subprocess.run(
            [sys.executable, str(script_path)] + argv,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if result.returncode != 0:
            print(f"  FAIL: {script_name} (exit code {result.returncode})")
            if result.stdout:
                print("--- stdout ---")
                print(result.stdout)
            if result.stderr:
                print("--- stderr ---")
                print(result.stderr)
            return 1
        print(f"  PASS: {script_name}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run tests.")
    parser.add_argument("--full", action="store_true",
                        help="Include slower parallelism/benchmark tests")
    parser.add_argument("--strict", action="store_true",
                        help="In quick/full mode, run calibration test with --strict (exit 1 if any scenario flagged)")
    args = parser.parse_args()

    # Build quick test list: calibration gets CALIBRATION_QUICK_ARGV + optional --strict
    calibration_argv = list(CALIBRATION_QUICK_ARGV)
    if args.strict:
        calibration_argv.append("--strict")
    quick_list = []
    for script_name, argv in QUICK_TESTS:
        if argv is None:
            quick_list.append((script_name, calibration_argv))
        else:
            quick_list.append((script_name, argv))

    print("=" * 60)
    mode = "full" if args.full else "quick"
    strict_note = " (calibration strict)" if args.strict else ""
    print(f"Tests ({mode}{strict_note})")
    print("=" * 60)

    rc = run_tests(quick_list)
    if rc != 0:
        sys.exit(rc)

    if args.full:
        print()
        print("Full-only tests:")
        rc = run_tests(FULL_ONLY_TESTS)
        if rc != 0:
            sys.exit(rc)

    print()
    print("All tests passed.")


if __name__ == "__main__":
    main()
