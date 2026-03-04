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
    ("test_freq_dict_sums_to_n.py", []),
    ("test_spearman_2d.py", []),
    ("test_batch_bootstrap_ci.py", []),
    ("test_calibration_accuracy.py", None),  # filled in main() with CALIBRATION_QUICK_ARGV ± --strict
    ("test_asymptotic_formulas.py", []),
    ("test_reproducibility.py", []),
    ("test_empirical_invalid_n.py", []),
    ("test_power_sanity.py", []),
    ("test_rng_data_independence.py", []),
]

FULL_ONLY_TESTS = [
    ("test_nested_parallelism.py", ["--compare"]),
    ("test_batch_sequential_vs_parallel.py", ["--compare"]),
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
