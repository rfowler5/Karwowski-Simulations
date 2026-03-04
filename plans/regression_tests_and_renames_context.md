# Implementation Context: Regression Tests and Renames

This document provides all the concrete details a model needs to implement the plan in `plans/regression_tests_and_renames_plan.md`. Work through the steps in order. Each step is self-contained with exact file paths, line numbers, old→new text, and code to write.

The repo root is the working directory. All paths are relative to it.

---

## Step 1: Rename test files

Use `git mv` for each rename so git tracks the history:

```
git mv tests/test_simulation_accuracy.py tests/test_calibration_accuracy.py
git mv tests/validation_test_spearman2d.py tests/test_spearman_2d.py
git mv tests/validate_batch_ci_three_steps.py tests/test_batch_bootstrap_ci.py
git mv tests/verify_nested_parallelism.py tests/test_nested_parallelism.py
git mv tests/verify_big_gap.py tests/test_batch_sequential_vs_parallel.py
```

### 1a: Update self-references inside renamed files

**`tests/test_calibration_accuracy.py`** (was `test_simulation_accuracy.py`):
- Line 16: `from test_simulation_accuracy import main` → `from test_calibration_accuracy import main`
- Line 23: `python tests/test_simulation_accuracy.py --n-sims 50 --case 3 --n-distinct 4` → `python tests/test_calibration_accuracy.py --n-sims 50 --case 3 --n-distinct 4`
- Line 26: `python tests/test_simulation_accuracy.py --n-sims 200` → `python tests/test_calibration_accuracy.py --n-sims 200`
- Line 29: `python tests/test_simulation_accuracy.py --case 3 --freq 19,18,18,18 --n-sims 50` → `python tests/test_calibration_accuracy.py --case 3 --freq 19,18,18,18 --n-sims 50`
- Line 32: `python tests/test_simulation_accuracy.py --generators copula,nonparametric` → `python tests/test_calibration_accuracy.py --generators copula,nonparametric`

**`tests/test_nested_parallelism.py`** (was `verify_nested_parallelism.py`):
- Line 6: `python tests/verify_nested_parallelism.py` → `python tests/test_nested_parallelism.py`
- Line 9: `python tests/verify_nested_parallelism.py --compare` → `python tests/test_nested_parallelism.py --compare`

**`tests/test_batch_sequential_vs_parallel.py`** (was `verify_big_gap.py`):
- Line 10: `python tests/verify_big_gap.py --compare` → `python tests/test_batch_sequential_vs_parallel.py --compare`
- Line 11: `python tests/verify_big_gap.py --seq` → `python tests/test_batch_sequential_vs_parallel.py --seq`
- Line 12: `python tests/verify_big_gap.py --par` → `python tests/test_batch_sequential_vs_parallel.py --par`

**`tests/test_spearman_2d.py`** and **`tests/test_batch_bootstrap_ci.py`**: No self-references to update.

### 1b: Update references in documentation and plans

**`README.md`** — Replace every occurrence of the old names with new names. The old names appear at these locations (use replace-all for each old→new pair):

| Old string | New string | Locations (approximate lines) |
|---|---|---|
| `test_simulation_accuracy` | `test_calibration_accuracy` | Lines 170, 248, 251, 254, 256, 259, 262, 265, 301, 318 |
| `validate_batch_ci_three_steps` | `test_batch_bootstrap_ci` | Lines 303, 430, 506 |
| `validation_test_spearman2d` | `test_spearman_2d` | Line 302 |
| `verify_nested_parallelism` | `test_nested_parallelism` | Lines 304, 430, 506 |
| `verify_big_gap` | `test_batch_sequential_vs_parallel` | Lines 304, 430, 506 |

Also update description text if needed:
- Line 301: `test_simulation_accuracy.py — Generator accuracy testing` → `test_calibration_accuracy.py — Calibration accuracy testing`

**`PROJECT_SUMMARY.md`** — Same pattern:

| Old string | New string | Locations (approximate lines) |
|---|---|---|
| `test_simulation_accuracy` | `test_calibration_accuracy` | Lines 83, 127 |
| `validate_batch_ci_three_steps` | `test_batch_bootstrap_ci` | Line 84 |
| `validation_test_spearman2d` | `test_spearman_2d` | Line 82 |
| `verify_nested_parallelism` | `test_nested_parallelism` | Line 84 |
| `verify_big_gap` | `test_batch_sequential_vs_parallel` | Line 84 |

**`docs/BENCHMARKING_FINDINGS.md`**:

| Old string | New string | Locations (approximate lines) |
|---|---|---|
| `validate_batch_ci_three_steps` | `test_batch_bootstrap_ci` | Lines 155, 159, 194 |
| `verify_nested_parallelism` | `test_nested_parallelism` | Lines 156, 194 |
| `verify_big_gap` | `test_batch_sequential_vs_parallel` | Lines 157, 194 |

**`plans/permutation_pvalue_implementation_plan.md`**:

| Old string | New string | Locations (approximate lines) |
|---|---|---|
| `test_simulation_accuracy` | `test_calibration_accuracy` | Lines 179, 196 |

**`plans/empirical_y_fixed_71_fix.plan.md`**:

| Old string | New string | Locations (approximate lines) |
|---|---|---|
| `test_simulation_accuracy` | `test_calibration_accuracy` | Lines 92, 108 |

### 1c: Final grep verification

After all renames and reference updates, grep the entire repo for each old name to confirm zero remaining references (excluding the rename plan itself and this context file):

```
rg "test_simulation_accuracy" --glob "!plans/regression_tests*"
rg "validate_batch_ci_three_steps" --glob "!plans/regression_tests*"
rg "validation_test_spearman2d" --glob "!plans/regression_tests*"
rg "verify_nested_parallelism" --glob "!plans/regression_tests*"
rg "verify_big_gap" --glob "!plans/regression_tests*"
```

Each should return zero results.

---

## Step 2: Add `--strict` and `sys.exit(1)` on failure

### 2a: `tests/test_calibration_accuracy.py` — add `--strict` flag

In the `if __name__ == "__main__"` block (starts at line 325 in the original), add a `--strict` argument to the parser and `sys.exit(1)` when flagged:

After `parser.add_argument("--threshold", ...)` (line 352-353), add:

```python
    parser.add_argument("--strict", action="store_true",
                        help="Exit with code 1 if any scenario is flagged")
```

Then change the tail of `if __name__ == "__main__"` from:

```python
    main(n_sims=args.n_sims, generators=gens, rho_targets=rhos,
         cases=cases_arg, n_distinct_values=nvals, dist_types=dtypes,
         custom_freq=custom_freq_arg, seed=args.seed, threshold=args.threshold,
         outfile=args.outfile, n_cal=args.n_cal,
         calibration_mode=args.calibration_mode)
```

to:

```python
    df = main(n_sims=args.n_sims, generators=gens, rho_targets=rhos,
              cases=cases_arg, n_distinct_values=nvals, dist_types=dtypes,
              custom_freq=custom_freq_arg, seed=args.seed, threshold=args.threshold,
              outfile=args.outfile, n_cal=args.n_cal,
              calibration_mode=args.calibration_mode)
    if args.strict and df["flagged"].any():
        sys.exit(1)
```

Note: `sys` is already imported (line 35). `main()` already returns a DataFrame with a `"flagged"` column.

### 2b: `tests/test_batch_bootstrap_ci.py` — exit 1 when Step 1 fails

At the very end of the file (after line 183 `print("\nDone.")`), add:

```python
if not ok1:
    print("FAIL: Step 1 (bit-identical) failed.")
    sys.exit(1)
```

`sys` is already imported (line 5). `ok1` is set at line 79.

### 2c: `tests/test_spearman_2d.py` — wrap in try/except for clean exit

The file currently uses bare `assert` statements (lines 63, 78-81). Wrap the entire test body in a try/except in `__main__`. The simplest approach: add at the end of the file:

The file doesn't have a `if __name__ == "__main__"` guard — all code runs at module level. The cleanest approach is to leave the asserts as-is (they already cause exit code 1 via uncaught AssertionError in Python). No change needed if the runner simply checks exit code. However, if you want a cleaner message, wrap the module-level test code (lines 51–84) in a function and call it from `__main__`:

**Recommended minimal change:** Leave as-is. Python's assert already causes a nonzero exit code (exit code 1) when it fails. The runner just checks exit codes, so this works.

### 2d: `tests/test_nested_parallelism.py` — already handled

`main_compare()` returns 1 on failure and is called via `sys.exit(main_compare())`. The `main_worker()` path is informational only. No change needed.

### 2e: `tests/test_batch_sequential_vs_parallel.py` — already handled

Same pattern as nested parallelism. `main_compare()` returns 1 on failure. No change needed.

---

## Step 3: Create `tests/run_regression.py`

Create a new file `tests/run_regression.py` with the following behavior:

- Accepts `--full` flag (default is quick mode)
- Runs tests in order via subprocess
- Skips any script that doesn't exist on disk
- On first failure (nonzero exit code), prints which test failed and exits 1
- Prints each test name before running
- On failure, prints the test's stdout and stderr

Here is the complete implementation:

```python
"""
Regression test runner.

Quick (default): runs core tests (fast, foundational first).
Full (--full):   adds slower parallelism/benchmarking tests.

Usage:
    python tests/run_regression.py          # quick
    python tests/run_regression.py --full   # full
"""

import argparse
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

QUICK_TESTS = [
    # (script_name, extra_argv)
    ("test_freq_dict_sums_to_n.py", []),
    ("test_spearman_2d.py", []),
    ("test_batch_bootstrap_ci.py", []),
    ("test_calibration_accuracy.py",
     ["--n-sims", "50", "--case", "3", "--n-distinct", "4",
      "--generators", "nonparametric", "--strict"]),
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
            text=True,
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
    parser = argparse.ArgumentParser(description="Run regression tests.")
    parser.add_argument("--full", action="store_true",
                        help="Include slower parallelism/benchmark tests")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Regression tests ({'full' if args.full else 'quick'})")
    print("=" * 60)

    rc = run_tests(QUICK_TESTS)
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
```

---

## Step 4: New test scripts (section 7)

Implement these in order. Each is a standalone script. All must:
- Add repo root to `sys.path` using the same pattern as existing tests
- Exit with code 1 on failure (via `sys.exit(1)` or uncaught exception)
- Exit with code 0 on success

### 4.1: `tests/test_freq_dict_sums_to_n.py`

```python
"""Verify every entry in config.FREQ_DICT has counts summing to n."""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import FREQ_DICT


def main():
    errors = []
    for n, by_k in FREQ_DICT.items():
        for k, by_dist in by_k.items():
            for dist_type, counts in by_dist.items():
                total = sum(counts)
                if total != n:
                    errors.append(
                        f"FREQ_DICT[{n}][{k}][{dist_type!r}]: sum={total}, expected {n}")
                if len(counts) != k:
                    errors.append(
                        f"FREQ_DICT[{n}][{k}][{dist_type!r}]: len={len(counts)}, expected {k}")

    if errors:
        print("FAIL: FREQ_DICT validation errors:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"PASS: All FREQ_DICT entries valid "
              f"({sum(len(bk) * len(bd) for bk in FREQ_DICT.values() for bd in bk.values())} entries checked).")


if __name__ == "__main__":
    main()
```

### 4.2: `tests/test_asymptotic_formulas.py`

Uses one fixed scenario: case 3 (n=73), k=4, even, alpha=0.05.

```python
"""Sanity/regression test for asymptotic power, CI, and min-detectable-rho formulas."""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import CASES
from power_asymptotic import asymptotic_power, asymptotic_ci, min_detectable_rho_asymptotic, get_x_counts


def main():
    case = CASES[3]
    n = case["n"]  # 73
    rho_true = case["observed_rho"]  # -0.13
    alpha = 0.05
    x_counts = get_x_counts(n, 4, distribution_type="even")

    # --- asymptotic_power ---
    power = asymptotic_power(n, rho_true, alpha=alpha, x_counts=x_counts)
    assert 0.0 <= power <= 1.0, f"asymptotic_power out of range: {power}"
    print(f"asymptotic_power(n={n}, rho={rho_true}): {power:.4f}")

    # --- asymptotic_ci ---
    ci_lower, ci_upper = asymptotic_ci(rho_true, n, alpha=alpha, x_counts=x_counts)
    assert -1.0 <= ci_lower <= 1.0, f"CI lower out of range: {ci_lower}"
    assert -1.0 <= ci_upper <= 1.0, f"CI upper out of range: {ci_upper}"
    assert ci_lower < ci_upper, f"CI lower >= upper: [{ci_lower}, {ci_upper}]"
    assert ci_lower < rho_true < ci_upper, (
        f"Observed rho {rho_true} not inside CI [{ci_lower}, {ci_upper}]")
    print(f"asymptotic_ci(rho_obs={rho_true}, n={n}): [{ci_lower:.4f}, {ci_upper:.4f}]")

    # --- min_detectable_rho_asymptotic ---
    min_rho = min_detectable_rho_asymptotic(
        n, target_power=0.80, alpha=alpha, x_counts=x_counts, direction="positive")
    assert 0.0 < min_rho < 1.0, f"min_detectable_rho out of range: {min_rho}"
    print(f"min_detectable_rho_asymptotic(n={n}, power=0.80): {min_rho:.4f}")

    print("PASS: All asymptotic formula checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, Exception) as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
```

**API notes for the implementer:**
- `asymptotic_power(n, rho_true, alpha=None, x_counts=None, tie_correction=True, two_sided=True)` returns a float in [0, 1].
- `asymptotic_ci(rho_obs, n, alpha=None, x_counts=None, tie_correction=True)` returns a tuple `(ci_lower, ci_upper)`.
- `min_detectable_rho_asymptotic(n, target_power=0.80, alpha=None, x_counts=None, tie_correction=True, direction="positive")` returns a float.
- `get_x_counts(n, n_distinct, distribution_type=None, ...)` returns a list/array of frequency counts. Defined in `power_asymptotic.py` line 269.

### 4.3: `tests/test_reproducibility.py`

```python
"""Verify same seed produces identical results for estimate_power and bootstrap_ci_averaged."""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import CASES
from power_simulation import estimate_power
from confidence_interval_calculator import bootstrap_ci_averaged

SEED = 42


def main():
    case = CASES[3]
    n, k, dt = case["n"], 4, "even"
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
    rho_s = case["observed_rho"]
    ok = True

    # --- Power reproducibility ---
    p1 = estimate_power(n, k, dt, rho_s, y_params, n_sims=200, seed=SEED,
                        calibration_mode="single")
    p2 = estimate_power(n, k, dt, rho_s, y_params, n_sims=200, seed=SEED,
                        calibration_mode="single")
    if p1 != p2:
        print(f"FAIL: estimate_power not reproducible: {p1} != {p2}")
        ok = False
    else:
        print(f"PASS: estimate_power reproducible (seed={SEED}): power={p1:.4f}")

    # --- CI reproducibility ---
    r1 = bootstrap_ci_averaged(n, k, dt, rho_s, y_params,
                                n_reps=10, n_boot=50, seed=SEED,
                                calibration_mode="single")
    r2 = bootstrap_ci_averaged(n, k, dt, rho_s, y_params,
                                n_reps=10, n_boot=50, seed=SEED,
                                calibration_mode="single")
    ci1_lo, ci1_hi = r1["ci_lower"], r1["ci_upper"]
    ci2_lo, ci2_hi = r2["ci_lower"], r2["ci_upper"]
    if ci1_lo != ci2_lo or ci1_hi != ci2_hi:
        print(f"FAIL: bootstrap_ci_averaged not reproducible: "
              f"[{ci1_lo}, {ci1_hi}] != [{ci2_lo}, {ci2_hi}]")
        ok = False
    else:
        print(f"PASS: bootstrap_ci_averaged reproducible (seed={SEED}): "
              f"CI=[{ci1_lo:.4f}, {ci1_hi:.4f}]")

    if not ok:
        sys.exit(1)
    print("PASS: All reproducibility checks passed.")


if __name__ == "__main__":
    main()
```

**API notes for the implementer:**
- `bootstrap_ci_averaged(...)` returns a dict with keys including `"ci_lower"` and `"ci_upper"`. Check the actual return value structure. The function is in `confidence_interval_calculator.py` line 174. To verify the exact return keys, read lines ~280–320 of that file to see what the function returns.
- Use `calibration_mode="single"` for speed (avoids expensive multipoint calibration).

### 4.4: `tests/test_empirical_invalid_n.py`

```python
"""Verify get_pool raises ValueError for unsupported sample sizes."""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data_generator import digitized_available


def main():
    if not digitized_available():
        print("SKIP: Digitized data not available; cannot test get_pool.")
        return

    from data_generator import get_pool

    invalid_ns = [70, 85, 100, 1]
    ok = True
    for n in invalid_ns:
        try:
            get_pool(n)
            print(f"FAIL: get_pool({n}) did not raise ValueError")
            ok = False
        except ValueError:
            print(f"PASS: get_pool({n}) correctly raised ValueError")
        except ImportError:
            print(f"SKIP: get_pool({n}) raised ImportError (digitized data issue)")

    if not ok:
        sys.exit(1)
    print("PASS: All invalid-n checks passed.")


if __name__ == "__main__":
    main()
```

**API notes for the implementer:**
- `get_pool(n)` is in `data_generator.py` line 188. It raises `ValueError` for n not in {73, 80, 81, 82} (line 226-227). It raises `ImportError` if digitized data is unavailable (line 205-206).
- `digitized_available()` is also in `data_generator.py` — it returns True/False.

### 4.5: `tests/test_power_sanity.py`

```python
"""Sanity check: under H0 (rho=0), power should be near alpha (type I error rate)."""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import CASES
from power_simulation import estimate_power


def main():
    case = CASES[3]
    n, k, dt = case["n"], 4, "even"
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}

    alpha = 0.05
    n_sims = 2000
    power = estimate_power(n, k, dt, rho_s=0.0, y_params=y_params,
                           n_sims=n_sims, alpha=alpha, seed=123,
                           calibration_mode="single")

    lo, hi = 0.02, 0.10
    print(f"Power at rho=0 (n_sims={n_sims}): {power:.4f}  (expected in [{lo}, {hi}])")
    if not (lo <= power <= hi):
        print(f"FAIL: Power {power:.4f} outside plausible range [{lo}, {hi}] for alpha={alpha}")
        sys.exit(1)
    print("PASS: Power under H0 is near alpha.")


if __name__ == "__main__":
    main()
```

### 4.6: `tests/test_rng_data_independence.py`

This test verifies that data RNG and bootstrap RNG are independent, so changing n_boot doesn't change datasets.

```python
"""Verify data generation is independent of bootstrap consumption (no RNG coupling)."""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from config import CASES
from data_generator import (
    generate_cumulative_aluminum_batch,
    generate_y_nonparametric_batch,
    calibrate_rho,
    _fit_lognormal,
)

SEED = 42


def test_ci_path_independence():
    """Data stream should be identical regardless of bootstrap consumption."""
    case = CASES[3]
    n, k, dt = case["n"], 4, "even"
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
    rho_s = case["observed_rho"]
    n_reps = 3

    ln_params = _fit_lognormal(y_params["median"], y_params["iqr"])
    cal_rho = calibrate_rho(n, k, dt, rho_s, y_params,
                            all_distinct=False, calibration_mode="single")

    # Run A: generate data only (no bootstrap consumption)
    ss_a = np.random.SeedSequence(SEED)
    data_rng_a, boot_rng_a = [np.random.default_rng(s) for s in ss_a.spawn(2)]
    x_a = generate_cumulative_aluminum_batch(n_reps, n, k, distribution_type=dt,
                                              all_distinct=False, rng=data_rng_a)
    y_a = generate_y_nonparametric_batch(x_a, rho_s, y_params, rng=data_rng_a,
                                          _calibrated_rho=cal_rho, _ln_params=ln_params)

    # Run B: same seed, but consume bootstrap RNG before generating data
    ss_b = np.random.SeedSequence(SEED)
    data_rng_b, boot_rng_b = [np.random.default_rng(s) for s in ss_b.spawn(2)]
    # Simulate bootstrap consumption (as if n_boot=100 for one rep)
    boot_rng_b.integers(0, n, size=(100, n))
    x_b = generate_cumulative_aluminum_batch(n_reps, n, k, distribution_type=dt,
                                              all_distinct=False, rng=data_rng_b)
    y_b = generate_y_nonparametric_batch(x_b, rho_s, y_params, rng=data_rng_b,
                                          _calibrated_rho=cal_rho, _ln_params=ln_params)

    assert np.array_equal(x_a, x_b), "x arrays differ after bootstrap consumption"
    assert np.array_equal(y_a, y_b), "y arrays differ after bootstrap consumption"
    print("PASS: CI path — data independent of bootstrap RNG consumption.")


def test_power_path_determinism():
    """Same seed gives identical first dataset when generating 1 vs 2 sims."""
    case = CASES[3]
    n, k, dt = case["n"], 4, "even"
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
    rho_s = case["observed_rho"]

    ln_params = _fit_lognormal(y_params["median"], y_params["iqr"])
    cal_rho = calibrate_rho(n, k, dt, rho_s, y_params,
                            all_distinct=False, calibration_mode="single")

    # 1 sim
    rng1 = np.random.default_rng(SEED)
    x1 = generate_cumulative_aluminum_batch(1, n, k, distribution_type=dt,
                                             all_distinct=False, rng=rng1)
    y1 = generate_y_nonparametric_batch(x1, rho_s, y_params, rng=rng1,
                                         _calibrated_rho=cal_rho, _ln_params=ln_params)

    # 2 sims
    rng2 = np.random.default_rng(SEED)
    x2 = generate_cumulative_aluminum_batch(2, n, k, distribution_type=dt,
                                             all_distinct=False, rng=rng2)
    y2 = generate_y_nonparametric_batch(x2, rho_s, y_params, rng=rng2,
                                         _calibrated_rho=cal_rho, _ln_params=ln_params)

    assert np.array_equal(x1[0], x2[0]), "x[0] differs between 1-sim and 2-sim runs"
    assert np.array_equal(y1[0], y2[0]), "y[0] differs between 1-sim and 2-sim runs"
    print("PASS: Power path — first dataset identical for 1-sim vs 2-sim runs.")


def main():
    ok = True
    try:
        test_ci_path_independence()
    except AssertionError as e:
        print(f"FAIL: {e}")
        ok = False

    try:
        test_power_path_determinism()
    except AssertionError as e:
        print(f"FAIL: {e}")
        ok = False

    if not ok:
        sys.exit(1)
    print("PASS: All RNG data-independence checks passed.")


if __name__ == "__main__":
    main()
```

**API notes for the implementer:**
- `generate_cumulative_aluminum_batch(n_batch, n, n_distinct, distribution_type=..., all_distinct=..., freq_dict=..., rng=...)` returns an `(n_batch, n)` array. Defined in `data_generator.py`.
- `generate_y_nonparametric_batch(x_all, rho_s, y_params, rng=..., _calibrated_rho=..., _ln_params=...)` returns an `(n_batch, n)` array. Defined in `data_generator.py`.
- `calibrate_rho(n, k, dt, rho_s, y_params, all_distinct=..., calibration_mode=...)` returns a float. Defined in `data_generator.py`.
- `_fit_lognormal(median, iqr)` returns lognormal parameters tuple. Defined in `data_generator.py`.
- The `SeedSequence.spawn(2)` pattern matches exactly what `bootstrap_ci_averaged` does at line 246-247 of `confidence_interval_calculator.py`.

---

## Step 5: Update README documentation

Add a new section (or update existing "Validation" section) with the runner command. In `README.md`, near the existing test documentation:

Add after the validation section:

```markdown
### Regression test suite

Run all core tests (fast, ~1-2 min):

    python tests/run_regression.py

Run all tests including slower parallelism benchmarks:

    python tests/run_regression.py --full
```

Update the `tests/` listing in the project structure section to use the new names:

```
- `test_calibration_accuracy.py` — Calibration accuracy testing
- `test_spearman_2d.py` — Spearman 2D vs reference
- `test_batch_bootstrap_ci.py` — Batch CI bit-identical and timing
- `test_nested_parallelism.py`, `test_batch_sequential_vs_parallel.py` — Parallelism verification
- `run_regression.py` — Regression test runner (quick/full)
```

(The renames in Step 1b already handle the individual name changes; this step adds the runner to the listing and adds the "Regression test suite" section.)

---

## Verification checklist

After all steps, verify:

1. `python tests/run_regression.py` runs without error (some tests may SKIP if not yet implemented — that's fine because the runner skips missing scripts)
2. `rg "test_simulation_accuracy" --glob "!plans/regression_tests*"` returns nothing
3. `rg "validate_batch_ci_three_steps" --glob "!plans/regression_tests*"` returns nothing
4. `rg "validation_test_spearman2d" --glob "!plans/regression_tests*"` returns nothing
5. `rg "verify_nested_parallelism" --glob "!plans/regression_tests*"` returns nothing
6. `rg "verify_big_gap" --glob "!plans/regression_tests*"` returns nothing
7. Each new test script exits 0 when run individually

---

## Important notes for the implementer

- **Do not modify `main()` return behavior** in `test_calibration_accuracy.py`. The `--strict` / `sys.exit(1)` logic goes only in the `if __name__ == "__main__"` block so programmatic callers still get the DataFrame back.
- **Use `calibration_mode="single"`** in all new tests that call `calibrate_rho`, `estimate_power`, or `bootstrap_ci_averaged` for speed. Multipoint calibration is much slower and not needed for regression checks.
- **The runner skips missing scripts** (`Path(script).exists()` check), so you can implement incrementally — the runner works even before all new tests exist.
- **All tests use the same `sys.path` preamble** (lines 1-4 pattern from existing tests):
  ```python
  import sys
  from pathlib import Path
  _root = Path(__file__).resolve().parents[1]
  if str(_root) not in sys.path:
      sys.path.insert(0, str(_root))
  ```
- **`bootstrap_ci_averaged` return value**: Read lines ~280-320 of `confidence_interval_calculator.py` to confirm the exact dict keys returned. The test assumes `"ci_lower"` and `"ci_upper"` keys exist. Adjust if the actual keys differ.
