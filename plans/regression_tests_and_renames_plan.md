---
name: Regression tests and renames
overview: Add a single regression runner script, rename existing test scripts to reflect what they test (e.g. calibration accuracy), add new tests (asymptotic, reproducibility, empirical invalid n, power sanity, FREQ_DICT sums, RNG data-independence), and ensure all tests exit with code 1 on failure so the runner can detect breakage.
todos: []
isProject: false
---

# Regression test suite and test script renames

## 1. Test script renames

Rename files and update all references so names match what each test actually does.


| Current name                                                                     | New name                                     | Rationale                                                                                                                                 |
| -------------------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| [tests/test_simulation_accuracy.py](tests/test_simulation_accuracy.py)           | `tests/test_calibration_accuracy.py`         | It checks that **calibration** (plus generation) achieves target Spearman rho; it does not test the power simulation pipeline end-to-end. |
| [tests/validation_test_spearman2d.py](tests/validation_test_spearman2d.py)       | `tests/test_spearman_2d.py`                  | Numba 2D Spearman/rho/pvalue vs reference; shorter, consistent test_ prefix.                                                              |
| [tests/validate_batch_ci_three_steps.py](tests/validate_batch_ci_three_steps.py) | `tests/test_batch_bootstrap_ci.py`           | Batch bootstrap CI correctness (bit-identical to old path) and timing; name reflects content.                                               |
| [tests/verify_nested_parallelism.py](tests/verify_nested_parallelism.py)         | `tests/test_nested_parallelism.py`           | Keeps behavior; test_ prefix for consistency.                                                                                             |
| [tests/verify_big_gap.py](tests/verify_big_gap.py)                               | `tests/test_batch_sequential_vs_parallel.py`  | Describes what is compared (sequential vs parallel batch path).                                                                             |


**References to update after renames:** [README.md](README.md) (Accuracy testing, tests section, calibration failure note, Validation, BENCHMARKING), [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md), [docs/BENCHMARKING_FINDINGS.md](docs/BENCHMARKING_FINDINGS.md), [plans/permutation_pvalue_implementation_plan.md](plans/permutation_pvalue_implementation_plan.md), [plans/empirical_y_fixed_71_fix.plan.md](plans/empirical_y_fixed_71_fix.plan.md). Inside renamed files: docstrings and any self-references (e.g. `from test_simulation_accuracy import main` → `from test_calibration_accuracy import main`). **Final sweep:** After renames, grep for old script names (`test_simulation_accuracy`, `validate_batch_ci_three_steps`, `validation_test_spearman2d`, `verify_nested_parallelism`, `verify_big_gap`) across the repo and update any remaining references, including README code snippets (e.g. `from tests.test_simulation_accuracy import main`).

---

## 2. Single regression runner script

Add one script that runs tests in a fixed order and exits non-zero on first failure so you can quickly see if an update broke something.

**Recommendation:** A **Python** runner at **`tests/run_regression.py`** (or repo root `run_tests.py` that delegates to `tests/run_regression.py`). Python is cross-platform (Windows/PowerShell friendly), can invoke scripts as subprocesses, and can pass arguments (e.g. quick vs full).

**Behavior:**

- **Order of execution** (fast / foundational first):
  1. `test_freq_dict_sums_to_n.py` — fast; no RNG; validates config.
  2. `test_spearman_2d.py` — fast; no calibration, core Spearman correctness.
  3. `test_batch_bootstrap_ci.py` — correctness of batch CI (bit-identical). In quick mode, invoke with `--quick` when that flag exists (Step 1 only).
  4. `test_calibration_accuracy.py` — with quick args and `--strict`.
  5. New tests from section 7, in order: `test_asymptotic_formulas.py`, `test_reproducibility.py`, `test_empirical_invalid_n.py` (or in test_empirical_generator), `test_power_sanity.py`, `test_rng_data_independence.py`.
  6. *(Full only)* `test_nested_parallelism.py`, `test_batch_sequential_vs_parallel.py` — slower, diagnostic.
- **Missing scripts:** The runner must **skip any script that does not exist** (e.g. check `Path(script).exists()` before running). This allows the runner to work immediately after renames and exit-1 changes, and new tests can be added incrementally without breaking the runner.
- **Quick vs full (explicit):**
  - **Quick (default):** Steps 1–4 plus 7.5 (freq dict), 7.1 (asymptotic), 7.2 (reproducibility), 7.3 (empirical invalid n), 7.4 (power sanity, with small n_sims), 7.6 (RNG data-independence). Skip 6 (nested parallelism, sequential vs parallel).
  - **Full:** All of the above plus step 6 (test_nested_parallelism.py, test_batch_sequential_vs_parallel.py). If 7.4 uses larger n_sims in a "full" variant, include that in full only.
- **Exit code:** Runner exits 0 only if every launched test exits 0; on first test failure, print which test failed and exit 1.
- **Output:** Print each test name before running; capture and print test stdout/stderr on failure.

No pytest required: keep existing tests as standalone scripts. The runner runs them in order via `subprocess` and checks exit codes.

---

## 3. Make test scripts exit 1 on failure

For the runner to detect breakage, every test must exit with code 1 on failure.

- **test_spearman_2d.py:** Uses `assert`; optionally wrap `if __name__ == "__main__"` in try/except and `sys.exit(1)` for a clear exit code.
- **test_batch_bootstrap_ci.py:** Add explicit `sys.exit(1)` when Step 1 (or any critical step) fails (e.g. when `not ok1`).
- **test_calibration_accuracy.py:** Add **`--strict`** flag. In the `if __name__ == "__main__"` branch only, when `--strict` is set, after `print_report(df)`, if `df["flagged"].any()` then `sys.exit(1)`. Do not exit from `main()` when it is called programmatically, so callers can inspect the DataFrame.
- **test_nested_parallelism.py** / **test_batch_sequential_vs_parallel.py:** If they have any pass/fail notion, exit 1 on failure; otherwise they can remain informational.
- **All new tests (section 7):** Exit 1 on assertion or check failure.

---

## 4. Future tests (permutation, empirical)

Planned additions:

- **Permutation p-value** ([plans/permutation_pvalue_implementation_plan.md](plans/permutation_pvalue_implementation_plan.md)): smoke test for `estimate_power` with permutation; unit tests for precomputed null and MC p-value. Add to the runner in a sensible order.
- **Empirical 71 fix** ([plans/empirical_y_fixed_71_fix.plan.md](plans/empirical_y_fixed_71_fix.plan.md)): add test that empirical generator yields 71 fixed values once per dataset (and optional pool 73⊂80) in a new script (e.g. `tests/test_empirical_generator.py`). The 71-fix plan also **updates test 7.3** (`test_empirical_invalid_n.py`) to use `build_empirical_pool(n, rng)` for invalid n instead of `get_pool(n)`, so the invalid-n check covers the data-generation API after the fix.

The runner should be structured so adding a new test is just appending an entry to a list of `(name, [argv], optional_quick_skip)` and running them in sequence.

---

## 5. Implementation order

1. **Rename test files** and update all references (README, PROJECT_SUMMARY, docs, plans, and docstrings/imports inside the renamed files). Do a final grep for old script names and update any remaining references (including README code snippets).
2. **Add `--strict` to calibration test** and **add `sys.exit(1)` on failure** to batch CI (and optionally to Spearman 2D and verify scripts).
3. **Implement `tests/run_regression.py`** with quick/full modes, ordered list of tests, subprocess runs, and exit-on-first-failure.
4. **Add new tests (section 7):** Implement each new test script; register each in the runner. Suggested order: freq dict (7.5), asymptotic (7.1), reproducibility (7.2), empirical invalid n (7.3), power sanity (7.4), RNG data-independence (7.6).
5. **Document** in README: "Regression run: `python tests/run_regression.py` (quick) or `python tests/run_regression.py --full`," and update the tests section to use the new script names and mention the runner.

---

## 6. Optional: "Quick" for batch CI and calibration

- **test_batch_bootstrap_ci.py:** Consider a `--quick` that runs only Step 1 (bit-identical). Full run (Steps 2–3) remains available for deeper checks.
- **test_calibration_accuracy.py:** Runner already passes quick args; a single `--quick` flag in that script could simplify the runner's argv. Optional.

---

## 7. Additional tests to implement

The following tests should be added as new test scripts and registered in the regression runner.

### 7.1 Asymptotic formulas (sanity / regression)

- **Purpose:** Protect [power_asymptotic.py](power_asymptotic.py) from formula or tie-correction regressions; currently untested.
- **Test:** Call `asymptotic_power`, `asymptotic_ci`, and optionally `min_detectable_rho_asymptotic` for one fixed scenario (e.g. n=73, k=4, even, alpha=0.05, using `get_x_counts`). Assert: power in [0, 1], CI bounds in [-1, 1], min_detectable_rho in a plausible range. Optionally lock in known-good numeric results and assert equality.
- **Script:** e.g. `tests/test_asymptotic_formulas.py`.

### 7.2 Reproducibility (same seed, same result)

- **Purpose:** Catch general non-determinism (e.g. threading, uninitialized state).
- **Test:** Call `estimate_power(..., seed=42)` twice; assert the two power values are equal. Call `bootstrap_ci_averaged(..., seed=42)` twice; assert returned CI endpoints (ci_lower, ci_upper) are equal.
- **Script:** e.g. `tests/test_reproducibility.py` (can be combined with 7.6).

### 7.3 Empirical generator: invalid n

- **Purpose:** Enforce that the empirical pool API rejects unsupported n.
- **Test (regression done first):** Call **`get_pool(n)`** with n not in {73, 80, 81, 82} (e.g. n=70 or n=85). Assert the same ValueError (or error) the code raises for unsupported n. Use the current API so this test works before the 71 fix exists.
- **After the 71 fix:** The [71-fix plan](plans/empirical_y_fixed_71_fix.plan.md) will update this test to call **`build_empirical_pool(n, rng)`** instead of `get_pool(n)` for invalid n, so the invalid-n check covers the data-generation API. Same assertion (same ValueError).
- **Digitized optional:** Implement so that n is validated **before** any digitized data load (so the test can run and expect ValueError even when digitized data is unavailable). If validation currently happens after load, implement 7.3 to skip when digitized is not available and document the skip in the runner (e.g. try/import or check for digitized_available() before running this test).
- **Script:** Standalone `tests/test_empirical_invalid_n.py`. When the 71 fix is implemented, that plan updates this script to use `build_empirical_pool`; the 71 fix also adds a separate test for “71 fixed once per dataset” (e.g. in `tests/test_empirical_generator.py`).

### 7.4 Power sanity (rho=0 near alpha)

- **Purpose:** Catch a broken null so that under H0 power is near the nominal alpha.
- **Test:** Run `estimate_power(..., rho_s=0, n_sims=...)` for one scenario. Assert returned power is in a plausible range for type I error (e.g. [0.02, 0.10] for alpha=0.05), allowing Monte Carlo noise.
- **Script:** e.g. `tests/test_power_sanity.py`.

### 7.5 Pre-defined frequency distributions sum to N

- **Purpose:** Ensure every entry in [config.FREQ_DICT](config.py) has counts that sum to the corresponding sample size n. Catches typos when adding or editing predefined tie structures.
- **Test:** Iterate over all (n, n_distinct, distribution_type) in `config.FREQ_DICT`; for each, assert `sum(FREQ_DICT[n][k][dist_type]) == n`.
- **Script:** e.g. `tests/test_freq_dict_sums_to_n.py`. Fast, no RNG; run early in the runner.

### 7.6 RNG data-independence (no n_boot / n_sims coupling)

- **Purpose:** Regress the bug where data and bootstrap shared one RNG, so changing n_boot changed the datasets. Ensures data RNG is independent of bootstrap RNG (and of n_boot).
- **CI path:** Use the same pattern as [confidence_interval_calculator.bootstrap_ci_averaged](confidence_interval_calculator.py): `SeedSequence(seed).spawn(2)` for data_rng and boot_rng. In run A: generate n_reps datasets using only data_rng (same params as CI). In run B: spawn(2) again, then **consume the same randomness the bootstrap would use for one rep with n_boot=100**: call `boot_rng.integers(0, n, size=(100, n))` (or the same pattern the real bootstrap uses), then generate n_reps datasets with data_rng. Assert the two sets of (x_all, y_all) are identical. This proves the data stream does not depend on bootstrap stream consumption.
- **Power path:** Same seed, two calls to the same data-generation path used by estimate_power: generate "1 sim" and "2 sims" with default_rng(seed); assert the first sim's (x, y) in the 2-sim run equals the single (x, y) from the 1-sim run. Minimal test: data generation with same seed gives identical first dataset when called twice (determinism of data gen).
- **Script:** e.g. `tests/test_rng_data_independence.py`. Include both CI and power-path checks; exit 1 on failure.

---

## Summary

| Deliverable            | Description                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------- |
| Renamed tests          | 5 files renamed; all references updated.                                          |
| Calibration `--strict` | test_calibration_accuracy exits 1 when any scenario flagged.                       |
| Batch CI exit 1        | test_batch_bootstrap_ci exits 1 when Step 1 (or critical step) fails.              |
| Runner                 | `tests/run_regression.py` runs tests in order, quick/full, exit on first failure. |
| New tests (7.1–7.6)    | Asymptotic, reproducibility, empirical invalid n, power sanity, FREQ_DICT sums, RNG data-independence. |
| Docs                   | README (and related) updated with runner command and new script names.             |

Future permutation and empirical tests are added as new scripts and registered in the runner's list.
