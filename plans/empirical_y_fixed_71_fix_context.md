# Implementation Context: Empirical Y Fixed 71 Fix

> **Read the plan first:** [plans/empirical_y_fixed_71_fix.plan.md](empirical_y_fixed_71_fix.plan.md)
>
> This file provides the exact code locations, current code, and step-by-step
> instructions so an implementer can make all the changes without ambiguity.

---

## File inventory (all paths relative to repo root)

| File | Role |
|---|---|
| `data_generator.py` (1044 lines) | Core changes: new function, modify 5 existing functions |
| `power_simulation.py` (286 lines) | Call-site cleanup |
| `confidence_interval_calculator.py` (~481 lines) | Call-site cleanup |
| `tests/test_calibration_accuracy.py` (~383 lines) | Call-site cleanup |
| `tests/test_empirical_invalid_n.py` (37 lines) | Extend with new API checks |
| `tests/test_empirical_generator.py` (**new file**) | New test for 71-fixed and 73⊂80 |
| `tests/run_tests.py` (122 lines) | Register new test |
| `README.md` | Docstring update |

---

## Step 1: Add `build_empirical_pool(n, rng)` — data_generator.py

**Insert after** `get_pool` (after line 230, before `generate_y_empirical` on line 233).

The new function mirrors `get_pool`'s logic but takes `rng` instead of using
a fixed seed, and is not cached. For n=80 it calls `build_empirical_pool(73, rng)`
internally (ensuring 73⊂80); for n=82 it calls `build_empirical_pool(81, rng)`.

```python
def build_empirical_pool(n, rng):
    """Build an empirical Y pool of size *n*, advancing *rng*.

    Unlike get_pool (which is cached with a fixed seed for calibration),
    this function is called once per sim/rep so the 'remainder' values
    (beyond the 71 digitized) vary across replications.

    The 71 digitized values (B_AL71 or H_AL71) appear exactly once.
    Only the remainder is resampled from the 71 via rng.
    """
    if not _DIGITIZED_AVAILABLE:
        raise ImportError(
            "Digitized data not available. Create data/digitized.py or use --skip-empirical.")
    if n == 73:
        fill = rng.choice(B_AL71, size=2, replace=True)
        return np.concatenate([B_AL71, fill])
    elif n == 80:
        base = build_empirical_pool(73, rng)
        outliers = generate_b_al_outliers(n_missing=5, rng=rng)
        return np.concatenate([base, outliers])
    elif n == 81:
        fill = rng.choice(H_AL71, size=10, replace=True)
        return np.concatenate([H_AL71, fill])
    elif n == 82:
        base = build_empirical_pool(81, rng)
        return np.append(base, H_AL_OUTLIER)
    else:
        raise ValueError(
            f"Empirical generator only supports n in {{73, 80, 81, 82}}, got {n}")
```

Also update `get_pool`'s docstring (line 188-202) to add a note at the end:

> **Calibration only.** For data generation, use ``build_empirical_pool(n, rng)``
> which resamples the remainder per call.

---

## Step 2: Modify `_raw_rank_mix` — data_generator.py lines 415-418

**Current code** (lines 415-418):
```python
    elif marginal == "empirical":
        if pool is None:
            raise ValueError("pool required for empirical marginal")
        y_values = rng.choice(pool, size=n, replace=True)
```

**Replace with:**
```python
    elif marginal == "empirical":
        if pool is None:
            raise ValueError("pool required for empirical marginal")
        if len(pool) != n:
            raise ValueError(
                f"For empirical marginal, pool length must equal n={n}, got {len(pool)}")
        y_values = np.sort(pool)
```

The rest of the function (lines 421-425: `y_values.sort()`, argsort, assignment)
stays the same. Since `y_values` is already sorted by `np.sort(pool)`, the
existing `y_values.sort()` on line 421 is a no-op but harmless.

---

## Step 3: Modify `_raw_rank_mix_batch` — data_generator.py lines 968-971

**Current code** (lines 968-971):
```python
    elif marginal == "empirical":
        if pool is None:
            raise ValueError("pool required for empirical marginal")
        y_values = rng.choice(pool, size=(n_sims, n), replace=True)
```

**Replace with:**
```python
    elif marginal == "empirical":
        if pool is None:
            raise ValueError("pool required for empirical marginal")
        if pool.ndim != 2 or pool.shape != (n_sims, n):
            raise ValueError(
                f"For empirical marginal, pool must be 2D with shape ({n_sims}, {n}), "
                f"got shape {pool.shape}")
        y_values = np.sort(pool, axis=1)
```

The rest (lines 974-982: `y_values.sort(axis=1)`, argsort, assignment) stays.
`np.sort` already returns sorted rows so the existing `.sort(axis=1)` is a no-op.

---

## Step 4: Modify `_mean_rho_empirical` — data_generator.py line 760

**Current code** (line 760):
```python
        y_vals = cal_rng.choice(pool, size=nn, replace=True)
```

**Replace with:**
```python
        y_vals = np.sort(pool)
```

This changes calibration from sampling-with-replacement to using the fixed pool
directly (sorted, then assigned by mixed order). The shuffle of x and
noise_ranks (lines 744-748) still use `cal_rng`; only the y-value source changes.

---

## Step 5: Modify `generate_y_empirical` — data_generator.py lines 233-245

**Current code** (lines 233-245):
```python
def generate_y_empirical(x, rho_s, y_params, rng=None, _calibrated_rho=None):
    """Generate y via rank-mixing with empirical pool resampling.

    Same rank-mixing logic as generate_y_nonparametric, but draws y-marginal
    values from the empirical pool (via get_pool) instead of a lognormal.
    """
    if rng is None:
        rng = np.random.default_rng()
    pool = get_pool(len(x))
    rho_input = _calibrated_rho if _calibrated_rho is not None else rho_s
    x_ranks = rankdata(x, method="average")
    return _raw_rank_mix(x_ranks, rho_input, y_params, rng,
                         marginal="empirical", pool=pool)
```

**Replace with:**
```python
def generate_y_empirical(x, rho_s, y_params, rng=None, _calibrated_rho=None):
    """Generate y via rank-mixing with empirical marginal.

    Builds a fresh pool per call via build_empirical_pool: the 71 digitized
    values appear exactly once, remainder is resampled using rng.
    """
    if rng is None:
        rng = np.random.default_rng()
    pool = build_empirical_pool(len(x), rng)
    rho_input = _calibrated_rho if _calibrated_rho is not None else rho_s
    x_ranks = rankdata(x, method="average")
    return _raw_rank_mix(x_ranks, rho_input, y_params, rng,
                         marginal="empirical", pool=pool)
```

---

## Step 6: Modify `generate_y_empirical_batch` — data_generator.py lines 248-258

**Current code** (lines 248-258):
```python
def generate_y_empirical_batch(x_batch, rho_s, y_params, rng=None,
                               _calibrated_rho=None, pool=None):
    """Batch version of generate_y_empirical."""
    if rng is None:
        rng = np.random.default_rng()
    if pool is None:
        pool = get_pool(x_batch.shape[1])
    rho_input = _calibrated_rho if _calibrated_rho is not None else rho_s
    x_ranks_batch = _rank_rows(x_batch)
    return _raw_rank_mix_batch(x_ranks_batch, rho_input, y_params, rng,
                               marginal="empirical", pool=pool)
```

**Replace with:**
```python
def generate_y_empirical_batch(x_batch, rho_s, y_params, rng=None,
                               _calibrated_rho=None):
    """Batch version of generate_y_empirical.

    Builds a fresh pool per rep via build_empirical_pool (71 digitized
    values fixed, remainder resampled).  The pool parameter has been
    removed; callers no longer pass pool for data generation.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_reps, n = x_batch.shape
    pool_batch = np.array([build_empirical_pool(n, rng) for _ in range(n_reps)])
    rho_input = _calibrated_rho if _calibrated_rho is not None else rho_s
    x_ranks_batch = _rank_rows(x_batch)
    return _raw_rank_mix_batch(x_ranks_batch, rho_input, y_params, rng,
                               marginal="empirical", pool=pool_batch)
```

---

## Step 7: Call-site cleanup — remove `pool=` from data-generation calls

### 7a. power_simulation.py

**Import** (lines 38-39): keep `get_pool` in the import (still used for calibration on line 100).

**Line 100** (`pool = get_pool(n)`): KEEP — used for calibration on lines 101-104.

**Lines 118-120** — remove `pool=pool`:
```python
# BEFORE:
            y_all = generate_y_empirical_batch(
                x_all, rho_s, y_params, rng=rng,
                _calibrated_rho=cal_rho, pool=pool)
# AFTER:
            y_all = generate_y_empirical_batch(
                x_all, rho_s, y_params, rng=rng,
                _calibrated_rho=cal_rho)
```

**Lines 138-139** — no change needed (already has no `pool=`):
```python
                yi = generate_y_empirical(xi, rho_s, y_params, rng=rng,
                                          _calibrated_rho=cal_rho)
```

### 7b. confidence_interval_calculator.py

**Import** (lines 52-57): keep `get_pool` in the import (still used for calibration).

**Line 154** (`pool = get_pool(n)`): KEEP — used for calibration on lines 155-156.

**Lines 157-158** — no change needed (no `pool=` passed already):
```python
        y = generate_y_empirical(x, rho_s, y_params, rng=rng,
                                  _calibrated_rho=cal_rho)
```

**Line 264** (`pool = get_pool(n)`): KEEP — used for calibration on lines 265-268.

**Lines 285-287** — remove `pool=pool`:
```python
# BEFORE:
            y_all = generate_y_empirical_batch(
                x_all, rho_s, y_params, rng=data_rng,
                _calibrated_rho=cal_rho, pool=pool)
# AFTER:
            y_all = generate_y_empirical_batch(
                x_all, rho_s, y_params, rng=data_rng,
                _calibrated_rho=cal_rho)
```

**Lines 326-328** — remove `pool=pool`:
```python
# BEFORE:
            y_reps = generate_y_empirical_batch(
                x_reps, rho_s, y_params, rng=data_rng,
                _calibrated_rho=cal_rho, pool=pool)
# AFTER:
            y_reps = generate_y_empirical_batch(
                x_reps, rho_s, y_params, rng=data_rng,
                _calibrated_rho=cal_rho)
```

**Lines 354-355** — no change needed (no `pool=` passed already):
```python
                y = generate_y_empirical(x, rho_s, y_params, rng=data_rng,
                                         _calibrated_rho=cal_rho)
```

### 7c. tests/test_calibration_accuracy.py

**Import** (lines 48-51): keep `get_pool` in the import (still used for calibration on line 91).

**Line 91** (`pool = get_pool(n)`): KEEP — used for calibration on lines 92-95.

**Lines 109-110** — no change needed (no `pool=` passed already):
```python
            y = generate_y_empirical(x, rho_target, y_params, rng=rng,
                                      _calibrated_rho=cal_rho)
```

---

## Step 8: Extend tests/test_empirical_invalid_n.py

**Current code** (37 lines): Tests `get_pool(n)` for invalid n values.

**Add** a second loop that tests `build_empirical_pool(n, rng)` for the same
invalid n values. Import `build_empirical_pool` alongside `get_pool`. Both
should raise `ValueError`. Keep the existing `get_pool` tests.

```python
"""Verify get_pool and build_empirical_pool raise ValueError for unsupported sample sizes."""

import sys
from pathlib import Path
import numpy as np
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data_generator import digitized_available


def main():
    if not digitized_available():
        print("SKIP: Digitized data not available; cannot test get_pool/build_empirical_pool.")
        return

    from data_generator import get_pool, build_empirical_pool

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

    rng = np.random.default_rng(42)
    for n in invalid_ns:
        try:
            build_empirical_pool(n, rng)
            print(f"FAIL: build_empirical_pool({n}, rng) did not raise ValueError")
            ok = False
        except ValueError:
            print(f"PASS: build_empirical_pool({n}, rng) correctly raised ValueError")
        except ImportError:
            print(f"SKIP: build_empirical_pool({n}, rng) raised ImportError (digitized data issue)")

    if not ok:
        sys.exit(1)
    print("PASS: All invalid-n checks passed.")


if __name__ == "__main__":
    main()
```

---

## Step 9: Create tests/test_empirical_generator.py (new file)

This test verifies:
1. Each generated dataset's y contains all 71 digitized values exactly once.
2. Pool structure: 73⊂80 (first 73 elements of an n=80 pool equal the n=73 pool
   when built from two RNGs with the same seed).
3. Reproducibility with same seed.

```python
"""Verify empirical generator fixes the 71 digitized values and resamples only the remainder."""

import sys
from pathlib import Path
import numpy as np
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data_generator import digitized_available


def main():
    if not digitized_available():
        print("SKIP: Digitized data not available.")
        return

    from data_generator import build_empirical_pool
    from data.digitized import B_AL71, H_AL71

    ok = True
    seed = 12345

    # --- Test 1: 71 digitized values appear exactly once in each pool ---
    for n, base_arr, label in [(73, B_AL71, "B-Al"), (81, H_AL71, "H-Al")]:
        rng = np.random.default_rng(seed)
        pool = build_empirical_pool(n, rng)
        if len(pool) != n:
            print(f"FAIL: {label} pool length {len(pool)} != {n}")
            ok = False
            continue
        sorted_pool = np.sort(pool)
        sorted_base = np.sort(base_arr)
        # The first 71 of the sorted pool should be >= the 71 base values
        # More precisely: every base value must appear at least once in pool
        for val in sorted_base:
            count = np.sum(pool == val)
            if count < 1:
                print(f"FAIL: {label} n={n} missing digitized value {val}")
                ok = False
                break
        else:
            print(f"PASS: {label} n={n} contains all 71 digitized values")

    # --- Test 2: 73 subset 80 ---
    rng73 = np.random.default_rng(seed)
    rng80 = np.random.default_rng(seed)
    pool_73 = build_empirical_pool(73, rng73)
    pool_80 = build_empirical_pool(80, rng80)
    if np.array_equal(pool_80[:73], pool_73):
        print("PASS: pool_80[:73] == pool_73 (73 subset 80)")
    else:
        print("FAIL: pool_80[:73] != pool_73")
        ok = False

    # --- Test 3: 81 subset 82 ---
    rng81 = np.random.default_rng(seed)
    rng82 = np.random.default_rng(seed)
    pool_81 = build_empirical_pool(81, rng81)
    pool_82 = build_empirical_pool(82, rng82)
    if np.array_equal(pool_82[:81], pool_81):
        print("PASS: pool_82[:81] == pool_81 (81 subset 82)")
    else:
        print("FAIL: pool_82[:81] != pool_81")
        ok = False

    # --- Test 4: Reproducibility ---
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed)
    pool_a = build_empirical_pool(73, rng_a)
    pool_b = build_empirical_pool(73, rng_b)
    if np.array_equal(pool_a, pool_b):
        print("PASS: Same seed produces identical pools")
    else:
        print("FAIL: Same seed produced different pools")
        ok = False

    if not ok:
        sys.exit(1)
    print("PASS: All empirical generator checks passed.")


if __name__ == "__main__":
    main()
```

---

## Step 10: Register new test in tests/run_tests.py

**In the `QUICK_TESTS` list** (line 31-41), add the new test after
`test_empirical_invalid_n.py`:

```python
    ("test_empirical_invalid_n.py", []),
    ("test_empirical_generator.py", []),   # <-- ADD THIS LINE
    ("test_power_sanity.py", []),
```

---

## Step 11: Update README.md

Find any description of the empirical generator and update it to mention:
- The 71 digitized values (B-Al and H-Al each have 71) are fixed in every
  simulated dataset; only the remainder is resampled per sim/rep.
- Pool structure: 73⊂80 and 81⊂82.

Search for "empirical" or "digitized" or "71" in README.md to find the
relevant sections.

---

## Implementation order

1. **Step 1** — Add `build_empirical_pool` + update `get_pool` docstring
2. **Steps 2-4** — Modify `_raw_rank_mix`, `_raw_rank_mix_batch`, `_mean_rho_empirical` (no-replacement)
3. **Steps 5-6** — Modify `generate_y_empirical` and `generate_y_empirical_batch`
4. **Step 7** — Call-site cleanup (power_simulation.py, confidence_interval_calculator.py, test_calibration_accuracy.py)
5. **Steps 8-10** — Tests (extend invalid-n, create new generator test, register in runner)
6. **Step 11** — README update

## After implementation

Run the test suite to verify:
```
python tests/run_tests.py
```

The calibration test may show different rho values than before (this is
expected — see "Calibration outputs change" in the plan's Edge cases).
