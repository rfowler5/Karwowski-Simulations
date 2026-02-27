# Multi-point Calibration: Implementation Context & Spec

This document gives a cheaper AI everything it needs to implement multi-point calibration from scratch, without reading additional files beyond this spec. Read the whole document before writing any code.

---

## 1. What the codebase does (1-paragraph summary)

This project estimates the statistical power and confidence intervals of Spearman rank correlations for a vaccine-aluminum study. The key challenge is **tied x-values** (cumulative vaccine aluminum clusters into 4-10 distinct values). The nonparametric y-generator uses a "rank-mixing" formula to produce (x, y) pairs with a target Spearman rho, but ties **attenuate** the realised rho. A **calibration** step compensates by finding the input mixing weight that yields the desired output rho. Currently calibration probes at a single rho (0.30) and assumes a linear attenuation ratio -- this works well in most cases but can have 0.01-0.03 bias when attenuation is nonlinear. Multi-point calibration probes at three rho values and interpolates, fixing this.

---

## 2. Current single-point calibration (what exists today)

**File:** `data_generator.py`, function `calibrate_rho` (line 285-370).

**How it works:**
1. Pick a single probe rho = 0.30.
2. Use bisection (25 iterations) to find `rho_in` such that `_mean_rho(rho_in)` ≈ 0.30, where `_mean_rho` generates `n_cal` (default 300) datasets and averages the realised Spearman rho.
3. Compute `ratio = rho_in / 0.30`.
4. Cache `ratio` under key `(n, n_distinct, distribution_type, all_distinct, n_cal)` (plus custom counts if applicable) in module-level dict `_CALIBRATION_CACHE`.
5. For any `rho_target`, return `rho_target * ratio` (clipped to [-0.999, 0.999]).

**Key properties:**
- Symmetry: Only probes at positive 0.30. Negative targets get the same ratio applied (sign carried through).
- Cache is rho-independent -- one calibration per tie structure, reused for all rho values.
- `_mean_rho` is a closure defined inside `calibrate_rho`; it uses a fixed `seed` (default 99) and the x-template for the tie structure.

**Signature:**
```python
def calibrate_rho(n, n_distinct, distribution_type, rho_target, y_params,
                  all_distinct=False, n_cal=300, seed=99, freq_dict=None):
```

**Who calls it:**
- `power_simulation.py` → `estimate_power()` (line 67-69)
- `confidence_interval_calculator.py` → `bootstrap_ci_averaged()` (line 201-204)
- `test_simulation_accuracy.py` → `test_scenario()` (line 72-75)

All three call it identically:
```python
cal_rho = calibrate_rho(
    n, n_distinct, distribution_type, rho_target, y_params,
    all_distinct=all_distinct, freq_dict=freq_dict)
```
(No `n_cal` or `seed` overrides at the call sites -- they use the defaults.)

The test_simulation_accuracy.py call is slightly different -- it also passes `n_cal=n_cal`:
```python
cal_rho = calibrate_rho(
    n, n_distinct, distribution_type, rho_target, y_params,
    all_distinct=all_distinct, freq_dict=freq_dict, n_cal=n_cal)
```

---

## 3. What to implement

### 3a. New function: `_calibrate_rho_multipoint`

Add a **private** function in `data_generator.py` that builds a 3-point calibration curve. This is a helper called by `calibrate_rho` when mode is `"multipoint"`.

**Probes:** `[0.10, 0.30, 0.50]` (positive only; symmetry handles negatives).

**Algorithm for each probe `p`:**
1. Run the same bisection that single-point uses, but targeting `_mean_rho(rho_in) ≈ p`.
2. Store the result as a `(p, rho_in)` pair.
3. If a probe fails (i.e., `_mean_rho(0.999) < p` -- the tie structure cannot reach that target), skip it (don't include that pair).

**After probing all three:**
- You have a list of successful `(probe, rho_in)` pairs (at least 1, at most 3).
- Cache this list (not a ratio).

**Interpolation for a given `|rho_target|`:**
- If only 1 pair succeeded: fall back to single-point ratio logic using that pair.
- If 2+ pairs succeeded: use `np.interp(|rho_target|, probes, rho_ins)` where `probes` and `rho_ins` are the x and y arrays from the successful pairs. `np.interp` does linear interpolation and **linear extrapolation from the nearest two points** for values outside the range (this is the default behavior -- `np.interp` actually does flat extrapolation, so you need to handle extrapolation manually; see implementation detail below).

**Extrapolation detail:** `np.interp` clamps outside the range (flat, not linear). For `|rho_target|` outside `[min_probe, max_probe]`, extrapolate linearly from the two nearest points:
```python
def _interp_with_extrapolation(x, xp, fp):
    """Linear interpolation with linear extrapolation beyond endpoints."""
    if x <= xp[0]:
        if len(xp) >= 2:
            slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
            return fp[0] + slope * (x - xp[0])
        return fp[0]
    if x >= xp[-1]:
        if len(xp) >= 2:
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            return fp[-1] + slope * (x - xp[-1])
        return fp[-1]
    return float(np.interp(x, xp, fp))
```

**Symmetry for negative targets:** `rho_in = -interpolate(|rho_target|)`.

**Return:** `float(np.clip(rho_in, -0.999, 0.999))`.

**Cache:** Store the list of `(probe, rho_in)` pairs in a NEW module-level dict `_CALIBRATION_CACHE_MULTIPOINT = {}`. Cache key: `(n, n_distinct, distribution_type, all_distinct, n_cal, "multipoint")` (plus custom counts tuple if `freq_dict` is not None). The `"multipoint"` string in the key avoids collision with single-point cache.

### 3b. Modify `calibrate_rho` to accept `calibration_mode`

**New parameter:** `calibration_mode="multipoint"` (default).

**Behavior:**
- When `calibration_mode="multipoint"`: call `_calibrate_rho_multipoint` and return its result.
- When `calibration_mode="single"`: use existing single-point logic (unchanged).

**Updated signature:**
```python
def calibrate_rho(n, n_distinct, distribution_type, rho_target, y_params,
                  all_distinct=False, n_cal=300, seed=99, freq_dict=None,
                  calibration_mode="multipoint"):
```

**Implementation approach:** Extract the `_mean_rho` closure and bisection logic into a shared helper `_bisect_for_probe(probe, template, y_params, n_cal, seed)` that both single-point and multi-point can use. This avoids code duplication. The helper should return the calibrated `rho_in` for a given probe, or `None` if the probe is unreachable.

Actually, to minimize changes and risk, a simpler approach: keep the existing single-point code path completely intact. Add the new multipoint path as a separate branch:

```python
def calibrate_rho(n, n_distinct, distribution_type, rho_target, y_params,
                  all_distinct=False, n_cal=300, seed=99, freq_dict=None,
                  calibration_mode="multipoint"):
    if abs(rho_target) < 1e-12:
        return 0.0

    if calibration_mode == "multipoint":
        return _calibrate_rho_multipoint(
            n, n_distinct, distribution_type, rho_target, y_params,
            all_distinct=all_distinct, n_cal=n_cal, seed=seed,
            freq_dict=freq_dict)

    # --- existing single-point code below (unchanged) ---
    ...
```

### 3c. The `_mean_rho` helper

Both single-point and multi-point need to compute the mean realised Spearman rho for a given `rho_in`. Factor this out of `calibrate_rho` into a module-level function to avoid duplication:

```python
def _mean_rho(rho_in, template, y_params, n_cal, seed):
    """Mean realised Spearman rho over n_cal samples for given rho_in."""
    cal_rng = np.random.default_rng(seed)
    mu, sigma = _fit_lognormal(y_params["median"], y_params["iqr"])
    total = 0.0
    for _ in range(n_cal):
        x = template.copy()
        cal_rng.shuffle(x)
        x_ranks = rankdata(x, method="average")
        nn = len(x_ranks)
        noise_ranks = cal_rng.permutation(nn) + 1.0

        s_x = x_ranks - np.mean(x_ranks)
        sd_x = np.std(x_ranks, ddof=0)
        if sd_x > 0:
            s_x /= sd_x
        s_n = noise_ranks - np.mean(noise_ranks)
        s_n /= np.std(noise_ranks, ddof=0)

        rho_c = np.clip(rho_in, -0.999, 0.999)
        mixed = rho_c * s_x + np.sqrt(1.0 - rho_c ** 2) * s_n

        y_vals = cal_rng.lognormal(mean=mu, sigma=sigma, size=nn)
        y_vals.sort()
        y_final = np.empty(nn)
        y_final[np.argsort(mixed)] = y_vals

        total += _fast_spearman(x_ranks, y_final)
    return total / n_cal
```

This is **identical** to the current closure inside `calibrate_rho` (lines 321-348), just lifted to module scope with `template`, `y_params`, `n_cal`, `seed` as explicit parameters.

Then update the single-point code inside `calibrate_rho` to call `_mean_rho(rho_in, template, y_params, n_cal, seed)` instead of the closure. This is a refactor-only change that makes the function reusable.

### 3d. The `_bisect_for_probe` helper

Factor out the bisection logic:

```python
def _bisect_for_probe(probe, template, y_params, n_cal, seed,
                      n_iter=25, tol=5e-5):
    """Bisect to find rho_in such that _mean_rho(rho_in, ...) ≈ probe.

    Returns rho_in (float) or None if the probe is unreachable.
    """
    lo, hi = 0.0, min(probe * 2.0, 0.999)
    if _mean_rho(hi, template, y_params, n_cal, seed) < probe:
        hi = 0.999
    if _mean_rho(hi, template, y_params, n_cal, seed) < probe:
        return None  # unreachable

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        observed = _mean_rho(mid, template, y_params, n_cal, seed)
        if observed < probe:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    return (lo + hi) / 2.0
```

Note: the `_mean_rho(hi, ...) < probe` check happens twice (once after initial hi, once after expanding to 0.999). The second check determines if the probe is truly unreachable. This mirrors the existing single-point logic.

### 3e. `_calibrate_rho_multipoint` implementation

```python
_CALIBRATION_CACHE_MULTIPOINT = {}

_MULTIPOINT_PROBES = [0.10, 0.30, 0.50]


def _calibrate_rho_multipoint(n, n_distinct, distribution_type, rho_target,
                               y_params, all_distinct=False, n_cal=300,
                               seed=99, freq_dict=None):
    """Multi-point calibration: probe at 3 rho values, interpolate."""
    cache_key = (n, n_distinct, distribution_type, all_distinct, n_cal,
                 "multipoint")
    if freq_dict is not None:
        counts = tuple(freq_dict[n][n_distinct]["custom"])
        cache_key = cache_key + (counts,)

    if cache_key not in _CALIBRATION_CACHE_MULTIPOINT:
        template = _get_x_template(n, n_distinct, distribution_type,
                                    freq_dict, all_distinct)
        pairs = []
        for probe in _MULTIPOINT_PROBES:
            rho_in = _bisect_for_probe(probe, template, y_params,
                                        n_cal, seed)
            if rho_in is not None:
                pairs.append((probe, rho_in))

        _CALIBRATION_CACHE_MULTIPOINT[cache_key] = pairs

    pairs = _CALIBRATION_CACHE_MULTIPOINT[cache_key]

    if not pairs:
        # No probes succeeded -- return rho_target unmodified
        return float(np.clip(rho_target, -0.999, 0.999))

    abs_target = abs(rho_target)
    sign = 1.0 if rho_target >= 0 else -1.0

    if len(pairs) == 1:
        # Fall back to single-point ratio
        probe, rho_in = pairs[0]
        ratio = rho_in / probe
        result = abs_target * ratio
    else:
        probes = [p for p, _ in pairs]
        rho_ins = [r for _, r in pairs]
        result = _interp_with_extrapolation(abs_target, probes, rho_ins)

    return float(np.clip(sign * result, -0.999, 0.999))
```

### 3f. Add `CALIBRATION_MODE` to `config.py`

Add this line in the "Simulation parameters" section (after `ASYMPTOTIC_TIE_CORRECTION_MODE`):

```python
# "multipoint" (default, more accurate) or "single" (faster, ~3x less calibration cost)
CALIBRATION_MODE = "multipoint"
```

### 3g. Thread `calibration_mode` through callers

**All three callers** (`power_simulation.py`, `confidence_interval_calculator.py`, `test_simulation_accuracy.py`) need to import `CALIBRATION_MODE` from config and pass it to `calibrate_rho`. The changes are mechanical and identical in pattern.

#### `power_simulation.py` — `estimate_power()`

Current code (lines 66-69):
```python
    if generator == "nonparametric":
        cal_rho = calibrate_rho(
            n, n_distinct, distribution_type, rho_s, y_params,
            all_distinct=all_distinct, freq_dict=freq_dict)
```

Change to:
```python
    if generator == "nonparametric":
        cal_rho = calibrate_rho(
            n, n_distinct, distribution_type, rho_s, y_params,
            all_distinct=all_distinct, freq_dict=freq_dict,
            calibration_mode=calibration_mode)
```

And add `calibration_mode=None` parameter to `estimate_power` and `min_detectable_rho` signatures, with:
```python
    if calibration_mode is None:
        calibration_mode = CALIBRATION_MODE
```

Import `CALIBRATION_MODE` from config at the top:
```python
from config import (CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES,
                    N_SIMS, ALPHA, TARGET_POWER,
                    POWER_SEARCH_DIRECTION, CALIBRATION_MODE)
```

Also thread `calibration_mode` from `min_detectable_rho` → `estimate_power` (it calls `estimate_power` inside bisection).

Also thread from `_power_one_scenario` → `min_detectable_rho` and from `run_all_scenarios` → `_power_one_scenario`. The `run_all_scenarios` function should accept `calibration_mode=None`.

#### `confidence_interval_calculator.py` — `bootstrap_ci_averaged()`

Same pattern. Add `calibration_mode=None` to `bootstrap_ci_averaged()` signature. Inside:
```python
    if calibration_mode is None:
        calibration_mode = CALIBRATION_MODE
```

Pass `calibration_mode=calibration_mode` to the `calibrate_rho` call (line 202-204).

Thread through: `_ci_one_scenario` → `bootstrap_ci_averaged` → `calibrate_rho`. And `run_all_ci_scenarios` → `_ci_one_scenario` should accept and forward `calibration_mode`.

Import `CALIBRATION_MODE` from config.

#### `test_simulation_accuracy.py` — `test_scenario()`

Same pattern. Add `calibration_mode=None` to `test_scenario()`. Inside:
```python
    if calibration_mode is None:
        from config import CALIBRATION_MODE
        calibration_mode = CALIBRATION_MODE
```

Pass `calibration_mode=calibration_mode` to the `calibrate_rho` call (line 73-75).

Thread through: `run_accuracy_tests` → `test_scenario`. And `main` → `run_accuracy_tests` should accept and forward `calibration_mode`.

### 3h. CLI arguments (optional but nice)

Add `--calibration-mode` argument to `run_simulation.py`, `run_single_scenario.py`, and `test_simulation_accuracy.py`:

```python
parser.add_argument("--calibration-mode",
                    choices=["multipoint", "single"],
                    default=None,
                    help="Calibration mode (default: multipoint)")
```

Thread the parsed value into the respective `main()` calls.

---

## 4. Exact file-by-file change list

### `config.py`
- **Add 1 line** after `ASYMPTOTIC_TIE_CORRECTION_MODE = "with_tie_correction"` (line 247):
```python
CALIBRATION_MODE = "multipoint"
```

### `data_generator.py`
- **Add** module-level helper `_mean_rho(rho_in, template, y_params, n_cal, seed)` — extracted from the closure in `calibrate_rho`. Place it right after `_fast_spearman` (after line 282).
- **Add** module-level helper `_bisect_for_probe(probe, template, y_params, n_cal, seed, n_iter=25, tol=5e-5)` — extracted bisection logic. Place it after `_mean_rho`.
- **Add** module-level helper `_interp_with_extrapolation(x, xp, fp)` — linear interpolation with linear extrapolation. Place it after `_bisect_for_probe`.
- **Add** module-level dict `_CALIBRATION_CACHE_MULTIPOINT = {}` and constant `_MULTIPOINT_PROBES = [0.10, 0.30, 0.50]`.
- **Add** function `_calibrate_rho_multipoint(...)` as described in §3e.
- **Modify** `calibrate_rho`:
  - Add `calibration_mode="multipoint"` parameter.
  - At the top (after the `abs(rho_target) < 1e-12` check), add the multipoint branch that delegates to `_calibrate_rho_multipoint`.
  - **Refactor** the existing single-point code to use `_mean_rho` and `_bisect_for_probe` helpers. This is optional -- you can also leave the existing single-point code unchanged as a self-contained block. Either way is fine; the key requirement is that `calibration_mode="single"` produces **identical** results to the current code.

### `power_simulation.py`
- **Import** `CALIBRATION_MODE` from config.
- **Add** `calibration_mode=None` param to: `estimate_power`, `min_detectable_rho`, `_power_one_scenario`, `run_all_scenarios`.
- **Add** `if calibration_mode is None: calibration_mode = CALIBRATION_MODE` at the top of `estimate_power` and `min_detectable_rho`.
- **Pass** `calibration_mode=calibration_mode` from `estimate_power` → `calibrate_rho`.
- **Pass** `calibration_mode=calibration_mode` from `min_detectable_rho` → `estimate_power`.
- **Thread** through `run_all_scenarios` → `_power_one_scenario` → `min_detectable_rho`.

### `confidence_interval_calculator.py`
- **Import** `CALIBRATION_MODE` from config.
- **Add** `calibration_mode=None` param to: `bootstrap_ci_averaged`, `_ci_one_scenario`, `run_all_ci_scenarios`.
- **Add** `if calibration_mode is None: calibration_mode = CALIBRATION_MODE` at the top of `bootstrap_ci_averaged`.
- **Pass** `calibration_mode=calibration_mode` from `bootstrap_ci_averaged` → `calibrate_rho`.
- **Thread** through `run_all_ci_scenarios` → `_ci_one_scenario` → `bootstrap_ci_averaged`.

### `test_simulation_accuracy.py`
- **Import** `CALIBRATION_MODE` from config (add to existing config import).
- **Add** `calibration_mode=None` param to: `test_scenario`, `run_accuracy_tests`, `main`.
- **Add** `if calibration_mode is None: calibration_mode = CALIBRATION_MODE` at the top of `test_scenario`.
- **Pass** `calibration_mode=calibration_mode` from `test_scenario` → `calibrate_rho`.
- **Thread** through `main` → `run_accuracy_tests` → `test_scenario`.
- **Add** `--calibration-mode` CLI argument.

### `run_simulation.py`
- **Add** `calibration_mode=None` param to `main()`.
- **Thread** into `mc_scenarios(...)` and `run_all_ci_scenarios(...)` calls.
- **Add** `--calibration-mode` CLI argument.

### `run_single_scenario.py`
- **Add** `calibration_mode=None` param to `main()`, `run_power()`, `run_ci()`.
- **Thread** into `min_detectable_rho(...)` and `bootstrap_ci_averaged(...)` calls.
- **Add** `--calibration-mode` CLI argument.

---

## 5. Things that do NOT change

- `calibrate_rho_copula` — copula stays single-point only. Do not modify it.
- `generate_y_nonparametric`, `generate_y_copula`, `generate_y_linear` — no changes.
- `_raw_rank_mix` — no changes.
- `spearman_helpers.py` — no changes.
- `power_asymptotic.py` — no changes.
- `table_outputs.py` — no changes.

---

## 6. Testing & validation

After implementation, run:

```bash
# Quick accuracy test with multipoint (default)
python test_simulation_accuracy.py --n-sims 50 --case 3 --n-distinct 4 --generators nonparametric

# Same with single-point (verify no regression)
python test_simulation_accuracy.py --n-sims 50 --case 3 --n-distinct 4 --generators nonparametric --calibration-mode single

# Broader test
python test_simulation_accuracy.py --n-sims 200 --generators nonparametric

# Custom tie structure (where single-point may show more bias)
python test_simulation_accuracy.py --case 3 --freq 19,18,18,18 --n-sims 200 --generators nonparametric
```

**Expected results:** All scenarios should pass (|diff| < 0.01) for both modes. Multipoint may show smaller mean|diff| than single-point, especially for custom or extreme tie structures.

---

## 7. Implementation ordering

1. `config.py` — add `CALIBRATION_MODE` (1 line).
2. `data_generator.py` — add helpers + `_calibrate_rho_multipoint` + modify `calibrate_rho`. This is the core work.
3. `power_simulation.py` — thread `calibration_mode` (mechanical).
4. `confidence_interval_calculator.py` — thread `calibration_mode` (mechanical).
5. `test_simulation_accuracy.py` — thread `calibration_mode` + CLI arg.
6. `run_simulation.py` — thread `calibration_mode` + CLI arg.
7. `run_single_scenario.py` — thread `calibration_mode` + CLI arg.
8. Validate with test commands from §6.

---

## 8. Edge cases to handle

1. **`rho_target` near zero:** Already handled — `calibrate_rho` returns 0.0 for `abs(rho_target) < 1e-12` before branching on mode.
2. **No probes succeed:** Return `rho_target` unmodified (clipped). This means the tie structure has a very low ceiling -- extremely unlikely with the built-in structures but possible with pathological custom ones.
3. **Only 1 probe succeeds:** Fall back to ratio logic (like single-point).
4. **Extrapolation beyond [0.10, 0.50]:** Linear extrapolation from two nearest points. For targets near 0, extrapolation goes toward (0, 0) naturally. For targets > 0.50, extrapolation from the 0.30/0.50 pair.
5. **`all_distinct=True`:** Attenuation is minimal (ratio ≈ 1.0), so multipoint and single-point should produce nearly identical results. Both should work fine.
6. **Negative `rho_target`:** Use symmetry: `rho_in = -interpolate(|rho_target|)`. This is handled in `_calibrate_rho_multipoint`.

---

## 9. Performance notes

Multi-point probes at 3 values (vs 1 for single-point), so calibration takes ~3x longer. Since calibration is cached per tie structure and the probing itself takes ~3s per tie structure for single-point, multi-point will take ~9s per tie structure on first run. This is a one-time cost per unique (n, k, dist_type) combination. Subsequent calls for different `rho_target` values reuse the cached curve at zero extra cost.
