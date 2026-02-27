---
name: Multi-point Calibration Plan
overview: Add multi-point calibration for the nonparametric generator as the default; single-point remains available for faster runs when less accuracy is needed.
todos: []
isProject: false
---

# Multi-point Calibration for Nonparametric Generator

## Goal

Add multi-point calibration as the **default** for the nonparametric rank-mixing method to handle tie structures where attenuation is nonlinear (ratio varies with rho). Single-point remains available as an opt-in for quicker runs when less accuracy is needed.

## Design

### Option selection

- Add `calibration_mode` parameter: `"multipoint"` (default) or `"single"`.
- Thread through: `config.py` (or function arg), `calibrate_rho`, callers (`power_simulation.py`, `confidence_interval_calculator.py`, `test_simulation_accuracy.py`).
- Copula stays single-point only (attenuation is linear).

### Single-point (opt-in, for speed)

- Current logic in [data_generator.py](c:\Users\raymondf\Documents\Vaccines\Aluminum\KarwowskiSpearmanPowerSims\data_generator.py) `calibrate_rho`: probe at 0.30, bisection, ratio = calibrated_probe / probe, result = rho_target * ratio.
- Cache key: `(n, n_distinct, distribution_type, all_distinct, n_cal)` (+ counts if custom).
- **Use when:** Quick exploratory runs or when second-decimal accuracy is not required. Single-point is ~3× faster (one probe vs three) but can have model bias (0.01–0.03) when attenuation is nonlinear.

### Multi-point implementation

**Probes:** 0.10, 0.30, 0.50 only (three points). Use symmetry for negative targets: the input rho needed for -|rho| is the negative of the input needed for +|rho|.

**Per-probe bisection:** Reuse existing `_mean_rho`-style logic; for each positive probe, find rho_in that yields mean_rho = probe. If a probe fails (e.g. `_mean_rho(0.999) < probe`), record `None` or skip; fall back to nearest successful probe when interpolating.

**Interpolation:** Build curve from (probe, rho_in) pairs for positive probes. For rho_target > 0: interpolate using the curve. For rho_target < 0: `rho_in = -interpolate(|rho_target|)`. For |rho_target| outside [0.10, 0.50], extrapolate linearly from the two nearest points.

**Cache:** Store list of (probe, rho_in) pairs per tie structure (three pairs only). Cache key includes `"multipoint"` to avoid collision with single-point cache.

**API:** Same signature `calibrate_rho(..., calibration_mode="multipoint")`. When `"multipoint"`, return interpolated value (negated for negative rho_target).

### Config / CLI

- Add `CALIBRATION_MODE = "multipoint"` (default) to [config.py](c:\Users\raymondf\Documents\Vaccines\Aluminum\KarwowskiSpearmanPowerSims\config.py). When `"single"`, use single-point calibration.
- Optional: `--calibration-mode single` in `run_simulation.py`, `run_single_scenario.py`, `test_simulation_accuracy.py` for faster runs.
- Both modes are always available; the config/arg selects which to use. Default is multipoint.

## Token-saving implementation approach

1. **Use expensive model once** to produce this spec (or a short design doc).
2. **Use cheaper model** to implement from the spec. The work is localized to `calibrate_rho` and config/CLI; the pattern is repetitive.
3. **Validate** with `test_simulation_accuracy` on built-in and custom tie structures.
4. **Escalate to expensive model** only if the cheaper model's implementation fails validation or hits edge cases.

**Spec to give cheaper model:** "In data_generator.py, add `calibrate_rho_multipoint` that probes at [0.10, 0.30, 0.50] only (three points). Use symmetry for negative rho_target: rho_in = -interpolate(|rho_target|). Cache (probe, rho_in) pairs and use `np.interp` for lookup. Add `calibration_mode` parameter to `calibrate_rho`; when 'multipoint' (default), call `calibrate_rho_multipoint` and return interpolated value (negated for negative targets). When 'single', use existing single-point logic (faster, less accurate when attenuation is nonlinear). Add CALIBRATION_MODE = 'multipoint' to config.py."

## File changes

| File                                          | Changes                                                                                   |
| --------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `data_generator.py`                           | Add `calibrate_rho_multipoint`; add `calibration_mode` to `calibrate_rho`; branch on mode |
| `config.py`                                   | Add `CALIBRATION_MODE = "multipoint"`                                                     |
| `power_simulation.py`                         | Pass `calibration_mode` to `calibrate_rho`                                                |
| `confidence_interval_calculator.py`           | Pass `calibration_mode` to `calibrate_rho`                                                |
| `test_simulation_accuracy.py`                 | Pass `calibration_mode` to `calibrate_rho`                                                |
| `run_simulation.py`, `run_single_scenario.py` | Optional: add `--calibration-mode single|multipoint` CLI arg (default multipoint)           |
| `README.md`                                   | Document `calibration_mode`; multipoint default; single-point for faster runs when less accuracy needed |

## Validation

- Run `test_simulation_accuracy` with `calibration_mode="multipoint"` (default): all scenarios should pass.
- Run with `calibration_mode="single"`: verify no regression; compare to multipoint on custom tie structures where single-point may show bias.
