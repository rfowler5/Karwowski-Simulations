---
name: Multi-point Calibration Plan
overview: Add optional multi-point calibration for the nonparametric generator while keeping single-point as default. Use a token-saving implementation approach (detailed spec + cheaper model).
todos: []
isProject: false
---

# Multi-point Calibration for Nonparametric Generator

## Goal

Add an optional multi-point calibration mode for the nonparametric rank-mixing method to handle tie structures where attenuation is nonlinear (ratio varies with rho). **Keep the single-point approach as default**; multi-point is an opt-in option for custom or extreme tie structures.

## Design

### Option selection

- Add `calibration_mode` parameter: `"single"` (default) or `"multipoint"`.
- Thread through: `config.py` (or function arg), `calibrate_rho`, callers (`power_simulation.py`, `confidence_interval_calculator.py`, `test_simulation_accuracy.py`).
- Copula stays single-point only (attenuation is linear).

### Single-point (unchanged, default)

- Current logic in [data_generator.py](c:\Users\raymondf\Documents\Vaccines\Aluminum\KarwowskiSpearmanPowerSims\data_generator.py) `calibrate_rho`: probe at 0.30, bisection, ratio = calibrated_probe / probe, result = rho_target * ratio.
- Cache key: `(n, n_distinct, distribution_type, all_distinct, n_cal)` (+ counts if custom).
- No changes when `calibration_mode="single"`.

### Multi-point implementation

**Probes:** 0.10, 0.30, 0.50 (and negative: -0.10, -0.30, -0.50 for symmetry).

**Per-probe bisection:** Reuse existing `_mean_rho`-style logic; for each probe, find rho_in that yields mean_rho = probe. If a probe fails (e.g. `_mean_rho(0.999) < probe`), record `None` or skip; fall back to nearest successful probe when interpolating.

**Interpolation:** Linear between (probe, rho_in) pairs. For |rho_target| outside [0.10, 0.50], extrapolate linearly from the two nearest points.

**Cache:** Store list of (probe, rho_in) pairs per tie structure. Cache key includes `"multipoint"` to avoid collision with single-point cache.

**API:** Same signature `calibrate_rho(..., calibration_mode="single")`. When `"multipoint"`, return `interpolate(rho_target, calibration_curve)` instead of `rho_target * ratio`.

### Config / CLI

- Add `CALIBRATION_MODE = "single"` to [config.py](c:\Users\raymondf\Documents\Vaccines\Aluminum\KarwowskiSpearmanPowerSims\config.py).
- Optional: `--calibration-mode multipoint` in `run_simulation.py`, `run_single_scenario.py`, `test_simulation_accuracy.py` for testing.

## Token-saving implementation approach

1. **Use expensive model once** to produce this spec (or a short design doc).
2. **Use cheaper model** to implement from the spec. The work is localized to `calibrate_rho` and config/CLI; the pattern is repetitive.
3. **Validate** with `test_simulation_accuracy` on built-in and custom tie structures.
4. **Escalate to expensive model** only if the cheaper model's implementation fails validation or hits edge cases.

**Spec to give cheaper model:** "In data_generator.py, add `calibrate_rho_multipoint` that probes at [0.10, 0.30, 0.50] and [-0.10, -0.30, -0.50], caches (probe, rho_in) pairs, and uses `np.interp` for lookup. Add `calibration_mode` parameter to `calibrate_rho`; when 'multipoint', call `calibrate_rho_multipoint` and return interpolated value. Keep single-point as default. Add CALIBRATION_MODE to config.py."

## File changes

| File                                          | Changes                                                                                   |
| --------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `data_generator.py`                           | Add `calibrate_rho_multipoint`; add `calibration_mode` to `calibrate_rho`; branch on mode |
| `config.py`                                   | Add `CALIBRATION_MODE = "single"`                                                         |
| `power_simulation.py`                         | Pass `calibration_mode` to `calibrate_rho`                                                |
| `confidence_interval_calculator.py`           | Pass `calibration_mode` to `calibrate_rho`                                                |
| `test_simulation_accuracy.py`                 | Pass `calibration_mode` to `calibrate_rho`                                                |
| `run_simulation.py`, `run_single_scenario.py` | Optional: add `--calibration-mode` CLI arg                                                |
| `README.md`                                   | Document `calibration_mode` and when to use multipoint                                    |

## Validation

- Run `test_simulation_accuracy` with `calibration_mode="single"` (default): all scenarios should pass (no regression).
- Run with `calibration_mode="multipoint"`: built-in scenarios should pass; compare to single-point on custom tie structures where single-point fails.
