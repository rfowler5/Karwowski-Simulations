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
    """Same seed + same n_sims gives identical (x, y) datasets."""
    case = CASES[3]
    n, k, dt = case["n"], 4, "even"
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
    rho_s = case["observed_rho"]
    n_sims = 3

    ln_params = _fit_lognormal(y_params["median"], y_params["iqr"])
    cal_rho = calibrate_rho(n, k, dt, rho_s, y_params,
                            all_distinct=False, calibration_mode="single")

    # Run A
    rng_a = np.random.default_rng(SEED)
    x_a = generate_cumulative_aluminum_batch(n_sims, n, k, distribution_type=dt,
                                              all_distinct=False, rng=rng_a)
    y_a = generate_y_nonparametric_batch(x_a, rho_s, y_params, rng=rng_a,
                                          _calibrated_rho=cal_rho, _ln_params=ln_params)

    # Run B: same seed, same n_sims
    rng_b = np.random.default_rng(SEED)
    x_b = generate_cumulative_aluminum_batch(n_sims, n, k, distribution_type=dt,
                                              all_distinct=False, rng=rng_b)
    y_b = generate_y_nonparametric_batch(x_b, rho_s, y_params, rng=rng_b,
                                          _calibrated_rho=cal_rho, _ln_params=ln_params)

    assert np.array_equal(x_a, x_b), "x arrays differ for same seed + same n_sims"
    assert np.array_equal(y_a, y_b), "y arrays differ for same seed + same n_sims"
    print("PASS: Power path — same seed + same n_sims gives identical datasets.")


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
