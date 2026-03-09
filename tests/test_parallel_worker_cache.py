"""Smoke test: parallel worker cache injection via joblib initializer/initargs.

Verifies:
  1. run_all_scenarios(n_jobs=2) completes without PicklingError or other
     worker-init failures.
  2. Results are bitwise identical to n_jobs=1 when both runs share the same
     pre-warmed calibration and null caches (same curves, same null arrays
     => same calibrated rho, same simulation outcomes for fixed seeds).
  3. run_all_ci_scenarios(n_jobs=2) completes without errors and returns the
     expected number of scenarios, with plausible CI values.

Test grid for power: 1 case, 2 k-values, 1 dist_type => 3 scenarios
(2 tied + 1 all-distinct).  Keeps the test under ~60 s on typical hardware.
Full 88-scenario grid is used for CI but with minimal n_reps/n_boot.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES

_TEST_CASE_IDS = [1]
_TEST_CASES = {k: v for k, v in CASES.items() if k in _TEST_CASE_IDS}
_TEST_K = [4, 5]
_TEST_DT = ["even"]
_SEED = 42
_N_SIMS = 200


def _warm_power_caches():
    """Warm null and calibration caches for the restricted power test grid.

    Called once before both sequential and parallel runs so they share
    exactly the same cached values, making results bitwise comparable.
    """
    from data_generator import warm_calibration_cache
    from config import USE_PERMUTATION_PVALUE

    warm_calibration_cache(
        "nonparametric",
        cases=_TEST_CASES,
        n_distinct_values=_TEST_K,
        dist_types=_TEST_DT,
    )
    if USE_PERMUTATION_PVALUE:
        from permutation_pvalue import warm_precomputed_null_cache
        warm_precomputed_null_cache(
            cases=_TEST_CASES,
            n_distinct_values=_TEST_K,
            dist_types=_TEST_DT,
            seed=42,
        )


def test_power_parallel_matches_sequential():
    """run_all_scenarios(n_jobs=2) is bitwise identical to n_jobs=1.

    Both runs use pre_warm=False so they draw from the same already-warm
    module-level caches (sequential hits them directly; parallel snapshots
    and injects them into workers via the initializer).
    """
    from power_simulation import run_all_scenarios

    _warm_power_caches()

    kwargs = dict(
        generator="nonparametric",
        n_sims=_N_SIMS,
        seed=_SEED,
        cases=_TEST_CASE_IDS,
        n_distinct_values=_TEST_K,
        dist_types=_TEST_DT,
        pre_warm=False,
    )

    seq = run_all_scenarios(n_jobs=1, **kwargs)
    par = run_all_scenarios(n_jobs=2, **kwargs)

    assert len(seq) == len(par), (
        f"Scenario count mismatch: seq={len(seq)}, par={len(par)}"
    )

    sort_key = lambda r: (r["case"], r["n_distinct"], r["dist_type"], r["direction"])
    seq_sorted = sorted(seq, key=sort_key)
    par_sorted = sorted(par, key=sort_key)

    mismatches = []
    for s, p in zip(seq_sorted, par_sorted):
        if s["min_detectable_rho"] != p["min_detectable_rho"]:
            mismatches.append(
                f"  case={s['case']} k={s['n_distinct']} dt={s['dist_type']} "
                f"dir={s['direction']}: "
                f"seq={s['min_detectable_rho']:.6f}  par={p['min_detectable_rho']:.6f}"
            )

    assert not mismatches, (
        "Sequential and parallel min_detectable_rho differ:\n"
        + "\n".join(mismatches)
    )

    print(f"  PASS: {len(seq)} power scenarios identical (n_jobs=1 vs n_jobs=2)")


def test_ci_parallel_no_crash():
    """run_all_ci_scenarios(n_jobs=2) completes and returns plausible results.

    Runs the full 88-scenario grid with tiny n_reps/n_boot so it finishes
    quickly.  Checks: correct count, no exceptions, and ci_lower < ci_upper
    for every scenario.
    """
    from confidence_interval_calculator import run_all_ci_scenarios

    expected = len(CASES) * (len(N_DISTINCT_VALUES) * len(DISTRIBUTION_TYPES) + 1)

    results = run_all_ci_scenarios(
        generator="nonparametric",
        n_reps=3,
        n_boot=20,
        seed=_SEED,
        n_jobs=2,
        # pre_warm=True is the default; exercises the full warm+snapshot+inject path
    )

    assert len(results) == expected, (
        f"Expected {expected} CI scenarios, got {len(results)}"
    )

    inverted = [
        r for r in results
        if r["boot_ci_lower"] >= r["boot_ci_upper"]
    ]
    assert not inverted, (
        f"{len(inverted)} scenarios have inverted CI (lower >= upper): "
        + str(inverted[:3])
    )

    print(
        f"  PASS: {len(results)} CI scenarios completed without error "
        f"(n_jobs=2, pre_warm=True)"
    )


if __name__ == "__main__":
    print("Smoke test: parallel worker cache injection")
    print("-" * 50)

    print("1. Power: sequential vs parallel exact match...")
    test_power_parallel_matches_sequential()

    print("2. CI: parallel no-crash + result sanity (88 scenarios, n_reps=3)...")
    test_ci_parallel_no_crash()

    print()
    print("All smoke tests passed.")
