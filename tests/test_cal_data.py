import numpy as np
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from data_generator import (_get_x_template, _precompute_calibration_arrays,
                            _eval_mean_rho,
                            _precompute_calibration_arrays_fast,
                            _eval_mean_rho_fast,
                            _interp_with_extrapolation,
                            _MULTIPOINT_PROBES)
from spearman_helpers import _rank_rows
from config import CASES

n = 80
case = CASES[1]
y_params = {"median": case["median"], "iqr": case["iqr"],
            "range": case["range"]}
target_rhos = np.arange(0.05, 0.61, 0.01)

structures = [
    ("k=4 even", 4, "even", False),
    ("k=9 even", 9, "even", False),
    ("all_distinct", n, None, True),
]

def test_table_consistency(template, n_cal=500_000, seeds=(99, 100), label=""):
    """Check reproducibility of F(rho_in) between two seeds at large n_cal."""
    arrays1 = _precompute_calibration_arrays_fast(template, n_cal, seeds[0])
    arrays2 = _precompute_calibration_arrays_fast(template, n_cal, seeds[1])
    diffs = []
    for rho_in in np.arange(0.1436, 0.1445, 0.0000075):
        v1 = _eval_mean_rho_fast(rho_in, *arrays1)
        v2 = _eval_mean_rho_fast(rho_in, *arrays2)
        diffs.append(abs(v1 - v2))
        print(f"{label}: F_seed1 = {v1:.5f})")
        print(f"{label}: F_seed2 = {v2:.5f})")
    print(f"{label}: max |F_seed1 - F_seed2| = {max(diffs):.5f}")


def _precompute_calibration_arrays_fast_jittered(template, n_cal, seed):
    """Like _precompute_calibration_arrays_fast but adds Uniform(-0.49, 0.49)
    jitter to the noise permutation before standardisation.

    This breaks the integer commensurability of the noise ranks that causes
    the discrete staircase in E[Spearman(rho_in)] for heavily tied x
    (e.g. k=4 even).  The jitter is small enough (<0.5) that rank ordering
    within each noise permutation row is never changed, so the noise still
    represents a uniform random permutation of positions.
    """
    cal_rng = np.random.default_rng(seed)
    n = len(template)

    x_batch = np.tile(template, (n_cal, 1))
    if hasattr(cal_rng, 'permuted'):
        x_batch = cal_rng.permuted(x_batch, axis=1)
    else:
        perm = np.argsort(cal_rng.random((n_cal, n)), axis=1)
        x_batch = np.take_along_axis(x_batch, perm, axis=1)

    x_ranks_batch = _rank_rows(x_batch)

    noise_base = np.tile(np.arange(1.0, n + 1.0), (n_cal, 1))
    if hasattr(cal_rng, 'permuted'):
        noise_batch = cal_rng.permuted(noise_base, axis=1)
    else:
        perm = np.argsort(cal_rng.random((n_cal, n)), axis=1)
        noise_batch = np.take_along_axis(noise_base, perm, axis=1)

    # Add per-element jitter in (-0.49, 0.49); adjacent integers can never
    # cross (gap = 1.0 > 2 * 0.49), so rank order within each row is
    # preserved and the noise permutation semantics are unchanged.
    noise_batch = noise_batch + cal_rng.uniform(-0.49, 0.49, size=(n_cal, n))

    s_x = x_ranks_batch - x_ranks_batch.mean(axis=1, keepdims=True)
    sd_x = x_ranks_batch.std(axis=1, keepdims=True, ddof=0)
    sd_x = np.where(sd_x > 0, sd_x, 1.0)
    s_x /= sd_x

    noise_mean = (n + 1) / 2.0
    noise_std = np.sqrt((n * n - 1) / 12.0)
    s_n = (noise_batch - noise_mean) / noise_std

    return s_x, s_n, x_ranks_batch


def test_jitter_smoothing(template, n_cal=500_000, seed=99, label="",
                          grid=None):
    """Compare F(rho_in) on a fine grid with and without noise jitter.

    Without jitter the function should be a step function (flat segments
    separated by a sharp jump).  With jitter it should vary smoothly.

    Parameters
    ----------
    grid : array-like or None
        rho_in values to probe.  Defaults to 200 points in [0.1436, 0.1445].
    """
    if grid is None:
        grid = np.linspace(0.1436, 0.1445, 200)

    arrays_plain = _precompute_calibration_arrays_fast(template, n_cal, seed)
    arrays_jitter = _precompute_calibration_arrays_fast_jittered(template, n_cal, seed)

    print(f"\n--- {label}: jitter smoothing diagnostic ---")
    print(f"{'rho_in':>10}  {'F_plain':>10}  {'F_jitter':>10}  {'diff':>8}")
    for rho_in in grid:
        vp = _eval_mean_rho_fast(rho_in, *arrays_plain)
        vj = _eval_mean_rho_fast(rho_in, *arrays_jitter)
        print(f"{rho_in:10.6f}  {vp:10.5f}  {vj:10.5f}  {vj - vp:+8.5f}")


def _bisect_jittered(tr, cal_arrays):
    """Bisect cal_arrays to find rho_in such that F_jitter(rho_in) ≈ tr."""
    lo, hi = 0.0, min(tr * 2.0, 0.999)
    if _eval_mean_rho_fast(hi, *cal_arrays) < tr:
        hi = 0.999
    if _eval_mean_rho_fast(hi, *cal_arrays) < tr:
        return tr  # unreachable; return raw target
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _eval_mean_rho_fast(mid, *cal_arrays) < tr:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    vl = _eval_mean_rho_fast(lo, *cal_arrays)
    vh = _eval_mean_rho_fast(hi, *cal_arrays)
    return hi if abs(vh - tr) <= abs(vl - tr) else lo


def _print_deviation_table(label, tag, target_rhos, cal_arrays, eval_arrays, n_cal,
                            get_rho_in_fn):
    """Shared printer for bisection and multipoint accuracy tests."""
    deviations = []
    print(f"\n--- {label} ({tag}, n_cal={n_cal:,}) ---")
    print("target_rho  calibrated_rho_in  realised_rho   deviation")
    for tr in target_rhos:
        rho_in = get_rho_in_fn(tr)
        realised = _eval_mean_rho_fast(rho_in, *eval_arrays)
        deviation = realised - tr
        deviations.append(deviation)
        print(f"  {tr:.2f}       {rho_in:.6f}         {realised:.5f}      {deviation:+.5f}")
    max_abs = max(abs(d) for d in deviations)
    mean_abs = float(np.mean(np.abs(deviations)))
    print(f"max_abs_deviation:  {max_abs:.5f}")
    print(f"mean_abs_deviation: {mean_abs:.5f}")
    for thresh, label_tier in [(0.001, "+/-0.001"), (0.002, "+/-0.002"), (0.010, "+/-0.010")]:
        if max_abs <= thresh:
            print(f"  OK: within {label_tier} tier")
            break
    else:
        print(f"  WARNING: exceeds +/-0.010 tier")


def test_bisection_accuracy_jittered(template, target_rhos, n_cal=500_000,
                                     cal_seed=99, eval_seed=12345, label="",
                                     cal_arrays=None, eval_arrays=None):
    """Per-target bisection accuracy test using jittered noise.

    Mirrors the plain bisection block at the bottom of this file but uses
    _precompute_calibration_arrays_fast_jittered + _eval_mean_rho_fast.
    With jitter the calibration curve is smooth, so bisection should reach
    every target to within MC noise (~0.0005 at n_cal=500k) rather than
    being capped by the discrete staircase step (~0.010-0.013 plain).

    Pass pre-built cal_arrays / eval_arrays to avoid redundant precomputation
    when calling multiple tests for the same template and n_cal.
    """
    if cal_arrays is None:
        cal_arrays = _precompute_calibration_arrays_fast_jittered(template, n_cal, cal_seed)
    if eval_arrays is None:
        eval_arrays = _precompute_calibration_arrays_fast_jittered(template, n_cal, eval_seed)
    _print_deviation_table(label, "jittered bisection", target_rhos,
                           cal_arrays, eval_arrays, n_cal,
                           lambda tr: _bisect_jittered(tr, cal_arrays))


def test_multipoint_accuracy_jittered(template, target_rhos, n_cal=500_000,
                                      cal_seed=99, eval_seed=12345, label="",
                                      cal_arrays=None, eval_arrays=None):
    """Multipoint calibration accuracy test using jittered noise.

    Simulates the full production multipoint pipeline end-to-end:
      1. Bisect at each of _MULTIPOINT_PROBES on jittered cal_arrays.
      2. Build (probe, rho_in) pairs.
      3. Interpolate with _interp_with_extrapolation for each target.
      4. Evaluate the resulting calibrated rho_in on independent eval_arrays.

    Pass pre-built cal_arrays / eval_arrays to share precomputed data with
    test_bisection_accuracy_jittered.
    """
    if cal_arrays is None:
        cal_arrays = _precompute_calibration_arrays_fast_jittered(template, n_cal, cal_seed)
    if eval_arrays is None:
        eval_arrays = _precompute_calibration_arrays_fast_jittered(template, n_cal, eval_seed)

    # Build probe -> rho_in map (mirrors _calibrate_rho_multipoint logic)
    pairs = []
    for probe in _MULTIPOINT_PROBES:
        rho_in = _bisect_jittered(probe, cal_arrays)
        pairs.append((probe, rho_in))
    pairs.sort(key=lambda p: p[0])

    probes = [p for p, _ in pairs]
    rho_ins = [r for _, r in pairs]

    print(f"\n  Multipoint probe map (jittered):")
    for p, r in pairs:
        print(f"    probe={p:.2f}  ->  rho_in={r:.6f}  "
              f"(F_cal={_eval_mean_rho_fast(r, *cal_arrays):.5f})")

    def _interp_rho_in(tr):
        result = _interp_with_extrapolation(abs(tr), probes, rho_ins)
        return float(np.clip(result, -0.999, 0.999))

    _print_deviation_table(label, "jittered multipoint", target_rhos,
                           cal_arrays, eval_arrays, n_cal, _interp_rho_in)


if __name__ == "__main__":
    # Optimized run: k=4 even only (worst case), n_cal=100k (50x faster than
    # 500k; MC noise ~0.0005, well below the 0.001 decision threshold).
    # Arrays built once and shared between both tests to avoid duplicate work.
    N_CAL = 100_000
    label, n_distinct, dist_type, all_dist = structures[0]  # k=4 even
    template = _get_x_template(n, n_distinct, dist_type, None, all_dist)
    print(f"Building jittered cal/eval arrays (n_cal={N_CAL:,}) ...")
    cal_arrays = _precompute_calibration_arrays_fast_jittered(template, N_CAL, 99)
    eval_arrays = _precompute_calibration_arrays_fast_jittered(template, N_CAL, 12345)
    print("Done.")

    # test_bisection_accuracy_jittered(
    #     template, target_rhos, n_cal=N_CAL, label=label,
    #     cal_arrays=cal_arrays, eval_arrays=eval_arrays)

    test_multipoint_accuracy_jittered(
        template, target_rhos, n_cal=N_CAL, label=label,
        cal_arrays=cal_arrays, eval_arrays=eval_arrays)

    sys.exit(0)

for label, n_distinct, dist_type, all_dist in structures:
    template = _get_x_template(n, n_distinct, dist_type, None, all_dist)
    cal_arrays = _precompute_calibration_arrays(template, y_params, 500000, 99)
    eval_arrays = _precompute_calibration_arrays(template, y_params, 500000, 12345)

    deviations = []
    print(f"\n--- {label} (per-target bisection) ---")
    print("target_rho  calibrated_rho_in  realised_rho   deviation")

    for tr in target_rhos:
        lo, hi = 0.0, min(tr * 2.0, 0.999)
        val_hi = _eval_mean_rho(hi, *cal_arrays)
        if val_hi < tr:
            hi = 0.999
            val_hi = _eval_mean_rho(hi, *cal_arrays)
        if val_hi < tr:
            rho_in = tr
        else:
            for _ in range(25):
                mid = (lo + hi) / 2.0
                observed = _eval_mean_rho(mid, *cal_arrays)
                if observed < tr:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 5e-5:
                    break
            vl = _eval_mean_rho(lo, *cal_arrays)
            vh = _eval_mean_rho(hi, *cal_arrays)
            rho_in = hi if abs(vh - tr) <= abs(vl - tr) else lo

        realised = _eval_mean_rho(rho_in, *eval_arrays)
        deviation = realised - tr
        deviations.append(deviation)
        print(f"  {tr:.2f}       {rho_in:.4f}           {realised:.4f}      {deviation:+.4f}")

    max_abs = max(abs(d) for d in deviations)
    mean_abs = float(np.mean(np.abs(deviations)))
    print(f"max_abs_deviation: {max_abs:.4f}")
    print(f"mean_abs_deviation: {mean_abs:.4f}")
    if max_abs > 0.010:
        print(f"  WARNING: exceeds +/-0.01 tier")
    else:
        print(f"  OK: within +/-0.01 tier")