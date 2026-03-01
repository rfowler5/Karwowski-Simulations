"""
Measure the 'big gap': batch sequential (default Numba) vs batch parallel (Numba=1).

The doc had: sequential ~112s, parallel (Numba=1) ~175s — so parallel was slower.
This script compares the same two configs in separate processes (so Numba thread
count is set correctly before import). If T_par > T_seq, we've confirmed that
most of the huge gap is joblib overhead / no benefit, not nested parallelism.

Usage:
  python tests/verify_big_gap.py --compare   # run both, print comparison
  python tests/verify_big_gap.py --seq      # run sequential (default Numba), print TIME=
  python tests/verify_big_gap.py --par      # run parallel (call with NUMBA_NUM_THREADS=1)
"""
import os
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import time
import subprocess

# Full grid 88 scenarios; moderate size so each run finishes in a few minutes
N_REPS, N_BOOT, SEED = 200, 1000, 42
kwargs = dict(
    generator="nonparametric",
    n_reps=N_REPS,
    n_boot=N_BOOT,
    seed=SEED,
    batch_bootstrap=True,
    calibration_mode="single",
)


def run_workload(n_jobs):
    """Run full grid with given n_jobs. Returns elapsed seconds."""
    from confidence_interval_calculator import run_all_ci_scenarios, bootstrap_ci_averaged
    from config import CASES

    # Warm up
    case = CASES[3]
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
    bootstrap_ci_averaged(
        case["n"], 4, "even", case["observed_rho"], y_params,
        generator="nonparametric", n_reps=2, n_boot=5, seed=0,
        batch_bootstrap=True, calibration_mode="single",
    )
    t0 = time.perf_counter()
    run_all_ci_scenarios(n_jobs=n_jobs, **kwargs)
    return time.perf_counter() - t0


def main_seq():
    """Sequential: default Numba (do not set NUMBA_NUM_THREADS)."""
    elapsed = run_workload(n_jobs=1)
    print(f"TIME={elapsed:.4f}")


def main_par():
    """Parallel: must be called with NUMBA_NUM_THREADS=1 (set by parent)."""
    elapsed = run_workload(n_jobs=-1)
    print(f"TIME={elapsed:.4f}")


def main_compare():
    """Spawn sequential (default Numba) and parallel (Numba=1), compare."""
    script = os.path.abspath(__file__)
    if script.endswith(".pyc"):
        script = script[:-1]
    py = sys.executable

    env_default = os.environ.copy()
    env_default.pop("NUMBA_NUM_THREADS", None)
    env_numba1 = os.environ.copy()
    env_numba1["NUMBA_NUM_THREADS"] = "1"
    cwd = _root  # run from repo root so imports work

    def run(args, env):
        r = subprocess.run(
            [py, script] + args,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=cwd,
        )
        t = None
        for line in (r.stdout or "").strip().splitlines():
            if line.startswith("TIME="):
                t = float(line.split("=", 1)[1])
                break
        return r.returncode, t

    print("1. Batch path, sequential (n_jobs=1), default Numba threads...")
    rc_seq, t_seq = run(["--seq"], env_default)
    if rc_seq != 0:
        print("   Failed:", subprocess.run([py, script, "--seq"], env=env_default, cwd=cwd).stderr)
        return 1
    print(f"   {t_seq:.2f}s\n")

    print("2. Batch path, parallel (n_jobs=-1), NUMBA_NUM_THREADS=1...")
    rc_par, t_par = run(["--par"], env_numba1)
    if rc_par != 0:
        print("   Failed")
        return 1
    print(f"   {t_par:.2f}s\n")

    gap = t_par - t_seq
    pct = (gap / t_seq * 100) if t_seq else 0
    print("=" * 60)
    print(f"  Sequential (default Numba): {t_seq:.2f}s")
    print(f"  Parallel   (Numba=1):      {t_par:.2f}s")
    print(f"  Gap (parallel - sequential): {gap:.2f}s  ({pct:+.0f}%)")
    print()
    if gap > 10:
        print("Conclusion: Parallel is slower — most of the 'huge gap' is joblib")
        print("overhead / no benefit (no nested parallelism in the parallel run).")
    elif gap < -10:
        print("Conclusion: Parallel is faster here (joblib helped with Numba=1).")
    else:
        print("Gap is small — inconclusive or run-to-run variance.")
    return 0


if __name__ == "__main__":
    if "--compare" in sys.argv:
        sys.exit(main_compare())
    if "--par" in sys.argv:
        main_par()
    else:
        main_seq()
