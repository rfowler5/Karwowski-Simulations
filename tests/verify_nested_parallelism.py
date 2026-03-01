"""
Verify that nested parallelism (joblib + Numba both parallel) causes the batch-path
slowdown when using n_jobs=-1.

Run once: prints time for batch path with n_jobs=-1 using current NUMBA_NUM_THREADS.
  python tests/verify_nested_parallelism.py

Run comparison: spawns two processes (default Numba threads vs Numba=1), same workload.
  python tests/verify_nested_parallelism.py --compare

If nested parallelism is the issue, (default threads) should be significantly slower
than (Numba=1) when both use n_jobs=-1.
"""
import os
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import time
import subprocess

# Workload: full grid would be 88 scenarios; use smaller for quicker verification
N_REPS, N_BOOT = 100, 500
SEED = 42


def run_workload():
    """Run batch path with n_jobs=-1. Returns (numba_threads, elapsed_seconds)."""
    import numba
    from confidence_interval_calculator import run_all_ci_scenarios, bootstrap_ci_averaged
    from config import CASES

    # Warm up Numba (both workers do this so we compare steady-state, not JIT cost)
    case = CASES[3]
    y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
    bootstrap_ci_averaged(
        case["n"], 4, "even", case["observed_rho"], y_params,
        generator="nonparametric", n_reps=2, n_boot=5, seed=0,
        batch_bootstrap=True, calibration_mode="single",
    )

    n_threads = numba.get_num_threads()
    t0 = time.perf_counter()
    run_all_ci_scenarios(
        generator="nonparametric",
        n_reps=N_REPS,
        n_boot=N_BOOT,
        seed=SEED,
        n_jobs=-1,
        batch_bootstrap=True,
        calibration_mode="single",
    )
    elapsed = time.perf_counter() - t0
    return n_threads, elapsed


def main_worker():
    """Single run: run workload and print machine-parseable lines."""
    n_threads, elapsed = run_workload()
    print(f"NUMBA_THREADS={n_threads}")
    print(f"TIME={elapsed:.4f}")


def main_compare():
    """Spawn two workers (default Numba threads vs Numba=1), compare times."""
    script = os.path.abspath(__file__)
    if script.endswith(".pyc"):
        script = script[:-1]
    cmd = [sys.executable, script]

    # Env for default Numba threads (do not set NUMBA_NUM_THREADS so Numba uses all cores)
    env_default = os.environ.copy()
    env_default.pop("NUMBA_NUM_THREADS", None)

    # Env for Numba=1 only
    env_numba1 = os.environ.copy()
    env_numba1["NUMBA_NUM_THREADS"] = "1"

    print("Running batch path (n_jobs=-1) with DEFAULT Numba threads...")
    result_default = subprocess.run(
        cmd,
        env=env_default,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=_root,
    )
    print("Running batch path (n_jobs=-1) with NUMBA_NUM_THREADS=1...")
    result_numba1 = subprocess.run(
        cmd,
        env=env_numba1,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=_root,
    )

    def parse_output(result):
        out = result.stdout or ""
        threads = None
        t = None
        for line in out.strip().splitlines():
            if line.startswith("NUMBA_THREADS="):
                threads = int(line.split("=", 1)[1])
            elif line.startswith("TIME="):
                t = float(line.split("=", 1)[1])
        return threads, t

    threads_default, time_default = parse_output(result_default)
    threads_numba1, time_numba1 = parse_output(result_numba1)

    if result_default.returncode != 0:
        print("Default-threads run failed:", result_default.stderr)
        return 1
    if result_numba1.returncode != 0:
        print("Numba=1 run failed:", result_numba1.stderr)
        return 1

    print()
    print("Results (batch path, n_jobs=-1, full 88-scenario grid)")
    print(f"  Numba default ({threads_default} threads): {time_default:.2f}s")
    print(f"  Numba=1:                              {time_numba1:.2f}s")
    diff = time_default - time_numba1
    pct = (diff / time_numba1 * 100) if time_numba1 else 0
    print()
    if diff > 5:  # Substantial slowdown
        print(f"  Default is {diff:.1f}s slower ({pct:.0f}%) — nested parallelism cost.")
        print("  Conclusion: nested parallelism (joblib + Numba) is the cause.")
    elif diff < -5:
        print(f"  Numba=1 is {-diff:.1f}s slower — run-to-run variance or machine load.")
    else:
        print(f"  Difference {diff:.1f}s — inconclusive (try larger workload or re-run).")
    return 0


if __name__ == "__main__":
    if "--compare" in sys.argv:
        sys.exit(main_compare())
    main_worker()
