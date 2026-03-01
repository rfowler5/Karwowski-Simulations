"""
Benchmark old vs new path on full 88-scenario grid.
Per plan: run sequential first, then parallel. One at a time.
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import time
from confidence_interval_calculator import run_all_ci_scenarios, bootstrap_ci_averaged, bootstrap_ci_single
import numba
from joblib.externals.loky import get_reusable_executor
from config import CASES
case = CASES[3]

# Warm up Numba
print("Warming up Numba (one scenario each path)...")
# bootstrap_ci_single(x, y, 0.3, n_boot=200, rng=np.random.default_rng(1))
y = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}
bootstrap_ci_averaged(73, 4, "even", -0.13, y, generator="nonparametric",
    n_reps=5, n_boot=10, seed=0, batch_bootstrap=False, calibration_mode="single")
bootstrap_ci_averaged(73, 4, "even", -0.13, y, generator="nonparametric",
    n_reps=5, n_boot=10, seed=0, batch_bootstrap=True, calibration_mode="single")
print("Warmup done.\n")

def shutdown_loky():
    try:
        executor = get_reusable_executor()
        if executor:
            executor.shutdown(wait=True, kill_workers=True)
    except:
        pass

kwargs = dict(
    generator="nonparametric",
    n_reps=200,
    n_boot=1000,
    seed=42,
    calibration_mode="single",
)

print("Full grid (88 scenarios): n_reps=200, n_boot=1000, single-point calibration")
print("=" * 60)

default_threads = numba.get_num_threads()
print(f"Default Numba threads: {default_threads}")

print("\n1. Old path, sequential (n_jobs=1)...")
t0 = time.perf_counter()
run_all_ci_scenarios(batch_bootstrap=False, n_jobs=1, **kwargs)
t_old_seq = time.perf_counter() - t0
print(f"   {t_old_seq:.2f}s")

print("\n2. New path, sequential (n_jobs=1)...")
t0 = time.perf_counter()
run_all_ci_scenarios(batch_bootstrap=True, n_jobs=1, **kwargs)
t_new_seq = time.perf_counter() - t0
print(f"   {t_new_seq:.2f}s")

print("\n3. Old path, parallel (n_jobs=-1)...")
t0 = time.perf_counter()
run_all_ci_scenarios(batch_bootstrap=False, n_jobs=-1, **kwargs)
t_old_par = time.perf_counter() - t0
print(f"   {t_old_par:.2f}s")
shutdown_loky()

print("\n4. New path, parallel (n_jobs=-1)...")
t0 = time.perf_counter()
run_all_ci_scenarios(batch_bootstrap=True, n_jobs=-1, **kwargs)
t_new_par = time.perf_counter() - t0
print(f"   {t_new_par:.2f}s")
shutdown_loky()

print("\n5. New path, parallel with Numba threads=1 (n_jobs=-1)...")
numba.set_num_threads(1)
t0 = time.perf_counter()
run_all_ci_scenarios(batch_bootstrap=True, n_jobs=-1, **kwargs)
t_new_seq_numba1 = time.perf_counter() - t0
numba.set_num_threads(default_threads)
print(f"   {t_new_seq_numba1:.2f}s")
shutdown_loky()

print("\n6. New path, parallel with Numba threads=1 (n_jobs=2)...")
numba.set_num_threads(1)
t0 = time.perf_counter()
run_all_ci_scenarios(batch_bootstrap=True, n_jobs=2, **kwargs)
t_new_seq_numba1_jobs2 = time.perf_counter() - t0
numba.set_num_threads(default_threads)
print(f"   {t_new_seq_numba1_jobs2:.2f}s")
shutdown_loky()

print("\n" + "=" * 60)
print("Summary")
print("-" * 40)
print(f"{'Config':<35} {'Time':>8}")
print("-" * 40)
print(f"{'Old path, sequential (n_jobs=1)':<35} {t_old_seq:>7.2f}s")
print(f"{'New path, sequential (n_jobs=1)':<35} {t_new_seq:>7.2f}s")
print(f"{'Old path, parallel (n_jobs=-1)':<35} {t_old_par:>7.2f}s")
print(f"{'New path, parallel (n_jobs=-1)':<35} {t_new_par:>7.2f}s")
print(f"{'New path, parallel Numba threads=1 (n_jobs=-1)':<45} {t_new_seq_numba1:>7.2f}s")
print(f"{'New path, parallel Numba threads=1 (n_jobs=2)':<45} {t_new_seq_numba1_jobs2:>7.2f}s")
print("-" * 40)
print(f"Sequential speedup: {t_old_seq/t_new_seq:.2f}x")
print(f"Parallel speedup:   {t_old_par/t_new_par:.2f}x")
print(f"Old path parallel speedup: {t_old_seq/t_old_par:.2f}x")
print(f"New path parallel speedup: {t_new_seq/t_new_par:.2f}x")
print(f"New parallel Numba=1 vs new seq: {t_new_seq/t_new_seq_numba1:.2f}x")
