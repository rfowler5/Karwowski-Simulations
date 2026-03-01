"""Compare single scenario: batch_bootstrap=True vs False."""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import time
from config import CASES
from confidence_interval_calculator import bootstrap_ci_averaged

case = CASES[3]
y_params = {"median": case["median"], "iqr": case["iqr"], "range": case["range"]}

print("Single scenario (Case 3, k=4, even): n_reps=200, n_boot=1000, single-point calibration")
print("-" * 60)

print("Without batching (batch_bootstrap=False)...")
t0 = time.perf_counter()
r_old = bootstrap_ci_averaged(
    73, 4, "even", -0.13, y_params,
    generator="nonparametric", n_reps=200, n_boot=1000, seed=42,
    batch_bootstrap=False, calibration_mode="single",
)
t_old = time.perf_counter() - t0
print(f"  {t_old:.2f}s  ci=[{r_old['ci_lower']:.4f}, {r_old['ci_upper']:.4f}]")

print("With batching (batch_bootstrap=True)...")
t0 = time.perf_counter()
r_new = bootstrap_ci_averaged(
    73, 4, "even", -0.13, y_params,
    generator="nonparametric", n_reps=200, n_boot=1000, seed=42,
    batch_bootstrap=True, calibration_mode="single",
)
t_new = time.perf_counter() - t0
print(f"  {t_new:.2f}s  ci=[{r_new['ci_lower']:.4f}, {r_new['ci_upper']:.4f}]")

print("-" * 60)
print(f"Without batching: {t_old:.2f}s")
print(f"With batching:    {t_new:.2f}s")
print(f"Speedup:          {t_old/t_new:.2f}x")
