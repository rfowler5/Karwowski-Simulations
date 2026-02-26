"""Warm up Numba JIT cache by running small workloads that trigger compilation."""

import argparse
import time

import numpy as np

import config

parser = argparse.ArgumentParser(description="Warm up Numba JIT cache.")
parser.add_argument("--no-numba", action="store_true",
                    help="Disable Numba (verify fallback works)")
args = parser.parse_args()
if args.no_numba:
    config.USE_NUMBA = False

from spearman_helpers import use_numba
from confidence_interval_calculator import bootstrap_ci_single
from power_simulation import estimate_power

if not use_numba():
    print("Numba is disabled or not installed -- nothing to warm up.")
    print("Fallback to pure NumPy is active.")
    raise SystemExit(0)

print("=== Warming up Numba (first run compiles; may take 5-15 s) ===")
t0 = time.time()

rng = np.random.default_rng(0)
x = rng.integers(0, 5, size=82).astype(np.float64)
y = rng.standard_normal(82)

bootstrap_ci_single(x, y, 0.3, n_boot=200, rng=np.random.default_rng(1))

estimate_power(82, 4, "heavy_center", 0.3,
               {"median": 2.5, "iqr": 1.2, "range": (0, 10)},
               n_sims=100, generator="nonparametric", seed=1)

elapsed = time.time() - t0
print(f"=== Numba cache written ({elapsed:.1f} s) ===")
print("Subsequent runs will load from cache and skip compilation.")
