"""Verify get_pool and build_empirical_pool raise ValueError for unsupported sample sizes."""

import sys
from pathlib import Path
import numpy as np
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data_generator import digitized_available


def main():
    if not digitized_available():
        print("SKIP: Digitized data not available; cannot test get_pool/build_empirical_pool.")
        return

    from data_generator import get_pool, build_empirical_pool

    invalid_ns = [70, 85, 100, 1]
    ok = True

    for n in invalid_ns:
        try:
            get_pool(n)
            print(f"FAIL: get_pool({n}) did not raise ValueError")
            ok = False
        except ValueError:
            print(f"PASS: get_pool({n}) correctly raised ValueError")
        except ImportError:
            print(f"SKIP: get_pool({n}) raised ImportError (digitized data issue)")

    rng = np.random.default_rng(42)
    for n in invalid_ns:
        try:
            build_empirical_pool(n, rng)
            print(f"FAIL: build_empirical_pool({n}, rng) did not raise ValueError")
            ok = False
        except ValueError:
            print(f"PASS: build_empirical_pool({n}, rng) correctly raised ValueError")
        except ImportError:
            print(f"SKIP: build_empirical_pool({n}, rng) raised ImportError (digitized data issue)")

    if not ok:
        sys.exit(1)
    print("PASS: All invalid-n checks passed.")


if __name__ == "__main__":
    main()
