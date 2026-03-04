"""Verify every entry in config.FREQ_DICT has counts summing to n."""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import FREQ_DICT


def main():
    errors = []
    for n, by_k in FREQ_DICT.items():
        for k, by_dist in by_k.items():
            for dist_type, counts in by_dist.items():
                total = sum(counts)
                if total != n:
                    errors.append(
                        f"FREQ_DICT[{n}][{k}][{dist_type!r}]: sum={total}, expected {n}")
                if len(counts) != k:
                    errors.append(
                        f"FREQ_DICT[{n}][{k}][{dist_type!r}]: len={len(counts)}, expected {k}")

    if errors:
        print("FAIL: FREQ_DICT validation errors:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"PASS: All FREQ_DICT entries valid "
              f"({sum(len(bk) * len(bd) for bk in FREQ_DICT.values() for bd in bk.values())} entries checked).")


if __name__ == "__main__":
    main()
