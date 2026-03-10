"""
Tests for the bulletproof calibration disk cache format.

Covers:
  1.  _cal_key_to_disk_key — all 8 key shapes
  2.  _disk_key_to_cal_key — round-trip (strip → reconstruct == original)
  3.  Save/load round-trip — disk keys have counts_tuple, no n_cal, no "multipoint"
  4.  n_cal mismatch → returns False
  5.  multipoint_probes mismatch → mp dicts empty, sp dicts loaded
  6.  config_hash mismatch (soft) → warning emitted, data still loaded
  7.  Unrecognized format (no multipoint_probes) → returns False with warning
  8.  Save-condition fix — save occurs even when load succeeded (new entries persisted)
  9.  cleanup_stale_calibration_entries — removes entries with changed counts_tuple
 10.  Load-before-warm integration — second precompute call merges old + new entries
 11.  Null cache: soft config_hash (warn + load), hard n_pre (reject)

CLI usage
---------
    python tests/test_cal_disk_cache.py
"""

import sys
import pickle
import tempfile
import warnings
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

import data_generator as _dg
from config import FREQ_DICT
from data_generator import (
    _cal_key_to_disk_key,
    _disk_key_to_cal_key,
    _MULTIPOINT_PROBES,
    _CALIBRATION_CACHE_MULTIPOINT,
    _CALIBRATION_CACHE_MULTIPOINT_COPULA,
    _CALIBRATION_CACHE_MULTIPOINT_EMP,
    _CALIBRATION_CACHE_MULTIPOINT_LINEAR,
    _CALIBRATION_CACHE,
    _CALIBRATION_CACHE_COPULA,
    _CALIBRATION_CACHE_EMP,
    _CALIBRATION_CACHE_LINEAR,
    save_calibration_caches_to_disk,
    load_calibration_caches_from_disk,
    cleanup_stale_calibration_entries,
    compute_config_hash,
)
from permutation_pvalue import (
    _NULL_CACHE,
    save_null_cache_to_disk,
    load_null_cache_from_disk,
    cleanup_stale_null_entries,
)

# Stable fixture values taken from config.FREQ_DICT
_N = 80
_K = 4
_DT = "heavy_center"
_COUNTS = tuple(FREQ_DICT[_N][_K][_DT])   # (12, 30, 29, 9)
_CUSTOM_COUNTS = (20, 20, 20, 20)
_N_CAL = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_all_cal_caches():
    for d in (
        _CALIBRATION_CACHE_MULTIPOINT,
        _CALIBRATION_CACHE_MULTIPOINT_COPULA,
        _CALIBRATION_CACHE_MULTIPOINT_EMP,
        _CALIBRATION_CACHE_MULTIPOINT_LINEAR,
        _CALIBRATION_CACHE,
        _CALIBRATION_CACHE_COPULA,
        _CALIBRATION_CACHE_EMP,
        _CALIBRATION_CACHE_LINEAR,
    ):
        d.clear()


def _make_file_with_raw_entry(path, disk_key, label="sp", value=1.0):
    """Write a minimal calibration file containing one raw disk entry."""
    payload = {
        "metadata": {
            "n_cal": _N_CAL,
            "multipoint_probes": tuple(_MULTIPOINT_PROBES),
            "config_hash": compute_config_hash(),
        },
        "caches": {
            "mp": {}, "mp_cop": {}, "mp_emp": {}, "mp_lin": {},
            "sp": {}, "sp_cop": {}, "sp_emp": {}, "sp_lin": {},
        },
    }
    payload["caches"][label][disk_key] = value
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# 1. _cal_key_to_disk_key — all 8 key shapes
# ---------------------------------------------------------------------------

def test_cal_key_to_disk_key_mp_standard():
    key = (_N, _K, _DT, False, _N_CAL, "multipoint")
    assert _cal_key_to_disk_key(key) == (_N, _K, _DT, False, _COUNTS)


def test_cal_key_to_disk_key_mp_all_distinct():
    key = (_N, _N, None, True, _N_CAL, "multipoint")
    assert _cal_key_to_disk_key(key) == (_N, _N, None, True)


def test_cal_key_to_disk_key_mp_custom():
    key = (_N, _K, "custom", False, _N_CAL, "multipoint", _CUSTOM_COUNTS)
    assert _cal_key_to_disk_key(key) == (_N, _K, "custom", False, _CUSTOM_COUNTS)


def test_cal_key_to_disk_key_sp_standard():
    key = (_N, _K, _DT, False, _N_CAL)
    assert _cal_key_to_disk_key(key) == (_N, _K, _DT, False, _COUNTS)


def test_cal_key_to_disk_key_sp_all_distinct():
    key = (_N, _N, None, True, _N_CAL)
    assert _cal_key_to_disk_key(key) == (_N, _N, None, True)


def test_cal_key_to_disk_key_sp_custom():
    key = (_N, _K, "custom", False, _N_CAL, _CUSTOM_COUNTS)
    assert _cal_key_to_disk_key(key) == (_N, _K, "custom", False, _CUSTOM_COUNTS)


def test_cal_key_to_disk_key_mp_emp_standard():
    key = ("emp", _N, _K, _DT, False, _N_CAL, "multipoint")
    assert _cal_key_to_disk_key(key) == (_N, _K, _DT, False, _COUNTS)


def test_cal_key_to_disk_key_sp_emp_standard():
    key = ("emp", _N, _K, _DT, False, _N_CAL)
    assert _cal_key_to_disk_key(key) == (_N, _K, _DT, False, _COUNTS)


def test_cal_key_to_disk_key_counts_always_present_for_non_all_distinct():
    """Non-all_distinct disk keys must always carry a counts_tuple."""
    for mem_key in [
        (_N, _K, _DT, False, _N_CAL, "multipoint"),
        (_N, _K, _DT, False, _N_CAL),
        ("emp", _N, _K, _DT, False, _N_CAL, "multipoint"),
        ("emp", _N, _K, _DT, False, _N_CAL),
    ]:
        dk = _cal_key_to_disk_key(mem_key)
        assert len(dk) == 5, f"Expected 5-tuple disk key, got {dk!r}"
        assert isinstance(dk[4], tuple), f"counts_tuple should be a tuple, got {dk[4]!r}"


# ---------------------------------------------------------------------------
# 2. _disk_key_to_cal_key round-trip
# ---------------------------------------------------------------------------

def _roundtrip(mem_key, label):
    dk = _cal_key_to_disk_key(mem_key)
    rebuilt = _disk_key_to_cal_key(dk, _N_CAL, label)
    assert rebuilt == mem_key, (
        f"Round-trip failed for label={label!r}:\n"
        f"  original : {mem_key!r}\n"
        f"  disk key : {dk!r}\n"
        f"  rebuilt  : {rebuilt!r}"
    )


def test_roundtrip_mp_standard():
    _roundtrip((_N, _K, _DT, False, _N_CAL, "multipoint"), "mp")


def test_roundtrip_mp_all_distinct():
    _roundtrip((_N, _N, None, True, _N_CAL, "multipoint"), "mp")


def test_roundtrip_mp_custom():
    _roundtrip((_N, _K, "custom", False, _N_CAL, "multipoint", _CUSTOM_COUNTS), "mp")


def test_roundtrip_mp_cop():
    _roundtrip((_N, _K, _DT, False, _N_CAL, "multipoint"), "mp_cop")


def test_roundtrip_mp_lin():
    _roundtrip((_N, _K, _DT, False, _N_CAL, "multipoint"), "mp_lin")


def test_roundtrip_mp_emp_standard():
    _roundtrip(("emp", _N, _K, _DT, False, _N_CAL, "multipoint"), "mp_emp")


def test_roundtrip_mp_emp_custom():
    _roundtrip(("emp", _N, _K, "custom", False, _N_CAL, "multipoint", _CUSTOM_COUNTS), "mp_emp")


def test_roundtrip_sp_standard():
    _roundtrip((_N, _K, _DT, False, _N_CAL), "sp")


def test_roundtrip_sp_all_distinct():
    _roundtrip((_N, _N, None, True, _N_CAL), "sp")


def test_roundtrip_sp_custom():
    _roundtrip((_N, _K, "custom", False, _N_CAL, _CUSTOM_COUNTS), "sp")


def test_roundtrip_sp_emp():
    _roundtrip(("emp", _N, _K, _DT, False, _N_CAL), "sp_emp")


def test_roundtrip_sp_lin():
    _roundtrip((_N, _K, _DT, False, _N_CAL), "sp_lin")


# ---------------------------------------------------------------------------
# 3. Save/load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    _clear_all_cal_caches()
    mem_key_mp = (_N, _K, _DT, False, _N_CAL, "multipoint")
    mem_key_sp = (_N, _K, _DT, False, _N_CAL)
    sentinel_mp = [(0.10, 0.11), (0.30, 0.33), (0.50, 0.55)]
    sentinel_sp = 1.1

    _CALIBRATION_CACHE_MULTIPOINT[mem_key_mp] = sentinel_mp
    _CALIBRATION_CACHE[mem_key_sp] = sentinel_sp

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "calibration_ncal300.pkl"
        save_calibration_caches_to_disk(path, _N_CAL)

        with open(path, "rb") as f:
            payload = pickle.load(f)
        meta = payload["metadata"]
        assert "multipoint_probes" in meta, "multipoint_probes must be in metadata"
        assert meta["n_cal"] == _N_CAL
        assert "config_hash" in meta

        disk_key = (_N, _K, _DT, False, _COUNTS)
        assert disk_key in payload["caches"]["mp"], "disk key not found in mp cache"
        assert disk_key in payload["caches"]["sp"], "disk key not found in sp cache"
        for part in disk_key:
            assert part != _N_CAL, "disk key must not contain n_cal"
            assert part != "multipoint", "disk key must not contain 'multipoint'"

        _clear_all_cal_caches()
        result = load_calibration_caches_from_disk(path, _N_CAL)
        assert result is True
        assert _CALIBRATION_CACHE_MULTIPOINT.get(mem_key_mp) == sentinel_mp
        assert _CALIBRATION_CACHE.get(mem_key_sp) == sentinel_sp

    _clear_all_cal_caches()


def test_load_missing_file_returns_false():
    result = load_calibration_caches_from_disk("/nonexistent/path/cal.pkl", _N_CAL)
    assert result is False


# ---------------------------------------------------------------------------
# 4. n_cal mismatch → False
# ---------------------------------------------------------------------------

def test_ncal_mismatch_returns_false():
    _clear_all_cal_caches()
    _CALIBRATION_CACHE[(_N, _K, _DT, False, _N_CAL)] = 1.0

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cal.pkl"
        save_calibration_caches_to_disk(path, _N_CAL)
        _clear_all_cal_caches()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_calibration_caches_from_disk(path, 99999)
        assert result is False, "n_cal mismatch must return False"
        assert any("n_cal" in str(warning.message) for warning in w), \
            "Expected a warning mentioning n_cal"

    _clear_all_cal_caches()


# ---------------------------------------------------------------------------
# 5. multipoint_probes mismatch → mp empty, sp loaded
# ---------------------------------------------------------------------------

def test_multipoint_probes_mismatch_skips_mp_loads_sp():
    _clear_all_cal_caches()
    mem_key_mp = (_N, _K, _DT, False, _N_CAL, "multipoint")
    mem_key_sp = (_N, _K, _DT, False, _N_CAL)
    _CALIBRATION_CACHE_MULTIPOINT[mem_key_mp] = [(0.3, 0.33)]
    _CALIBRATION_CACHE[mem_key_sp] = 1.1

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cal.pkl"
        save_calibration_caches_to_disk(path, _N_CAL)
        _clear_all_cal_caches()

        original_probes = list(_dg._MULTIPOINT_PROBES)
        _dg._MULTIPOINT_PROBES = [0.15, 0.30, 0.45]
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = load_calibration_caches_from_disk(path, _N_CAL)
            assert result is True, "Should still return True (sp was loaded)"
            assert any("multipoint_probes" in str(warning.message) for warning in w), \
                "Expected a warning about multipoint_probes mismatch"
            assert len(_CALIBRATION_CACHE_MULTIPOINT) == 0, "mp dict must be empty"
            assert _CALIBRATION_CACHE.get(mem_key_sp) == 1.1, "sp entry must be loaded"
        finally:
            _dg._MULTIPOINT_PROBES = original_probes

    _clear_all_cal_caches()


# ---------------------------------------------------------------------------
# 6. config_hash mismatch (soft) → warning, data still loaded
# ---------------------------------------------------------------------------

def test_config_hash_mismatch_warns_but_loads():
    _clear_all_cal_caches()
    mem_key_sp = (_N, _K, _DT, False, _N_CAL)
    _CALIBRATION_CACHE[mem_key_sp] = 1.1

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cal.pkl"
        save_calibration_caches_to_disk(path, _N_CAL)

        with open(path, "rb") as f:
            payload = pickle.load(f)
        payload["metadata"]["config_hash"] = "deadbeef"
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        _clear_all_cal_caches()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_calibration_caches_from_disk(path, _N_CAL)
        assert result is True, "Soft config_hash mismatch must still return True"
        assert any(
            "config_hash" in str(warning.message) or "hash mismatch" in str(warning.message)
            for warning in w
        ), "Expected a warning about config_hash mismatch"
        assert _CALIBRATION_CACHE.get(mem_key_sp) == 1.1, "Data must still be loaded"

    _clear_all_cal_caches()


# ---------------------------------------------------------------------------
# 7. Unrecognized format (no multipoint_probes) → False + warning
# ---------------------------------------------------------------------------

def test_unrecognized_format_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old_cal.pkl"
        payload = {
            "metadata": {"n_cal": _N_CAL, "config_hash": compute_config_hash()},
            "caches": {"mp": {}, "sp": {}},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_calibration_caches_from_disk(path, _N_CAL)
        assert result is False, "Old-format file must return False"
        assert any(
            "unrecognized format" in str(warning.message).lower() for warning in w
        ), "Expected a warning about unrecognized format"


# ---------------------------------------------------------------------------
# 8. Save-condition fix — new entries after successful load are persisted
# ---------------------------------------------------------------------------

def test_save_condition_new_entries_persisted_after_load():
    """
    Load succeeds, warm adds new entries, save runs unconditionally:
    verify that reload yields both original and new entries.
    """
    _clear_all_cal_caches()

    mem_key_orig = (_N, _K, _DT, False, _N_CAL)
    _CALIBRATION_CACHE[mem_key_orig] = 1.1

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"calibration_ncal{_N_CAL}.pkl"
        save_calibration_caches_to_disk(path, _N_CAL)
        _clear_all_cal_caches()

        # Simulate load (run_all_scenarios disk-load step)
        assert load_calibration_caches_from_disk(path, _N_CAL) is True
        assert _CALIBRATION_CACHE.get(mem_key_orig) == 1.1

        # Simulate warm adding a new entry
        mem_key_new = (_N, 5, "even", False, _N_CAL)
        _CALIBRATION_CACHE[mem_key_new] = 0.9

        # Save is not gated on _cal_loaded — always runs when _save is True
        save_calibration_caches_to_disk(path, _N_CAL)
        _clear_all_cal_caches()

        # Reload: both must survive
        assert load_calibration_caches_from_disk(path, _N_CAL) is True
        assert _CALIBRATION_CACHE.get(mem_key_orig) == 1.1, "Original entry must survive"
        assert _CALIBRATION_CACHE.get(mem_key_new) == 0.9, "New entry must be persisted"

    _clear_all_cal_caches()


# ---------------------------------------------------------------------------
# 9. cleanup_stale_calibration_entries
# ---------------------------------------------------------------------------

def test_cleanup_stale_entry_removed():
    stale_counts = (25, 25, 25, 5)
    assert stale_counts != _COUNTS, "Test setup: stale_counts must differ from FREQ_DICT"
    stale_disk_key = (_N, _K, _DT, False, stale_counts)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"calibration_ncal{_N_CAL}.pkl"
        _make_file_with_raw_entry(path, stale_disk_key, "sp", 1.0)

        result = cleanup_stale_calibration_entries(path, _N_CAL, dry_run=False)
        assert len(result["stale"]) == 1
        assert result["kept"] == 0
        assert result["total"] == 1

        with open(path, "rb") as f:
            payload = pickle.load(f)
        assert stale_disk_key not in payload["caches"]["sp"], "Stale entry must be removed"


def test_cleanup_fresh_entry_kept():
    fresh_disk_key = (_N, _K, _DT, False, _COUNTS)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"calibration_ncal{_N_CAL}.pkl"
        _make_file_with_raw_entry(path, fresh_disk_key, "sp", 1.0)

        result = cleanup_stale_calibration_entries(path, _N_CAL, dry_run=False)
        assert len(result["stale"]) == 0
        assert result["kept"] == 1


def test_cleanup_all_distinct_never_stale():
    all_dist_key = (_N, _N, None, True)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"calibration_ncal{_N_CAL}.pkl"
        _make_file_with_raw_entry(path, all_dist_key, "sp", 1.0)

        result = cleanup_stale_calibration_entries(path, _N_CAL, dry_run=False)
        assert len(result["stale"]) == 0
        assert result["kept"] == 1


def test_cleanup_custom_never_stale():
    custom_key = (_N, _K, "custom", False, _CUSTOM_COUNTS)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"calibration_ncal{_N_CAL}.pkl"
        _make_file_with_raw_entry(path, custom_key, "sp", 1.0)

        result = cleanup_stale_calibration_entries(path, _N_CAL, dry_run=False)
        assert len(result["stale"]) == 0


def test_cleanup_dry_run_does_not_modify():
    stale_counts = (25, 25, 25, 5)
    stale_disk_key = (_N, _K, _DT, False, stale_counts)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"calibration_ncal{_N_CAL}.pkl"
        _make_file_with_raw_entry(path, stale_disk_key, "sp", 1.0)

        result = cleanup_stale_calibration_entries(path, _N_CAL, dry_run=True)
        assert len(result["stale"]) == 1

        with open(path, "rb") as f:
            payload = pickle.load(f)
        assert stale_disk_key in payload["caches"]["sp"], "Dry-run must not modify file"


def test_cleanup_ncal_mismatch_skipped():
    fresh_disk_key = (_N, _K, _DT, False, _COUNTS)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"calibration_ncal{_N_CAL}.pkl"
        _make_file_with_raw_entry(path, fresh_disk_key, "sp", 1.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cleanup_stale_calibration_entries(path, 99999, dry_run=False)
        assert result["total"] == 0
        assert any("n_cal" in str(warning.message) for warning in w)


def test_cleanup_missing_file_returns_empty():
    result = cleanup_stale_calibration_entries("/nonexistent/cal.pkl", _N_CAL)
    assert result == {"stale": [], "kept": 0, "total": 0}


# ---------------------------------------------------------------------------
# 10. Load-before-warm integration
# ---------------------------------------------------------------------------

def test_load_before_warm_merges_old_and_new():
    """
    Simulate a second precompute run that merges existing + new entries:
      First run:  save scenario A.
      Second run: load file → inject scenario B → save → both A+B survive.
    """
    _clear_all_cal_caches()

    mem_key_a = (_N, _K, _DT, False, _N_CAL)
    _CALIBRATION_CACHE[mem_key_a] = 1.1

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"calibration_ncal{_N_CAL}.pkl"
        save_calibration_caches_to_disk(path, _N_CAL)
        _clear_all_cal_caches()

        # Second run: load → warm adds B → save
        assert load_calibration_caches_from_disk(path, _N_CAL) is True
        assert _CALIBRATION_CACHE.get(mem_key_a) == 1.1

        mem_key_b = (_N, 5, "even", False, _N_CAL)
        _CALIBRATION_CACHE[mem_key_b] = 0.9

        save_calibration_caches_to_disk(path, _N_CAL)
        _clear_all_cal_caches()

        # Reload: both A and B must be present
        assert load_calibration_caches_from_disk(path, _N_CAL) is True
        assert _CALIBRATION_CACHE.get(mem_key_a) == 1.1, "Scenario A must survive second run"
        assert _CALIBRATION_CACHE.get(mem_key_b) == 0.9, "Scenario B must be present after merge"

    _clear_all_cal_caches()


# ---------------------------------------------------------------------------
# 11. Null cache: soft config_hash, hard n_pre
# ---------------------------------------------------------------------------

_N_PRE = 100  # small value for fast tests

def _make_null_entry(n, all_distinct, x_counts, n_pre):
    """Build one null cache entry and return the in-memory key + value."""
    rng = np.random.default_rng(42)
    sorted_abs = np.sort(np.abs(rng.normal(size=n_pre)))
    key = (n, all_distinct, tuple(int(c) for c in x_counts), n_pre)
    return key, sorted_abs


def test_null_config_hash_mismatch_warns_but_loads():
    """Null cache load with tampered config_hash must warn but still return True."""
    _NULL_CACHE.clear()
    x_counts = list(FREQ_DICT[_N][_K][_DT])
    mem_key, value = _make_null_entry(_N, False, x_counts, _N_PRE)
    _NULL_CACHE[mem_key] = value

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"null_npre{_N_PRE}.pkl"
        save_null_cache_to_disk(path, _N_PRE)

        # Tamper with config_hash.
        with open(path, "rb") as f:
            payload = pickle.load(f)
        payload["metadata"]["config_hash"] = "deadbeef"
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        _NULL_CACHE.clear()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_null_cache_from_disk(path, _N_PRE)
        assert result is True, "Soft config_hash mismatch must still return True"
        assert any(
            "config_hash" in str(warning.message) or "hash mismatch" in str(warning.message)
            for warning in w
        ), "Expected a warning about config_hash mismatch"
        assert mem_key in _NULL_CACHE, "Null entry must be loaded despite hash mismatch"
        assert np.array_equal(_NULL_CACHE[mem_key], value), "Null data must match"

    _NULL_CACHE.clear()


def test_null_npre_mismatch_returns_false():
    """Null cache load with wrong n_pre must return False (hard check)."""
    _NULL_CACHE.clear()
    x_counts = list(FREQ_DICT[_N][_K][_DT])
    mem_key, value = _make_null_entry(_N, False, x_counts, _N_PRE)
    _NULL_CACHE[mem_key] = value

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"null_npre{_N_PRE}.pkl"
        save_null_cache_to_disk(path, _N_PRE)
        _NULL_CACHE.clear()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_null_cache_from_disk(path, 99999)
        assert result is False, "n_pre mismatch must return False"
        assert any("n_pre" in str(warning.message) for warning in w), \
            "Expected a warning about n_pre"

    _NULL_CACHE.clear()


# ---------------------------------------------------------------------------
# 12. cleanup_stale_null_entries — standard_keys provenance
# ---------------------------------------------------------------------------

def _make_null_file(path, disk_cache, n_pre, standard_keys=None):
    """Write a minimal null cache pickle with optional standard_keys metadata."""
    payload = {
        "metadata": {
            "n_pre": n_pre,
            "config_hash": compute_config_hash(),
        },
        "cache": disk_cache,
    }
    if standard_keys is not None:
        payload["metadata"]["standard_keys"] = standard_keys
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def test_null_save_includes_standard_keys():
    """save_null_cache_to_disk must write standard_keys into metadata."""
    _NULL_CACHE.clear()
    x_counts = list(FREQ_DICT[_N][_K][_DT])
    mem_key, value = _make_null_entry(_N, False, x_counts, _N_PRE)
    _NULL_CACHE[mem_key] = value

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"null_npre{_N_PRE}.pkl"
        save_null_cache_to_disk(path, _N_PRE)

        with open(path, "rb") as f:
            payload = pickle.load(f)
        meta = payload["metadata"]
        assert "standard_keys" in meta, "standard_keys must be in metadata"
        sk = meta["standard_keys"]
        assert isinstance(sk, set), "standard_keys must be a set"
        # The entry we saved is a standard grid entry, so it must be in standard_keys
        disk_key = (_N, False, tuple(int(c) for c in x_counts))
        assert disk_key in sk, f"{disk_key} should be in standard_keys"

    _NULL_CACHE.clear()


def test_null_cleanup_stale_standard_entry_removed():
    """A standard entry whose x_counts no longer matches FREQ_DICT is removed."""
    stale_counts = (25, 25, 25, 5)  # n=80, not in any current FREQ_DICT[80]
    standard_counts = tuple(FREQ_DICT[_N][_K][_DT])
    assert stale_counts != standard_counts, "Test setup: counts must differ"

    disk_key_stale = (_N, False, stale_counts)
    disk_key_fresh = (_N, False, standard_counts)
    arr_stale = np.sort(np.abs(np.random.default_rng(0).normal(size=_N_PRE)))
    arr_fresh = np.sort(np.abs(np.random.default_rng(1).normal(size=_N_PRE)))

    # Both the stale and fresh entry were standard at save time
    standard_keys = {disk_key_stale, disk_key_fresh}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"null_npre{_N_PRE}.pkl"
        _make_null_file(path, {disk_key_stale: arr_stale, disk_key_fresh: arr_fresh},
                        _N_PRE, standard_keys=standard_keys)

        result = cleanup_stale_null_entries(path, _N_PRE, dry_run=False)
        assert len(result["stale"]) == 1
        assert disk_key_stale in result["stale"]
        assert result["kept"] == 1
        assert result["total"] == 2

        with open(path, "rb") as f:
            payload = pickle.load(f)
        assert disk_key_stale not in payload["cache"], "Stale standard entry must be removed"
        assert disk_key_fresh in payload["cache"], "Fresh standard entry must be kept"


def test_null_cleanup_custom_entry_protected():
    """A custom-freq_dict entry (not in standard_keys) must survive cleanup."""
    custom_counts = (30, 30, 10, 10)  # not from any standard FREQ_DICT
    stale_counts = (25, 25, 25, 5)    # also not in FREQ_DICT; but WAS standard

    disk_key_custom = (_N, False, custom_counts)
    disk_key_stale = (_N, False, stale_counts)
    arr = np.sort(np.abs(np.random.default_rng(0).normal(size=_N_PRE)))

    # Only the stale entry was standard; custom was not.
    standard_keys = {disk_key_stale}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"null_npre{_N_PRE}.pkl"
        _make_null_file(path, {disk_key_custom: arr.copy(), disk_key_stale: arr.copy()},
                        _N_PRE, standard_keys=standard_keys)

        result = cleanup_stale_null_entries(path, _N_PRE, dry_run=False)
        # Stale standard entry removed, custom entry kept
        assert len(result["stale"]) == 1
        assert disk_key_stale in result["stale"]

        with open(path, "rb") as f:
            payload = pickle.load(f)
        assert disk_key_custom in payload["cache"], "Custom entry must survive cleanup"
        assert disk_key_stale not in payload["cache"], "Stale standard entry must be removed"


def test_null_cleanup_no_standard_keys_falls_back():
    """Files without standard_keys in metadata use legacy behavior (flag all invalid)."""
    custom_counts = (30, 30, 10, 10)  # not from any standard FREQ_DICT
    disk_key = (_N, False, custom_counts)
    arr = np.sort(np.abs(np.random.default_rng(0).normal(size=_N_PRE)))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"null_npre{_N_PRE}.pkl"
        # No standard_keys → legacy file
        _make_null_file(path, {disk_key: arr}, _N_PRE, standard_keys=None)

        result = cleanup_stale_null_entries(path, _N_PRE, dry_run=False)
        # Without standard_keys, everything invalid is flagged (legacy behavior)
        assert len(result["stale"]) == 1
        assert disk_key in result["stale"]


def test_null_cleanup_all_distinct_never_stale():
    """All-distinct entries should not be flagged as stale."""
    all_distinct_counts = tuple([1] * _N)
    disk_key = (_N, True, all_distinct_counts)
    arr = np.sort(np.abs(np.random.default_rng(0).normal(size=_N_PRE)))

    # Mark it as standard (it is)
    standard_keys = {disk_key}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"null_npre{_N_PRE}.pkl"
        _make_null_file(path, {disk_key: arr}, _N_PRE, standard_keys=standard_keys)

        result = cleanup_stale_null_entries(path, _N_PRE, dry_run=False)
        assert len(result["stale"]) == 0
        assert result["kept"] == 1


def test_null_cleanup_dry_run_does_not_modify():
    """Dry-run must report stale entries without modifying the file."""
    stale_counts = (25, 25, 25, 5)
    disk_key = (_N, False, stale_counts)
    arr = np.sort(np.abs(np.random.default_rng(0).normal(size=_N_PRE)))
    standard_keys = {disk_key}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"null_npre{_N_PRE}.pkl"
        _make_null_file(path, {disk_key: arr}, _N_PRE, standard_keys=standard_keys)

        result = cleanup_stale_null_entries(path, _N_PRE, dry_run=True)
        assert len(result["stale"]) == 1

        with open(path, "rb") as f:
            payload = pickle.load(f)
        assert disk_key in payload["cache"], "Dry-run must not modify the file"


def test_null_cleanup_npre_mismatch_skipped():
    """cleanup_stale_null_entries with wrong n_pre must skip the file."""
    disk_key = (_N, False, _COUNTS)
    arr = np.sort(np.abs(np.random.default_rng(0).normal(size=_N_PRE)))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"null_npre{_N_PRE}.pkl"
        _make_null_file(path, {disk_key: arr}, _N_PRE, standard_keys={disk_key})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cleanup_stale_null_entries(path, 99999, dry_run=False)
        assert result["total"] == 0
        assert any("n_pre" in str(warning.message) for warning in w)


def test_null_cleanup_missing_file_returns_empty():
    result = cleanup_stale_null_entries("/nonexistent/null.pkl", _N_PRE)
    assert result == {"stale": [], "kept": 0, "total": 0}


def test_null_cleanup_end_to_end_via_save():
    """End-to-end: save standard + custom entries, then cleanup correctly."""
    _NULL_CACHE.clear()

    # Standard entry (from FREQ_DICT)
    std_counts = list(FREQ_DICT[_N][_K][_DT])
    std_key, std_val = _make_null_entry(_N, False, std_counts, _N_PRE)
    _NULL_CACHE[std_key] = std_val

    # Custom entry (not from any standard grid combo)
    custom_counts = [30, 30, 10, 10]
    custom_key, custom_val = _make_null_entry(_N, False, custom_counts, _N_PRE)
    _NULL_CACHE[custom_key] = custom_val

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"null_npre{_N_PRE}.pkl"
        save_null_cache_to_disk(path, _N_PRE)

        # Verify standard_keys only includes the standard entry
        with open(path, "rb") as f:
            payload = pickle.load(f)
        sk = payload["metadata"]["standard_keys"]
        std_disk_key = (_N, False, tuple(int(c) for c in std_counts))
        custom_disk_key = (_N, False, tuple(int(c) for c in custom_counts))
        assert std_disk_key in sk, "Standard entry must be in standard_keys"
        assert custom_disk_key not in sk, "Custom entry must NOT be in standard_keys"

        # Cleanup: standard entry is still valid, custom is unknown but protected
        result = cleanup_stale_null_entries(path, _N_PRE, dry_run=False)
        assert len(result["stale"]) == 0, "Neither entry should be stale"
        assert result["kept"] == 2

    _NULL_CACHE.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cal_key_to_disk_key_mp_standard()
    test_cal_key_to_disk_key_mp_all_distinct()
    test_cal_key_to_disk_key_mp_custom()
    test_cal_key_to_disk_key_sp_standard()
    test_cal_key_to_disk_key_sp_all_distinct()
    test_cal_key_to_disk_key_sp_custom()
    test_cal_key_to_disk_key_mp_emp_standard()
    test_cal_key_to_disk_key_sp_emp_standard()
    test_cal_key_to_disk_key_counts_always_present_for_non_all_distinct()
    print("1. _cal_key_to_disk_key: OK")

    test_roundtrip_mp_standard()
    test_roundtrip_mp_all_distinct()
    test_roundtrip_mp_custom()
    test_roundtrip_mp_cop()
    test_roundtrip_mp_lin()
    test_roundtrip_mp_emp_standard()
    test_roundtrip_mp_emp_custom()
    test_roundtrip_sp_standard()
    test_roundtrip_sp_all_distinct()
    test_roundtrip_sp_custom()
    test_roundtrip_sp_emp()
    test_roundtrip_sp_lin()
    print("2. _disk_key_to_cal_key round-trips: OK")

    test_save_load_roundtrip()
    test_load_missing_file_returns_false()
    print("3. Save/load round-trip: OK")

    test_ncal_mismatch_returns_false()
    print("4. n_cal mismatch: OK")

    test_multipoint_probes_mismatch_skips_mp_loads_sp()
    print("5. multipoint_probes mismatch: OK")

    test_config_hash_mismatch_warns_but_loads()
    print("6. config_hash mismatch (soft): OK")

    test_unrecognized_format_rejected()
    print("7. Unrecognized format rejection: OK")

    test_save_condition_new_entries_persisted_after_load()
    print("8. Save-condition fix: OK")

    test_cleanup_stale_entry_removed()
    test_cleanup_fresh_entry_kept()
    test_cleanup_all_distinct_never_stale()
    test_cleanup_custom_never_stale()
    test_cleanup_dry_run_does_not_modify()
    test_cleanup_ncal_mismatch_skipped()
    test_cleanup_missing_file_returns_empty()
    print("9. cleanup_stale_calibration_entries: OK")

    test_load_before_warm_merges_old_and_new()
    print("10. Load-before-warm integration: OK")

    test_null_config_hash_mismatch_warns_but_loads()
    test_null_npre_mismatch_returns_false()
    print("11. Null cache soft config_hash / hard n_pre: OK")

    test_null_save_includes_standard_keys()
    test_null_cleanup_stale_standard_entry_removed()
    test_null_cleanup_custom_entry_protected()
    test_null_cleanup_no_standard_keys_falls_back()
    test_null_cleanup_all_distinct_never_stale()
    test_null_cleanup_dry_run_does_not_modify()
    test_null_cleanup_npre_mismatch_skipped()
    test_null_cleanup_missing_file_returns_empty()
    test_null_cleanup_end_to_end_via_save()
    print("12. cleanup_stale_null_entries (standard_keys provenance): OK")

    print("\nAll tests passed.")
