# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
String storage tests. String values are stored as int64 dictionary codes reusing the ordinary
int64 write/read path with no C or block-format changes.

These run in SQLite mode only (no MariaDB/Docker needed). They exercise:

1. round-trip incl. unicode / empty / >10KB / embedded newlines-commas-quotes / duplicates
2. dictionary append stability (existing codes stable, only new strings appended, no dupes)
3. persistence across a fresh AtriumSDK on the same location
4. sub-block-size writes that MERGE into one block still round-trip
5. get_interval_array returns coarse presence for a string measure
6. guard-rail errors (analog / return_nan_filled) raise
7. a numeric round-trip works
8. the MeasureStringDictionary class in isolation (encode/decode/append/errors/concurrency)
"""
import json
import os
import shutil
import tempfile
import threading

import numpy as np
import pytest

from atriumdb import AtriumSDK
from atriumdb.string_dictionary import MeasureStringDictionary

SEC = 1_000_000_000


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def sdk():
    loc = tempfile.mkdtemp(prefix="atrium_strstore_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    s._loc = loc  # stash for tests that reopen the same location
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def new_string_measure(sdk, tag="strmeas"):
    # unit is NOT NULL; a "string" sentinel marks intent (there's no schema column for it).
    m = sdk.insert_measure(measure_tag=tag, freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag=f"dev_{tag}")
    return m, d


def times_for(n, start=1_600_000_000 * SEC, step=SEC):
    return start + np.arange(n, dtype=np.int64) * step


# --------------------------------------------------------------------------- #
# 1. Round-trip: unicode / empty / >10KB / newlines-commas-quotes / duplicates
# --------------------------------------------------------------------------- #
def test_roundtrip_varied_strings(sdk):
    m, d = new_string_measure(sdk)
    values = [
        "Anesthesia START",
        "café ☕ – ünïçödé 汉字 😀",       # unicode
        "",                                 # empty string
        "line1\nline2\r\nline3",            # embedded newlines
        'has "quotes", commas, and \t tabs',
        "x" * 20000,                        # > 10 KB
        "Anesthesia START",                 # duplicate of first
        "café ☕ – ünïçödé 汉字 😀",       # duplicate unicode
    ]
    t = times_for(len(values))
    sdk.write_time_value_pairs(m, d, t, np.array(values, dtype=object))

    r_times, r_values = sdk.get_string_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d)
    assert np.array_equal(r_times.astype(np.int64), t)
    assert list(r_values) == values
    # duplicates collapse in the vocabulary
    assert len(MeasureStringDictionary.load(sdk._meta_dir, m)) == 6


def test_roundtrip_list_input(sdk):
    """A plain python list[str] (not an ndarray) must be accepted."""
    m, d = new_string_measure(sdk, tag="listinput")
    values = ["alpha", "beta", "gamma", "alpha"]
    t = times_for(len(values))
    sdk.write_time_value_pairs(m, d, list(t), values)  # both list inputs
    _, r_values = sdk.get_string_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d)
    assert list(r_values) == values


def test_roundtrip_via_write_data(sdk):
    """The advanced write_data entry point also accepts string arrays."""
    m, d = new_string_measure(sdk, tag="writedata")
    values = np.array(["one", "two", "three", "two"], dtype=object)
    t = times_for(len(values))
    sdk.write_data(m, d, t, values, freq_nhz=SEC, time_0=int(t[0]), raw_time_type=1)
    _, r_values = sdk.get_string_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d)
    assert list(r_values) == list(values)


# --------------------------------------------------------------------------- #
# 2. Dictionary append / stability
# --------------------------------------------------------------------------- #
def test_dict_append_stability(sdk):
    m, d = new_string_measure(sdk, tag="append")
    first = ["red", "green", "blue"]
    t1 = times_for(len(first))
    sdk.write_time_value_pairs(m, d, t1, np.array(first, dtype=object))

    dict_path = MeasureStringDictionary.path_for(sdk._meta_dir, m)
    codes_before = {s: MeasureStringDictionary.load(sdk._meta_dir, m)._code_of[s] for s in first}

    # Second write: mix of old + new strings.
    second = ["green", "purple", "red", "orange"]
    t2 = times_for(len(second), start=int(t1[-1]) + SEC)
    sdk.write_time_value_pairs(m, d, t2, np.array(second, dtype=object))

    sd = MeasureStringDictionary.load(sdk._meta_dir, m)
    # Existing codes are unchanged.
    for s, c in codes_before.items():
        assert sd._code_of[s] == c
    # Only the two genuinely new strings were appended, in insertion order.
    assert sd._strings == ["red", "green", "blue", "purple", "orange"]

    # No duplicate lines in the .jsonl.
    with open(dict_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip("\n") != ""]
    assert lines == ["red", "green", "blue", "purple", "orange"]
    assert len(lines) == len(set(lines))

    # And the second write still round-trips.
    _, r_values = sdk.get_string_data(m, int(t2[0]), int(t2[-1]) + SEC, device_id=d)
    assert list(r_values) == second


# --------------------------------------------------------------------------- #
# 3. Persistence across a fresh AtriumSDK
# --------------------------------------------------------------------------- #
def test_persistence_new_sdk_object(sdk):
    m, d = new_string_measure(sdk, tag="persist")
    values = ["state_a", "state_b", "state_c", "state_a"]
    t = times_for(len(values))
    sdk.write_time_value_pairs(m, d, t, np.array(values, dtype=object))
    loc = sdk._loc
    sdk.close()

    sdk2 = AtriumSDK(dataset_location=loc, metadata_connection_type="sqlite")
    try:
        _, r_values = sdk2.get_string_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d)
        assert list(r_values) == values
        # codes are identical to what was written
        sd = MeasureStringDictionary.load(sdk2._meta_dir, m)
        assert sd._strings == ["state_a", "state_b", "state_c"]
    finally:
        sdk2.close()


# --------------------------------------------------------------------------- #
# 4. Merge: several sub-block-size writes merge into one block and round-trip
# --------------------------------------------------------------------------- #
def test_submlock_writes_merge_and_roundtrip(sdk):
    m, d = new_string_measure(sdk, tag="merge")
    vocab = ["alpha", "bravo", "charlie", "delta", "echo"]
    all_values = []
    t0 = 1_600_000_000 * SEC
    # 6 small writes, each far below block_size, contiguous in time so they merge.
    idx = 0
    for w in range(6):
        chunk = [vocab[(idx + j) % len(vocab)] for j in range(5)]
        ts = t0 + np.arange(idx, idx + 5, dtype=np.int64) * SEC
        sdk.write_time_value_pairs(m, d, ts, np.array(chunk, dtype=object))
        all_values.extend(chunk)
        idx += 5

    # They should have merged into a single block.
    block_list = sdk.sql_handler.select_blocks(m, t0, t0 + idx * SEC + SEC, d, None)
    assert len(block_list) == 1, f"expected merge into 1 block, got {len(block_list)}"

    _, r_values = sdk.get_string_data(m, t0, t0 + idx * SEC + SEC, device_id=d)
    assert list(r_values) == all_values


# --------------------------------------------------------------------------- #
# 5. Interval index: coarse presence for a string measure
# --------------------------------------------------------------------------- #
def test_interval_array_string_measure(sdk):
    m, d = new_string_measure(sdk, tag="intervals")
    values = ["e1", "e2", "e3", "e4", "e5"]
    t = times_for(len(values))
    sdk.write_time_value_pairs(m, d, t, np.array(values, dtype=object))
    iv = sdk.get_interval_array(m, device_id=d)
    assert iv.shape[0] >= 1
    # Coarse presence spans the written data.
    assert int(iv[0][0]) == int(t[0])
    assert int(iv[-1][1]) >= int(t[-1])


# --------------------------------------------------------------------------- #
# 6. Guard rails
# --------------------------------------------------------------------------- #
def test_get_data_decodes_string_measure_with_default_analog(sdk):
    m, d = new_string_measure(sdk, tag="guard_analog")
    t = times_for(3)
    values = ["a", "b", "c"]
    sdk.write_time_value_pairs(m, d, t, np.array(values, dtype=object))

    headers, read_times, read_values = sdk.get_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d)

    assert headers
    assert np.array_equal(read_times.astype(np.int64), t)
    assert list(read_values) == values


def test_guardrail_nan_filled_raises(sdk):
    m, d = new_string_measure(sdk, tag="guard_nan")
    t = times_for(3)
    sdk.write_time_value_pairs(m, d, t, np.array(["a", "b", "c"], dtype=object))
    with pytest.raises(ValueError, match="string measure"):
        sdk.get_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d, analog=False, return_nan_filled=True)


def test_guardrail_explicit_numeric_raw_value_type_raises(sdk):
    m, d = new_string_measure(sdk, tag="guard_rvt")
    t = times_for(3)
    with pytest.raises(ValueError, match="raw_value_type"):
        sdk.write_data(m, d, t, np.array(["a", "b", "c"], dtype=object),
                       freq_nhz=SEC, time_0=int(t[0]), raw_time_type=1,
                       raw_value_type=3)  # V_TYPE_DOUBLE


# --------------------------------------------------------------------------- #
# 7. An ordinary numeric round-trip works
# --------------------------------------------------------------------------- #
def test_numeric_roundtrip_unchanged(sdk):
    m = sdk.insert_measure(measure_tag="hr_numeric", freq=1.0, freq_units="Hz", units="bpm")
    d = sdk.insert_device(device_tag="dev_numeric")
    t = times_for(50)
    v = (np.arange(50) * 3).astype(np.int64)
    sdk.write_time_value_pairs(m, d, t, v)
    # Numeric measure is NOT flagged as string; ordinary get_data works with defaults.
    _, r_times, r_values = sdk.get_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d)
    assert np.array_equal(r_times.astype(np.int64), t)
    assert np.array_equal(r_values.astype(np.int64), v)
    # No string dictionary was created for a numeric measure.
    assert not MeasureStringDictionary.exists(sdk._meta_dir, m)


def test_numeric_float_roundtrip_unchanged(sdk):
    m = sdk.insert_measure(measure_tag="temp_f", freq=1.0, freq_units="Hz", units="C")
    d = sdk.insert_device(device_tag="dev_float")
    t = times_for(30)
    v = np.linspace(36.0, 38.0, 30).astype(np.float64)
    sdk.write_time_value_pairs(m, d, t, v)
    _, r_times, r_values = sdk.get_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d)
    assert np.array_equal(r_times.astype(np.int64), t)
    assert np.allclose(r_values, v)


# --------------------------------------------------------------------------- #
# 8. MeasureStringDictionary in isolation
# --------------------------------------------------------------------------- #
def test_dict_encode_decode_isolation(tmp_path):
    meta = tmp_path / "meta"
    sd = MeasureStringDictionary.load(meta, 7)
    assert len(sd) == 0
    assert not MeasureStringDictionary.exists(meta, 7)

    codes = sd.encode(["a", "b", "a", "c", "b"])
    assert codes.dtype == np.int64
    assert codes.tolist() == [0, 1, 0, 2, 1]
    assert len(sd) == 3
    assert MeasureStringDictionary.exists(meta, 7)

    decoded = sd.decode(codes)
    assert decoded.dtype == object
    assert list(decoded) == ["a", "b", "a", "c", "b"]


def test_dict_encode_non_string_raises(tmp_path):
    sd = MeasureStringDictionary.load(tmp_path / "meta", 1)
    with pytest.raises(TypeError):
        sd.encode(["ok", 5, "also_ok"])


def test_dict_decode_out_of_range_raises(tmp_path):
    sd = MeasureStringDictionary.load(tmp_path / "meta", 1)
    sd.encode(["x", "y"])
    with pytest.raises(ValueError, match="out of range"):
        sd.decode(np.array([0, 5], dtype=np.int64))


def test_dict_bytes_input(tmp_path):
    sd = MeasureStringDictionary.load(tmp_path / "meta", 1)
    codes = sd.encode(np.array([b"hello", b"world", b"hello"], dtype="S5"))
    assert codes.tolist() == [0, 1, 0]
    assert list(sd.decode(codes)) == ["hello", "world", "hello"]


def test_dict_concurrent_appends_consistent(tmp_path):
    """Two threads appending overlapping vocabularies under the lock must produce a
    consistent, duplicate-free dictionary where every string has exactly one code."""
    meta = tmp_path / "meta"
    vocab_a = [f"a{i}" for i in range(50)]
    vocab_b = [f"b{i}" for i in range(50)]
    shared = [f"s{i}" for i in range(50)]

    def worker(extra):
        sd = MeasureStringDictionary.load(meta, 42)
        sd.encode(extra + shared)

    threads = [threading.Thread(target=worker, args=(v,)) for v in (vocab_a, vocab_b)]
    for _ in range(4):  # repeat to increase contention
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        threads = [threading.Thread(target=worker, args=(v,)) for v in (vocab_a, vocab_b)]

    path = MeasureStringDictionary.path_for(meta, 42)
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip("\n") != ""]
    # No duplicates: each string appears exactly once (one code).
    assert len(lines) == len(set(lines))
    # Every expected string is present.
    assert set(lines) == set(vocab_a) | set(vocab_b) | set(shared)
