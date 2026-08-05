# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
"""
ADVERSARIAL / audit tests for Phase 1 string storage.

This file is an *independent audit* written to try to break the string-storage
implementation (docs/design/aperiodic-and-text-support.md section 17). It probes
edge cases and failure modes that the acceptance suite (test_string_storage.py)
does not: dictionary corner cases, block-merge / ordering semantics, cross-measure
isolation, on-disk corruption resilience, every read entry point + guard rail, and
interactions with numeric writes / windowing.

Tests marked ``xfail`` document confirmed defects or spec-vs-implementation gaps.
No source under sdk/atriumdb/ is modified by this audit.

Run (SQLite only, no MariaDB):
    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_string_storage_audit.py -q
"""
import json
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
    loc = tempfile.mkdtemp(prefix="atrium_straudit_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    s._loc = loc
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def new_string_measure(sdk, tag="strmeas"):
    m = sdk.insert_measure(measure_tag=tag, freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag=f"dev_{tag}")
    return m, d


def times_for(n, start=1_600_000_000 * SEC, step=SEC):
    return start + np.arange(n, dtype=np.int64) * step


# =========================================================================== #
# A. Dictionary corner cases (MeasureStringDictionary in isolation)
# =========================================================================== #
def test_whitespace_and_newline_only_strings_are_distinct(tmp_path):
    """Strings differing only by leading/trailing whitespace or a newline must
    map to distinct codes (no normalization / stripping)."""
    sd = MeasureStringDictionary.load(tmp_path / "meta", 1)
    vals = ["a", "a ", " a", "a\n", "a\t", "a  ", "", " "]
    codes = sd.encode(vals)
    # every distinct input has its own code
    assert len(set(codes.tolist())) == len(vals)
    assert list(sd.decode(codes)) == vals


def test_json_escape_and_literal_backslash_roundtrip(tmp_path):
    """Backslashes, quotes, and JSON escape sequences survive the JSONL encoding
    verbatim (the file stores JSON, not the raw string)."""
    sd = MeasureStringDictionary.load(tmp_path / "meta", 1)
    vals = [
        r"C:\path\to\file",          # literal backslashes
        "a\\nb",                      # literal backslash-n (NOT a newline)
        "a\nb",                       # actual newline
        'he said "hi"',              # embedded quotes
        "\u0000null-byte",           # embedded NUL
        "tab\tafter",
        "unicode-\\u0041-literal",   # literal \u0041 text, not 'A'
        "🙂",
    ]
    codes = sd.encode(vals)
    assert list(sd.decode(codes)) == vals
    # a fresh load from disk reproduces them exactly
    sd2 = MeasureStringDictionary.load(tmp_path / "meta", 1)
    assert sd2._strings == vals


def test_bytes_and_str_of_same_content_collapse(tmp_path):
    """b'x' and 'x' decode to the same string, hence the same code."""
    sd = MeasureStringDictionary.load(tmp_path / "meta", 1)
    codes = sd.encode(np.array(["hello", b"hello", "world"], dtype=object))
    assert codes.tolist() == [0, 0, 1]


def test_encode_empty_sequence_returns_empty_int64(tmp_path):
    sd = MeasureStringDictionary.load(tmp_path / "meta", 1)
    out = sd.encode([])
    assert out.dtype == np.int64
    assert out.size == 0
    # no file should be created for a zero-append encode
    assert not MeasureStringDictionary.exists(tmp_path / "meta", 1)


def test_encode_rejects_none_and_numbers_in_object_array(tmp_path):
    sd = MeasureStringDictionary.load(tmp_path / "meta", 1)
    with pytest.raises(TypeError):
        sd.encode(np.array(["ok", None, "ok2"], dtype=object))
    with pytest.raises(TypeError):
        sd.encode(np.array(["ok", 3.14], dtype=object))


def test_decode_preserves_shape_and_rejects_negative(tmp_path):
    sd = MeasureStringDictionary.load(tmp_path / "meta", 1)
    sd.encode(["a", "b", "c"])
    # 2-D codes -> 2-D strings, same shape
    codes2d = np.array([[0, 1], [2, 0]], dtype=np.int64)
    out = sd.decode(codes2d)
    assert out.shape == (2, 2)
    assert out.tolist() == [["a", "b"], ["c", "a"]]
    # negative code is out of range
    with pytest.raises(ValueError, match="out of range"):
        sd.decode(np.array([-1], dtype=np.int64))


def test_large_vocabulary_roundtrip(tmp_path):
    """A large vocabulary assigns dense sequential codes 0..N-1 and round-trips."""
    sd = MeasureStringDictionary.load(tmp_path / "meta", 9)
    vocab = [f"string_value_{i:05d}" for i in range(1500)]
    codes = sd.encode(vocab)
    assert codes.tolist() == list(range(1500))
    assert list(sd.decode(codes)) == vocab
    # reload sees all of them
    assert len(MeasureStringDictionary.load(tmp_path / "meta", 9)) == 1500


def test_second_instance_sees_appended_codes(tmp_path):
    """A separately-loaded instance must observe codes appended by another
    instance once it reloads (the file is the source of truth)."""
    a = MeasureStringDictionary.load(tmp_path / "meta", 1)
    b = MeasureStringDictionary.load(tmp_path / "meta", 1)
    a.encode(["x", "y"])
    # b has not reloaded yet
    assert len(b) == 0
    b2 = MeasureStringDictionary.load(tmp_path / "meta", 1)
    assert b2._strings == ["x", "y"]


# =========================================================================== #
# B. On-disk corruption / robustness
# =========================================================================== #
def test_trailing_blank_line_is_ignored(sdk):
    m, d = new_string_measure(sdk, tag="trailblank")
    t = times_for(3)
    sdk.write_time_value_pairs(m, d, t, np.array(["w", "x", "y"], dtype=object))
    p = MeasureStringDictionary.path_for(sdk._meta_dir, m)
    with open(p, "a", encoding="utf-8") as f:
        f.write("\n\n")  # extra blank trailing lines
    sd = MeasureStringDictionary.load(sdk._meta_dir, m)
    assert sd._strings == ["w", "x", "y"]
    _, rv = sdk.get_string_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d)
    assert list(rv) == ["w", "x", "y"]


def test_stray_midfile_blank_line_does_not_shift_codes(sdk):
    """Because encode never writes a blank line and reload skips every blank line,
    a stray blank line injected anywhere leaves the surviving code assignment
    intact -- reads still decode correctly. (Robustness probe.)"""
    m, d = new_string_measure(sdk, tag="midblank")
    t = times_for(4)
    sdk.write_time_value_pairs(m, d, t, np.array(["w", "x", "y", "z"], dtype=object))
    p = MeasureStringDictionary.path_for(sdk._meta_dir, m)
    lines = p.read_text(encoding="utf-8").split("\n")
    # insert a blank line in the middle (after the 2nd entry)
    injected = "\n".join(lines[:2] + [""] + lines[2:])
    p.write_text(injected, encoding="utf-8")
    sd = MeasureStringDictionary.load(sdk._meta_dir, m)
    assert sd._strings == ["w", "x", "y", "z"]
    _, rv = sdk.get_string_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d)
    assert list(rv) == ["w", "x", "y", "z"]


def test_reopen_sdk_midstream_and_continue(sdk):
    """Write, drop the SDK object, reopen the same location, write more (reusing
    old strings + adding new), and read the union back."""
    m, d = new_string_measure(sdk, tag="midstream")
    t1 = times_for(3)
    sdk.write_time_value_pairs(m, d, t1, np.array(["alpha", "beta", "gamma"], dtype=object))
    loc = sdk._loc
    sdk.close()

    sdk2 = AtriumSDK(dataset_location=loc, metadata_connection_type="sqlite")
    try:
        t2 = times_for(3, start=int(t1[-1]) + SEC)
        # "beta" already exists (must keep its code), "delta" is new
        sdk2.write_time_value_pairs(m, d, t2, np.array(["beta", "delta", "alpha"], dtype=object))
        _, rv = sdk2.get_string_data(m, int(t1[0]), int(t2[-1]) + SEC, device_id=d)
        assert list(rv) == ["alpha", "beta", "gamma", "beta", "delta", "alpha"]
        sd = MeasureStringDictionary.load(sdk2._meta_dir, m)
        assert sd._strings == ["alpha", "beta", "gamma", "delta"]
    finally:
        # re-attach to the original fixture SDK so its teardown cleans up
        sdk2.close()


# =========================================================================== #
# C. Merge / ordering semantics
# =========================================================================== #
def test_newest_wins_across_separate_merged_writes(sdk):
    """Two separate sub-block writes at the SAME timestamp merge; under the default
    'overwrite' policy the newest write's value wins."""
    m, d = new_string_measure(sdk, tag="newestwins")
    ts = np.array([100 * SEC], dtype=np.int64)
    sdk.write_time_value_pairs(m, d, ts, np.array(["OLD"], dtype=object))
    sdk.write_time_value_pairs(m, d, ts, np.array(["NEW"], dtype=object))
    _, rv = sdk.get_string_data(m, 100 * SEC, 300 * SEC, device_id=d)
    assert list(rv) == ["NEW"]
    # both strings live in the dict (append-only); the OLD code is simply unused now
    assert MeasureStringDictionary.load(sdk._meta_dir, m)._strings == ["OLD", "NEW"]


def test_duplicate_timestamp_within_single_call_first_wins(sdk):
    """DOCUMENTS ACTUAL BEHAVIOR: for duplicate timestamps inside ONE
    write_time_value_pairs call, np.unique keeps the FIRST array element -- so the
    *earliest listed* value wins, which is the opposite of the cross-write
    'newest-wins' merge policy. Callers who expect newest-wins within a single
    array will be surprised."""
    m, d = new_string_measure(sdk, tag="dupsingle")
    t = np.array([100 * SEC, 100 * SEC, 200 * SEC], dtype=np.int64)
    v = np.array(["FIRST", "SECOND", "third"], dtype=object)
    sdk.write_time_value_pairs(m, d, t, v)
    rt, rv = sdk.get_string_data(m, 100 * SEC, 300 * SEC, device_id=d)
    assert rt.astype(np.int64).tolist() == [100 * SEC, 200 * SEC]
    assert list(rv) == ["FIRST", "third"]


def test_out_of_order_timestamps_sorted_with_values(sdk):
    """Values must stay glued to their timestamps when the write path sorts an
    out-of-order timestamp array."""
    m, d = new_string_measure(sdk, tag="ooo")
    t = np.array([300 * SEC, 100 * SEC, 200 * SEC, 400 * SEC], dtype=np.int64)
    v = np.array(["at300", "at100", "at200", "at400"], dtype=object)
    sdk.write_time_value_pairs(m, d, t, v)
    rt, rv = sdk.get_string_data(m, 100 * SEC, 500 * SEC, device_id=d)
    assert rt.astype(np.int64).tolist() == [100 * SEC, 200 * SEC, 300 * SEC, 400 * SEC]
    assert list(rv) == ["at100", "at200", "at300", "at400"]


def test_code_stability_across_blocks_each_write_adds_vocab(sdk):
    """Several contiguous sub-block writes, each introducing brand-new strings,
    merge into one block. After merge every value must decode correctly, proving
    codes assigned in earlier blocks stay valid once the dict has grown."""
    m, d = new_string_measure(sdk, tag="crossblock")
    t0 = 1_600_000_000 * SEC
    all_vals = []
    idx = 0
    for w in range(6):
        # each write introduces 3 unique-to-this-write strings
        chunk = [f"w{w}_s{j}" for j in range(3)]
        ts = t0 + np.arange(idx, idx + 3, dtype=np.int64) * SEC
        sdk.write_time_value_pairs(m, d, ts, np.array(chunk, dtype=object))
        all_vals.extend(chunk)
        idx += 3

    block_list = sdk.sql_handler.select_blocks(m, t0, t0 + idx * SEC + SEC, d, None)
    assert len(block_list) == 1, f"expected merge into 1 block, got {len(block_list)}"
    _, rv = sdk.get_string_data(m, t0, t0 + idx * SEC + SEC, device_id=d)
    assert list(rv) == all_vals
    # dict has exactly the 18 distinct strings, in first-seen order
    assert MeasureStringDictionary.load(sdk._meta_dir, m)._strings == all_vals


def test_same_string_reused_across_blocks_keeps_one_code(sdk):
    """The same string written in two separate (non-merging) writes must use a
    single code, and read back identically in both time ranges."""
    m, d = new_string_measure(sdk, tag="reuse")
    # first write far in the past
    t1 = times_for(3, start=100 * SEC)
    sdk.write_time_value_pairs(m, d, t1, np.array(["shared", "u1", "u2"], dtype=object))
    # second write far in the future so it does not merge with the first block
    t2 = times_for(3, start=10_000 * SEC)
    sdk.write_time_value_pairs(m, d, t2, np.array(["u3", "shared", "u4"], dtype=object))
    sd = MeasureStringDictionary.load(sdk._meta_dir, m)
    assert sd._strings.count("shared") == 1
    _, rv1 = sdk.get_string_data(m, 100 * SEC, 200 * SEC, device_id=d)
    _, rv2 = sdk.get_string_data(m, 10_000 * SEC, 10_100 * SEC, device_id=d)
    assert rv1[0] == "shared" and rv2[1] == "shared"


# =========================================================================== #
# D. Cross-measure isolation
# =========================================================================== #
def test_two_string_measures_have_independent_dicts(sdk):
    m1, d1 = new_string_measure(sdk, tag="iso1")
    m2, d2 = new_string_measure(sdk, tag="iso2")
    # write different-order overlapping vocab to each
    t = times_for(3)
    sdk.write_time_value_pairs(m1, d1, t, np.array(["shared", "only1", "x"], dtype=object))
    sdk.write_time_value_pairs(m2, d2, t, np.array(["only2", "shared", "y"], dtype=object))

    sd1 = MeasureStringDictionary.load(sdk._meta_dir, m1)
    sd2 = MeasureStringDictionary.load(sdk._meta_dir, m2)
    # "shared" gets code 0 in m1 but code 1 in m2 -- independent numbering
    assert sd1._code_of["shared"] == 0
    assert sd2._code_of["shared"] == 1
    # cross reads are correct despite the code collision on int value 0/1
    _, rv1 = sdk.get_string_data(m1, int(t[0]), int(t[-1]) + SEC, device_id=d1)
    _, rv2 = sdk.get_string_data(m2, int(t[0]), int(t[-1]) + SEC, device_id=d2)
    assert list(rv1) == ["shared", "only1", "x"]
    assert list(rv2) == ["only2", "shared", "y"]
    # dict files are separate
    assert MeasureStringDictionary.path_for(sdk._meta_dir, m1) != \
        MeasureStringDictionary.path_for(sdk._meta_dir, m2)


# =========================================================================== #
# E. Concurrency
# =========================================================================== #
def test_concurrent_encode_two_measures_no_crosstalk(tmp_path):
    """Threads encoding into DIFFERENT measures in parallel must not corrupt each
    other's dictionaries."""
    meta = tmp_path / "meta"
    results = {}

    def worker(measure_id, vocab):
        sd = MeasureStringDictionary.load(meta, measure_id)
        sd.encode(vocab)
        results[measure_id] = sd

    threads = [
        threading.Thread(target=worker, args=(i, [f"m{i}_v{j}" for j in range(100)]))
        for i in range(6)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    for i in range(6):
        sd = MeasureStringDictionary.load(meta, i)
        expected = [f"m{i}_v{j}" for j in range(100)]
        assert sd._strings == expected
        assert len(sd._strings) == len(set(sd._strings))


def test_concurrent_appends_same_measure_no_duplicate_codes(tmp_path):
    """Overlapping vocab appended from many threads to ONE measure yields a
    duplicate-free dictionary in which every string has exactly one code."""
    meta = tmp_path / "meta"
    shared = [f"s{i}" for i in range(60)]

    def worker(prefix):
        sd = MeasureStringDictionary.load(meta, 3)
        sd.encode([f"{prefix}{i}" for i in range(40)] + shared)

    for _ in range(3):
        threads = [threading.Thread(target=worker, args=(p,)) for p in ("a", "b", "c")]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    path = MeasureStringDictionary.path_for(meta, 3)
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip("\n") != ""]
    assert len(lines) == len(set(lines))
    expected = set(shared) | {f"{p}{i}" for p in ("a", "b", "c") for i in range(40)}
    assert set(lines) == expected


# =========================================================================== #
# F. Read entry points / selectors
# =========================================================================== #
def test_read_by_patient_id_and_mrn(sdk):
    m, d = new_string_measure(sdk, tag="ptread")
    t = times_for(4)
    sdk.write_time_value_pairs(m, d, t, np.array(["s1", "s2", "s3", "s4"], dtype=object))
    pid = sdk.insert_patient(mrn="AUDIT_MRN")
    sdk.insert_device_patient_data([(d, pid, int(t[0]), int(t[-1]) + SEC)])

    _, rv_pid = sdk.get_string_data(m, int(t[0]), int(t[-1]) + SEC, patient_id=pid)
    _, rv_mrn = sdk.get_string_data(m, int(t[0]), int(t[-1]) + SEC, mrn="AUDIT_MRN")
    assert list(rv_pid) == ["s1", "s2", "s3", "s4"]
    assert list(rv_mrn) == ["s1", "s2", "s3", "s4"]


def test_read_by_device_tag_and_measure_tag_resolution(sdk):
    m, d = new_string_measure(sdk, tag="tagread")
    t = times_for(3)
    sdk.write_time_value_pairs(m, d, t, np.array(["a", "b", "c"], dtype=object))
    _, rv = sdk.get_string_data(
        measure_tag="tagread", freq=1.0, freq_units="Hz", units="string",
        start_time_n=int(t[0]), end_time_n=int(t[-1]) + SEC, device_tag="dev_tagread")
    assert list(rv) == ["a", "b", "c"]


def test_read_time_units_scaling(sdk):
    m, d = new_string_measure(sdk, tag="tunits")
    t = times_for(3)
    sdk.write_time_value_pairs(m, d, t, np.array(["a", "b", "c"], dtype=object))
    st_s, en_s = int(t[0]) // SEC, int(t[-1]) // SEC + 1
    rt, rv = sdk.get_string_data(m, st_s, en_s, device_id=d, time_units="s")
    assert list(rv) == ["a", "b", "c"]
    assert rt.tolist() == [1_600_000_000.0, 1_600_000_001.0, 1_600_000_002.0]


def test_read_sort_false_and_empty_range(sdk):
    m, d = new_string_measure(sdk, tag="sortempty")
    t = times_for(3)
    sdk.write_time_value_pairs(m, d, t, np.array(["a", "b", "c"], dtype=object))
    _, rv = sdk.get_string_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d, sort=False)
    assert set(rv) == {"a", "b", "c"}
    # empty range returns empty object array, not an error
    rt, rv2 = sdk.get_string_data(m, int(t[-1]) + 100 * SEC, int(t[-1]) + 200 * SEC, device_id=d)
    assert rv2.dtype == object and len(rv2) == 0 and len(rt) == 0


# =========================================================================== #
# G. Guard rails (every entry point) + documented raw-code behavior
# =========================================================================== #
def test_guard_analog_default_raises(sdk):
    m, d = new_string_measure(sdk, tag="g_analog")
    t = times_for(3)
    sdk.write_time_value_pairs(m, d, t, np.array(["a", "b", "c"], dtype=object))
    with pytest.raises(ValueError, match="string measure"):
        sdk.get_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d)


def test_guard_nan_filled_bool_raises(sdk):
    m, d = new_string_measure(sdk, tag="g_nan")
    t = times_for(3)
    sdk.write_time_value_pairs(m, d, t, np.array(["a", "b", "c"], dtype=object))
    with pytest.raises(ValueError, match="string measure"):
        sdk.get_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d,
                     analog=False, return_nan_filled=True)


def test_guard_nan_filled_ndarray_raises_like_windowing(sdk):
    """The windowing iterator calls get_data with return_nan_filled=<ndarray>
    (light_mapped_iterator). That exact call must trip the guard so a string
    measure cannot be silently rasterized into a float NaN buffer."""
    m, d = new_string_measure(sdk, tag="g_nanarr")
    t = times_for(3)
    sdk.write_time_value_pairs(m, d, t, np.array(["a", "b", "c"], dtype=object))
    out = np.full(3, np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="string measure"):
        sdk.get_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d,
                     analog=False, return_nan_filled=out)


def test_get_data_analog_false_returns_raw_codes(sdk):
    """DOCUMENTED behavior: get_data(analog=False) on a string measure returns the
    raw int64 dictionary codes (this is exactly how get_string_data reads)."""
    m, d = new_string_measure(sdk, tag="rawcodes")
    t = times_for(3)
    sdk.write_time_value_pairs(m, d, t, np.array(["p", "q", "p"], dtype=object))
    _, _, codes = sdk.get_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d, analog=False)
    assert codes.dtype == np.int64
    assert codes.tolist() == [0, 1, 0]


def test_guard_does_not_fire_for_numeric_measure(sdk):
    """A numeric measure (no dict file) must read normally with analog defaults."""
    m = sdk.insert_measure(measure_tag="num_ok", freq=1.0, freq_units="Hz", units="bpm")
    d = sdk.insert_device(device_tag="dev_num_ok")
    t = times_for(10)
    v = (np.arange(10) * 2).astype(np.int64)
    sdk.write_time_value_pairs(m, d, t, v)
    _, _, rv = sdk.get_data(m, int(t[0]), int(t[-1]) + SEC, device_id=d)  # analog=True default
    assert np.array_equal(rv.astype(np.int64), v)
    assert not MeasureStringDictionary.exists(sdk._meta_dir, m)


# =========================================================================== #
# H. Interaction with existing features
# =========================================================================== #
def test_get_interval_array_on_string_measure(sdk):
    m, d = new_string_measure(sdk, tag="ivl")
    t = times_for(5)
    sdk.write_time_value_pairs(m, d, t, np.array(["e1", "e2", "e3", "e4", "e5"], dtype=object))
    iv = sdk.get_interval_array(m, device_id=d)
    assert iv.shape[0] >= 1
    assert int(iv[0][0]) == int(t[0])
    assert int(iv[-1][1]) >= int(t[-1])


# --------------------------------------------------------------------------- #
# BUGS / spec gaps (xfail): mixing numeric and string data on ONE measure.
# Spec section 17/13 treats a measure as either string-typed (has a dict file)
# or numeric; the design says a measure must not mix the two. The implementation
# does NOT enforce this on write, and the result is silently unreadable data.
# --------------------------------------------------------------------------- #
def test_numeric_then_string_mixing_now_rejected_and_numeric_stays_readable(sdk):
    """Phase 2 fix: the mixed write is now REJECTED on write (§19.3), so the
    earlier numeric data is never corrupted and stays fully readable.

    (Formerly ``test_numeric_then_string_corrupts_readability_actual_behavior``,
    which documented the P1 defect where the mixed write was silently accepted.)"""
    m = sdk.insert_measure(measure_tag="mix_ns", freq=1.0, freq_units="Hz", units="x")
    d = sdk.insert_device(device_tag="dev_mix_ns")
    t1 = times_for(4)
    sdk.write_time_value_pairs(m, d, t1, (np.arange(4) * 3).astype(np.int64))  # values 0,3,6,9
    t2 = times_for(3, start=int(t1[-1]) + SEC)
    # The conflicting string write is now rejected instead of accepted.
    with pytest.raises(ValueError, match="numeric"):
        sdk.write_time_value_pairs(m, d, t2, np.array(["a", "b", "c"], dtype=object))

    # The original numeric data is intact and still readable via get_data.
    _, _, vals = sdk.get_data(m, int(t1[0]), int(t1[-1]) + SEC, device_id=d)
    assert list(np.asarray(vals).astype(np.int64)) == [0, 3, 6, 9]


def test_numeric_then_string_should_be_rejected(sdk):
    m = sdk.insert_measure(measure_tag="mix_ns2", freq=1.0, freq_units="Hz", units="x")
    d = sdk.insert_device(device_tag="dev_mix_ns2")
    t1 = times_for(4)
    sdk.write_time_value_pairs(m, d, t1, (np.arange(4) * 3).astype(np.int64))
    t2 = times_for(3, start=int(t1[-1]) + SEC)
    with pytest.raises(Exception):
        sdk.write_time_value_pairs(m, d, t2, np.array(["a", "b", "c"], dtype=object))


def test_string_then_numeric_should_be_rejected(sdk):
    m = sdk.insert_measure(measure_tag="mix_sn", freq=1.0, freq_units="Hz", units="string")
    d = sdk.insert_device(device_tag="dev_mix_sn")
    t1 = times_for(3)
    sdk.write_time_value_pairs(m, d, t1, np.array(["a", "b", "c"], dtype=object))
    t2 = times_for(3, start=int(t1[-1]) + SEC)
    with pytest.raises(Exception):
        sdk.write_time_value_pairs(m, d, t2, (np.arange(3) + 10).astype(np.int64))
