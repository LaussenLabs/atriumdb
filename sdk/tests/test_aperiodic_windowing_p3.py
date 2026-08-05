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
Phase 3 aperiodic / string rasterization tests (design section 21.6).

Runs in SQLite mode only (no MariaDB/Docker DB needed). Covers per-kind
rasterization into the Window contract with the correct unknown-sentinel
placement, the 1 s default raster period, mixed-rate batching, and the string
int64-code + decode accessor path. The critical numeric-path regression is
guarded by the existing sdk/tests/test_iterator.py (byte-for-byte grid path).
"""
import shutil
import tempfile

import numpy as np
import pytest

from atriumdb import AtriumSDK, DatasetDefinition
from atriumdb.string_dictionary import UNKNOWN_STRING_CODE, UNKNOWN_STRING_VALUE
from atriumdb.windowing.windowing_functions import get_signal_dictionary

SEC = 1_000_000_000
BASE = 1_600_000_000 * SEC


@pytest.fixture
def sdk():
    loc = tempfile.mkdtemp(prefix="atrium_p3_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def _region_def(sdk, tag, device_id, start_ns, end_ns):
    d = DatasetDefinition(
        measures=[tag],
        device_ids={device_id: [{"start": int(start_ns), "end": int(end_ns)}]},
    )
    d.validate(sdk)
    return d


def _one_signal(window):
    # windows in these tests have exactly one measure
    (key,) = list(window.signals.keys())
    return key, window.signals[key]


# --------------------------------------------------------------------------- #
# 1 s default period when a measure has no usable grid period
# --------------------------------------------------------------------------- #
def test_default_period_one_second_for_aperiodic(sdk):
    # freq 4 Hz -> freq-derived period is 250 ms, but an aperiodic measure must
    # grid at the 1 s default, NOT the freq-derived period.
    m = sdk.insert_measure("nibp", freq=4.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    t = BASE + np.array([2, 5, 9], dtype=np.int64) * SEC
    sdk.write_time_value_pairs(m, dev, t, np.array([100.0, 110.0, 120.0]))

    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)
    it = sdk.get_iterator(defn, 10 * SEC, 10 * SEC)
    assert it.render_config[m]["period_ns"] == SEC
    assert it.lowest_period_ns == SEC
    # override wins
    it2 = sdk.get_iterator(defn, 10 * SEC, 10 * SEC, period_overrides={m: 2}, time_units="s")
    assert it2.render_config[m]["period_ns"] == 2 * SEC


# --------------------------------------------------------------------------- #
# sample: carry-forward (default), sparse, aggregate
# --------------------------------------------------------------------------- #
def test_sample_carry_forward_default_with_left_censor(sdk):
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    t = BASE + np.array([2, 5], dtype=np.int64) * SEC
    sdk.write_time_value_pairs(m, dev, t, np.array([100.0, 110.0]))

    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)
    it = sdk.get_iterator(defn, 10 * SEC, 10 * SEC)
    win = next(iter(it))
    _, sig = _one_signal(win)
    vals = sig["values"]
    assert vals.shape == (10,)
    # cells 0,1 before first reading -> unknown sentinel NaN (left-censored)
    assert np.isnan(vals[0]) and np.isnan(vals[1])
    # carry-forward
    assert list(vals[2:5]) == [100.0, 100.0, 100.0]
    assert list(vals[5:10]) == [110.0] * 5
    assert sig["actual_count"] == 8  # cells 2..9 known


def test_sample_sparse(sdk):
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    t = BASE + np.array([2, 5], dtype=np.int64) * SEC
    sdk.write_time_value_pairs(m, dev, t, np.array([100.0, 110.0]))

    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)
    it = sdk.get_iterator(defn, 10 * SEC, 10 * SEC, fill_overrides={m: "sparse"})
    _, sig = _one_signal(next(iter(it)))
    vals = sig["values"]
    known_idx = np.where(~np.isnan(vals))[0]
    assert list(known_idx) == [2, 5]
    assert vals[2] == 100.0 and vals[5] == 110.0


def test_sample_aggregate_mean(sdk):
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    # two readings inside cell 2 -> mean; one in cell 5
    t = BASE + np.array([2 * SEC, 2 * SEC + 100, 5 * SEC], dtype=np.int64)
    sdk.write_time_value_pairs(m, dev, t, np.array([100.0, 200.0, 50.0]))

    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)
    it = sdk.get_iterator(defn, 10 * SEC, 10 * SEC, fill_overrides={m: "aggregate:mean"})
    _, sig = _one_signal(next(iter(it)))
    vals = sig["values"]
    assert vals[2] == 150.0
    assert vals[5] == 50.0
    assert np.isnan(vals[0])


# --------------------------------------------------------------------------- #
# state: carry-forward with left-censoring (numeric)
# --------------------------------------------------------------------------- #
def test_state_left_censored_numeric(sdk):
    m = sdk.insert_measure("mode", freq=1.0, freq_units="Hz", units="code", signal_kind="state")
    dev = sdk.insert_device(device_tag="dev1")
    # first observed transition at cell 3 -> before that is genuinely unknown
    t = BASE + np.array([3, 7], dtype=np.int64) * SEC
    sdk.write_time_value_pairs(m, dev, t, np.array([1.0, 2.0]))

    defn = _region_def(sdk, "mode", dev, BASE, BASE + 10 * SEC)
    it = sdk.get_iterator(defn, 10 * SEC, 10 * SEC)
    _, sig = _one_signal(next(iter(it)))
    vals = sig["values"]
    assert np.all(np.isnan(vals[:3]))         # left-censored, NOT the first value
    assert list(vals[3:7]) == [1.0] * 4
    assert list(vals[7:10]) == [2.0] * 3


# --------------------------------------------------------------------------- #
# event: presence (default) and count
# --------------------------------------------------------------------------- #
def test_event_presence_and_count(sdk):
    m = sdk.insert_measure("alarm", freq=1.0, freq_units="Hz", units="n", signal_kind="event")
    dev = sdk.insert_device(device_tag="dev1")
    # two events land in cell 2, one in cell 6
    t = BASE + np.array([2 * SEC, 2 * SEC + 5, 6 * SEC], dtype=np.int64)
    sdk.write_time_value_pairs(m, dev, t, np.array([1.0, 1.0, 1.0]))

    defn = _region_def(sdk, "alarm", dev, BASE, BASE + 10 * SEC)
    it = sdk.get_iterator(defn, 10 * SEC, 10 * SEC)  # default presence
    _, sig = _one_signal(next(iter(it)))
    vals = sig["values"]
    assert not np.any(np.isnan(vals))          # no unknown sentinel for events
    expected = np.zeros(10); expected[2] = 1; expected[6] = 1
    assert np.array_equal(vals, expected)

    it_c = sdk.get_iterator(defn, 10 * SEC, 10 * SEC, fill_overrides={m: "count"})
    _, sig_c = _one_signal(next(iter(it_c)))
    assert sig_c["values"][2] == 2.0 and sig_c["values"][6] == 1.0


# --------------------------------------------------------------------------- #
# string state: int64 codes in the window + decode accessor + reserved sentinel
# --------------------------------------------------------------------------- #
def test_string_state_codes_and_decode(sdk):
    m = sdk.insert_measure("anes", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="state", value_type="string")
    dev = sdk.insert_device(device_tag="dev1")
    t = BASE + np.array([3, 7], dtype=np.int64) * SEC
    sdk.write_time_value_pairs(m, dev, t, np.array(["START", "STOP"], dtype=object))

    defn = _region_def(sdk, "anes", dev, BASE, BASE + 10 * SEC)
    it = sdk.get_iterator(defn, 10 * SEC, 10 * SEC)
    win = next(iter(it))
    key, sig = _one_signal(win)
    vals = sig["values"]
    # window carries int64 codes, not decoded strings
    assert vals.dtype == np.int64
    # left-censored cells carry the reserved unknown code, never a real code
    assert np.all(vals[:3] == UNKNOWN_STRING_CODE)
    assert np.all(vals[:3] != vals[3])         # sentinel distinct from real code

    # decode accessor (both the Window method and the iterator method)
    decoded = win.decode_string_signal(sdk, key)
    assert list(decoded[:3]) == [UNKNOWN_STRING_VALUE] * 3
    assert list(decoded[3:7]) == ["START"] * 4
    assert list(decoded[7:10]) == ["STOP"] * 3
    decoded2 = it.decode_window_strings(win, key)
    assert np.array_equal(decoded.astype(str), decoded2.astype(str))
    # "<unknown>" is never a genuine vocabulary entry
    assert UNKNOWN_STRING_VALUE not in ["START", "STOP"]


def test_string_sample_carry_forward(sdk):
    m = sdk.insert_measure("txt", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="sample", value_type="string")
    dev = sdk.insert_device(device_tag="dev1")
    t = BASE + np.array([1, 4], dtype=np.int64) * SEC
    sdk.write_time_value_pairs(m, dev, t, np.array(["A", "B"], dtype=object))

    defn = _region_def(sdk, "txt", dev, BASE, BASE + 6 * SEC)
    it = sdk.get_iterator(defn, 6 * SEC, 6 * SEC)
    win = next(iter(it))
    key, sig = _one_signal(win)
    decoded = win.decode_string_signal(sdk, key)
    assert decoded[0] == UNKNOWN_STRING_VALUE
    assert list(decoded[1:4]) == ["A"] * 3
    assert list(decoded[4:6]) == ["B"] * 2


# --------------------------------------------------------------------------- #
# mixed-rate window: a waveform + an aperiodic measure in one definition
# --------------------------------------------------------------------------- #
def test_mixed_rate_window_batches(sdk):
    # waveform at 4 Hz
    wf = sdk.insert_measure("ecg", freq=4.0, freq_units="Hz", units="mV")  # default waveform/numeric
    # aperiodic sample (grids at 1 s)
    sm = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")

    wf_t = BASE + np.arange(40, dtype=np.int64) * (SEC // 4)  # 10 s @ 4 Hz
    sdk.write_time_value_pairs(wf, dev, wf_t, np.arange(40, dtype=np.float64), freq=4.0, freq_units="Hz")
    sm_t = BASE + np.array([2, 5], dtype=np.int64) * SEC
    sdk.write_time_value_pairs(sm, dev, sm_t, np.array([100.0, 110.0]))

    defn = DatasetDefinition(
        measures=["ecg", "nibp"],
        device_ids={dev: [{"start": int(BASE), "end": int(BASE + 10 * SEC)}]},
    )
    defn.validate(sdk)
    it = sdk.get_iterator(defn, 10 * SEC, 10 * SEC)
    # lowest period must come from the waveform (250 ms), not the aperiodic 1 s
    assert it.lowest_period_ns == SEC // 4
    win = next(iter(it))
    sig_by_tag = {k[0]: v for k, v in win.signals.items()}
    assert sig_by_tag["ecg"]["values"].shape == (40,)   # 10 s @ 4 Hz
    assert sig_by_tag["nibp"]["values"].shape == (10,)  # 10 s @ 1 Hz
    # waveform stays a float NaN grid, fully populated here
    assert sig_by_tag["ecg"]["values"].dtype == np.float64
    assert not np.any(np.isnan(sig_by_tag["ecg"]["values"]))
    # aperiodic carried-forward
    assert np.isnan(sig_by_tag["nibp"]["values"][0])
    assert sig_by_tag["nibp"]["values"][2] == 100.0


# --------------------------------------------------------------------------- #
# waveform numeric path stays a plain NaN grid (local regression sanity)
# --------------------------------------------------------------------------- #
def test_waveform_numeric_unchanged(sdk):
    wf = sdk.insert_measure("ecg", freq=1.0, freq_units="Hz", units="mV")
    dev = sdk.insert_device(device_tag="dev1")
    t = BASE + np.array([0, 1, 2, 5], dtype=np.int64) * SEC
    sdk.write_time_value_pairs(wf, dev, t, np.array([1.0, 2.0, 3.0, 6.0]))

    defn = _region_def(sdk, "ecg", dev, BASE, BASE + 10 * SEC)
    it = sdk.get_iterator(defn, 10 * SEC, 10 * SEC)
    assert it.render_config[wf]["fill_rule"] == "grid"
    _, sig = _one_signal(next(iter(it)))
    vals = sig["values"]
    assert vals.dtype == np.float64
    # gaps stay NaN (no carry-forward on a waveform)
    assert np.isnan(vals[3]) and np.isnan(vals[4])
    assert vals[0] == 1.0 and vals[5] == 6.0


# --------------------------------------------------------------------------- #
# CRITICAL: the waveform-numeric grid path is byte-for-byte identical whether or
# not a render_config is supplied (proves the P3 refactor did not touch it, with
# no physionet download needed).
# --------------------------------------------------------------------------- #
def test_waveform_grid_byte_for_byte_identical(sdk):
    wf = sdk.insert_measure("ecg", freq=4.0, freq_units="Hz", units="mV")
    dev = sdk.insert_device(device_tag="dev1")
    # 8 s of 4 Hz data with an internal gap (skip 3..5 s) so NaN-fill matters
    keep = np.array([i for i in range(32) if not (12 <= i < 20)], dtype=np.int64)
    t = BASE + keep * (SEC // 4)
    v = keep.astype(np.float64) + 0.5
    sdk.write_time_value_pairs(wf, dev, t, v, freq=4.0, freq_units="Hz")

    defn = _region_def(sdk, "ecg", dev, BASE, BASE + 8 * SEC)
    it = sdk.get_iterator(defn, 4 * SEC, 2 * SEC)  # overlapping windows

    measures = it.measures
    wcfg = it.render_config  # waveform -> fill_rule 'grid'
    common = dict(sdk=sdk, device_id=dev, query_patient_id=None,
                  window_duration_ns=4 * SEC, window_slide_ns=2 * SEC, measures=measures,
                  batch_start_time=int(BASE), batch_end_time=int(BASE + 8 * SEC),
                  batch_num_windows=3, range_start_time=int(BASE), range_end_time=int(BASE + 8 * SEC))
    legacy = get_signal_dictionary(render_config=None, **common)
    p3 = get_signal_dictionary(render_config=wcfg, **common)

    lt, lv, lc = legacy[wf]
    pt, pv, pc = p3[wf]
    assert lc == pc
    assert np.array_equal(lt, pt)
    # equal_nan so NaN gap cells compare equal position-for-position
    assert np.array_equal(lv, pv, equal_nan=True)


# --------------------------------------------------------------------------- #
# fill-rule validation
# --------------------------------------------------------------------------- #
def test_incompatible_override_raises(sdk):
    m = sdk.insert_measure("alarm", freq=1.0, freq_units="Hz", units="n", signal_kind="event")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([1], dtype=np.int64) * SEC, np.array([1.0]))
    defn = _region_def(sdk, "alarm", dev, BASE, BASE + 5 * SEC)
    with pytest.raises(ValueError):
        sdk.get_iterator(defn, 5 * SEC, 5 * SEC, fill_overrides={m: "carry_forward"})


def test_period_larger_than_window_clear_error(sdk):
    # Bug 3: a nominal period larger than the window duration must raise a clear,
    # measure-named error (not an opaque "slice step cannot be zero").
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([1], dtype=np.int64) * SEC, np.array([1.0]))
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)
    # period 6 s > window 5 s -> zero grid cells per window.
    with pytest.raises(ValueError, match=r"larger than the window duration") as exc:
        sdk.get_iterator(defn, 5 * SEC, 5 * SEC, period_overrides={m: 6 * SEC})
    assert str(m) in str(exc.value)  # names the measure


def test_carry_forward_batch_independent(sdk):
    # Bug 1 regression (writer-side): carry-forward is deterministic regardless of
    # num_windows_prefetch. Reading at 2s carries into a later window's batch.
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2], dtype=np.int64) * SEC, np.array([100.0]))
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 20 * SEC)
    default_batch = [sig["values"] for sig in
                     (_one_signal(w)[1] for w in sdk.get_iterator(defn, 10 * SEC, 10 * SEC))]
    one_per_batch = [sig["values"] for sig in
                     (_one_signal(w)[1] for w in sdk.get_iterator(defn, 10 * SEC, 10 * SEC, num_windows_prefetch=1))]
    assert np.array_equal(default_batch[1], one_per_batch[1])
    assert list(default_batch[1]) == [100.0] * 10  # [10s,20s) carries 100


def test_string_numeric_aggregate_rejected(sdk):
    m = sdk.insert_measure("txt", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="sample", value_type="string")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([1], dtype=np.int64) * SEC,
                               np.array(["A"], dtype=object))
    defn = _region_def(sdk, "txt", dev, BASE, BASE + 5 * SEC)
    with pytest.raises(ValueError):
        sdk.get_iterator(defn, 5 * SEC, 5 * SEC, fill_overrides={m: "aggregate:mean"})
