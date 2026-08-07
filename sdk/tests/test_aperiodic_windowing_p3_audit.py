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
ADVERSARIAL audit of Phase 3 "rasterize into the Window contract"
(design section 21). Written by an independent auditor: the goal is to *break*
the aperiodic/string windowing path, not to re-confirm the happy path the
writer's own ``test_aperiodic_windowing_p3.py`` already covers.

Runs in SQLite mode only (no MariaDB / physionet needed). Synthetic data.

Layout:
  * ``test_pass_*`` -- probes the auditor ran that the code handles CORRECTLY.
  * ``test_bug_*``  -- confirmed defects, captured as ``xfail(strict=True)`` so
                       they fail loudly (become XPASS) the day they are fixed.
                       Each carries a minimal repro + observed-vs-expected note.

Headline defect (see ``test_bug_carry_forward_*``): carry-forward / left-censoring
is computed over the per-BATCH read grid, and the batch only reads data inside
``[range_start, batch_end]``. A reading that precedes a window's batch is never
read, so a value that IS known (per carry-forward semantics: "most recent prior
reading") is emitted as the unknown sentinel. Because batch boundaries are set by
``num_windows_prefetch`` -- documented as a pure RAM/speed knob -- the SAME window
of the SAME dataset yields different values depending on that knob.
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
    loc = tempfile.mkdtemp(prefix="atrium_p3_audit_")
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


def _one(window):
    (key,) = list(window.signals.keys())
    return key, window.signals[key]


def _win_values(iterator):
    return [_one(w)[1]["values"] for w in iterator]


# =========================================================================== #
# BUGS (confirmed) -- xfail(strict=True)
# =========================================================================== #
def test_bug_carry_forward_sample_batch_boundary_inconsistent(sdk):
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    # single reading at t=2s; nothing afterwards.
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2], dtype=np.int64) * SEC, np.array([100.0]))
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 20 * SEC)

    default_batch = _win_values(sdk.get_iterator(defn, 10 * SEC, 10 * SEC))
    one_per_batch = _win_values(sdk.get_iterator(defn, 10 * SEC, 10 * SEC, num_windows_prefetch=1))

    # Window 1 = [10s,20s): most recent prior reading is 100 -> carry-forward = 100.
    # The two configs MUST agree; they do not (default=100, prefetch1=NaN).
    np.testing.assert_array_equal(default_batch[1], one_per_batch[1])


def test_bug_state_carry_forward_batch_boundary_inconsistent(sdk):
    m = sdk.insert_measure("mode", freq=1.0, freq_units="Hz", units="code", signal_kind="state")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2], dtype=np.int64) * SEC, np.array([7.0]))
    defn = _region_def(sdk, "mode", dev, BASE, BASE + 20 * SEC)

    default_batch = _win_values(sdk.get_iterator(defn, 10 * SEC, 10 * SEC))
    one_per_batch = _win_values(sdk.get_iterator(defn, 10 * SEC, 10 * SEC, num_windows_prefetch=1))
    # default -> [7]*10 (state in effect), prefetch=1 -> [nan]*10.
    np.testing.assert_array_equal(default_batch[1], one_per_batch[1])


def test_bug_string_state_sentinel_leaks_across_batch_boundary(sdk):
    m = sdk.insert_measure("anes", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="state", value_type="string")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2], dtype=np.int64) * SEC,
                               np.array(["ON"], dtype=object))
    defn = _region_def(sdk, "anes", dev, BASE, BASE + 20 * SEC)

    default_batch = _win_values(sdk.get_iterator(defn, 10 * SEC, 10 * SEC))
    one_per_batch = _win_values(sdk.get_iterator(defn, 10 * SEC, 10 * SEC, num_windows_prefetch=1))
    # default win1 -> [0]*10 (code for "ON"); prefetch1 win1 -> [-1]*10 (sentinel).
    assert UNKNOWN_STRING_CODE not in default_batch[1]
    np.testing.assert_array_equal(default_batch[1], one_per_batch[1])


# =========================================================================== #
# DOCUMENTED-BEHAVIOR probes (these PASS; they pin down current semantics and
# the acknowledged sentinel/censoring limitations so they can't silently drift)
# =========================================================================== #
def test_pass_carry_forward_sees_reading_before_range_start(sdk):
    """A reading BEFORE the definition's range start IS the value in effect at
    the range start, and carry-forward now seeds from it.

    This probe used to pin the opposite ("value known-before-range is dropped"),
    on the reasoning that a hard floor at the range start was defensible as a
    range restriction. It is not: it made the same wall-clock window render
    differently depending on where the cohort's region happened to begin, so the
    flagship "N minutes either side of every event" recipe returned NaN for the
    whole pre-anchor half of every window on any slow measure. Design 21.3 says a
    `sample` cell holds "the most recent prior reading"; the unknown sentinel is
    for genuinely-unknown cells, and this cell is not one. The lookback is bounded
    (windowing_functions.CARRY_FORWARD_LOOKBACK_NS) and is a pure function of the
    definition's range start, so batch independence is untouched -- the tests
    above still pin that."""
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2], dtype=np.int64) * SEC, np.array([100.0]))
    # range starts at 5s, reading is at 2s (before it)
    defn = _region_def(sdk, "nibp", dev, BASE + 5 * SEC, BASE + 15 * SEC)
    _, sig = _one(next(iter(sdk.get_iterator(defn, 10 * SEC, 10 * SEC))))
    assert list(sig["values"]) == [100.0] * 10


def test_pass_real_nan_reading_conflated_with_unknown(sdk):
    """Design 21.2 #2(a): a genuine NaN reading is indistinguishable from an
    unknown/censored cell on a float channel. Confirm the conflation is EXACTLY
    that (value is NaN, and actual_count treats it as not-known) and nothing
    worse leaks."""
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    # real reading NaN at cell 2, real 110 at cell 5
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2, 5], dtype=np.int64) * SEC,
                               np.array([np.nan, 110.0]))
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)
    _, sig = _one(next(iter(sdk.get_iterator(defn, 10 * SEC, 10 * SEC, fill_overrides={m: "sparse"}))))
    vals = sig["values"]
    assert np.isnan(vals[2])                       # genuine NaN reading
    # the real-NaN cell is counted as NOT known -- the documented conflation
    assert sig["actual_count"] == 1                # only cell 5 counts as known


def test_pass_state_right_censoring_is_noop(sdk):
    """Design note: right-censoring is a no-op in P3. A state that never closes
    is carried forward to the end of the batch rather than marked unknown. Pinned
    so a future P4 change is a deliberate, visible edit."""
    m = sdk.insert_measure("mode", freq=1.0, freq_units="Hz", units="code", signal_kind="state")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([3], dtype=np.int64) * SEC, np.array([1.0]))
    defn = _region_def(sdk, "mode", dev, BASE, BASE + 10 * SEC)
    _, sig = _one(next(iter(sdk.get_iterator(defn, 10 * SEC, 10 * SEC))))
    assert list(sig["values"][3:]) == [1.0] * 7    # carried forward, not censored


# =========================================================================== #
# UNKNOWN-SENTINEL INTEGRITY (these PASS)
# =========================================================================== #
def test_pass_string_sentinel_never_a_real_code(sdk):
    """Real dictionary codes are line indices >= 0; the sentinel is -1. Even with
    many distinct strings, -1 is never assigned, and decode maps it to
    '<unknown>' which is not a vocabulary member."""
    m = sdk.insert_measure("txt", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="state", value_type="string")
    dev = sdk.insert_device(device_tag="dev1")
    strings = [f"S{i}" for i in range(50)]
    t = BASE + np.arange(1, 51, dtype=np.int64) * SEC
    sdk.write_time_value_pairs(m, dev, t, np.array(strings, dtype=object))
    defn = _region_def(sdk, "txt", dev, BASE, BASE + 51 * SEC)
    win = next(iter(sdk.get_iterator(defn, 51 * SEC, 51 * SEC)))
    key, sig = _one(win)
    real_codes = sig["values"][sig["values"] != UNKNOWN_STRING_CODE]
    assert real_codes.min() >= 0                   # no real code collides with -1
    decoded = win.decode_string_signal(sdk, key)
    assert UNKNOWN_STRING_VALUE not in set(strings)
    # cell 0 is left-censored (first reading at t=1s) -> sentinel -> "<unknown>"
    assert sig["values"][0] == UNKNOWN_STRING_CODE
    assert decoded[0] == UNKNOWN_STRING_VALUE


def test_pass_repeated_strings_have_stable_codes(sdk):
    m = sdk.insert_measure("txt", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="sample", value_type="string")
    dev = sdk.insert_device(device_tag="dev1")
    t = BASE + np.array([1, 2, 3, 4], dtype=np.int64) * SEC
    sdk.write_time_value_pairs(m, dev, t, np.array(["A", "B", "A", "B"], dtype=object))
    defn = _region_def(sdk, "txt", dev, BASE, BASE + 5 * SEC)
    _, sig = _one(next(iter(sdk.get_iterator(defn, 5 * SEC, 5 * SEC, fill_overrides={m: "sparse"}))))
    vals = sig["values"]
    assert vals[1] == vals[3]                       # both "A"? no -> A at 1,3
    # readings: A@1,B@2,A@3,B@4 ; sparse -> cells 1..4
    assert vals[1] == vals[3]                        # code("A") stable
    assert vals[2] == vals[4]                        # code("B") stable
    assert vals[1] != vals[2]                        # A != B


def test_pass_event_presence_and_count_never_sentinel(sdk):
    m = sdk.insert_measure("alarm", freq=1.0, freq_units="Hz", units="n", signal_kind="event")
    dev = sdk.insert_device(device_tag="dev1")
    t = BASE + np.array([2 * SEC, 2 * SEC + 7, 6 * SEC], dtype=np.int64)
    sdk.write_time_value_pairs(m, dev, t, np.array([1.0, 1.0, 1.0]))
    defn = _region_def(sdk, "alarm", dev, BASE, BASE + 10 * SEC)
    _, sig = _one(next(iter(sdk.get_iterator(defn, 10 * SEC, 10 * SEC))))
    assert not np.any(np.isnan(sig["values"]))       # no sentinel for events
    assert sig["actual_count"] == 10                 # every cell "known" for events
    _, sig_c = _one(next(iter(sdk.get_iterator(defn, 10 * SEC, 10 * SEC, fill_overrides={m: "count"}))))
    assert sig_c["values"][2] == 2.0 and sig_c["values"][6] == 1.0
    assert not np.any(np.isnan(sig_c["values"]))


# =========================================================================== #
# AGGREGATE / FILL edge cases (these PASS)
# =========================================================================== #
def test_pass_sparse_is_batch_independent(sdk):
    """Contrast with the carry-forward bug: sparse/aggregate touch only a
    reading's own cell, so they are correctly batch-boundary independent."""
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2, 12], dtype=np.int64) * SEC,
                               np.array([100.0, 200.0]))
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 20 * SEC)
    a = _win_values(sdk.get_iterator(defn, 10 * SEC, 10 * SEC, fill_overrides={m: "sparse"}))
    b = _win_values(sdk.get_iterator(defn, 10 * SEC, 10 * SEC, fill_overrides={m: "sparse"},
                                     num_windows_prefetch=1))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y, )
    assert b[1][2] == 200.0                          # reading at 12s -> window1 cell 2


def test_pass_aggregate_min_max_mean_and_empty_cell(sdk):
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    # cell 2: three readings 10,30,20 ; cell 5: single 99 ; others empty
    t = BASE + np.array([2 * SEC, 2 * SEC + 1, 2 * SEC + 2, 5 * SEC], dtype=np.int64)
    v = np.array([10.0, 30.0, 20.0, 99.0])
    sdk.write_time_value_pairs(m, dev, t, v)
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)
    for rule, expect2 in [("aggregate:min", 10.0), ("aggregate:max", 30.0), ("aggregate:mean", 20.0)]:
        _, sig = _one(next(iter(sdk.get_iterator(defn, 10 * SEC, 10 * SEC, fill_overrides={m: rule}))))
        vals = sig["values"]
        assert vals[2] == expect2, rule
        assert vals[5] == 99.0, rule
        assert np.isnan(vals[0]) and np.isnan(vals[3])  # empty cells stay unknown


def test_pass_string_aggregate_last_keeps_most_recent_in_cell(sdk):
    m = sdk.insert_measure("txt", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="sample", value_type="string")
    dev = sdk.insert_device(device_tag="dev1")
    t = BASE + np.array([2 * SEC, 2 * SEC + 100, 2 * SEC + 200], dtype=np.int64)
    sdk.write_time_value_pairs(m, dev, t, np.array(["A", "B", "C"], dtype=object))
    defn = _region_def(sdk, "txt", dev, BASE, BASE + 5 * SEC)
    win = next(iter(sdk.get_iterator(defn, 5 * SEC, 5 * SEC, fill_overrides={m: "aggregate:last"})))
    key, sig = _one(win)
    assert win.decode_string_signal(sdk, key)[2] == "C"   # newest in the cell wins


def test_pass_global_fill_silent_fallback_on_string_but_applies_numeric(sdk):
    """aperiodic_fill='aggregate:mean' as a GLOBAL default: silently falls back
    to carry_forward for a string measure (incompatible) yet applies to a numeric
    sample measure. (fill_overrides would instead RAISE for the string.)"""
    sm = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    st = sdk.insert_measure("txt", freq=1.0, freq_units="Hz", units="string",
                            signal_kind="sample", value_type="string")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(sm, dev, BASE + np.array([2 * SEC, 2 * SEC + 1], dtype=np.int64),
                               np.array([10.0, 30.0]))
    sdk.write_time_value_pairs(st, dev, BASE + np.array([1], dtype=np.int64) * SEC,
                               np.array(["A"], dtype=object))
    defn = DatasetDefinition(measures=["nibp", "txt"],
                             device_ids={dev: [{"start": int(BASE), "end": int(BASE + 5 * SEC)}]})
    defn.validate(sdk)
    it = sdk.get_iterator(defn, 5 * SEC, 5 * SEC, aperiodic_fill="aggregate:mean")
    assert it.render_config[sm]["fill_rule"] == "aggregate:mean"     # numeric applies
    assert it.render_config[st]["fill_rule"] == "carry_forward"      # string falls back
    win = next(iter(it))
    sig_by_tag = {k[0]: v for k, v in win.signals.items()}
    assert sig_by_tag["nibp"]["values"][2] == 20.0                   # mean(10,30)
    # string carried forward from cell 1
    assert sig_by_tag["txt"]["values"][1] != UNKNOWN_STRING_CODE
    assert sig_by_tag["txt"]["values"][2] == sig_by_tag["txt"]["values"][1]


def test_pass_invalid_override_rule_raises(sdk):
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([1], dtype=np.int64) * SEC, np.array([1.0]))
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 5 * SEC)
    with pytest.raises(ValueError):
        sdk.get_iterator(defn, 5 * SEC, 5 * SEC, fill_overrides={m: "bogus_rule"})


def test_pass_wrong_kind_override_raises(sdk):
    # carry_forward is invalid for an event kind
    m = sdk.insert_measure("alarm", freq=1.0, freq_units="Hz", units="n", signal_kind="event")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([1], dtype=np.int64) * SEC, np.array([1.0]))
    defn = _region_def(sdk, "alarm", dev, BASE, BASE + 5 * SEC)
    with pytest.raises(ValueError):
        sdk.get_iterator(defn, 5 * SEC, 5 * SEC, fill_overrides={m: "carry_forward"})


def test_unknown_measure_override_is_rejected(sdk):
    """A fill_override / period_override keyed by a measure id NOT in the
    definition is rejected.

    This previously *silently ignored* the key, so a typo (or a measure TAG,
    which is how definitions identify measures everywhere else) produced a
    differently rasterized dataset with no error and no warning."""
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2], dtype=np.int64) * SEC, np.array([100.0]))
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)
    with pytest.raises(ValueError, match="fill_overrides"):
        sdk.get_iterator(defn, 10 * SEC, 10 * SEC, fill_overrides={999999: "sparse"})
    with pytest.raises(ValueError, match="period_overrides"):
        sdk.get_iterator(defn, 10 * SEC, 10 * SEC, period_overrides={999999: 3})

    # the real measure, keyed correctly, still resolves as before
    it = sdk.get_iterator(defn, 10 * SEC, 10 * SEC)
    _, sig = _one(next(iter(it)))
    assert it.render_config[m]["fill_rule"] == "carry_forward"
    assert sig["values"][2] == 100.0


# =========================================================================== #
# NOMINAL PERIOD (these PASS, plus one ugly-error edge)
# =========================================================================== #
def test_pass_period_override_unit_conversion(sdk):
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([1], dtype=np.int64) * SEC, np.array([1.0]))
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)
    # 500 ms
    it_ms = sdk.get_iterator(defn, 10 * SEC, 10 * SEC, period_overrides={m: 500}, time_units="ms")
    assert it_ms.render_config[m]["period_ns"] == 500_000_000
    # 250000 us
    it_us = sdk.get_iterator(defn, 10 * SEC, 10 * SEC, period_overrides={m: 250_000}, time_units="us")
    assert it_us.render_config[m]["period_ns"] == 250_000_000
    # ns default
    it_ns = sdk.get_iterator(defn, 10 * SEC, 10 * SEC, period_overrides={m: 2_000_000_000})
    assert it_ns.render_config[m]["period_ns"] == 2_000_000_000


def test_pass_period_override_larger_than_window_raises(sdk):
    """Robustness edge: a nominal period larger than the window duration yields a
    zero-length row (window_duration // period == 0) and surfaces as an opaque
    'slice step cannot be zero' ValueError rather than a clear message. Captured
    so the behavior is at least a clean raise, not a crash/hang."""
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([1], dtype=np.int64) * SEC, np.array([1.0]))
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)
    with pytest.raises(ValueError):
        it = sdk.get_iterator(defn, 5 * SEC, 5 * SEC, period_overrides={m: 6 * SEC})
        next(iter(it))


def test_pass_mixed_waveform_aperiodic_no_row_size_distortion(sdk):
    """An aperiodic measure gridded at 1 s must NOT distort row_size / batch
    sizing: the waveform's finer period drives lowest_period_ns."""
    wf = sdk.insert_measure("ecg", freq=4.0, freq_units="Hz", units="mV")  # waveform/numeric
    sm = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    wf_t = BASE + np.arange(40, dtype=np.int64) * (SEC // 4)
    sdk.write_time_value_pairs(wf, dev, wf_t, np.arange(40, dtype=np.float64), freq=4.0, freq_units="Hz")
    sdk.write_time_value_pairs(sm, dev, BASE + np.array([2, 5], dtype=np.int64) * SEC,
                               np.array([100.0, 110.0]))
    defn = DatasetDefinition(measures=["ecg", "nibp"],
                             device_ids={dev: [{"start": int(BASE), "end": int(BASE + 10 * SEC)}]})
    defn.validate(sdk)
    it = sdk.get_iterator(defn, 10 * SEC, 10 * SEC)
    assert it.lowest_period_ns == SEC // 4          # waveform, not the aperiodic 1 s
    assert it.row_size == 40
    win = next(iter(it))
    by_tag = {k[0]: v for k, v in win.signals.items()}
    assert by_tag["ecg"]["values"].shape == (40,)
    assert by_tag["nibp"]["values"].shape == (10,)


# =========================================================================== #
# REGRESSION: waveform-numeric path unchanged (independent of writer's test)
# =========================================================================== #
def test_pass_waveform_numeric_byte_identical_dense_overlap(sdk):
    """Independent equivalence check with a DIFFERENT gap pattern and slide than
    the writer's test: render_config=None vs the waveform 'grid' config must
    produce byte-identical windows (equal_nan)."""
    wf = sdk.insert_measure("ecg", freq=8.0, freq_units="Hz", units="mV")
    dev = sdk.insert_device(device_tag="dev1")
    # 6 s @ 8 Hz with two separate internal gaps
    keep = np.array([i for i in range(48) if i not in range(5, 9) and i not in range(30, 33)],
                    dtype=np.int64)
    t = BASE + keep * (SEC // 8)
    v = (keep.astype(np.float64) * 1.5) - 3.0
    sdk.write_time_value_pairs(wf, dev, t, v, freq=8.0, freq_units="Hz")

    defn = _region_def(sdk, "ecg", dev, BASE, BASE + 6 * SEC)
    it = sdk.get_iterator(defn, 3 * SEC, 1 * SEC)   # heavily overlapping windows
    common = dict(sdk=sdk, device_id=dev, query_patient_id=None,
                  window_duration_ns=3 * SEC, window_slide_ns=1 * SEC, measures=it.measures,
                  batch_start_time=int(BASE), batch_end_time=int(BASE + 6 * SEC),
                  batch_num_windows=4, range_start_time=int(BASE), range_end_time=int(BASE + 6 * SEC))
    legacy = get_signal_dictionary(render_config=None, **common)
    p3 = get_signal_dictionary(render_config=it.render_config, **common)
    lt, lv, lc = legacy[wf]
    pt, pv, pc = p3[wf]
    assert lc == pc
    assert np.array_equal(lt, pt)
    assert np.array_equal(lv, pv, equal_nan=True)


# =========================================================================== #
# ITERATOR VARIANTS thread the fill config
# =========================================================================== #
def _first_window_values_from(iterator):
    for w in iterator:
        return _one(w)[1]["values"]
    return None


def test_pass_mapped_and_filtered_variants_thread_fill_config(sdk):
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2, 5], dtype=np.int64) * SEC,
                               np.array([100.0, 110.0]))
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)

    base = _first_window_values_from(sdk.get_iterator(defn, 10 * SEC, 10 * SEC))

    mapped = sdk.get_iterator(defn, 10 * SEC, 10 * SEC, iterator_type="mapped")
    assert mapped.render_config[m]["fill_rule"] == "carry_forward"
    np.testing.assert_array_equal(mapped[0].signals[list(mapped[0].signals.keys())[0]]["values"], base)

    filt = sdk.get_iterator(defn, 10 * SEC, 10 * SEC, iterator_type="filtered",
                            window_filter_fn=lambda w: True)
    assert filt.render_config[m]["fill_rule"] == "carry_forward"
    np.testing.assert_array_equal(_first_window_values_from(filt), base)


def test_pass_num_iterators_thread_fill_config(sdk):
    """num_iterators>1 recursion must thread aperiodic_fill/fill_overrides to
    every partitioned sub-iterator. (partition_dataset stratifies by patient, so
    the sources are patient-mapped here.)"""
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev1 = sdk.insert_device(device_tag="dev1")
    dev2 = sdk.insert_device(device_tag="dev2")
    p1 = sdk.insert_patient(mrn="mrn1")
    p2 = sdk.insert_patient(mrn="mrn2")
    sdk.insert_device_patient_data([(dev1, p1, int(BASE), int(BASE + 10 * SEC)),
                                    (dev2, p2, int(BASE), int(BASE + 10 * SEC))])
    for d in (dev1, dev2):
        sdk.write_time_value_pairs(m, d, BASE + np.array([2, 5], dtype=np.int64) * SEC,
                                   np.array([100.0, 110.0]))
    defn = DatasetDefinition(
        measures=["nibp"],
        patient_ids={p1: [{"start": int(BASE), "end": int(BASE + 10 * SEC)}],
                     p2: [{"start": int(BASE), "end": int(BASE + 10 * SEC)}]},
    )
    defn.validate(sdk)
    iters = sdk.get_iterator(defn, 10 * SEC, 10 * SEC, num_iterators=2,
                             fill_overrides={m: "sparse"})
    assert isinstance(iters, list) and len(iters) == 2
    for sub in iters:
        assert sub.render_config[m]["fill_rule"] == "sparse"


def test_lightmapped_warns_then_rejects_an_aperiodic_measure(sdk):
    """'lightmapped' is the numeric NaN-grid path only.

    It still warns that the P3 fill config is not applied (never silently
    pretends to honor it), and it now refuses an aperiodic/string measure at
    construction with a measure-named error instead of failing deep inside
    iteration with an opaque block-codec message."""
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2], dtype=np.int64) * SEC, np.array([100.0]))
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match=f"Measure {m}"):
            sdk.get_iterator(defn, 10 * SEC, 10 * SEC, iterator_type="lightmapped",
                             aperiodic_fill="sparse")

    # a plain waveform definition still works on 'lightmapped'
    m_wave = sdk.insert_measure("ecg", freq=1.0, freq_units="Hz", units="mV")
    sdk.write_time_value_pairs(m_wave, dev, BASE + np.arange(10, dtype=np.int64) * SEC,
                               np.arange(10.0))
    wave_defn = _region_def(sdk, "ecg", dev, BASE, BASE + 10 * SEC)
    assert sdk.get_iterator(wave_defn, 10 * SEC, 10 * SEC, iterator_type="lightmapped") is not None


# =========================================================================== #
# ALL-UNKNOWN window (no data) + string decode round trip over a gap
# =========================================================================== #
def test_pass_no_data_window_is_all_unknown(sdk):
    m = sdk.insert_measure("nibp", freq=1.0, freq_units="Hz", units="mmHg", signal_kind="sample")
    dev = sdk.insert_device(device_tag="dev1")
    # write one reading far outside the requested region so the region has none
    sdk.write_time_value_pairs(m, dev, BASE + np.array([100], dtype=np.int64) * SEC, np.array([1.0]))
    defn = _region_def(sdk, "nibp", dev, BASE, BASE + 10 * SEC)
    _, sig = _one(next(iter(sdk.get_iterator(defn, 10 * SEC, 10 * SEC))))
    assert np.all(np.isnan(sig["values"]))
    assert sig["actual_count"] == 0


def test_pass_string_carry_forward_holds_across_internal_gap(sdk):
    """Within one batch, a string state carries its last value across a data gap
    (no reading in the gap -> carried, not sentinel)."""
    m = sdk.insert_measure("anes", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="state", value_type="string")
    dev = sdk.insert_device(device_tag="dev1")
    # readings only at 1s and 8s; the 2..7 gap must carry "ON"
    t = BASE + np.array([1, 8], dtype=np.int64) * SEC
    sdk.write_time_value_pairs(m, dev, t, np.array(["ON", "OFF"], dtype=object))
    defn = _region_def(sdk, "anes", dev, BASE, BASE + 10 * SEC)
    win = next(iter(sdk.get_iterator(defn, 10 * SEC, 10 * SEC)))
    key, sig = _one(win)
    decoded = win.decode_string_signal(sdk, key)
    assert decoded[0] == UNKNOWN_STRING_VALUE        # before first reading
    assert list(decoded[1:8]) == ["ON"] * 7          # carried across the gap
    assert list(decoded[8:10]) == ["OFF"] * 2
