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
Regression guards for the READ / WINDOWING / ITERATOR / EXPORT defects found by
the wave-2 adversarial review of the aperiodic + text feature (design sections
21, 23). One minimal test per fixed defect.

Every test here asserts the FIXED behaviour and must PASS. SQLite only; no
MariaDB and no physionet download needed.
"""
import os
import shutil
import tempfile
import warnings

import numpy as np
import pytest

from atriumdb import AtriumSDK, DatasetDefinition

SEC = 1_000_000_000
BASE = 1_600_000_000 * SEC


@pytest.fixture
def sdk():
    loc = tempfile.mkdtemp(prefix="atrium_read_fixes_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def _region_def(sdk, tags, device_id, start_ns, end_ns):
    d = DatasetDefinition(
        measures=list(tags),
        device_ids={device_id: [{"start": int(start_ns), "end": int(end_ns)}]},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d.validate(sdk)
    return d


def _windows(iterator):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return list(iterator)


def _one(window):
    (key,) = list(window.signals.keys())
    return key, window.signals[key]


def _signal(window, tag):
    return next(s for k, s in window.signals.items() if k[0] == tag)


# --------------------------------------------------------------------------- #
# 1. decode_string_signal must never fabricate clinical strings
# --------------------------------------------------------------------------- #
def test_decode_string_signal_refuses_a_presence_channel(sdk):
    """An 'event' string measure rasterizes to presence FLOATS, not codes.

    Blindly astype(int64)-ing them turned 0/1 occupancy into dictionary codes
    0/1 and returned real vocabulary words for cells where nothing happened --
    silent fabrication of clinical values through a documented API.
    """
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("alarm", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="event")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([1, 3, 7], dtype=np.int64) * SEC,
                               np.array(["ASYSTOLE", "VTACH", "ASYSTOLE"]))

    d = _region_def(sdk, ["alarm"], dev, BASE, BASE + 10 * SEC)
    iterator = sdk.get_iterator(d, 10 * SEC, 10 * SEC)
    window = _windows(iterator)[0]
    key, sig = _one(window)

    # precondition: this really is a presence channel, not a code channel
    assert np.issubdtype(np.asarray(sig["values"]).dtype, np.floating)

    with pytest.raises(ValueError, match="dictionary codes"):
        window.decode_string_signal(sdk, key)
    with pytest.raises(ValueError, match="dictionary codes"):
        iterator.decode_window_strings(window, key)


def test_decode_string_signal_still_decodes_a_real_code_channel(sdk):
    """The guard must not block the legitimate case: a 'state'/'sample' string
    measure carries int64 codes and still decodes."""
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("mode", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="state")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([0], dtype=np.int64) * SEC,
                               np.array(["SIMV"]))

    d = _region_def(sdk, ["mode"], dev, BASE, BASE + 5 * SEC)
    iterator = sdk.get_iterator(d, 5 * SEC, 5 * SEC)
    window = _windows(iterator)[0]
    key, sig = _one(window)

    assert np.issubdtype(np.asarray(sig["values"]).dtype, np.integer)
    assert list(window.decode_string_signal(sdk, key)) == ["SIMV"] * 5


# --------------------------------------------------------------------------- #
# 2. cached_windows_per_source is a RAM knob and must not change any value
# --------------------------------------------------------------------------- #
def test_carry_forward_is_independent_of_cached_windows_per_source(sdk):
    """_split_time_ranges rewrites self.sources, so every split piece used to
    become its own carry-forward seed floor -- a pure RAM/shuffle knob silently
    turned carried-forward state into the unknown sentinel from the first split
    onward. This is the shuffle-on ML training path."""
    dev = sdk.insert_device(device_tag="d1")
    m_state = sdk.insert_measure("mode", freq=1.0, freq_units="Hz", units="string",
                                 signal_kind="state")
    m_sample = sdk.insert_measure("lab", freq=1.0, freq_units="Hz", units="mmol/L",
                                  signal_kind="sample")
    # one transition / one reading, both at the very start of the range
    sdk.write_time_value_pairs(m_state, dev, np.array([BASE], dtype=np.int64),
                               np.array(["SIMV"]))
    sdk.write_time_value_pairs(m_sample, dev, np.array([BASE], dtype=np.int64),
                               np.array([5.0]))

    def run(**kwargs):
        d = _region_def(sdk, ["mode", "lab"], dev, BASE, BASE + 60 * SEC)
        out = {}
        for w in _windows(sdk.get_iterator(d, 10 * SEC, 10 * SEC, **kwargs)):
            out[int(w.start_time)] = {k[0]: np.asarray(s["values"]).tolist()
                                      for k, s in w.signals.items()}
        return out

    reference = run()
    # sanity: unsplit, the state carries forward across all 6 windows
    assert all(v["mode"] == [0] * 10 for v in reference.values())
    assert all(v["lab"] == [5.0] * 10 for v in reference.values())

    for kwargs in ({"shuffle": 1, "cached_windows_per_source": 1},
                   {"shuffle": 1, "cached_windows_per_source": 3}):
        assert run(**kwargs) == reference, f"carry-forward output changed with {kwargs}"


# --------------------------------------------------------------------------- #
# 3. event presence must not fabricate zeros outside the definition range
# --------------------------------------------------------------------------- #
def test_event_presence_does_not_fabricate_zeros_past_the_range_end(sdk):
    """A trailing partial window overhangs the definition range. Those cells used
    to read as a confident 0 ("no event occurred") with actual_count reporting
    full coverage -- systematic false-negative label injection at every cohort
    boundary. They must be the unknown sentinel, like every other channel."""
    dev = sdk.insert_device(device_tag="d1")
    m_ev = sdk.insert_measure("alarm", freq=1.0, freq_units="Hz", units="string",
                              signal_kind="event")
    m_ecg = sdk.insert_measure("ecg", freq=1.0, freq_units="Hz", units="mV")
    sdk.write_time_value_pairs(m_ev, dev, BASE + np.array([1, 22], dtype=np.int64) * SEC,
                               np.array(["ASYSTOLE", "ASYSTOLE"]))
    sdk.write_time_value_pairs(m_ecg, dev, BASE + np.arange(25, dtype=np.int64) * SEC,
                               np.arange(25.0))

    # range ends at 25 s -> the third 10 s window covers 20..30 s
    d = _region_def(sdk, ["alarm", "ecg"], dev, BASE, BASE + 25 * SEC)
    windows = _windows(sdk.get_iterator(d, 10 * SEC, 10 * SEC))

    last_alarm = _signal(windows[-1], "alarm")
    last_ecg = _signal(windows[-1], "ecg")
    assert last_ecg["actual_count"] == 5, "precondition: waveform half-covers this window"
    assert last_alarm["actual_count"] == 5
    values = np.asarray(last_alarm["values"])
    assert values[2] == 1.0                      # the event at t=22 s
    assert values[0] == 0.0 and values[1] == 0.0  # genuine in-range absence stays a hard 0
    assert np.all(np.isnan(values[5:]))          # out of range -> unknown, not 0

    # a fully-covered window is unaffected: absence is still a meaningful 0
    first_alarm = _signal(windows[0], "alarm")
    assert first_alarm["actual_count"] == 10
    assert not np.any(np.isnan(np.asarray(first_alarm["values"])))


# --------------------------------------------------------------------------- #
# 4. the pre-flight guard must use the resolved nominal period, not freq_nhz
# --------------------------------------------------------------------------- #
def test_iterator_accepts_an_honestly_declared_aperiodic_measure(sdk):
    """An NIBP declared at its true 1/300 Hz used to be rejected outright by
    get_iterator's freq-derived sample-count guard, and period_overrides could
    not rescue it because the guard runs before the iterator is built."""
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("nibp", freq=1 / 300, freq_units="Hz", units="mmHg",
                           signal_kind="sample")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([0, 300, 600], dtype=np.int64) * SEC,
                               np.array([120.0, 118.0, 122.0]))

    d = _region_def(sdk, ["nibp"], dev, BASE, BASE + 900 * SEC)
    windows = _windows(sdk.get_iterator(d, 60, 60, time_units="s"))
    assert len(windows) == 15
    # 1 s nominal raster; the reading at t=0 carries forward through window 0
    assert np.asarray(_one(windows[0])[1]["values"]).tolist() == [120.0] * 60


def test_period_larger_than_the_window_still_names_the_measure(sdk):
    """The guard must not swallow the P3 audit's measure-named Bug-3 error."""
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("lab", freq=1.0, freq_units="Hz", units="mmol/L",
                           signal_kind="sample")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2], dtype=np.int64) * SEC,
                               np.array([5.0]))
    d = _region_def(sdk, ["lab"], dev, BASE, BASE + 60 * SEC)
    with pytest.raises(ValueError, match=f"Measure {m}"):
        sdk.get_iterator(d, 10 * SEC, 10 * SEC, period_overrides={m: 30 * SEC})


# --------------------------------------------------------------------------- #
# 5. fill / period override validation
# --------------------------------------------------------------------------- #
def test_aperiodic_fill_rejects_an_unknown_rule_name(sdk):
    """resolve_fill_rule only checked kind-compatibility, so a name that is not a
    fill rule at all (garbage, or the hyphen typo 'carry-forward') fell through
    to the per-kind default with no error and no warning."""
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("mode", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="state")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2], dtype=np.int64) * SEC,
                               np.array(["A"]))
    d = _region_def(sdk, ["mode"], dev, BASE, BASE + 10 * SEC)

    for bad in ("carry-forward", "nonsense"):
        with pytest.raises(ValueError, match="not a supported fill rule"):
            sdk.get_iterator(d, 10 * SEC, 10 * SEC, aperiodic_fill=bad)

    # a valid-but-incompatible global default still falls back silently (documented)
    it = sdk.get_iterator(d, 10 * SEC, 10 * SEC, aperiodic_fill="presence")
    assert it.render_config[m]["fill_rule"] == "carry_forward"


def test_overrides_reject_a_key_that_matches_no_definition_measure(sdk):
    """Override dicts are keyed by measure ID; an unknown key (typically a
    measure TAG, which is how definitions identify measures everywhere else)
    used to be a silent no-op."""
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("lab", freq=1.0, freq_units="Hz", units="mmol/L",
                           signal_kind="sample")
    sdk.write_time_value_pairs(m, dev, BASE + np.array([2], dtype=np.int64) * SEC,
                               np.array([5.0]))
    d = _region_def(sdk, ["lab"], dev, BASE, BASE + 10 * SEC)

    with pytest.raises(ValueError, match="fill_overrides"):
        sdk.get_iterator(d, 10 * SEC, 10 * SEC, fill_overrides={"lab": "sparse"})
    with pytest.raises(ValueError, match="period_overrides"):
        sdk.get_iterator(d, 10 * SEC, 10 * SEC, period_overrides={9999: 2})

    # the correctly-keyed override still applies
    it = sdk.get_iterator(d, 10 * SEC, 10 * SEC, fill_overrides={m: "sparse"})
    assert it.render_config[m]["fill_rule"] == "sparse"


def test_period_override_on_a_waveform_measure_names_the_measure(sdk):
    """period_overrides is an aperiodic-raster concept. On a waveform it used to
    re-grid the legacy NaN-fill path while get_data still filled at the real
    frequency, producing an opaque, measure-less block-codec error."""
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("ecg", freq=4.0, freq_units="Hz", units="mV")
    sdk.write_time_value_pairs(m, dev, BASE + np.arange(40, dtype=np.int64) * (SEC // 4),
                               np.arange(40.0))
    d = _region_def(sdk, ["ecg"], dev, BASE, BASE + 10 * SEC)

    with pytest.raises(ValueError, match="period_overrides"):
        sdk.get_iterator(d, 10 * SEC, 10 * SEC, period_overrides={m: SEC})


# --------------------------------------------------------------------------- #
# 6. the carry-forward seed lookback must be bounded
# --------------------------------------------------------------------------- #
def test_carry_forward_seed_lookback_is_bounded(sdk):
    """The seed read used to span [definition_range_start, batch_start) on EVERY
    batch and keep only the last element: O(N) per batch, O(N^2) overall, with a
    code comment falsely claiming bounded RAM."""
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("mode", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="state")
    n_readings = 600
    sdk.write_time_value_pairs(
        m, dev, BASE + np.arange(n_readings, dtype=np.int64) * SEC,
        np.array(["SIMV" if i % 2 == 0 else "PRVC" for i in range(n_readings)]))

    d = _region_def(sdk, ["mode"], dev, BASE, BASE + n_readings * SEC)

    decoded = []
    original = AtriumSDK.get_data

    def counting_get_data(self, measure_id, start_time_n, end_time_n, *args, **kwargs):
        result = original(self, measure_id, start_time_n, end_time_n, *args, **kwargs)
        decoded.append(int(np.asarray(result[1]).size))
        return result

    AtriumSDK.get_data = counting_get_data
    try:
        windows = _windows(sdk.get_iterator(d, 10 * SEC, 10 * SEC, num_windows_prefetch=1))
    finally:
        AtriumSDK.get_data = original

    assert sum(decoded) <= 4 * n_readings, (
        f"decoded {sum(decoded)} values for a {n_readings}-reading stream over "
        f"{len(decoded)} get_data calls (max single read: {max(decoded)})")

    # ... and the bounded lookback still produces the same values as one big batch
    d2 = _region_def(sdk, ["mode"], dev, BASE, BASE + n_readings * SEC)
    reference = _windows(sdk.get_iterator(d2, 10 * SEC, 10 * SEC, num_windows_prefetch=10_000))
    assert len(windows) == len(reference)
    for got, want in zip(windows, reference):
        assert np.array_equal(_one(got)[1]["values"], _one(want)[1]["values"])


# --------------------------------------------------------------------------- #
# 7. the PyTorch on-ramp ('lightmapped') must fail actionably, not opaquely
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,units,expect", [
    ("state", "string", "string measure"),      # AtriumDBMapDataset's documented path
    ("sample", "mmHg", "aperiodic"),            # numeric aperiodic: used to be opaque
])
def test_lightmapped_rejects_aperiodic_measures_actionably(sdk, kind, units, expect):
    """AtriumDBMapDataset hard-coded iterator_type='lightmapped', which is the
    numeric NaN-grid path. A string measure failed deep inside iteration; a
    numeric aperiodic one failed with a measure-less block-codec message. Both
    now fail at construction, naming the measure and pointing at 'mapped'."""
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("m1", freq=1.0, freq_units="Hz", units=units, signal_kind=kind)
    sdk.write_time_value_pairs(
        m, dev, BASE + np.array([2, 6], dtype=np.int64) * SEC,
        np.array(["A", "B"]) if units == "string" else np.array([120.0, 130.0]))
    d = _region_def(sdk, ["m1"], dev, BASE, BASE + 10 * SEC)

    with pytest.raises(ValueError, match=expect) as exc:
        sdk.get_iterator(d, 10 * SEC, 10 * SEC, iterator_type="lightmapped")
    assert "mapped" in str(exc.value)
    assert f"Measure {m}" in str(exc.value)

    # the pointed-to path works
    windows = _windows(sdk.get_iterator(d, 10 * SEC, 10 * SEC, iterator_type="mapped"))
    assert len(windows) == 1


def test_lightmapped_still_works_for_waveform_measures(sdk):
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("ecg", freq=1.0, freq_units="Hz", units="mV")
    sdk.write_time_value_pairs(m, dev, BASE + np.arange(10, dtype=np.int64) * SEC,
                               np.arange(10.0))
    d = _region_def(sdk, ["ecg"], dev, BASE, BASE + 10 * SEC)
    iterator = sdk.get_iterator(d, 10 * SEC, 10 * SEC, iterator_type="lightmapped")
    assert np.array_equal(np.asarray(_one(iterator[0])[1]["values"]), np.arange(10.0))


# --------------------------------------------------------------------------- #
# 8. event-region definition validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("extra", [
    {"start": BASE + 400 * SEC, "end": BASE + 500 * SEC},
    {"time0": BASE + 400 * SEC},
])
def test_event_region_rejects_silently_ignored_keys(extra):
    """_get_validated_entries branches on 'anchor'/'from' and never reads
    'start'/'end'/'time0' in the same dict, so "anchor within this shift" used to
    validate cleanly and silently return the whole range."""
    region = {"anchor": "START", "measure": "anes", "pre": 10 * SEC, "post": 10 * SEC}
    region.update(extra)
    with pytest.raises(ValueError, match="cannot be combined"):
        DatasetDefinition(measures=["ECG"], device_ids={1: [region]})


def test_event_region_rejects_max_duration_zero():
    """max_duration=0 caps every interval to zero length; the `if start < end`
    filter then dropped the whole region with no error and no warning."""
    with pytest.raises(ValueError, match="max_duration must be greater than 0"):
        DatasetDefinition(measures=["ECG"], device_ids={1: [
            {"from": "START", "to": "STOP", "measure": "anes", "within": "none",
             "max_duration": 0}]})

    # a positive cap is still accepted
    DatasetDefinition(measures=["ECG"], device_ids={1: [
        {"from": "START", "to": "STOP", "measure": "anes", "within": "none",
         "max_duration": 60 * SEC}]})


# --------------------------------------------------------------------------- #
# 9. carry_forward must not lose a reading that lands inside a cell
# --------------------------------------------------------------------------- #
def test_carry_forward_renders_a_reading_in_the_cell_it_falls_in(sdk):
    """A cell reports the value in effect DURING that cell.

    carry_forward used to give a cell the most recent reading at or before the
    cell's START (searchsorted side='right' on the grid), so a reading landing
    mid-cell only surfaced from the NEXT cell -- and a reading in the FINAL cell of
    the definition range had no next cell, so it vanished from every window: an
    all-sentinel window with actual_count 0 despite a genuine in-range observation.
    """
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("lactate", freq=1.0, freq_units="Hz", units="mmol/L",
                           signal_kind="sample")
    # one reading 9.5 s into a 10 s range -> inside the last 1 s cell
    sdk.write_time_value_pairs(m, dev, np.array([BASE + 9_500_000_000], dtype=np.int64),
                               np.array([4.2]))

    d = _region_def(sdk, ["lactate"], dev, BASE, BASE + 10 * SEC)
    _key, sig = _one(_windows(sdk.get_iterator(d, 10 * SEC, 10 * SEC))[0])

    vals = sig["values"]
    assert sig["actual_count"] == 1, f"the range's only reading is absent: {vals}"
    assert vals[9] == 4.2                       # rendered in the cell it falls in
    assert np.all(np.isnan(vals[:9]))           # earlier cells stay left-censored


def test_carry_forward_mid_cell_reading_is_not_delayed_by_one_cell(sdk):
    """The same off-by-one, away from the range boundary, for a string state.

    Also pins the invariant that a reading landing exactly ON a cell boundary is
    unaffected by the fix (that is what every pre-existing P3 test writes).
    """
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("mode", freq=1.0, freq_units="Hz", units="string",
                           signal_kind="state", value_type="string")
    # transitions at 2.5 s (mid-cell) and 6.0 s (exactly on a cell start)
    sdk.write_time_value_pairs(
        m, dev, np.array([BASE + 2_500_000_000, BASE + 6 * SEC], dtype=np.int64),
        np.array(["A", "B"]))

    d = _region_def(sdk, ["mode"], dev, BASE, BASE + 10 * SEC)
    win = _windows(sdk.get_iterator(d, 10 * SEC, 10 * SEC))[0]
    key, _sig = _one(win)
    decoded = list(win.decode_string_signal(sdk, key))

    # cell 2 spans [2 s, 3 s) and contains the transition -> it reports "A"
    assert decoded[2:6] == ["A"] * 4
    # a reading exactly on a cell boundary is unchanged: cell 6 opens with "B"
    assert decoded[6:10] == ["B"] * 4
    # cells strictly before the first observation are still left-censored
    assert decoded[0] != "A" and decoded[1] != "A"


# --------------------------------------------------------------------------- #
# 11. carry-forward must see the reading that precedes the region start
# --------------------------------------------------------------------------- #
def test_carry_forward_seeds_from_before_the_region_start(sdk):
    """A cohort region that begins mid-stream must render the same values as the
    whole stream at the same wall-clock time.

    Reported by a user driving the "N minutes either side of every event" recipe:
    with one unbroken availability interval and a reading at 330 s, windows from a
    region starting at 400 s came back NaN with actual_count == 0 for three
    minutes, while region='all' rendered the same wall-clock windows as the real
    value. The carry-forward seed floor was the definition's own range start, so
    the value in effect at the region start was unreachable. The floor is now that
    range start minus a bounded lookback horizon
    (windowing_functions.CARRY_FORWARD_LOOKBACK_NS) -- still a pure function of
    the definition, so batching independence is unaffected.
    """
    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("lab", freq=1.0, freq_units="Hz", units="x",
                           signal_kind="sample", value_type="numeric")
    sdk.write_time_value_pairs(
        m, dev, BASE + np.array([0, 330, 630], dtype=np.int64) * SEC,
        np.array([100.0, 110.0, 120.0]))

    def values_at(defn, start_ns):
        for w in _windows(sdk.get_iterator(defn, 60 * SEC, 60 * SEC)):
            if int(w.start_time) == int(start_ns):
                return _one(w)[1]
        raise AssertionError(f"no window starting at {start_ns}")

    whole = DatasetDefinition(measures=["lab"], device_ids={dev: "all"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole.validate(sdk)
    scoped = _region_def(sdk, ["lab"], dev, BASE + 420 * SEC, BASE + 480 * SEC)

    scoped_sig = values_at(scoped, BASE + 420 * SEC)
    whole_sig = values_at(whole, BASE + 420 * SEC)

    assert list(scoped_sig["values"]) == [110.0] * 60
    assert scoped_sig["actual_count"] == 60
    assert list(scoped_sig["values"]) == list(whole_sig["values"])


def test_carry_forward_seed_lookback_does_not_reach_past_the_horizon(sdk):
    """The lookback is bounded: a reading older than CARRY_FORWARD_LOOKBACK_NS is
    not resurrected, so the seed can never turn into an unbounded backwards scan."""
    from atriumdb.windowing.windowing_functions import CARRY_FORWARD_LOOKBACK_NS

    dev = sdk.insert_device(device_tag="d1")
    m = sdk.insert_measure("lab", freq=1.0, freq_units="Hz", units="x",
                           signal_kind="sample", value_type="numeric")
    ancient = BASE
    region_start = ancient + CARRY_FORWARD_LOOKBACK_NS + 60 * SEC
    sdk.write_time_value_pairs(m, dev, np.array([ancient], dtype=np.int64), np.array([100.0]))
    sdk.write_time_value_pairs(m, dev, np.array([region_start + 30 * SEC], dtype=np.int64),
                               np.array([200.0]))

    defn = _region_def(sdk, ["lab"], dev, region_start, region_start + 60 * SEC)
    sig = _one(_windows(sdk.get_iterator(defn, 60 * SEC, 60 * SEC))[0])[1]
    values = np.asarray(sig["values"], dtype=np.float64)
    # cells before the in-range reading stay unknown -- the >24 h old value is not
    # carried -- and the in-range reading itself still renders.
    assert np.all(np.isnan(values[:30]))
    assert list(values[30:]) == [200.0] * 30


# --------------------------------------------------------------------------- #
# 12. on_censored='clip' must clip to real data, not to an event index's padding
# --------------------------------------------------------------------------- #
def test_clipped_censored_event_region_stops_at_the_end_of_the_recording(sdk):
    """A never-closed from/to interval must not extend past the recording.

    An aperiodic block's index entry ends at ``last_sample + period``. For an
    EVENT measure that trailing period is pure padding -- no event can be in it --
    but it was still the whole-stream container end that ``on_censored='clip'``
    (the default, and the docs' own example) clipped a censored interval to. A
    2 h recording whose event stream had a 30 min nominal period therefore
    produced 20 consecutive all-NaN waveform windows out to 2 h 20 min, which
    reads as missing data rather than as the end of the record.
    """
    dev = sdk.insert_device(device_tag="d1")
    ecg = sdk.insert_measure("ECG", freq=1.0, freq_units="Hz", units="mV")
    ev = sdk.insert_measure("evt", freq=1.0, freq_units="Hz", units="event",
                            signal_kind="event", value_type="string")

    # ECG covers [0, 90); events at 10 (START), 40 (STOP), 70 (START, never closed).
    # The event stream's detected period is 30 s, so its index runs to 100 s.
    sdk.write_time_value_pairs(
        ecg, dev, BASE + np.arange(90, dtype=np.int64) * SEC, np.arange(90.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdk.write_time_value_pairs(
            ev, dev, BASE + np.array([10, 40, 70], dtype=np.int64) * SEC,
            np.array(["START", "STOP", "START"]))

    # The event measure's own index reaches past the ECG: that padding is the
    # phantom bound the region must NOT be clipped to.
    ev_index_end = int(np.asarray(sdk.get_interval_array(measure_id=ev, device_id=dev))[-1][1])
    assert ev_index_end > BASE + 90 * SEC

    defn = DatasetDefinition(
        measures=["ECG", "evt"],
        device_ids={dev: [{"from": "START", "to": "STOP", "measure": "evt",
                           "within": "none", "on_censored": "clip"}]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        defn.validate(sdk)

    ranges = sorted(defn.validated_data_dict["sources"]["device_ids"][dev])
    assert [[(s - BASE) // SEC, (e - BASE) // SEC] for s, e in ranges] == [[10, 40], [70, 90]]

    # ... and no emitted window runs past the end of the ECG recording.
    last_end = max(int(w.start_time) + 10 * SEC
                   for w in _windows(sdk.get_iterator(defn, 10 * SEC, 10 * SEC)))
    assert last_end <= BASE + 90 * SEC


# --------------------------------------------------------------------------- #
# 13. a file export must never drop a string measure silently
# --------------------------------------------------------------------------- #
def _export_src(sdk):
    """One numeric + one string measure on one device."""
    dev = sdk.insert_device(device_tag="d1")
    hr = sdk.insert_measure("hr", freq=1.0, freq_units="Hz", units="bpm",
                            signal_kind="sample", value_type="numeric")
    alarm = sdk.insert_measure("alarm_text", freq=1.0, freq_units="Hz", units="alarm",
                               signal_kind="event", value_type="string")
    sdk.write_time_value_pairs(hr, dev, BASE + np.arange(10, dtype=np.int64) * SEC,
                               np.arange(60.0, 70.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdk.write_time_value_pairs(alarm, dev, BASE + np.array([2, 5], dtype=np.int64) * SEC,
                                   np.array(["ASYSTOLE", "SPO2 LOW"]))
    return dev


def test_csv_export_writes_string_measures_instead_of_dropping_them(sdk):
    """CSV / Parquet / NPZ used to export only the numeric measure.

    Requesting a numeric plus a string measure produced a bundle containing just
    the numeric one -- no warning, no error -- while the bundle's own
    meta/definition.yaml still listed the string measure under ``measures:``. A
    user would ship that extract believing the events were in it. These formats
    all have a text-capable value column, so the decoded strings are written.
    """
    from atriumdb.transfer.adb.dataset import transfer_data

    dev = _export_src(sdk)
    out = tempfile.mkdtemp(prefix="atrium_export_csv_")
    shutil.rmtree(out, ignore_errors=True)
    dest = AtriumSDK.create_dataset(dataset_location=out)
    try:
        defn = DatasetDefinition(measures=["hr", "alarm_text"], device_ids={dev: "all"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transfer_data(src_sdk=sdk, dest_sdk=dest, definition=defn, export_format="csv")

        written = [f for _r, _d, files in os.walk(os.path.join(out, "csv")) for f in files]
        assert any(f.endswith(".csv") for f in written)

        alarm_csv = [os.path.join(r, f)
                     for r, _d, files in os.walk(os.path.join(out, "csv"))
                     for f in files if "alarm_text" in r]
        assert alarm_csv, f"no alarm_text export produced; got {written}"
        text = open(alarm_csv[0]).read()
        assert "alarm_text" in text
        assert "ASYSTOLE" in text and "SPO2 LOW" in text

        # the manifest still legitimately lists it, because the data IS there
        manifest = open(os.path.join(out, "meta", "definition.yaml")).read()
        assert "alarm_text" in manifest
    finally:
        dest.close()
        shutil.rmtree(out, ignore_errors=True)


def test_export_that_cannot_carry_strings_warns_and_does_not_claim_them(sdk):
    """wfdb has no text-capable value column. Omitting the measure is fine;
    omitting it SILENTLY -- and still naming it in the exported definition -- is
    not, because the extract then looks complete."""
    from atriumdb.transfer.adb.dataset import transfer_data

    dev = _export_src(sdk)
    out = tempfile.mkdtemp(prefix="atrium_export_wfdb_")
    shutil.rmtree(out, ignore_errors=True)
    dest = AtriumSDK.create_dataset(dataset_location=out)
    try:
        defn = DatasetDefinition(measures=["hr", "alarm_text"], device_ids={dev: "all"})
        with pytest.warns(UserWarning, match="alarm_text"):
            transfer_data(src_sdk=sdk, dest_sdk=dest, definition=defn, export_format="wfdb")

        manifest = open(os.path.join(out, "meta", "definition.yaml")).read()
        assert "alarm_text" not in manifest
        assert "hr" in manifest
    finally:
        dest.close()
        shutil.rmtree(out, ignore_errors=True)
