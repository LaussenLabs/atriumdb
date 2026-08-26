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
Event-anchored DatasetDefinition region tests.

Covers:
  * ``anchor`` regions emit ``[t - pre, t + post]`` per occurrence, merged with each
    other and clipped to the source data union + global bounds.
  * ``from``/``to`` regions match ``get_event_intervals`` then honor ``pre``/``post``/
    ``max_duration``.
  * a region's ``within`` scopes the emitted windows (device_patient populated, and the
    empty-device_patient fall-through).
  * ``on_censored`` clip / drop / keep.
  * unknown event-measure tag and out-of-vocabulary values raise at ``validate()``.
  * a full anchor region resolves through to a real ``DatasetIterator`` producing the
    expected windows.

SQLite only:
    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_event_anchored_definition.py -q
"""
import shutil
import tempfile
import warnings

import numpy as np
import pytest

from atriumdb import AtriumSDK, DatasetDefinition

SEC = 1_000_000_000
BASE = 1_600_000_000 * SEC
FREQ_NHZ = 1_000_000_000          # 1 Hz in nHz
PERIOD_NS = (10 ** 18) // FREQ_NHZ  # 1 s


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def sdk():
    loc = tempfile.mkdtemp(prefix="atrium_event_anchored_definition_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    s._loc = loc
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def setup_source(sdk, numeric_span_s=(0, 100), num_tag="hr", evt_tag="evt"):
    """A device with a continuous numeric measure over ``numeric_span_s`` (1 Hz) and a
    co-located string/event measure. Returns (numeric_measure_id, event_measure_id,
    device_id)."""
    device_id = sdk.insert_device(device_tag=f"dev_{num_tag}")
    numeric_id = sdk.insert_measure(measure_tag=num_tag, freq=FREQ_NHZ, units="bpm")
    event_id = sdk.insert_measure(measure_tag=evt_tag, freq=FREQ_NHZ, units="string",
                                  signal_kind="event", value_type="string")

    start = int(BASE + numeric_span_s[0] * SEC)
    end = int(BASE + numeric_span_s[1] * SEC)
    times = np.arange(start, end, PERIOD_NS)
    values = np.sin(times.astype(np.float64) / 1e9).astype(np.float64)
    sdk.write_data_easy(numeric_id, device_id, times, values, FREQ_NHZ)
    return numeric_id, event_id, device_id


def write_events(sdk, m, d, pairs):
    """pairs: list of (offset_seconds, value_str). Writes at BASE + offset*SEC."""
    times = BASE + np.array([p[0] for p in pairs], dtype=np.int64) * SEC
    values = [p[1] for p in pairs]
    sdk.write_time_value_pairs(m, d, times, values, time_units="ns")


def _device_ranges(sources, device_id):
    """Collect all resolved (start, end) ranges for a device, whether they landed under
    the unmapped 'device_ids' key or were mapped into 'device_patient_tuples' (when a
    device_patient row covers them). Returns them sorted, relative seconds from BASE."""
    ranges = list(sources.get('device_ids', {}).get(device_id, []))
    for (dev_id, _pat_id), tuple_ranges in sources.get('device_patient_tuples', {}).items():
        if dev_id == device_id:
            ranges.extend(tuple_ranges)
    ranges = sorted(ranges)
    return [((s - BASE) / SEC, (e - BASE) / SEC) for s, e in ranges]


def validated_device_ranges(defn, sdk, device_id, **validate_kwargs):
    """Validate and return the (start, end) ranges the definition resolved for a device
    source (as relative seconds from BASE)."""
    defn.validate(sdk=sdk, **validate_kwargs)
    return _device_ranges(defn.validated_data_dict['sources'], device_id)


# --------------------------------------------------------------------------- #
# 1. anchor regions
# --------------------------------------------------------------------------- #
def test_anchor_emits_pre_post_window(sdk):
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 5 * SEC, "post": 5 * SEC}]})
    assert validated_device_ranges(defn, sdk, d) == [(45.0, 55.0)]


def test_anchor_multiple_occurrences_merge(sdk):
    numeric_id, event_id, d = setup_source(sdk)
    # Two anchors 2s apart with +/-3s windows -> [17,23] and [19,25] overlap -> [17,25].
    write_events(sdk, event_id, d, [(20, "MARK"), (22, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 3 * SEC, "post": 3 * SEC}]})
    assert validated_device_ranges(defn, sdk, d) == [(17.0, 25.0)]


def test_anchor_clipped_to_data_union(sdk):
    # Numeric data only spans 0..100s; an anchor near the end with a long post is clipped
    # to the data union (100s), not extended past it.
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    write_events(sdk, event_id, d, [(99, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 4 * SEC, "post": 20 * SEC}]})
    # [95, 119] intersect data union [0,100] -> [95, 100].
    assert validated_device_ranges(defn, sdk, d) == [(95.0, 100.0)]


def test_anchor_by_measure_id_accepted(sdk):
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(30, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": event_id,  # int id, not tag
                         "pre": 2 * SEC, "post": 2 * SEC}]})
    assert validated_device_ranges(defn, sdk, d) == [(28.0, 32.0)]


def test_anchor_no_occurrences_warns_and_no_ranges(sdk):
    numeric_id, event_id, d = setup_source(sdk)
    # "MARK" is in the vocabulary (written once) but not for the queried window/source
    # region... here we simply never write it near data: write it far outside the union.
    write_events(sdk, event_id, d, [(500, "MARK")])  # outside numeric union [0,100]
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 1 * SEC, "post": 1 * SEC}]})
    with pytest.warns(UserWarning, match="no occurrences"):
        ranges = validated_device_ranges(defn, sdk, d)
    assert ranges == []


# --------------------------------------------------------------------------- #
# 2. from/to regions
# --------------------------------------------------------------------------- #
def test_from_to_matches_event_intervals(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 60))
    write_events(sdk, event_id, d, [(10, "START"), (30, "STOP")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"from": "START", "to": "STOP", "measure": "evt",
                         "within": "none"}]})
    assert validated_device_ranges(defn, sdk, d) == [(10.0, 30.0)]


def test_from_to_pre_post_padding(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 60))
    write_events(sdk, event_id, d, [(10, "START"), (30, "STOP")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"from": "START", "to": "STOP", "measure": "evt",
                         "within": "none", "pre": 2 * SEC, "post": 3 * SEC}]})
    # [10,30] padded by pre=2/post=3 -> [8,33].
    assert validated_device_ranges(defn, sdk, d) == [(8.0, 33.0)]


def test_from_to_max_duration_caps_length(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 60))
    write_events(sdk, event_id, d, [(10, "START"), (30, "STOP")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"from": "START", "to": "STOP", "measure": "evt",
                         "within": "none", "max_duration": 5 * SEC}]})
    # [10,30] length 20 capped at 5 -> [10,15].
    assert validated_device_ranges(defn, sdk, d) == [(10.0, 15.0)]


# --------------------------------------------------------------------------- #
# 3. on_censored clip / drop / keep
# --------------------------------------------------------------------------- #
def _censored_defn(d, on_censored):
    return DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"from": "START", "to": "STOP", "measure": "evt",
                         "within": "none", "on_censored": on_censored}]})


def test_on_censored_keep(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 60))
    # Clean [10,20]; then START at 40 with no STOP -> right-censored to union end 60.
    write_events(sdk, event_id, d, [(10, "START"), (20, "STOP"), (40, "START")])
    ranges = validated_device_ranges(_censored_defn(d, "keep"), sdk, d)
    assert ranges == [(10.0, 20.0), (40.0, 60.0)]


def test_on_censored_drop(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 60))
    write_events(sdk, event_id, d, [(10, "START"), (20, "STOP"), (40, "START")])
    ranges = validated_device_ranges(_censored_defn(d, "drop"), sdk, d)
    # The censored [40,60] interval is dropped; only the clean pair remains.
    assert ranges == [(10.0, 20.0)]


def test_on_censored_clip_default_warns(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 60))
    write_events(sdk, event_id, d, [(10, "START"), (20, "STOP"), (40, "START")])
    # Default on_censored is clip -> keep the (already clipped) censored interval + warn.
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"from": "START", "to": "STOP", "measure": "evt",
                         "within": "none"}]})
    with pytest.warns(UserWarning, match="censored"):
        defn.validate(sdk=sdk)
    ranges = _device_ranges(defn.validated_data_dict['sources'], d)
    assert ranges == [(10.0, 20.0), (40.0, 60.0)]


# --------------------------------------------------------------------------- #
# 4. within scoping
# --------------------------------------------------------------------------- #
def test_anchor_within_device_patient_scopes(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    p = sdk.insert_patient(mrn="mrn_dp")
    # device_patient maps only [40s, 60s]; the anchor window is intersected with it.
    sdk.insert_device_patient_data([(d, p, int(BASE + 40 * SEC), int(BASE + 60 * SEC))])
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 20 * SEC, "post": 20 * SEC,
                         "within": "device_patient"}]})
    # [30,70] intersect device_patient [40,60] -> [40,60].
    assert validated_device_ranges(defn, sdk, d) == [(40.0, 60.0)]


def test_anchor_within_empty_device_patient_falls_back(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 5 * SEC, "post": 5 * SEC,
                         "within": "device_patient"}]})
    # Empty device_patient table -> within resolver warns + falls back to whole-stream,
    # so the window survives unscoped: [45,55].
    with pytest.warns(UserWarning):
        ranges = validated_device_ranges(defn, sdk, d)
    assert ranges == [(45.0, 55.0)]


# --------------------------------------------------------------------------- #
# 5. Validation errors (section 23.3)
# --------------------------------------------------------------------------- #
def test_unknown_event_measure_tag_raises(sdk):
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "not_a_measure",
                         "pre": 1 * SEC, "post": 1 * SEC}]})
    with pytest.raises(ValueError, match="No matching measures"):
        defn.validate(sdk=sdk)


def test_non_string_event_measure_raises(sdk):
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "hr",  # numeric measure
                         "pre": 1 * SEC, "post": 1 * SEC}]})
    with pytest.raises(ValueError, match="not 'string'"):
        defn.validate(sdk=sdk)


def test_out_of_vocab_anchor_raises(sdk):
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "NOPE", "measure": "evt",
                         "pre": 1 * SEC, "post": 1 * SEC}]})
    with pytest.raises(ValueError, match="not in the vocabulary"):
        defn.validate(sdk=sdk)


def test_out_of_vocab_from_to_raises(sdk):
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(10, "START"), (20, "STOP")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"from": "START", "to": "NOPE", "measure": "evt",
                         "within": "none"}]})
    with pytest.raises(ValueError, match="not in the string vocabulary"):
        defn.validate(sdk=sdk)


def test_missing_measure_key_raises_at_construction(sdk):
    # _check_times_and_warn (definition.py) requires 'measure' for an event region.
    with pytest.raises(ValueError, match="requires a 'measure'"):
        DatasetDefinition(
            measures=["hr"],
            device_ids={1: [{"anchor": "MARK", "pre": 1 * SEC, "post": 1 * SEC}]})


def test_from_without_to_raises_at_construction(sdk):
    with pytest.raises(ValueError, match="both 'from' and 'to'"):
        DatasetDefinition(
            measures=["hr"],
            device_ids={1: [{"from": "START", "measure": "evt"}]})


# --------------------------------------------------------------------------- #
# 6. Full resolution through a real iterator
# --------------------------------------------------------------------------- #
def test_anchor_region_resolves_through_iterator(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 5 * SEC, "post": 5 * SEC}]})
    defn.validate(sdk=sdk)

    window_dur = window_slide = 5 * SEC
    starts = []
    for window in sdk.get_iterator(defn, window_dur, window_slide, time_units="ns"):
        starts.append((window.start_time - BASE) / SEC)
    # Window range [45,55] with 5s windows -> starts at 45 and 50.
    assert sorted(starts) == [45.0, 50.0]


# --------------------------------------------------------------------------- #
# 7. Existing (non-event) definitions still resolve unchanged
# --------------------------------------------------------------------------- #
def test_classic_start_end_region_unchanged(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"start": int(BASE + 20 * SEC), "end": int(BASE + 40 * SEC)}]})
    assert validated_device_ranges(defn, sdk, d) == [(20.0, 40.0)]


def test_classic_time0_region_unchanged(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"time0": int(BASE + 50 * SEC),
                         "pre": 5 * SEC, "post": 5 * SEC}]})
    assert validated_device_ranges(defn, sdk, d) == [(45.0, 55.0)]
