# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
"""
ADVERSARIAL / independent audit of Phase 5 event-anchored DatasetDefinition regions
(design section 23). This file was written by an auditor who did NOT write the feature
and does not trust it. It probes edges the writer's own suite
(test_event_anchored_definition_p5.py) does not, and pins observed behaviour so
regressions are caught.

Source under audit (uncommitted): sdk/atriumdb/windowing/verify_definition.py
(_resolve_event_region, _merge_windows) and sdk/atriumdb/windowing/definition.py
(_check_times_and_warn). NOTHING in sdk/atriumdb is modified here.

Tests grouped:
  A. _merge_windows unit tests (touching / nested / adjacent / reversed / dup)
  B. anchor edge cases (zero-width, exact-bounds, cross-source, gap, big pre/post)
  C. from/to edge cases (max_duration semantics, pre+max_duration, drop/keep parity)
  D. measure-reference hazards (int id, ambiguous tag, numeric-tag collision)
  E. shape / validation
  F. mixing event + classic regions; time_units interpretation
  G. classic-definition regression vs a direct get_interval_array expectation

Run (SQLite gate):
    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_event_anchored_definition_p5_audit.py -q
"""
import shutil
import tempfile
import warnings

import numpy as np
import pytest

from atriumdb import AtriumSDK, DatasetDefinition
from atriumdb.windowing.verify_definition import _merge_windows

SEC = 1_000_000_000
BASE = 1_600_000_000 * SEC
FREQ_NHZ = 1_000_000_000            # 1 Hz in nHz
PERIOD_NS = (10 ** 18) // FREQ_NHZ  # 1 s


# --------------------------------------------------------------------------- #
# Fixtures / helpers (mirrors the writer's harness so comparisons are apples-to-apples)
# --------------------------------------------------------------------------- #
@pytest.fixture
def sdk():
    loc = tempfile.mkdtemp(prefix="atrium_p5audit_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    s._loc = loc
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def setup_source(sdk, numeric_span_s=(0, 100), num_tag="hr", evt_tag="evt"):
    device_id = sdk.insert_device(device_tag=f"dev_{num_tag}_{evt_tag}")
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
    times = BASE + np.array([p[0] for p in pairs], dtype=np.int64) * SEC
    values = [p[1] for p in pairs]
    sdk.write_time_value_pairs(m, d, times, values, time_units="ns")


def _device_ranges(sources, device_id):
    ranges = list(sources.get('device_ids', {}).get(device_id, []))
    for (dev_id, _pat_id), tuple_ranges in sources.get('device_patient_tuples', {}).items():
        if dev_id == device_id:
            ranges.extend(tuple_ranges)
    ranges = sorted(ranges)
    return [((s - BASE) / SEC, (e - BASE) / SEC) for s, e in ranges]


def resolved(defn, sdk, device_id, **kw):
    defn.validate(sdk=sdk, **kw)
    return _device_ranges(defn.validated_data_dict['sources'], device_id)


# ========================================================================== #
# A. _merge_windows: unit-level correctness (no DB needed)
# ========================================================================== #
def test_merge_touching_are_merged():
    # touching [0,10],[10,20] -> single [0,20] (list_intersection wants disjoint spans)
    assert _merge_windows([[0, 10], [10, 20]]) == [[0, 20]]


def test_merge_overlapping():
    assert _merge_windows([[0, 10], [5, 15]]) == [[0, 15]]


def test_merge_nested_swallowed():
    # a fully-nested window must not shrink the outer one
    assert _merge_windows([[0, 20], [5, 10]]) == [[0, 20]]


def test_merge_adjacent_gap_kept_separate():
    assert _merge_windows([[0, 10], [11, 20]]) == [[0, 10], [11, 20]]


def test_merge_unsorted_input():
    assert _merge_windows([[19, 25], [17, 23]]) == [[17, 25]]


def test_merge_duplicate_windows():
    assert _merge_windows([[5, 9], [5, 9], [5, 9]]) == [[5, 9]]


def test_merge_empty():
    assert _merge_windows([]) == []


def test_merge_chain_with_one_gap():
    # [0,5],[5,10],[10,15] all touch -> [0,15]; [100,110] isolated
    assert _merge_windows([[10, 15], [0, 5], [5, 10], [100, 110]]) == [[0, 15], [100, 110]]


# ========================================================================== #
# B. anchor edge cases
# ========================================================================== #
def test_anchor_zero_width_pre_post_produces_no_range(sdk):
    # pre=0, post=0 -> [t, t]; the final `s < e` guard drops it. Document: zero-width
    # anchors silently contribute NO ranges (and, notably, no warning).
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt", "pre": 0, "post": 0}]})
    assert resolved(defn, sdk, d) == []


def test_anchor_pre_only_post_defaults_zero(sdk):
    # post omitted -> defaults to 0; window is [t-pre, t].
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt", "pre": 4 * SEC}]})
    assert resolved(defn, sdk, d) == [(46.0, 50.0)]


def test_anchor_occurrence_at_data_union_start(sdk):
    # Event exactly at the union start (t=0). No explicit global bounds, so clipping
    # relies on the data-union intersection: [-3,3] ∩ [0,100] -> [0,3].
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    write_events(sdk, event_id, d, [(0, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 3 * SEC, "post": 3 * SEC}]})
    assert resolved(defn, sdk, d) == [(0.0, 3.0)]


def test_anchor_three_overlapping_windows_merge_to_one(sdk):
    # 10,12,14 with +/-3 -> [7,13],[9,15],[11,17] chain-overlap -> [7,17].
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(10, "MARK"), (12, "MARK"), (14, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 3 * SEC, "post": 3 * SEC}]})
    assert resolved(defn, sdk, d) == [(7.0, 17.0)]


def test_anchor_non_overlapping_windows_stay_separate(sdk):
    # 10 and 80 with +/-3 -> two disjoint windows preserved.
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(10, "MARK"), (80, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 3 * SEC, "post": 3 * SEC}]})
    assert resolved(defn, sdk, d) == [(7.0, 13.0), (77.0, 83.0)]


def test_anchor_window_lands_in_data_gap_contributes_nothing(sdk):
    # Numeric data in [0,10] and [50,60]; an anchor in the [10,50] gap -> its window
    # intersects the union to nothing.
    device_id = sdk.insert_device(device_tag="dev_gap")
    numeric_id = sdk.insert_measure(measure_tag="hr", freq=FREQ_NHZ, units="bpm")
    event_id = sdk.insert_measure(measure_tag="evt", freq=FREQ_NHZ, units="string",
                                  signal_kind="event", value_type="string")
    for a, b in [(0, 10), (50, 60)]:
        t = np.arange(int(BASE + a * SEC), int(BASE + b * SEC), PERIOD_NS)
        sdk.write_data_easy(numeric_id, device_id, t,
                            np.ones(t.shape, dtype=np.float64), FREQ_NHZ)
    write_events(sdk, event_id, device_id, [(30, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={device_id: [{"anchor": "MARK", "measure": "evt",
                                 "pre": 2 * SEC, "post": 2 * SEC}]})
    # gap tolerance in validate defaults to 0 here; make it 0 so the two spans stay split.
    assert resolved(defn, sdk, device_id, gap_tolerance=0) == []


def test_anchor_value_written_only_to_other_device_warns(sdk):
    # "MARK" is in the (per-measure, global) vocabulary via device A, but device B's
    # stream has none -> validate() warns "no occurrences" for B and yields no ranges.
    numeric_a, event_a, da = setup_source(sdk, num_tag="hrA", evt_tag="evA")
    write_events(sdk, event_a, da, [(50, "MARK")])
    # second device, same event tag name would collide; use a shared measure instead:
    db = sdk.insert_device(device_tag="dev_B")
    tb = np.arange(int(BASE), int(BASE + 100 * SEC), PERIOD_NS)
    sdk.write_data_easy(numeric_a, db, tb, np.ones(tb.shape, np.float64), FREQ_NHZ)
    defn = DatasetDefinition(
        measures=["hrA"],
        device_ids={db: [{"anchor": "MARK", "measure": "evA",
                          "pre": 1 * SEC, "post": 1 * SEC}]})
    with pytest.warns(UserWarning, match="no occurrences"):
        assert resolved(defn, sdk, db) == []


def test_anchor_huge_pre_post_clipped_to_explicit_global_bounds(sdk):
    # Explicit global start/end (in ns) must clip an over-wide anchor window.
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 1000 * SEC, "post": 1000 * SEC}]})
    gstart = int(BASE + 30 * SEC)
    gend = int(BASE + 70 * SEC)
    assert resolved(defn, sdk, d, start_time=gstart, end_time=gend) == [(30.0, 70.0)]


# ========================================================================== #
# C. from/to edge cases
# ========================================================================== #
def test_from_to_max_duration_longer_is_noop(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 60))
    write_events(sdk, event_id, d, [(10, "START"), (30, "STOP")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"from": "START", "to": "STOP", "measure": "evt",
                         "within": "none", "max_duration": 100 * SEC}]})
    assert resolved(defn, sdk, d) == [(10.0, 30.0)]


def test_from_to_max_duration_measured_from_PADDED_start(sdk):
    # AUDIT NOTE: max_duration caps AFTER pre-padding, measured from the padded start.
    # [10,30] with pre=5 -> padded [5,30]; max_duration=10 -> [5,15], i.e. the cap
    # eats into the real event interval, not just the padding. Pinning observed behaviour.
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 60))
    write_events(sdk, event_id, d, [(10, "START"), (30, "STOP")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"from": "START", "to": "STOP", "measure": "evt",
                         "within": "none", "pre": 5 * SEC, "post": 0,
                         "max_duration": 10 * SEC}]})
    assert resolved(defn, sdk, d) == [(5.0, 15.0)]


def test_from_to_max_duration_zero_is_rejected(sdk):
    # max_duration=0 used to pass validation (it is not negative) and then
    # collapse every interval to zero width -> the whole region silently
    # dropped. It is now rejected when the definition is built.
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 60))
    write_events(sdk, event_id, d, [(10, "START"), (30, "STOP")])
    with pytest.raises(ValueError, match="max_duration"):
        DatasetDefinition(
            measures=["hr"],
            device_ids={d: [{"from": "START", "to": "STOP", "measure": "evt",
                             "within": "none", "max_duration": 0}]})


def test_from_to_left_censored_start(sdk):
    # A STOP before any START (span opened already inside the state) -> left-censored,
    # clipped to the range start. clip(default) keeps [0,20] and warns.
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 60))
    write_events(sdk, event_id, d, [(20, "STOP"), (40, "START"), (50, "STOP")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"from": "START", "to": "STOP", "measure": "evt",
                         "within": "none"}]})
    with pytest.warns(UserWarning, match="censored"):
        defn.validate(sdk=sdk)
    assert _device_ranges(defn.validated_data_dict['sources'], d) == [(0.0, 20.0), (40.0, 50.0)]


def test_from_to_keep_equals_clip_for_ranges(sdk):
    # AUDIT NOTE: get_event_intervals ALREADY clips censored ends to the container, so
    # on_censored='keep' yields identical RANGES to 'clip' (only the warning differs).
    # 'keep' does not recover an un-clipped boundary because none is available.
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 60))
    write_events(sdk, event_id, d, [(10, "START"), (20, "STOP"), (40, "START")])
    keep = DatasetDefinition(measures=["hr"], device_ids={d: [
        {"from": "START", "to": "STOP", "measure": "evt", "within": "none",
         "on_censored": "keep"}]})
    clip = DatasetDefinition(measures=["hr"], device_ids={d: [
        {"from": "START", "to": "STOP", "measure": "evt", "within": "none",
         "on_censored": "clip"}]})
    keep_ranges = resolved(keep, sdk, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clip_ranges = resolved(clip, sdk, d)
    assert keep_ranges == clip_ranges == [(10.0, 20.0), (40.0, 60.0)]


def test_from_to_no_intervals_warns_no_ranges(sdk):
    # START/STOP are in the vocabulary, but the GLOBAL bounds passed to validate exclude
    # every event -> get_event_intervals returns [] -> warns "no ... intervals" and the
    # region contributes nothing. (numeric union still exists in [70,90], so the source
    # is not dropped earlier.)
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    write_events(sdk, event_id, d, [(10, "START"), (30, "STOP")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"from": "START", "to": "STOP", "measure": "evt",
                         "within": "none"}]})
    with pytest.warns(UserWarning, match="no .*intervals"):
        out = resolved(defn, sdk, d,
                       start_time=int(BASE + 70 * SEC), end_time=int(BASE + 90 * SEC))
    assert out == []


# ========================================================================== #
# D. measure-reference hazards
# ========================================================================== #
def test_anchor_by_int_measure_id(sdk):
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(30, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": event_id,
                         "pre": 2 * SEC, "post": 2 * SEC}]})
    assert resolved(defn, sdk, d) == [(28.0, 32.0)]


def test_unknown_int_measure_id_raises(sdk):
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(30, "MARK")])
    # a bogus int id is passed straight through get_measure_id_from_generic_measure
    # (no existence check for ints); _require_string_measure then rejects it.
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": 999999,
                         "pre": 1 * SEC, "post": 1 * SEC}]})
    with pytest.raises(ValueError, match="does not exist"):
        defn.validate(sdk=sdk)


def test_ambiguous_tag_both_string_resolves(sdk):
    # Two STRING measures share tag "evt" (different units). Events with the anchor
    # value are written to BOTH, so whichever "best" picks (by block count DESC) finds
    # occurrences and resolves.
    d = sdk.insert_device(device_tag="dev_amb")
    numeric_id = sdk.insert_measure(measure_tag="hr", freq=FREQ_NHZ, units="bpm")
    t = np.arange(int(BASE), int(BASE + 100 * SEC), PERIOD_NS)
    sdk.write_data_easy(numeric_id, d, t, np.ones(t.shape, np.float64), FREQ_NHZ)
    e1 = sdk.insert_measure(measure_tag="evt", freq=FREQ_NHZ, units="u1",
                            signal_kind="event", value_type="string")
    e2 = sdk.insert_measure(measure_tag="evt", freq=FREQ_NHZ, units="u2",
                            signal_kind="event", value_type="string")
    write_events(sdk, e1, d, [(50, "MARK")])
    write_events(sdk, e2, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 5 * SEC, "post": 5 * SEC}]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert resolved(defn, sdk, d) == [(45.0, 55.0)]


def test_tag_collision_numeric_and_string_prefers_string(sdk):
    # A numeric measure and a string measure share the tag "sig". Event-region
    # resolution now PREFERS the string measure (P5 fix): the numeric measure no longer
    # shadows it via the value_type-blind "best" rule, so the anchor resolves correctly
    # regardless of which id "best" would pick by block count.
    d = sdk.insert_device(device_tag="dev_coll")
    num = sdk.insert_measure(measure_tag="sig", freq=FREQ_NHZ, units="bpm")
    strm = sdk.insert_measure(measure_tag="sig", freq=FREQ_NHZ, units="string",
                              signal_kind="event", value_type="string")
    t = np.arange(int(BASE), int(BASE + 100 * SEC), PERIOD_NS)
    sdk.write_data_easy(num, d, t, np.ones(t.shape, np.float64), FREQ_NHZ)
    write_events(sdk, strm, d, [(50, "MARK")])

    defn = DatasetDefinition(
        measures=[{"tag": "sig", "freq_hz": 1.0, "units": "bpm"}],
        device_ids={d: [{"anchor": "MARK", "measure": "sig",
                         "pre": 5 * SEC, "post": 5 * SEC}]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert resolved(defn, sdk, d) == [(45.0, 55.0)]


# ========================================================================== #
# E. shape / validation (construction-time, section 23.3)
# ========================================================================== #
def test_anchor_and_from_together_rejected():
    with pytest.raises(ValueError, match="cannot combine 'anchor'"):
        DatasetDefinition(
            measures=["hr"],
            device_ids={1: [{"anchor": "M", "from": "A", "to": "B", "measure": "evt"}]})


def test_to_without_from_rejected():
    with pytest.raises(ValueError, match="both 'from' and 'to'"):
        DatasetDefinition(
            measures=["hr"],
            device_ids={1: [{"to": "STOP", "measure": "evt"}]})


def test_negative_pre_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        DatasetDefinition(
            measures=["hr"],
            device_ids={1: [{"anchor": "M", "measure": "evt", "pre": -1, "post": 1}]})


def test_negative_max_duration_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        DatasetDefinition(
            measures=["hr"],
            device_ids={1: [{"from": "A", "to": "B", "measure": "evt",
                             "max_duration": -5}]})


def test_bogus_on_censored_rejected_at_construction():
    with pytest.raises(ValueError, match="on_censored must be one of"):
        DatasetDefinition(
            measures=["hr"],
            device_ids={1: [{"from": "A", "to": "B", "measure": "evt",
                             "on_censored": "explode"}]})


def test_pre_post_without_anchor_rejected():
    # pre/post with neither time0 nor an event anchor is rejected.
    with pytest.raises(ValueError, match="cannot be provided without"):
        DatasetDefinition(
            measures=["hr"],
            device_ids={1: [{"pre": 1 * SEC, "post": 1 * SEC}]})


def test_from_to_missing_measure_rejected():
    with pytest.raises(ValueError, match="requires a 'measure'"):
        DatasetDefinition(
            measures=["hr"],
            device_ids={1: [{"from": "A", "to": "B"}]})


def test_unknown_within_with_occurrences_raises(sdk):
    # A bogus `within` on an anchor region with real occurrences must surface an error
    # (via _resolve_within_windows). NOTE: it is NOT validated at construction time.
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt", "pre": 1 * SEC,
                         "post": 1 * SEC, "within": "bogus_container"}]})
    with pytest.raises(ValueError, match="Unknown within option"):
        defn.validate(sdk=sdk)


def test_unknown_within_raises_even_with_no_occurrences(sdk):
    # P5 fix: `within` is validated up front, so a bogus value raises deterministically
    # even when the source has zero occurrences (previously the anchor path returned
    # early before checking `within`, letting the bad value silently escape).
    numeric_id, event_id, d = setup_source(sdk)
    write_events(sdk, event_id, d, [(500, "MARK")])  # outside data union
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt", "pre": 1 * SEC,
                         "post": 1 * SEC, "within": "bogus_container"}]})
    with pytest.raises(ValueError, match="Unknown within option"):
        defn.validate(sdk=sdk)


# ========================================================================== #
# F. mixing regions & time-unit interpretation
# ========================================================================== #
def test_event_and_classic_region_in_same_list(sdk):
    # One source list carrying BOTH an event region and a classic {start,end} region;
    # each resolves independently and both survive.
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [
            {"anchor": "MARK", "measure": "evt", "pre": 2 * SEC, "post": 2 * SEC},
            {"start": int(BASE + 80 * SEC), "end": int(BASE + 90 * SEC)},
        ]})
    assert resolved(defn, sdk, d) == [(48.0, 52.0), (80.0, 90.0)]


def test_region_pre_post_are_NOT_scaled_by_validate_time_units(sdk):
    # AUDIT NOTE: validate(time_units="s") scales the GLOBAL start/end only. Region
    # pre/post/max_duration are always raw nanoseconds (same as classic time0). Here
    # pre/post are given as whole nanoseconds (5*SEC) and stay 5 s regardless of the
    # time_units passed to validate -- i.e. units do NOT apply to region fields.
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    write_events(sdk, event_id, d, [(50, "MARK")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"anchor": "MARK", "measure": "evt",
                         "pre": 5 * SEC, "post": 5 * SEC}]})
    # global window in SECONDS: 0..100 s
    gstart_s = int(BASE // SEC)
    gend_s = int(BASE // SEC) + 100
    out = resolved(defn, sdk, d, start_time=gstart_s, end_time=gend_s, time_units="s")
    assert out == [(45.0, 55.0)]


# ========================================================================== #
# G. classic-definition regression (must be byte-identical to pre-P5 behaviour)
# ========================================================================== #
def _expected_all_ranges_from_interval_array(sdk, numeric_id, device_id):
    ia = sdk.get_interval_array(numeric_id, device_id=device_id, gap_tolerance_nano=0)
    return [((int(s) - BASE) / SEC, (int(e) - BASE) / SEC) for s, e in ia]


def test_classic_all_matches_interval_array(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    defn = DatasetDefinition(measures=["hr"], device_ids={d: "all"})
    got = resolved(defn, sdk, d, gap_tolerance=0)
    expected = _expected_all_ranges_from_interval_array(sdk, numeric_id, d)
    assert got == expected


def test_classic_start_end_unchanged(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"start": int(BASE + 20 * SEC), "end": int(BASE + 40 * SEC)}]})
    assert resolved(defn, sdk, d) == [(20.0, 40.0)]


def test_classic_time0_unchanged(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"time0": int(BASE + 50 * SEC), "pre": 5 * SEC, "post": 5 * SEC}]})
    assert resolved(defn, sdk, d) == [(45.0, 55.0)]


def test_classic_open_start_and_open_end(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    open_start = DatasetDefinition(
        measures=["hr"], device_ids={d: [{"start": int(BASE + 70 * SEC)}]})
    assert resolved(open_start, sdk, d) == [(70.0, 100.0)]
    open_end = DatasetDefinition(
        measures=["hr"], device_ids={d: [{"end": int(BASE + 30 * SEC)}]})
    assert resolved(open_end, sdk, d) == [(0.0, 30.0)]


# ========================================================================== #
# H. end-to-end iterator (from/to drives real windows)
# ========================================================================== #
def test_from_to_region_drives_iterator(sdk):
    numeric_id, event_id, d = setup_source(sdk, numeric_span_s=(0, 100))
    write_events(sdk, event_id, d, [(20, "START"), (40, "STOP")])
    defn = DatasetDefinition(
        measures=["hr"],
        device_ids={d: [{"from": "START", "to": "STOP", "measure": "evt",
                         "within": "none"}]})
    defn.validate(sdk=sdk)
    dur = slide = 5 * SEC
    starts = sorted((w.start_time - BASE) / SEC
                    for w in sdk.get_iterator(defn, dur, slide, time_units="ns"))
    # [20,40] with 5s windows -> 20,25,30,35.
    assert starts == [20.0, 25.0, 30.0, 35.0]
