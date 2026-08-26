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
Event query and from->to pairing edge-case tests.

Structure:
  A. Direct unit tests of the pure pairing engine ``AtriumSDK._pair_from_to`` and the
     container clip ``AtriumSDK._clip_intervals_to_containers`` with adversarial arrays.
  B. A brute-force reference state machine cross-checked against the vectorized public
     path over many random seeds.
  C. Property test: every censored edge lands on a REAL range/container boundary,
     never a fabricated value.
  D. ``within`` cascade integration (forced levels, patient source, empty
     device_patient table, multi-window union, boundary-spanning split).
  E. Censoring correctness (window fully inside an open state -> both ends censored).
  F. Vocabulary / validation (numeric rejection on all three methods, unknown value,
     empty vocabulary, range-scoped distinct + time_units).
  G. Source resolution & units.
  H. Expected-failure cases: same-timestamp from/to fabricates a
     phantom open interval and drops the observed close; get_string_values_present
     raises a cryptic TypeError (not a clear ValueError) for a missing range.

SQLite only:
    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_event_intervals_edge_cases.py -q
"""
import shutil
import tempfile
import warnings

import numpy as np
import pytest

from atriumdb import AtriumSDK

SEC = 1_000_000_000
BASE = 1_600_000_000 * SEC


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def sdk():
    loc = tempfile.mkdtemp(prefix="atrium_event_intervals_edges_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    s._loc = loc
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def new_event_measure(sdk, tag="evt"):
    m = sdk.insert_measure(measure_tag=tag, freq=1.0, freq_units="Hz", units="string",
                           signal_kind="event", value_type="string")
    d = sdk.insert_device(device_tag=f"dev_{tag}")
    return m, d


def write_events(sdk, m, d, pairs):
    """pairs: list of (offset_seconds, value_str). Writes at BASE + offset*SEC."""
    times = BASE + np.array([p[0] for p in pairs], dtype=np.int64) * SEC
    values = [p[1] for p in pairs]
    sdk.write_time_value_pairs(m, d, times, values, time_units="ns")


def rel(iv):
    return (
        (iv["start_time_n"] - BASE) / SEC,
        (iv["end_time_n"] - BASE) / SEC,
        iv["start_censored"],
        iv["end_censored"],
    )


def quiet_intervals(sdk, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sdk.get_event_intervals(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Brute-force reference (distinct-timestamp inputs).
# Collapse semantics: a run of `from`s until the next `to` is ONE interval
# (first-open -> first-close); a leading `to` before any `from` left-censors ONCE;
# a trailing open `from` right-censors; stray `to`s while closed are no-ops;
# values that are neither from nor to are ignored.
# --------------------------------------------------------------------------- #
def reference_pairs(events, c0, c1, from_val, to_val):
    """events: list of (time, value), STRICTLY increasing distinct times."""
    intervals = []
    open_start = None
    ever_opened = False
    left_censor_used = False
    for t, v in events:
        if v == from_val:
            ever_opened = True
            if open_start is None:
                open_start = t
        elif v == to_val:
            if open_start is not None:
                intervals.append((open_start, t, False, False))
                open_start = None
            elif not ever_opened and not left_censor_used:
                intervals.append((c0, t, True, False))
                left_censor_used = True
            # stray `to` while closed -> ignored
    if open_start is not None:
        intervals.append((open_start, c1, False, True))
    intervals.sort(key=lambda r: r[0])
    return intervals


# =========================================================================== #
# A. Pure pairing engine (adversarial arrays)
# =========================================================================== #
def arr(xs):
    return np.array(xs, dtype=np.int64)


def test_pair_empty_stream():
    assert AtriumSDK._pair_from_to(arr([]), arr([]), 0, 1000) == []


def test_pair_only_froms_collapse_to_one_right_censored():
    # every `from`, no `to` -> ONE right-censored interval opened at the first from.
    assert AtriumSDK._pair_from_to(arr([1, 2, 3]), arr([]), 0, 1000) == [
        (1, 1000, False, True)]


def test_pair_only_tos_one_leading_left_censor_rest_ignored():
    assert AtriumSDK._pair_from_to(arr([]), arr([1, 2, 3]), 0, 1000) == [
        (0, 1, True, False)]


def test_pair_leading_to_then_normal_pair():
    # to(50) before from(100); the leading to left-censors once, the trailing pair closes.
    assert AtriumSDK._pair_from_to(arr([100]), arr([50, 200]), 0, 1000) == [
        (0, 50, True, False), (100, 200, False, False)]


def test_pair_long_nested_run_collapses_to_first_from():
    froms = arr([10, 11, 12, 13, 14])
    tos = arr([20])
    assert AtriumSDK._pair_from_to(froms, tos, 0, 1000) == [(10, 20, False, False)]


def test_pair_alternating_runs():
    froms = arr([2, 3, 8])
    tos = arr([1, 4, 5])   # to(1) leads, to(4) closes run{2,3}, to(5) stray, from(8) open
    assert AtriumSDK._pair_from_to(froms, tos, 0, 1000) == [
        (0, 1, True, False), (2, 4, False, False), (8, 1000, False, True)]


def test_clip_splits_real_interval_across_container_gap_at_real_boundaries():
    # A single non-censored interval [10,50] intersected with two windows with a gap:
    # must split into two pieces, each censored at the REAL window boundary it was cut at.
    raw = [(10, 50, False, False)]
    windows = [[0, 20], [30, 60]]
    out = AtriumSDK._clip_intervals_to_containers(raw, windows)
    assert out == [(10, 20, False, True), (30, 50, True, False)]


def test_clip_interval_entirely_outside_container_is_dropped():
    raw = [(10, 20, False, False)]
    assert AtriumSDK._clip_intervals_to_containers(raw, [[0, 5]]) == []


# =========================================================================== #
# B. Vectorized public path vs independent brute force (many seeds)
# =========================================================================== #
@pytest.mark.parametrize("seed", [1, 7, 42, 100, 2024, 999999, 13, 271828])
def test_vectorized_matches_independent_reference(sdk, seed):
    rng = np.random.default_rng(seed)
    vocab = ["ON", "OFF", "NOISE"]  # NOISE is neither from nor to -> must be ignored
    for trial in range(12):
        m, d = new_event_measure(sdk, tag=f"s{seed}_{trial}")
        n = int(rng.integers(1, 45))
        # strictly increasing distinct offsets in [1, 400)
        offsets = np.sort(rng.choice(np.arange(1, 400), size=n, replace=False))
        vals = [vocab[int(i)] for i in rng.integers(0, len(vocab), size=n)]
        # Seed ON/OFF into the vocabulary OUT of the query range so code_for() never
        # rejects a trial that omitted one; these are never read back in range.
        write_events(sdk, m, d, [(900, "ON"), (901, "OFF")])
        write_events(sdk, m, d, list(zip(offsets.tolist(), vals)))

        start_n, end_n = int(BASE), int(BASE + 420 * SEC)
        ivals = quiet_intervals(
            sdk, m, "ON", "OFF", device_id=d, start_time=start_n, end_time=end_n,
            within="none", time_units="ns")
        got = [(i["start_time_n"], i["end_time_n"], i["start_censored"], i["end_censored"])
               for i in ivals]

        events = [(int(BASE + off * SEC), v) for off, v in zip(offsets.tolist(), vals)]
        expected = reference_pairs(events, start_n, end_n, "ON", "OFF")
        assert got == expected, f"seed {seed} trial {trial}: {got} != {expected}"


# =========================================================================== #
# C. Property: every censored edge lands on a REAL boundary (never fabricated)
# =========================================================================== #
@pytest.mark.parametrize("seed", [3, 17, 555, 90210])
def test_censored_edges_are_always_real_boundaries(sdk, seed):
    rng = np.random.default_rng(seed)
    vocab = ["ON", "OFF", "X"]
    for trial in range(15):
        m, d = new_event_measure(sdk, tag=f"c{seed}_{trial}")
        n = int(rng.integers(1, 30))
        offsets = np.sort(rng.choice(np.arange(1, 300), size=n, replace=False))
        vals = [vocab[int(i)] for i in rng.integers(0, len(vocab), size=n)]
        write_events(sdk, m, d, [(900, "ON"), (901, "OFF")])
        write_events(sdk, m, d, list(zip(offsets.tolist(), vals)))
        start_n, end_n = int(BASE), int(BASE + 320 * SEC)
        # within="none" -> the only container is the whole query range.
        ivals = quiet_intervals(
            sdk, m, "ON", "OFF", device_id=d, start_time=start_n, end_time=end_n,
            within="none", time_units="ns")
        boundaries = {start_n, end_n}
        for i in ivals:
            assert i["start_time_n"] < i["end_time_n"], "no zero/negative-length intervals"
            if i["start_censored"]:
                assert i["start_time_n"] in boundaries, \
                    f"fabricated censored START {i['start_time_n']} not in {boundaries}"
            if i["end_censored"]:
                assert i["end_time_n"] in boundaries, \
                    f"fabricated censored END {i['end_time_n']} not in {boundaries}"


# =========================================================================== #
# D. within cascade integration
# =========================================================================== #
def test_forced_none_ignores_populated_device_patient(sdk):
    m, d = new_event_measure(sdk)
    p = sdk.insert_patient(mrn="a")
    sdk.insert_device_patient_data([(d, p, int(BASE + 2 * SEC), int(BASE + 4 * SEC))])
    write_events(sdk, m, d, [(1, "ON"), (9, "OFF")])
    # within='none' must NOT warn and must NOT clip to the dp window.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ivals = sdk.get_event_intervals(
            m, "ON", "OFF", device_id=d, start_time=int(BASE),
            end_time=int(BASE + 12 * SEC), within="none", time_units="ns")
    assert [rel(i) for i in ivals] == [(1.0, 9.0, False, False)]


def test_forced_device_patient_returns_dp_windows(sdk):
    m, d = new_event_measure(sdk)
    p = sdk.insert_patient(mrn="b")
    sdk.insert_device_patient_data([(d, p, int(BASE + 2 * SEC), int(BASE + 8 * SEC))])
    write_events(sdk, m, d, [(1, "ON"), (9, "OFF")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # dp populated -> no warning
        ivals = sdk.get_event_intervals(
            m, "ON", "OFF", device_id=d, start_time=int(BASE),
            end_time=int(BASE + 12 * SEC), within="device_patient", time_units="ns")
    assert [rel(i) for i in ivals] == [(2.0, 8.0, True, True)]


def test_forced_device_patient_empty_warns_and_falls_to_whole_stream(sdk):
    m, d = new_event_measure(sdk)
    write_events(sdk, m, d, [(1, "ON"), (6, "OFF")])
    with pytest.warns(UserWarning, match="device_patient"):
        ivals = sdk.get_event_intervals(
            m, "ON", "OFF", device_id=d, start_time=int(BASE),
            end_time=int(BASE + 10 * SEC), within="device_patient", time_units="ns")
    assert [rel(i) for i in ivals] == [(1.0, 6.0, False, False)]


def test_forced_encounter_empty_warns_and_falls_to_whole_stream(sdk):
    m, d = new_event_measure(sdk)
    write_events(sdk, m, d, [(1, "ON"), (6, "OFF")])
    with pytest.warns(UserWarning, match="encounter"):
        ivals = sdk.get_event_intervals(
            m, "ON", "OFF", device_id=d, start_time=int(BASE),
            end_time=int(BASE + 10 * SEC), within="encounter", time_units="ns")
    assert [rel(i) for i in ivals] == [(1.0, 6.0, False, False)]


def test_unknown_within_raises_clear_error(sdk):
    m, d = new_event_measure(sdk)
    write_events(sdk, m, d, [(1, "ON"), (2, "OFF")])
    with pytest.raises(ValueError, match="Unknown within option"):
        sdk.get_event_intervals(
            m, "ON", "OFF", device_id=d, start_time=int(BASE),
            end_time=int(BASE + 10 * SEC), within="bogus", time_units="ns")


def test_patient_source_uses_device_patient_window(sdk):
    m, d = new_event_measure(sdk)
    p = sdk.insert_patient(mrn="pp")
    sdk.insert_device_patient_data([(d, p, int(BASE + 1 * SEC), int(BASE + 11 * SEC))])
    write_events(sdk, m, d, [(2, "ON"), (6, "OFF")])
    ivals = quiet_intervals(
        sdk, m, "ON", "OFF", patient_id=p, start_time=int(BASE),
        end_time=int(BASE + 12 * SEC), within=None, time_units="ns")
    assert [rel(i) for i in ivals] == [(2.0, 6.0, False, False)]


def test_adjacent_device_patient_windows_union_not_split(sdk):
    # Two dp rows that TOUCH at 5s must union into one window; a pair spanning 5s must
    # NOT be split at the (merged-away) internal boundary.
    m, d = new_event_measure(sdk)
    p1 = sdk.insert_patient(mrn="u1")
    p2 = sdk.insert_patient(mrn="u2")
    sdk.insert_device_patient_data([
        (d, p1, int(BASE), int(BASE + 5 * SEC)),
        (d, p2, int(BASE + 5 * SEC), int(BASE + 12 * SEC)),
    ])
    write_events(sdk, m, d, [(2, "ON"), (9, "OFF")])
    ivals = quiet_intervals(
        sdk, m, "ON", "OFF", device_id=d, start_time=int(BASE),
        end_time=int(BASE + 12 * SEC), within=None, time_units="ns")
    assert [rel(i) for i in ivals] == [(2.0, 9.0, False, False)]


def test_pair_spanning_container_gap_is_split_at_real_boundaries(sdk):
    m, d = new_event_measure(sdk)
    p1 = sdk.insert_patient(mrn="g1")
    p2 = sdk.insert_patient(mrn="g2")
    sdk.insert_device_patient_data([
        (d, p1, int(BASE), int(BASE + 5 * SEC)),
        (d, p2, int(BASE + 6 * SEC), int(BASE + 12 * SEC)),
    ])
    write_events(sdk, m, d, [(2, "ON"), (9, "OFF")])
    ivals = quiet_intervals(
        sdk, m, "ON", "OFF", device_id=d, start_time=int(BASE),
        end_time=int(BASE + 12 * SEC), within=None, time_units="ns")
    got = sorted(rel(i) for i in ivals)
    assert got == [(2.0, 5.0, False, True), (6.0, 9.0, True, False)]


def test_runs_with_empty_device_patient_table(sdk):
    m, d = new_event_measure(sdk)
    write_events(sdk, m, d, [(1, "ON"), (4, "OFF")])
    with pytest.warns(UserWarning):
        ivals = sdk.get_event_intervals(
            m, "ON", "OFF", device_id=d, start_time=int(BASE),
            end_time=int(BASE + 10 * SEC), within=None, time_units="ns")
    assert [rel(i) for i in ivals] == [(1.0, 4.0, False, False)]


# =========================================================================== #
# E. Censoring correctness: window fully inside an open state
# =========================================================================== #
def test_window_inside_open_state_both_ends_censored(sdk):
    # A single long open state [ON@1 .. OFF@1000]; a dp window [400,500] sits entirely
    # inside it with NO events of its own -> must be reported fully in-state with BOTH
    # ends censored by _collapse_event_intervals.
    m, d = new_event_measure(sdk)
    p = sdk.insert_patient(mrn="inside")
    sdk.insert_device_patient_data([(d, p, int(BASE + 400 * SEC), int(BASE + 500 * SEC))])
    write_events(sdk, m, d, [(1, "ON"), (1000, "OFF")])
    ivals = quiet_intervals(
        sdk, m, "ON", "OFF", device_id=d, start_time=int(BASE),
        end_time=int(BASE + 2000 * SEC), within=None, time_units="ns")
    assert [rel(i) for i in ivals] == [(400.0, 500.0, True, True)]


def test_window_inside_right_censored_state_both_ends_censored(sdk):
    # ON@1, never closes; dp window [400,500] inside -> both censored.
    m, d = new_event_measure(sdk)
    p = sdk.insert_patient(mrn="inside2")
    sdk.insert_device_patient_data([(d, p, int(BASE + 400 * SEC), int(BASE + 500 * SEC))])
    write_events(sdk, m, d, [(1, "ON"), (900, "OFF")])  # OFF@900 seeds vocab, outside window
    # Query range ends at 800s so OFF@900 is NOT read -> state is right-censored in range.
    ivals = quiet_intervals(
        sdk, m, "ON", "OFF", device_id=d, start_time=int(BASE),
        end_time=int(BASE + 800 * SEC), within=None, time_units="ns")
    assert [rel(i) for i in ivals] == [(400.0, 500.0, True, True)]


# =========================================================================== #
# F. Vocabulary / validation
# =========================================================================== #
def test_vocabulary_string_measure_never_written_is_empty(sdk):
    m = sdk.insert_measure(measure_tag="empty_str", freq=1.0, freq_units="Hz",
                           units="string", signal_kind="event", value_type="string")
    assert sdk.get_measure_string_vocabulary(m) == []


def test_vocabulary_rejects_numeric_measure(sdk):
    m = sdk.insert_measure(measure_tag="numv", freq=1.0, freq_units="Hz", units="mmHg")
    with pytest.raises(ValueError, match="not 'string'"):
        sdk.get_measure_string_vocabulary(m)


def test_values_present_rejects_numeric_measure(sdk):
    m = sdk.insert_measure(measure_tag="numv2", freq=1.0, freq_units="Hz", units="bpm")
    d = sdk.insert_device(device_tag="dnum")
    sdk.write_time_value_pairs(m, d, BASE + np.array([0, 1], dtype=np.int64) * SEC,
                               np.array([80.0, 82.0]))
    with pytest.raises(ValueError, match="not 'string'"):
        sdk.get_string_values_present(m, int(BASE), int(BASE + 5 * SEC), device_id=d)


def test_event_intervals_rejects_numeric_measure(sdk):
    m = sdk.insert_measure(measure_tag="numv3", freq=1.0, freq_units="Hz", units="bpm")
    d = sdk.insert_device(device_tag="dnum3")
    sdk.write_time_value_pairs(m, d, BASE + np.array([0, 1], dtype=np.int64) * SEC,
                               np.array([80.0, 82.0]))
    with pytest.raises(ValueError, match="not 'string'"):
        sdk.get_event_intervals(m, "ON", "OFF", device_id=d, start_time=int(BASE),
                                end_time=int(BASE + 5 * SEC), within="none")


def test_event_intervals_rejects_out_of_vocab_from_and_to(sdk):
    m, d = new_event_measure(sdk)
    write_events(sdk, m, d, [(1, "ON"), (2, "OFF")])
    with pytest.raises(ValueError, match="not in the string vocabulary"):
        sdk.get_event_intervals(m, "NOPE", "OFF", device_id=d, start_time=int(BASE),
                                end_time=int(BASE + 5 * SEC), within="none")
    with pytest.raises(ValueError, match="not in the string vocabulary"):
        sdk.get_event_intervals(m, "ON", "NOPE", device_id=d, start_time=int(BASE),
                                end_time=int(BASE + 5 * SEC), within="none")


def test_values_present_range_scoped_distinct_and_sorted(sdk):
    m, d = new_event_measure(sdk)
    write_events(sdk, m, d, [(0, "ON"), (2, "OFF"), (5, "PAUSE"), (20, "RESUME")])
    present = sdk.get_string_values_present(
        m, int(BASE), int(BASE + 10 * SEC), device_id=d, time_units="ns")
    assert present == ["OFF", "ON", "PAUSE"]  # sorted, distinct, RESUME@20 excluded
    allp = sdk.get_string_values_present(
        m, int(BASE), int(BASE + 30 * SEC), device_id=d, time_units="ns")
    assert allp == ["OFF", "ON", "PAUSE", "RESUME"]


def test_values_present_time_units_scales_only_input_range(sdk):
    m, d = new_event_measure(sdk)
    write_events(sdk, m, d, [(0, "ON"), (2, "OFF"), (20, "LATE")])
    # Range given in SECONDS; must scale to ns internally and exclude LATE@20s.
    present = sdk.get_string_values_present(
        m, 1_600_000_000, 1_600_000_010, device_id=d, time_units="s")
    assert present == ["OFF", "ON"]


def test_values_present_empty_range_returns_empty_list(sdk):
    m, d = new_event_measure(sdk)
    write_events(sdk, m, d, [(0, "ON"), (2, "OFF")])
    present = sdk.get_string_values_present(
        m, int(BASE + 100 * SEC), int(BASE + 200 * SEC), device_id=d, time_units="ns")
    assert present == []


# =========================================================================== #
# G. Source resolution & units
# =========================================================================== #
def test_source_resolution_device_tag_and_mrn(sdk):
    m, d = new_event_measure(sdk, tag="src")
    p = sdk.insert_patient(mrn="mrn_src")
    sdk.insert_device_patient_data([(d, p, int(BASE), int(BASE + 20 * SEC))])
    write_events(sdk, m, d, [(1, "ON"), (5, "OFF")])
    by_tag = quiet_intervals(
        sdk, m, "ON", "OFF", device_tag="dev_src", start_time=int(BASE),
        end_time=int(BASE + 10 * SEC), within="none", time_units="ns")
    assert [rel(i) for i in by_tag] == [(1.0, 5.0, False, False)]
    by_mrn = quiet_intervals(
        sdk, m, "ON", "OFF", mrn="mrn_src", start_time=int(BASE),
        end_time=int(BASE + 10 * SEC), within="none", time_units="ns")
    assert [rel(i) for i in by_mrn] == [(1.0, 5.0, False, False)]


def test_unknown_device_tag_raises(sdk):
    m, d = new_event_measure(sdk, tag="src2")
    write_events(sdk, m, d, [(1, "ON"), (5, "OFF")])
    with pytest.raises(ValueError, match="device_tag"):
        sdk.get_event_intervals(m, "ON", "OFF", device_tag="does_not_exist",
                                start_time=int(BASE), end_time=int(BASE + 10 * SEC),
                                within="none", time_units="ns")


def test_no_source_raises(sdk):
    m, d = new_event_measure(sdk, tag="src3")
    write_events(sdk, m, d, [(1, "ON"), (5, "OFF")])
    with pytest.raises(ValueError, match="data source is required"):
        sdk.get_event_intervals(m, "ON", "OFF", start_time=int(BASE),
                                end_time=int(BASE + 10 * SEC), within="none",
                                time_units="ns")


def test_time_units_scales_input_range_result_stays_ns(sdk):
    m, d = new_event_measure(sdk, tag="units")
    write_events(sdk, m, d, [(1, "ON"), (5, "OFF")])
    ivals = quiet_intervals(
        sdk, m, "ON", "OFF", device_id=d, start_time=1_600_000_000,
        end_time=1_600_000_010, within="none", time_units="s")
    assert ivals[0]["start_time_n"] == int(BASE + 1 * SEC)
    assert ivals[0]["end_time_n"] == int(BASE + 5 * SEC)


def test_measure_as_tuple_raises_not_crash(sdk):
    m, d = new_event_measure(sdk, tag="tup")
    write_events(sdk, m, d, [(1, "ON"), (5, "OFF")])
    with pytest.raises((TypeError, ValueError)):
        sdk.get_event_intervals((m,), "ON", "OFF", device_id=d, start_time=int(BASE),
                                end_time=int(BASE + 10 * SEC), within="none")


def test_start_end_required_get_event_intervals(sdk):
    m, d = new_event_measure(sdk, tag="req")
    write_events(sdk, m, d, [(1, "ON"), (5, "OFF")])
    with pytest.raises(ValueError, match="required"):
        sdk.get_event_intervals(m, "ON", "OFF", device_id=d, start_time=None,
                                end_time=int(BASE + 10 * SEC), within="none")
    with pytest.raises(ValueError, match="required"):
        sdk.get_event_intervals(m, "ON", "OFF", device_id=d, start_time=int(BASE),
                                end_time=None, within="none")


# =========================================================================== #
# H. Expected-failure cases and characterization of degenerate inputs
# =========================================================================== #
def test_same_timestamp_from_and_to_is_a_documented_precondition():
    # _pair_from_to assumes DISTINCT from/to timestamps. Storage guarantees this
    # (coincident values at one ns dedup to a single code, newest wins), so a `from`
    # and `to` can never share an exact ns via get_event_intervals -- it is a
    # documented precondition on the helper. Characterize the degenerate helper-only
    # behavior: a coincident `to` (side='right') does not close the `from`.
    out = AtriumSDK._pair_from_to(arr([100]), arr([100]), 0, 1000)
    assert out == [(100, 1000, False, True)]


def test_values_present_missing_range_should_raise_clear_error(sdk):
    m, d = new_event_measure(sdk, tag="reqp")
    write_events(sdk, m, d, [(1, "ON"), (5, "OFF")])
    with pytest.raises(ValueError):
        sdk.get_string_values_present(m, None, None, device_id=d)


def test_from_equals_to_degenerate_chaining_characterization():
    # CHARACTERIZATION of the low-level helper: given identical from/to timestamps the
    # engine chains consecutive events. The public method guards against
    # from_value == to_value (see test below), so this chaining is unreachable through
    # get_event_intervals; kept as a helper-level reference.
    out = AtriumSDK._pair_from_to(arr([10, 20, 30]), arr([10, 20, 30]), 0, 1000)
    assert out == [(10, 20, False, False), (20, 30, False, False), (30, 1000, False, True)]


def test_from_equals_to_rejected_by_public_method(sdk):
    # The public method rejects from_value == to_value with a clear error.
    m, d = new_event_measure(sdk, tag="eqguard")
    write_events(sdk, m, d, [(1, "ON"), (5, "OFF")])
    with pytest.raises(ValueError, match="must differ"):
        sdk.get_event_intervals(m, "ON", "ON", device_id=d, start_time=0, end_time=10)
