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
Event query and from->to pairing tests.

Covers:
  * get_measure_string_vocabulary -- ALL values from the dict file (no data scan)
    and numeric-measure rejection.
  * get_string_values_present -- range/source-scoped distinct values.
  * get_event_intervals COLLAPSE pairing (section 22.2 #2): repeats collapse to one
    interval, back-to-back from/to, a from with no to -> right-censored, a to with no
    from -> left-censored.
  * the within cascade (section 22.2 #3): device_patient used when populated; the
    cascade helper falls back to encounter, then whole-stream when device_patient is
    empty (with a warning); a pair spanning a container boundary is split/clipped.
  * numeric-measure input rejected with a clear error.
  * a vectorized-vs-brute-force cross-check on a randomized event sequence.

SQLite only:
    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_event_intervals.py -q
"""
import shutil
import tempfile

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
    loc = tempfile.mkdtemp(prefix="atrium_event_intervals_")
    shutil.rmtree(loc, ignore_errors=True)
    s = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    s._loc = loc
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


def new_event_measure(sdk, tag="anesthesia_events"):
    m = sdk.insert_measure(measure_tag=tag, freq=1.0, freq_units="Hz", units="string",
                           signal_kind="event", value_type="string")
    d = sdk.insert_device(device_tag=f"dev_{tag}")
    return m, d


def write_events(sdk, m, d, pairs):
    """pairs: list of (offset_seconds, value_str). Writes at BASE + offset*SEC."""
    times = BASE + np.array([p[0] for p in pairs], dtype=np.int64) * SEC
    values = [p[1] for p in pairs]
    sdk.write_time_value_pairs(m, d, times, values, time_units="ns")


def rel(interval_n):
    """Interval dict -> (start_offset_s, end_offset_s, sc, ec) for easy assertions."""
    return (
        (interval_n["start_time_n"] - BASE) / SEC,
        (interval_n["end_time_n"] - BASE) / SEC,
        interval_n["start_censored"],
        interval_n["end_censored"],
    )


# --------------------------------------------------------------------------- #
# Brute-force reference for the cross-check (per-event Python loop; the SDK path
# is vectorized). Implements the SAME state model: collapse repeats, leading-to
# left-censored once, trailing-open right-censored.
# --------------------------------------------------------------------------- #
def brute_force_pairs(events, c0, c1, from_val, to_val):
    """events: list of (time_ns, value) sorted by time, all within [c0, c1)."""
    out = []
    is_open = False
    open_start = None
    seen_from = False
    left_done = False
    for t, v in events:
        if v == from_val:
            seen_from = True
            if not is_open:
                is_open = True
                open_start = t
        elif v == to_val:
            if is_open:
                out.append((open_start, t, False, False))
                is_open = False
            elif not seen_from and not left_done:
                out.append((c0, t, True, False))
                left_done = True
            # else: stray `to` while out -> no-op
    if is_open:
        out.append((open_start, c1, False, True))
    out.sort(key=lambda r: r[0])
    return out


# --------------------------------------------------------------------------- #
# 1. Vocabulary enumeration
# --------------------------------------------------------------------------- #
def test_vocabulary_all_values_from_dict_file(sdk):
    m, d = new_event_measure(sdk)
    # Write across two calls; all distinct strings must appear (no data scan).
    write_events(sdk, m, d, [(0, "START"), (1, "STOP"), (2, "PAUSE")])
    write_events(sdk, m, d, [(10, "RESUME"), (11, "START")])
    vocab = sdk.get_measure_string_vocabulary(m)
    # Order is code order (insertion order); every distinct value present exactly once.
    assert vocab == ["START", "STOP", "PAUSE", "RESUME"]


def test_vocabulary_rejects_numeric_measure(sdk):
    m = sdk.insert_measure(measure_tag="nibp", freq=1.0, freq_units="Hz", units="mmHg")
    d = sdk.insert_device(device_tag="dev_num")
    sdk.write_time_value_pairs(m, d, BASE + np.array([0, 1], dtype=np.int64) * SEC,
                               np.array([100.0, 110.0]))
    with pytest.raises(ValueError, match="not 'string'"):
        sdk.get_measure_string_vocabulary(m)


def test_string_values_present_range_scoped(sdk):
    m, d = new_event_measure(sdk)
    write_events(sdk, m, d, [(0, "START"), (2, "STOP"), (5, "PAUSE"), (20, "RESUME")])
    # Window [0, 10) sees START/STOP/PAUSE but not RESUME (at 20s); sorted distinct.
    present = sdk.get_string_values_present(
        m, start_time=int(BASE), end_time=int(BASE + 10 * SEC), device_id=d, time_units="ns")
    assert present == ["PAUSE", "START", "STOP"]
    # Full window includes RESUME.
    present_all = sdk.get_string_values_present(
        m, start_time=int(BASE), end_time=int(BASE + 30 * SEC), device_id=d, time_units="ns")
    assert present_all == ["PAUSE", "RESUME", "START", "STOP"]


# --------------------------------------------------------------------------- #
# 2. COLLAPSE pairing (section 22.2 #2)
# --------------------------------------------------------------------------- #
def test_pairing_repeats_collapse_to_one_interval(sdk):
    m, d = new_event_measure(sdk)
    # from, from, to -> ONE interval [first from, to]
    write_events(sdk, m, d, [(1, "START"), (3, "START"), (6, "STOP")])
    ivals = sdk.get_event_intervals(
        m, "START", "STOP", device_id=d, start_time=int(BASE),
        end_time=int(BASE + 10 * SEC), within="none", time_units="ns")
    assert [rel(i) for i in ivals] == [(1.0, 6.0, False, False)]


def test_pairing_back_to_back(sdk):
    m, d = new_event_measure(sdk)
    # START STOP START STOP -> two disjoint intervals
    write_events(sdk, m, d, [(1, "START"), (2, "STOP"), (5, "START"), (7, "STOP")])
    ivals = sdk.get_event_intervals(
        m, "START", "STOP", device_id=d, start_time=int(BASE),
        end_time=int(BASE + 10 * SEC), within="none", time_units="ns")
    assert [rel(i) for i in ivals] == [
        (1.0, 2.0, False, False),
        (5.0, 7.0, False, False),
    ]


def test_pairing_from_no_to_right_censored(sdk):
    m, d = new_event_measure(sdk)
    # trailing START with no STOP -> right-censored, clipped to range end (10s)
    write_events(sdk, m, d, [(1, "START"), (2, "STOP"), (8, "START")])
    ivals = sdk.get_event_intervals(
        m, "START", "STOP", device_id=d, start_time=int(BASE),
        end_time=int(BASE + 10 * SEC), within="none", time_units="ns")
    assert [rel(i) for i in ivals] == [
        (1.0, 2.0, False, False),
        (8.0, 10.0, False, True),  # end_censored, clipped to range end
    ]


def test_pairing_to_no_from_left_censored(sdk):
    m, d = new_event_measure(sdk)
    # leading STOP with no preceding START -> left-censored, clipped to range start (0s)
    write_events(sdk, m, d, [(3, "STOP"), (5, "START"), (7, "STOP")])
    ivals = sdk.get_event_intervals(
        m, "START", "STOP", device_id=d, start_time=int(BASE),
        end_time=int(BASE + 10 * SEC), within="none", time_units="ns")
    assert [rel(i) for i in ivals] == [
        (0.0, 3.0, True, False),   # start_censored, clipped to range start
        (5.0, 7.0, False, False),
    ]


def test_event_intervals_rejects_numeric_measure(sdk):
    m = sdk.insert_measure(measure_tag="hr", freq=1.0, freq_units="Hz", units="bpm")
    d = sdk.insert_device(device_tag="dev_hr")
    sdk.write_time_value_pairs(m, d, BASE + np.array([0, 1], dtype=np.int64) * SEC,
                               np.array([80.0, 82.0]))
    with pytest.raises(ValueError, match="not 'string'"):
        sdk.get_event_intervals(m, "START", "STOP", device_id=d,
                                start_time=int(BASE), end_time=int(BASE + 10 * SEC),
                                within="none", time_units="ns")


def test_event_intervals_rejects_unknown_value(sdk):
    m, d = new_event_measure(sdk)
    write_events(sdk, m, d, [(1, "START"), (2, "STOP")])
    with pytest.raises(ValueError, match="not in the string vocabulary"):
        sdk.get_event_intervals(m, "START", "NOPE", device_id=d,
                                start_time=int(BASE), end_time=int(BASE + 10 * SEC),
                                within="none", time_units="ns")


# --------------------------------------------------------------------------- #
# 3. within cascade (section 22.2 #3)
# --------------------------------------------------------------------------- #
def _make_patient_device(sdk, mrn="mrn_a"):
    p = sdk.insert_patient(mrn=mrn)
    return p


def test_within_device_patient_used_when_populated(sdk):
    m, d = new_event_measure(sdk)
    p = _make_patient_device(sdk)
    # device_patient maps the device to the patient for [2s, 8s]; the START at 1s
    # and STOP at 9s straddle that window, so the interval is clipped to [2s, 8s]
    # and BOTH ends are censored (open before the window, still open after it).
    sdk.insert_device_patient_data([(d, p, int(BASE + 2 * SEC), int(BASE + 8 * SEC))])
    write_events(sdk, m, d, [(1, "START"), (9, "STOP")])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # cascade must NOT warn when dp is populated
        ivals = sdk.get_event_intervals(
            m, "START", "STOP", device_id=d, start_time=int(BASE),
            end_time=int(BASE + 12 * SEC), within=None, time_units="ns")
    # START at 1s is outside dp window -> the window opens already inside the state
    # (left-censored from 2s); no STOP inside -> right-censored to 8s.
    assert [rel(i) for i in ivals] == [(2.0, 8.0, True, True)]


def test_within_cascade_falls_to_whole_stream_with_warning(sdk):
    m, d = new_event_measure(sdk)
    # No device_patient rows, no encounters -> cascade must fall to whole-stream and WARN.
    write_events(sdk, m, d, [(1, "START"), (6, "STOP")])
    with pytest.warns(UserWarning, match="whole-stream"):
        ivals = sdk.get_event_intervals(
            m, "START", "STOP", device_id=d, start_time=int(BASE),
            end_time=int(BASE + 10 * SEC), within=None, time_units="ns")
    assert [rel(i) for i in ivals] == [(1.0, 6.0, False, False)]


def test_within_cascade_helper_falls_to_encounter(sdk):
    """The cascade helper: empty device_patient but a populated encounter -> use the
    encounter windows and warn (device_patient empty). Exercised directly on the
    resolver so it does not depend on a patient data read (which itself needs a
    device_patient mapping)."""
    p = sdk.insert_patient(mrn="mrn_enc")
    inst = sdk.sql_handler.insert_institution(name="Inst")
    unit = sdk.sql_handler.insert_unit(institution_id=inst, name="Unit", unit_type="T")
    sdk.sql_handler.insert_bed(unit_id=unit, name="Bed1", bed_id=1)
    sdk.insert_encounter(start_time=int(BASE + 3 * SEC), end_time=int(BASE + 7 * SEC),
                         patient_id=p, bed_id=1, time_units="ns")
    with pytest.warns(UserWarning, match="encounter"):
        windows, label = sdk._resolve_within_windows(
            None, device_id=None, patient_id=p,
            start_n=int(BASE), end_n=int(BASE + 10 * SEC))
    assert label == "encounter"
    assert windows == [[int(BASE + 3 * SEC), int(BASE + 7 * SEC)]]


def test_within_forced_encounter_scopes_intervals(sdk):
    """Forced within='encounter' with a device source: the patient is resolved through
    device_patient, the encounter window clips the interval."""
    m, d = new_event_measure(sdk)
    p = sdk.insert_patient(mrn="mrn_fe")
    inst = sdk.sql_handler.insert_institution(name="Inst2")
    unit = sdk.sql_handler.insert_unit(institution_id=inst, name="Unit2", unit_type="T")
    sdk.sql_handler.insert_bed(unit_id=unit, name="Bed2", bed_id=2)
    sdk.insert_device_patient_data([(d, p, int(BASE), int(BASE + 12 * SEC))])
    sdk.insert_encounter(start_time=int(BASE + 2 * SEC), end_time=int(BASE + 8 * SEC),
                         patient_id=p, bed_id=2, time_units="ns")
    write_events(sdk, m, d, [(3, "START"), (6, "STOP"), (10, "START"), (11, "STOP")])
    ivals = sdk.get_event_intervals(
        m, "START", "STOP", device_id=d, start_time=int(BASE),
        end_time=int(BASE + 12 * SEC), within="encounter", time_units="ns")
    # Only the [3s,6s] pair falls in the encounter [2s,8s]; the [10s,11s] pair is outside.
    assert [rel(i) for i in ivals] == [(3.0, 6.0, False, False)]


def test_within_pair_spanning_container_boundary_is_split(sdk):
    """A from in one device_patient window and its to in the next window: the pair is
    split at the boundary -- right-censored in the first window, left-censored in the
    second -- rather than crossing it."""
    m, d = new_event_measure(sdk)
    p1 = sdk.insert_patient(mrn="mrn_s1")
    p2 = sdk.insert_patient(mrn="mrn_s2")
    # Two disjoint device_patient windows: [0s,5s] and [6s,12s].
    sdk.insert_device_patient_data([
        (d, p1, int(BASE), int(BASE + 5 * SEC)),
        (d, p2, int(BASE + 6 * SEC), int(BASE + 12 * SEC)),
    ])
    # START at 2s (window A), STOP at 9s (window B): no to in A, no from in B.
    write_events(sdk, m, d, [(2, "START"), (9, "STOP")])
    ivals = sdk.get_event_intervals(
        m, "START", "STOP", device_id=d, start_time=int(BASE),
        end_time=int(BASE + 12 * SEC), within=None, time_units="ns")
    got = sorted(rel(i) for i in ivals)
    assert got == [
        (2.0, 5.0, False, True),   # window A: right-censored at A's end (5s)
        (6.0, 9.0, True, False),   # window B: left-censored from B's start (6s)
    ]


def test_runs_with_empty_device_patient_table(sdk):
    """Sanity: the whole path runs with an empty device_patient table (device source,
    within=None) without raising -- it warns and uses whole-stream."""
    m, d = new_event_measure(sdk)
    write_events(sdk, m, d, [(1, "START"), (4, "STOP")])
    with pytest.warns(UserWarning):
        ivals = sdk.get_event_intervals(
            m, "START", "STOP", device_id=d, start_time=int(BASE),
            end_time=int(BASE + 10 * SEC), within=None, time_units="ns")
    assert len(ivals) == 1


# --------------------------------------------------------------------------- #
# 4. Vectorized path vs brute-force reference (randomized, fixed seed)
# --------------------------------------------------------------------------- #
def test_vectorized_matches_brute_force_random(sdk):
    rng = np.random.default_rng(20240804)
    vocab = ["START", "STOP", "OTHER"]
    for trial in range(25):
        m, d = new_event_measure(sdk, tag=f"rand_{trial}")
        n = int(rng.integers(1, 40))
        # Strictly increasing distinct offsets (1..n mapped from a random sample).
        offsets = np.sort(rng.choice(np.arange(1, 200), size=n, replace=False))
        vals = [vocab[int(i)] for i in rng.integers(0, len(vocab), size=n)]
        # Seed the vocabulary with a guaranteed START/STOP pair written OUTSIDE the
        # query range [0, 210)s (at 900/901s), so a random trial that happens to omit
        # one of them still has it in the vocabulary. These seeds are never read back
        # (out of range) so they do not affect pairing or the brute-force reference.
        write_events(sdk, m, d, [(900, "START"), (901, "STOP")])
        write_events(sdk, m, d, list(zip(offsets.tolist(), vals)))

        start_n = int(BASE)
        end_n = int(BASE + 210 * SEC)
        ivals = sdk.get_event_intervals(
            m, "START", "STOP", device_id=d, start_time=start_n, end_time=end_n,
            within="none", time_units="ns")
        got = [(i["start_time_n"], i["end_time_n"], i["start_censored"], i["end_censored"])
               for i in ivals]

        events = [(int(BASE + off * SEC), v) for off, v in zip(offsets.tolist(), vals)]
        expected = brute_force_pairs(events, start_n, end_n, "START", "STOP")
        assert got == expected, f"trial {trial}: {got} != {expected}"


# --------------------------------------------------------------------------- #
# 5. A measure TAG where a measure id belongs
# --------------------------------------------------------------------------- #
def test_measure_tag_instead_of_id_names_the_parameter_and_the_fix(sdk):
    """A measure tag passed where an id is required must raise a clear error:
    a bare `ValueError: invalid literal for int() with base 10` naming neither the
    parameter, the method, nor how to get from a tag to an id."""
    m, d = new_event_measure(sdk)
    write_events(sdk, m, d, [(1, "START"), (4, "STOP")])

    with pytest.raises(TypeError) as excinfo:
        sdk.get_event_intervals(
            "anesthesia_events", "START", "STOP", device_id=d, start_time=int(BASE),
            end_time=int(BASE + 10 * SEC), within="none", time_units="ns")

    message = str(excinfo.value)
    assert "measure must be a measure id" in message
    assert "'anesthesia_events'" in message
    assert "get_measure_id" in message


def test_measure_tag_rejected_by_the_other_string_query_entry_points(sdk):
    """Same diagnostic from every entry point onto the string/event surface, so a caller
    does not have to learn three different failure modes."""
    m, _d = new_event_measure(sdk, tag="vent_mode")

    with pytest.raises(TypeError, match="measure_id must be a measure id"):
        sdk.get_measure_string_vocabulary("vent_mode")

    with pytest.raises(TypeError, match="measure_id must be a measure id"):
        sdk.get_string_values_present("vent_mode", start_time=int(BASE),
                                      end_time=int(BASE + 10 * SEC), device_id=1)


def test_measure_id_as_a_numeric_string_is_still_rejected(sdk):
    """'3' is a tag-shaped argument that int() would have accepted silently. The
    parameter is typed as an id; a str is a caller mistake either way."""
    m, d = new_event_measure(sdk, tag="strnum")
    write_events(sdk, m, d, [(1, "START"), (4, "STOP")])

    with pytest.raises(TypeError, match="not the measure tag"):
        sdk.get_event_intervals(
            str(m), "START", "STOP", device_id=d, start_time=int(BASE),
            end_time=int(BASE + 10 * SEC), within="none", time_units="ns")


def test_non_int_non_str_measure_is_rejected_with_its_type(sdk):
    with pytest.raises(TypeError, match="measure must be a measure id"):
        sdk.get_event_intervals(
            None, "START", "STOP", device_id=1, start_time=int(BASE),
            end_time=int(BASE + 10 * SEC), within="none", time_units="ns")


def test_numpy_integer_measure_id_still_works(sdk):
    """The guard must not break the ordinary integer-like ids callers really pass."""
    m, d = new_event_measure(sdk, tag="npint")
    write_events(sdk, m, d, [(1, "START"), (4, "STOP")])

    ivals = sdk.get_event_intervals(
        np.int64(m), "START", "STOP", device_id=d, start_time=int(BASE),
        end_time=int(BASE + 10 * SEC), within="none", time_units="ns")
    assert [rel(i) for i in ivals] == [(1.0, 4.0, False, False)]
