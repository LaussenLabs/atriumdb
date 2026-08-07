# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
"""
Regression guards for the Wave-2 write / storage / transfer fixes on the
aperiodic + text feature. One minimal test per defect that was actually fixed;
every test here must PASS.

  W1  a string write that fails AFTER the dictionary encode must not establish a
      value_type, must not retain the rejected batch's free text, and must leave
      the in-process cache agreeing with a fresh SDK.
  W2  buffered string writes of differing unicode width must round-trip, and one
      failing batch must not discard the other measures in the same buffer.
  W3  a de-identified transfer must not ship free-text vocabularies verbatim,
      with a documented opt-in for controlled vocabularies.
  W4  a string/numeric identity collision must abort BEFORE anything is written
      to the destination, naming the source measure.
  W6  insert_measure(freq=0) must be rejected before any row is written, leaving
      the dataset openable.
  W7  insert_measure / transfer must apply signal_kind to an existing measure,
      and set_measure_kind must exist to repair one.
  W8  a truncated dictionary must raise an error naming the measure and a remedy.
  W12 write_data_easy must reject strings with actionable advice.
  D6  'waveform' + 'string' -- the one combination the design forbids -- must be
      repaired where the measure is classified, not left to fail inside get_iterator.

SQLite only (no backend difference is at stake in any of these):
    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_aperiodic_write_fixes.py -q
"""
import json
import logging
import shutil
import tempfile

import numpy as np
import pytest

from atriumdb import AtriumSDK, DatasetDefinition
from atriumdb.string_dictionary import MeasureStringDictionary
from atriumdb.transfer.adb.dataset import transfer_data, REDACTED_STRING_VALUE

SEC = 1_000_000_000
BASE = 1_600_000_000 * SEC

STR = lambda values: np.array(values, dtype=object)
NUM = lambda values: np.array(values, dtype=np.float64)


def _new_dataset(prefix):
    loc = tempfile.mkdtemp(prefix=f"atrium_apwrite_{prefix}_")
    shutil.rmtree(loc, ignore_errors=True)
    sdk = AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite")
    sdk._loc = loc
    return sdk, loc


@pytest.fixture
def sdk():
    s, loc = _new_dataset("main")
    try:
        yield s
    finally:
        s.close()
        shutil.rmtree(loc, ignore_errors=True)


@pytest.fixture
def sdk_pair():
    """A (source, destination) dataset pair for the transfer tests."""
    src, src_loc = _new_dataset("src")
    dst, dst_loc = _new_dataset("dst")
    try:
        yield src, dst
    finally:
        src.close()
        dst.close()
        shutil.rmtree(src_loc, ignore_errors=True)
        shutil.rmtree(dst_loc, ignore_errors=True)


def _times(n, start=BASE, step=SEC):
    return start + np.arange(n, dtype=np.int64) * step


def _string_measure(sdk, tag="alarm", kind="event"):
    return sdk.insert_measure(measure_tag=tag, freq=1, freq_units="Hz", units="string",
                              signal_kind=kind, value_type="string")


def _numeric_measure(sdk, tag="hr", kind="sample"):
    return sdk.insert_measure(measure_tag=tag, freq=1, freq_units="Hz", units="bpm",
                              signal_kind=kind)


def _fail_string_write_after_encode(sdk, measure_id, device_id, text="PATIENT_JOHN_DOE"):
    """A string write that gets PAST the dictionary encode and then fails: the
    time/value length mismatch is caught in _sort_write_data, well after encode."""
    with pytest.raises(ValueError, match="equal size"):
        sdk.write_data(measure_id, device_id, _times(3), STR([text, "B"]),
                       period_ns=SEC, time_0=int(BASE), raw_time_type=1)


# =========================================================================== #
# W1 -- a failed string write must establish and persist nothing
# =========================================================================== #
def test_failed_string_write_leaves_measure_numeric_and_writable(sdk):
    """The measure stays numeric-capable, the dictionary keeps no orphan free text,
    and the in-process view matches a fresh SDK (no self-contradiction)."""
    measure_id = _numeric_measure(sdk, "poison")
    device_id = sdk.insert_device(device_tag="poison_dev")

    _fail_string_write_after_encode(sdk, measure_id, device_id)

    # Nothing was established, on either channel.
    assert sdk.sql_handler.select_measure(measure_id=measure_id)[11] is None
    assert sdk._established_value_type(measure_id) is None

    # The rejected batch's free text is not on disk.
    path = MeasureStringDictionary.path_for(sdk._meta_dir, measure_id)
    on_disk = [] if not path.is_file() else \
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert on_disk == []

    # The cached view and a fresh SDK agree...
    fresh = AtriumSDK(dataset_location=sdk._loc, metadata_connection_type="sqlite")
    try:
        assert sdk.get_measure_kind(measure_id) == fresh.get_measure_kind(measure_id)
    finally:
        fresh.close()

    # ... and the measure still takes the numeric data it was created for.
    sdk.write_time_value_pairs(measure_id, device_id, _times(3), NUM([1.0, 2.0, 3.0]))
    _, read_times, _ = sdk.get_data(measure_id, BASE - SEC, BASE + 10 * SEC, device_id=device_id)
    assert read_times.size == 3


def test_successful_string_write_still_establishes_string(sdk):
    """Do-no-harm counterpart: the rollback must not weaken the normal path."""
    measure_id = sdk.insert_measure(measure_tag="alarm", freq=1, freq_units="Hz", units="string")
    device_id = sdk.insert_device(device_tag="alarm_dev")

    sdk.write_time_value_pairs(measure_id, device_id, _times(2), STR(["ASYSTOLE", "VTACH"]))

    assert sdk._established_value_type(measure_id) == "string"
    assert sdk.get_measure_kind(measure_id)[1] == "string"
    assert sdk.get_measure_string_vocabulary(measure_id) == ["ASYSTOLE", "VTACH"]
    with pytest.raises(ValueError, match="cannot write 'numeric' values"):
        sdk.write_time_value_pairs(measure_id, device_id, _times(2, start=BASE + 100 * SEC),
                                   NUM([1.0, 2.0]))


# =========================================================================== #
# W2 -- buffered string writes
# =========================================================================== #
def test_buffered_string_writes_of_differing_widths_round_trip(sdk):
    """'OK' ('<U2') and 'ASYSTOLE' ('<U8') are one data type, not two."""
    measure_id = _string_measure(sdk, "buf_width")
    device_id = sdk.insert_device(device_tag="buf_width_dev")

    with sdk.write_buffer():
        sdk.write_time_value_pairs(measure_id, device_id, np.array([BASE], dtype=np.int64), ["OK"])
        sdk.write_time_value_pairs(measure_id, device_id, np.array([BASE + SEC], dtype=np.int64),
                                   ["ASYSTOLE"])

    _, values = sdk.get_string_data(measure_id, BASE - SEC, BASE + 10 * SEC, device_id=device_id)
    assert sorted(map(str, values)) == ["ASYSTOLE", "OK"]


def test_buffered_failure_does_not_discard_other_measures(sdk):
    """A batch the write path rejects must not take unrelated buffered data with it."""
    string_id = _string_measure(sdk, "blast_str")
    numeric_id = _numeric_measure(sdk, "blast_num")
    device_id = sdk.insert_device(device_tag="blast_dev")

    with pytest.raises(ValueError):
        with sdk.write_buffer():
            # Mixing numeric values into a string measure is a genuine error.
            sdk.write_time_value_pairs(string_id, device_id, np.array([BASE], dtype=np.int64),
                                       STR(["OK"]))
            sdk.write_time_value_pairs(string_id, device_id,
                                       np.array([BASE + SEC], dtype=np.int64), NUM([1.0]))
            sdk.write_time_value_pairs(numeric_id, device_id, _times(5),
                                       NUM([1.0, 2.0, 3.0, 4.0, 5.0]))

    _, read_times, _ = sdk.get_data(numeric_id, BASE - SEC, BASE + 10 * SEC, device_id=device_id)
    assert read_times.size == 5, "an unrelated measure's buffered batch was discarded"
    # The buffer detached even though the flush raised.
    assert sdk._active_buffer is None


# =========================================================================== #
# W6 / W11 -- insert_measure frequency validation
# =========================================================================== #
def test_insert_measure_freq_zero_rejected_and_dataset_stays_openable():
    sdk, loc = _new_dataset("freq0")
    try:
        with pytest.raises(ValueError, match="signal_kind"):
            sdk.insert_measure(measure_tag="ap0", freq=0, freq_units="Hz", units="string")
        assert sdk.sql_handler.select_all_measures() == [], "a freq_nhz=0 row was committed"
        sdk.close()
        # The dataset must still open: get_all_measures divides by freq_nhz.
        AtriumSDK(dataset_location=loc, metadata_connection_type="sqlite").close()
    finally:
        shutil.rmtree(loc, ignore_errors=True)


def test_insert_measure_negative_freq_rejected_leaves_no_row(sdk):
    with pytest.raises(ValueError):
        sdk.insert_measure(measure_tag="neg", freq=-1, freq_units="Hz", units="u")
    assert sdk.sql_handler.select_all_measures() == []


def test_aperiodic_measure_has_a_working_supported_route(sdk):
    """The route the freq=0 error points at actually works end to end."""
    measure_id = sdk.insert_measure(measure_tag="ap_ok", freq=1, freq_units="Hz", units="string",
                                    signal_kind="event", value_type="string")
    device_id = sdk.insert_device(device_tag="ap_ok_dev")
    irregular = np.array([BASE, BASE + 7 * SEC, BASE + 300 * SEC], dtype=np.int64)

    sdk.write_time_value_pairs(measure_id, device_id, irregular, STR(["A", "B", "A"]))

    times, values = sdk.get_string_data(measure_id, int(BASE), int(BASE) + 400 * SEC,
                                        device_id=device_id)
    assert times.tolist() == irregular.tolist()
    assert list(map(str, values)) == ["A", "B", "A"]
    assert sdk.get_measure_kind(measure_id) == ("event", "string")


# =========================================================================== #
# W7 -- signal_kind on an existing measure + the public setter
# =========================================================================== #
def test_insert_measure_applies_kind_to_an_existing_measure(sdk):
    first = sdk.insert_measure(measure_tag="vent", freq=1, freq_units="Hz", units="string")
    again = sdk.insert_measure(measure_tag="vent", freq=1, freq_units="Hz", units="string",
                               signal_kind="state", value_type="string")
    assert again == first
    assert sdk.get_measure_kind(again) == ("state", "string")


def test_set_measure_kind_repairs_and_refuses_to_relabel_written_data(sdk):
    measure_id = sdk.insert_measure(measure_tag="mode", freq=1, freq_units="Hz", units="string")
    device_id = sdk.insert_device(device_tag="mode_dev")

    assert sdk.set_measure_kind(measure_id, signal_kind="state") == ("state", "numeric")

    sdk.write_time_value_pairs(measure_id, device_id, _times(2), STR(["SIMV", "PRVC"]))
    assert sdk.get_measure_kind(measure_id) == ("state", "string")

    # Relabelling data that already exists is refused, not silently applied.
    with pytest.raises(ValueError, match="already holds 'string' data"):
        sdk.set_measure_kind(measure_id, value_type="numeric")
    with pytest.raises(ValueError):
        sdk.set_measure_kind(measure_id, signal_kind="not_a_kind")


# =========================================================================== #
# W3 / W4 / W7 -- transfer
# =========================================================================== #
def _seed_transfer_source(src, string_values, string_tag="note", with_numeric=False):
    measure_id = _string_measure(src, string_tag)
    device_id = src.insert_device(device_tag="dev_1")
    src.insert_patient(patient_id=1, mrn="MRN0001", first_name="Jane", last_name="Doe")
    src.insert_device_patient_data([(device_id, 1, BASE, BASE + 1000 * SEC)])
    src.write_time_value_pairs(measure_id, device_id, _times(len(string_values)),
                               STR(string_values), period=SEC)
    if with_numeric:
        numeric_id = _numeric_measure(src, "hr")
        src.write_time_value_pairs(numeric_id, device_id, _times(3), NUM([1.0, 2.0, 3.0]),
                                   period=SEC)
    return measure_id, device_id


def _definition(device_id, tags):
    return DatasetDefinition(measures=list(tags),
                             device_ids={device_id: [{"start": BASE, "end": BASE + 100 * SEC}]})


def test_deidentified_transfer_leaves_string_values_verbatim(sdk_pair):
    """RE-POINTED (was ``test_deidentified_transfer_redacts_string_values_by_default``).

    De-identification covers patient-level PHI and time-shifting; it does not alter signal
    content. A string measure's values are signal DATA in exactly the way a numeric
    measure's samples are, so ``deidentify=True`` must transfer them verbatim -- the
    previous default rewrote every one of them to "<redacted>".
    """
    src, dst = sdk_pair
    _, device_id = _seed_transfer_source(src, ["Vent mode SIMV", "pt stable"])

    transfer_data(src, dst, definition=_definition(device_id, ["note"]), deidentify=True)

    dest_measure = dst.get_measure_id("note", freq=1, freq_units="Hz", units="string")
    vocabulary = dst.get_measure_string_vocabulary(dest_measure)
    assert sorted(vocabulary) == ["Vent mode SIMV", "pt stable"]
    assert REDACTED_STRING_VALUE not in vocabulary
    times, values = dst.get_string_data(dest_measure, int(BASE), int(BASE) + 100 * SEC, device_id=1)
    assert times.size == 2
    assert list(values) == ["Vent mode SIMV", "pt stable"]


def test_deidentified_transfer_opt_in_and_callable_policies(sdk_pair):
    src, dst = sdk_pair
    _, device_id = _seed_transfer_source(src, ["ASYSTOLE", "VTACH"], string_tag="alarm")

    # "transfer" is the default in every mode; naming it explicitly changes nothing.
    transfer_data(src, dst, definition=_definition(device_id, ["alarm"]), deidentify=True,
                  string_value_policy="transfer")
    dest_measure = dst.get_measure_id("alarm", freq=1, freq_units="Hz", units="string")
    assert sorted(dst.get_measure_string_vocabulary(dest_measure)) == ["ASYSTOLE", "VTACH"]

    # "redact" remains available, but only when explicitly asked for.
    redact_dst, redact_loc = _new_dataset("redact_dst")
    try:
        transfer_data(src, redact_dst, definition=_definition(device_id, ["alarm"]),
                      deidentify=True, string_value_policy="redact")
        redact_measure = redact_dst.get_measure_id("alarm", freq=1, freq_units="Hz", units="string")
        assert redact_dst.get_measure_string_vocabulary(redact_measure) == [REDACTED_STRING_VALUE]
    finally:
        redact_dst.close()
        shutil.rmtree(redact_loc, ignore_errors=True)

    # A callable hook scrubs per value; returning None drops it.
    other, other_loc = _new_dataset("cb_dst")
    try:
        transfer_data(src, other, definition=_definition(device_id, ["alarm"]), deidentify=True,
                      string_value_policy=lambda value, info: None if value == "VTACH"
                      else value.lower())
        cb_measure = other.get_measure_id("alarm", freq=1, freq_units="Hz", units="string")
        assert other.get_measure_string_vocabulary(cb_measure) == ["asystole"]
    finally:
        other.close()
        shutil.rmtree(other_loc, ignore_errors=True)

    with pytest.raises(ValueError, match="string_value_policy"):
        transfer_data(src, dst, definition=_definition(device_id, ["alarm"]),
                      string_value_policy="not_a_policy")


def test_transfer_string_into_numeric_destination_aborts_before_writing(sdk_pair):
    src, dst = sdk_pair
    _, device_id = _seed_transfer_source(src, ["SIMV", "PRVC"], string_tag="mode",
                                         with_numeric=True)

    clash = dst.insert_measure(measure_tag="mode", freq=1, freq_units="Hz", units="string")
    clash_device = dst.insert_device(device_tag="dev_1")
    dst.write_time_value_pairs(clash, clash_device, _times(2), NUM([1.0, 2.0]), period=SEC)
    measures_before = set(dst.get_all_measures().keys())

    with pytest.raises(ValueError) as excinfo:
        transfer_data(src, dst, definition=_definition(device_id, ["mode", "hr"]))

    message = str(excinfo.value)
    assert "'mode'" in message, "the error must name the offending source measure tag"
    assert "before writing anything" in message
    # Nothing half-transferred: the numeric measure was never created either.
    assert set(dst.get_all_measures().keys()) == measures_before


def test_transfer_carries_signal_kind_into_an_existing_destination_measure(sdk_pair):
    src, dst = sdk_pair
    _, device_id = _seed_transfer_source(src, ["SIMV", "PRVC"], string_tag="mode")

    existing = dst.insert_measure(measure_tag="mode", freq=1, freq_units="Hz", units="string")
    assert dst.get_measure_kind(existing) == ("waveform", "numeric")

    transfer_data(src, dst, definition=_definition(device_id, ["mode"]),
                  string_value_policy="transfer")

    assert dst.get_measure_kind(existing) == src.get_measure_kind(
        src.get_measure_id("mode", freq=1, freq_units="Hz", units="string"))


# =========================================================================== #
# W8 / W12 -- actionable errors
# =========================================================================== #
def test_truncated_dictionary_error_names_the_measure_and_a_remedy(sdk):
    measure_id = _string_measure(sdk, "trunc")
    device_id = sdk.insert_device(device_tag="trunc_dev")
    sdk.write_time_value_pairs(measure_id, device_id, _times(2), STR(["ALPHA", "BETA"]))

    path = MeasureStringDictionary.path_for(sdk._meta_dir, measure_id)
    content = path.read_text(encoding="utf-8")
    path.write_text(content[:-3], encoding="utf-8")  # truncate the final line

    with pytest.raises(ValueError) as excinfo:
        MeasureStringDictionary.load(sdk._meta_dir, measure_id)
    message = str(excinfo.value)
    assert not isinstance(excinfo.value, json.JSONDecodeError)
    assert str(path) in message
    assert "Restore" in message


def test_write_data_easy_rejects_strings_with_actionable_advice(sdk):
    measure_id = _string_measure(sdk, "easy")
    device_id = sdk.insert_device(device_tag="easy_dev")

    with pytest.raises(ValueError, match="write_time_value_pairs"):
        sdk.write_data_easy(measure_id, device_id, _times(2), STR(["A", "B"]), 1,
                            freq_units="Hz")


# =========================================================================== #
# W1 (residual) -- the dictionary-file signal must mean the same thing to every
# consumer, or a killed write poisons the measure through the backfill instead
# =========================================================================== #
def _orphan_dictionary(sdk, measure_id, text="PATIENT_JOHN_DOE"):
    """The on-disk footprint of a string write killed between the dictionary append
    and the block commit (SIGKILL / power loss, or a rollback that could not run
    because a concurrent appender owned the file's tail): vocabulary on disk, no
    block, no value_type column."""
    path = MeasureStringDictionary.path_for(sdk._meta_dir, measure_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(text) + "\n", encoding="utf-8")
    return path


def test_orphan_dictionary_does_not_establish_string_anywhere(sdk):
    """A dictionary with no committed blocks behind it is not evidence of anything.

    ``_established_value_type`` already required blocks, but ``_resolve_measure_kind``
    (what ``get_measure_kind`` serves) did not, so the public API reported 'string' for a
    measure the write path considered unestablished -- and happily accepted numeric data
    for. All three consumers of the signal must now agree."""
    measure_id = _numeric_measure(sdk, "orphan_kind")
    device_id = sdk.insert_device(device_tag="orphan_kind_dev")
    _orphan_dictionary(sdk, measure_id)

    assert sdk._established_value_type(measure_id) is None
    assert sdk.get_measure_kind(measure_id) == ("sample", "numeric")

    # And the measure is still usable for what it was created for.
    sdk.write_time_value_pairs(measure_id, device_id, _times(3), NUM([1.0, 2.0, 3.0]))
    assert sdk.get_measure_kind(measure_id) == ("sample", "numeric")


def test_auto_upgrade_does_not_persist_an_orphan_dictionary_as_string(sdk):
    """The severe half: ``_backfill_string_value_types`` used to WRITE
    ``value_type='string'`` into the column for any measure with a dictionary file. A
    killed write plus a routine ``AtriumSDK(auto_upgrade=True)`` therefore bricked a
    numeric measure permanently -- exactly the poisoning W1 was meant to close, arriving
    through the schema upgrade instead of through the write."""
    measure_id = _numeric_measure(sdk, "orphan_backfill")
    device_id = sdk.insert_device(device_tag="orphan_backfill_dev")
    _orphan_dictionary(sdk, measure_id)
    location = sdk._loc

    upgraded = AtriumSDK(dataset_location=location, metadata_connection_type="sqlite",
                         auto_upgrade=True)
    try:
        assert upgraded.sql_handler.select_measure(measure_id=measure_id)[11] is None
        assert upgraded.get_measure_kind(measure_id) == ("sample", "numeric")
        # Not bricked: the numeric write the measure exists for still succeeds.
        upgraded.write_time_value_pairs(measure_id, device_id, _times(3), NUM([1.0, 2.0, 3.0]))
        _, read_times, _ = upgraded.get_data(measure_id, BASE - SEC, BASE + 10 * SEC,
                                             device_id=device_id)
        assert read_times.size == 3
    finally:
        upgraded.close()


def test_auto_upgrade_still_backfills_a_real_p1_string_measure(sdk):
    """Do-no-harm counterpart: a genuine P1 dictionary -- one with committed blocks
    behind it -- must still be backfilled to value_type='string'."""
    measure_id = sdk.insert_measure(measure_tag="alarm", freq=1, freq_units="Hz", units="string")
    device_id = sdk.insert_device(device_tag="p1_dev")
    sdk.write_time_value_pairs(measure_id, device_id, _times(2), STR(["ALPHA", "BETA"]))
    # Simulate the pre-P2 state: data and dictionary present, the column not yet set.
    # update_measure_metadata only writes non-None fields, so clear it directly.
    with sdk.sql_handler.sqlite_db_connection(begin=True) as (conn, cursor):
        cursor.execute("UPDATE measure SET value_type = NULL WHERE id = ?", (measure_id,))
    location = sdk._loc

    upgraded = AtriumSDK(dataset_location=location, metadata_connection_type="sqlite",
                         auto_upgrade=True)
    try:
        assert upgraded.sql_handler.select_measure(measure_id=measure_id)[11] == "string"
        _, values = upgraded.get_string_data(measure_id, BASE - SEC, BASE + 10 * SEC,
                                             device_id=device_id)
        assert list(map(str, values)) == ["ALPHA", "BETA"]
    finally:
        upgraded.close()


# =========================================================================== #
# W8 -- a lost string dictionary must never be silently re-issued
# =========================================================================== #
def test_lost_dictionary_is_refused_instead_of_re_issuing_codes(sdk):
    """Losing ``meta/string_dict/`` (a DB + tsc restore that omits ``meta/``) used to be
    invisible: the next write took code 0 again, so every historical code silently began
    decoding to a DIFFERENT string. The vocabulary size is now recorded in the metadata
    database, which survives the loss, and a shorter dictionary is refused."""
    measure_id = _string_measure(sdk, "lost_dict")
    device_id = sdk.insert_device(device_tag="lost_dict_dev")
    sdk.write_time_value_pairs(measure_id, device_id, _times(2), STR(["ALPHA", "BETA"]))
    assert sdk.sql_handler.get_string_dict_watermark(measure_id) == 2

    MeasureStringDictionary.path_for(sdk._meta_dir, measure_id).unlink()

    with pytest.raises(ValueError) as excinfo:
        sdk.write_time_value_pairs(measure_id, device_id,
                                   np.array([BASE + 10 * SEC], dtype=np.int64),
                                   STR(["OMEGA"]), period=SEC)
    message = str(excinfo.value)
    assert "lost data" in message and "2 were committed" in message
    assert "Restore meta/string_dict/" in message

    # The historical code is now unreadable -- but LOUDLY so, not silently wrong.
    with pytest.raises(ValueError, match="out of range"):
        sdk.get_string_data(measure_id, BASE, BASE + SEC, device_id=device_id)


def test_truncated_dictionary_is_refused_before_re_issuing_tail_codes(sdk):
    """The subtler half: a dictionary truncated from 3 entries to 1 would re-issue codes
    1 and 2, whose historical meaning still exists in the blocks."""
    measure_id = _string_measure(sdk, "trunc_reissue")
    device_id = sdk.insert_device(device_tag="trunc_reissue_dev")
    sdk.write_time_value_pairs(measure_id, device_id, _times(3), STR(["A", "B", "C"]))

    path = MeasureStringDictionary.path_for(sdk._meta_dir, measure_id)
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lost data"):
        sdk.write_time_value_pairs(measure_id, device_id,
                                   np.array([BASE + 10 * SEC], dtype=np.int64),
                                   STR(["Z"]), period=SEC)


def test_dictionary_loss_guard_does_not_fire_on_healthy_writes(sdk):
    """Do-no-harm counterpart, and the case the guard must never break: repeated string
    writes to a growing vocabulary, plus a first write to a measure that was DECLARED
    string (so it is 'established') but has never been written."""
    measure_id = _string_measure(sdk, "healthy")
    device_id = sdk.insert_device(device_tag="healthy_dev")
    for chunk in range(4):
        sdk.write_time_value_pairs(
            measure_id, device_id, _times(2, start=BASE + chunk * 2 * SEC),
            STR([f"S{2 * chunk}", f"S{2 * chunk + 1}"]), period=SEC)
    _, values = sdk.get_string_data(measure_id, BASE - SEC, BASE + 100 * SEC, device_id=device_id)
    assert list(map(str, values)) == [f"S{i}" for i in range(8)]
    assert sdk.sql_handler.get_string_dict_watermark(measure_id) == 8

    declared = sdk.insert_measure(measure_tag="declared", freq=1, freq_units="Hz", units="string")
    sdk.set_measure_kind(declared, signal_kind="event", value_type="string")
    sdk.write_time_value_pairs(declared, device_id, np.array([BASE], dtype=np.int64),
                               STR(["FIRST"]), period=SEC)
    _, values = sdk.get_string_data(declared, BASE - SEC, BASE + SEC, device_id=device_id)
    assert list(map(str, values)) == ["FIRST"]


# =========================================================================== #
# D6 -- 'waveform' + 'string' must be impossible to create through the API
# =========================================================================== #
# Design section 4 / 21.3: a string measure needs a signal_kind of event/state/sample. A
# 'waveform' string measure passed insert_measure, passed get_string_data and then died
# deep inside get_iterator's fill path ("its values cannot be NaN-filled") hours after the
# mistake was made. Every route that could classify a measure now repairs the combination
# where it is created, loudly, naming signal_kind and set_measure_kind.
def test_string_write_repairs_a_waveform_measure_it_would_have_stranded(sdk, caplog):
    """The route the docs' own former string example took: insert_measure with no
    signal_kind (so it defaults to waveform), then write text. The first write establishes
    value_type='string' -- and used to leave the measure at waveform+string."""
    device_id = sdk.insert_device(device_tag="d6_dev")
    measure_id = sdk.insert_measure(measure_tag="vent_mode_lazy", freq=1, freq_units="Hz",
                                    units="mode")
    assert sdk.get_measure_kind(measure_id) == ("waveform", "numeric")

    with caplog.at_level(logging.WARNING, logger="atriumdb.atrium_sdk"):
        sdk.write_time_value_pairs(measure_id, device_id, _times(2), STR(["SIMV", "PRVC"]),
                                   period=SEC)

    assert sdk.get_measure_kind(measure_id) == ("event", "string")
    assert "signal_kind" in caplog.text, "the report must name the field that was wrong"
    assert "set_measure_kind" in caplog.text, "and the public method that repairs it"
    # The data itself is untouched by the repair.
    _, values = sdk.get_string_data(measure_id, BASE - SEC, BASE + 100 * SEC, device_id=device_id)
    assert list(map(str, values)) == ["SIMV", "PRVC"]


def test_declaring_a_string_measure_without_a_signal_kind_is_repaired(sdk):
    """``insert_measure(..., value_type='string')`` with no signal_kind produced the same
    dead end, because a NULL signal_kind read-time-defaults to waveform. It is also the
    shape a legacy dataset carries into transfer_measures -> insert_measure, which is why
    this repairs rather than raises: a transfer of an already-broken source measure must
    fix it, not abort."""
    measure_id = sdk.insert_measure(measure_tag="explicit_str", freq=1, freq_units="Hz",
                                    units="string", value_type="string")
    assert sdk.get_measure_kind(measure_id) == ("event", "string")

    # An explicitly stated waveform+string is repaired the same way.
    both = sdk.insert_measure(measure_tag="stated_waveform_str", freq=1, freq_units="Hz",
                              units="string", signal_kind="waveform", value_type="string")
    assert sdk.get_measure_kind(both) == ("event", "string")


def test_set_measure_kind_cannot_recreate_the_combination_it_repairs(sdk):
    """The setter exists to fix a stranded measure, so it must not be able to make one --
    including via the half-stated call ``set_measure_kind(m, value_type='string')`` on a
    waveform measure, where the resulting combination is invalid even though neither
    argument is."""
    half = sdk.insert_measure(measure_tag="half_stated", freq=1, freq_units="Hz", units="s")
    assert sdk.set_measure_kind(half, value_type="string") == ("event", "string")

    # Pushing a real string measure back to waveform is refused the same way.
    real = _string_measure(sdk, "already_event")
    assert sdk.set_measure_kind(real, signal_kind="waveform") == ("event", "string")

    # Do no harm: a numeric measure may still be declared a waveform.
    numeric = _numeric_measure(sdk, "plain_numeric")
    assert sdk.set_measure_kind(numeric, signal_kind="waveform") == ("waveform", "numeric")


def test_repaired_string_measure_is_iterable(sdk):
    """The point of the repair: the failure used to surface in get_iterator, hours later.
    The measure the lazy write produces must now iterate."""
    device_id = sdk.insert_device(device_tag="d6_iter_dev")
    measure_id = sdk.insert_measure(measure_tag="lazy_iter", freq=1, freq_units="Hz", units="mode")
    sdk.write_time_value_pairs(measure_id, device_id, _times(3), STR(["A", "B", "C"]), period=SEC)

    definition = DatasetDefinition(
        measures=["lazy_iter"],
        device_ids={device_id: [{"start": BASE, "end": BASE + 10 * SEC}]})
    windows = list(sdk.get_iterator(definition, window_duration=10 * SEC, window_slide=10 * SEC))
    assert len(windows) >= 1
