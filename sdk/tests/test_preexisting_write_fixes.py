# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
"""
Regression guards for PRE-EXISTING write-path defects -- bugs that are present on
``main`` and are not caused by the aperiodic + text feature, though that feature is what
made two of them routine rather than exotic.

  W5  concurrent small writes to the same (measure, device) took an unserialized
      read-modify-write through the block merge, losing writes and duplicating data.
  --  ``transfer_devices`` keyed its returned map by ``None`` when the destination
      already held a different device at the source's device id.
  D4  a replayed batch of ``block_size`` values or more skipped the block merge and so
      silently DUPLICATED data that a smaller batch would have deduplicated.
  D5  ``transfer_data`` died with ``KeyError: None`` on a label with no measure, and
      built its ``insert_labels`` tuples with the fields one position out of step.

Every test here must PASS.

    docker run --rm -v "<repo>:/atriumdb" -e PYTHONPATH=/atriumdb/sdk \
        atriumdb-test:latest python -m pytest \
        /atriumdb/sdk/tests/test_preexisting_write_fixes.py -q
"""
import logging
import multiprocessing
import shutil
import tempfile

import numpy as np
import pytest

from atriumdb import DatasetDefinition
from atriumdb.transfer.adb.dataset import transfer_data

from atriumdb import AtriumSDK
from atriumdb.transfer.adb.devices import transfer_devices

SEC = 1_000_000_000
BASE = 1_600_000_000 * SEC

# Small enough to stay fast in CI, large enough that the unserialized merge lost data on
# every observed run (4 processes x 10 writes lost 20-35% before the fix).
CONCURRENT_PROCS = 4
CONCURRENT_WRITES = 10
CONCURRENT_TOTAL = CONCURRENT_PROCS * CONCURRENT_WRITES


def _new_dataset(prefix):
    location = tempfile.mkdtemp(prefix=f"atrium_prewrite_{prefix}_")
    shutil.rmtree(location, ignore_errors=True)
    sdk = AtriumSDK.create_dataset(dataset_location=location, database_type="sqlite")
    return sdk, location


# =========================================================================== #
# W5 -- the block-merge read-modify-write must be serialized
# =========================================================================== #
def _concurrent_worker(args):
    """One process drip-writing single events, INTERLEAVED with the others' timestamps so
    every process is handed the same closest block and the merges collide."""
    process_index, location = args
    sdk = AtriumSDK(dataset_location=location, metadata_connection_type="sqlite")
    failures = 0
    for write_index in range(CONCURRENT_WRITES):
        stamp = BASE + (write_index * CONCURRENT_PROCS + process_index) * SEC
        try:
            sdk.write_time_value_pairs(1, 1, np.array([stamp], dtype=np.int64),
                                       [f"P{process_index}_{write_index}"], period=SEC)
        except Exception:
            failures += 1
    sdk.close()
    return failures


def test_concurrent_event_writes_lose_no_values():
    """Four processes writing single events into the same (measure, device).

    Every event batch is far below ``block_size``, so each write takes the merge path: a
    SELECT of the closest block, a decode of its file, a merge, and a DELETE of the old
    block row -- with no isolation between those steps. Concurrent writers therefore
    merged into the SAME old block and the loser's transaction aborted with
    ``TypeError: 'NoneType' object is not subscriptable``. Observed before the fix:
    65-77 of 100 events readable across four trials, plus large-scale silent
    DUPLICATION (6742 rows read back for 320 writes) from the surviving overlapping
    merged blocks.

    Pre-existing on main; events merely make the merge path certain instead of rare.
    """
    sdk, location = _new_dataset("concurrent")
    try:
        sdk.insert_measure(measure_tag="alarm", freq=1, freq_units="Hz", units="string",
                           signal_kind="event", value_type="string")
        sdk.insert_device(device_tag="conc_dev")
        sdk.close()

        with multiprocessing.Pool(CONCURRENT_PROCS) as pool:
            failures = pool.map(_concurrent_worker,
                                [(i, str(location)) for i in range(CONCURRENT_PROCS)])
        assert sum(failures) == 0, f"{sum(failures)} writes raised"

        reader = AtriumSDK(dataset_location=location, metadata_connection_type="sqlite")
        try:
            _, values = reader.get_string_data(1, BASE - SEC, BASE + 100_000 * SEC, device_id=1)
        finally:
            reader.close()

        expected = {f"P{i}_{j}" for i in range(CONCURRENT_PROCS)
                    for j in range(CONCURRENT_WRITES)}
        # Nothing lost ...
        assert set(map(str, values)) == expected
        # ... and nothing duplicated (the other half of the corruption).
        assert len(values) == CONCURRENT_TOTAL
    finally:
        shutil.rmtree(location, ignore_errors=True)


def test_concurrent_numeric_writes_lose_no_values():
    """The same race with NUMERIC values, pinning that the fix is not string-specific:
    any stream whose writes are smaller than one block (trickle ingest, per-reading
    writes) was equally exposed. Observed before the fix: 33 raises and 287 of 320
    distinct values across 8 processes."""
    sdk, location = _new_dataset("concurrent_num")
    try:
        sdk.insert_measure(measure_tag="hr", freq=1, freq_units="Hz", units="bpm",
                           signal_kind="sample")
        sdk.insert_device(device_tag="conc_num_dev")
        sdk.close()

        with multiprocessing.Pool(CONCURRENT_PROCS) as pool:
            failures = pool.map(_numeric_worker,
                                [(i, str(location)) for i in range(CONCURRENT_PROCS)])
        assert sum(failures) == 0, f"{sum(failures)} writes raised"

        reader = AtriumSDK(dataset_location=location, metadata_connection_type="sqlite")
        try:
            _, _, values = reader.get_data(1, BASE - SEC, BASE + 100_000 * SEC, device_id=1)
        finally:
            reader.close()

        expected = {float(i * 1000 + j) for i in range(CONCURRENT_PROCS)
                    for j in range(CONCURRENT_WRITES)}
        assert set(values.tolist()) == expected
        assert values.size == CONCURRENT_TOTAL
    finally:
        shutil.rmtree(location, ignore_errors=True)


def _numeric_worker(args):
    process_index, location = args
    sdk = AtriumSDK(dataset_location=location, metadata_connection_type="sqlite")
    failures = 0
    for write_index in range(CONCURRENT_WRITES):
        stamp = BASE + (write_index * CONCURRENT_PROCS + process_index) * SEC
        try:
            sdk.write_time_value_pairs(1, 1, np.array([stamp], dtype=np.int64),
                                       np.array([float(process_index * 1000 + write_index)]),
                                       period=SEC)
        except Exception:
            failures += 1
    sdk.close()
    return failures


def test_sequential_small_writes_are_unaffected_by_the_merge_lock():
    """Do-no-harm counterpart: the ordinary single-writer merge path still merges, still
    round-trips, and is not disturbed by the added lock."""
    sdk, location = _new_dataset("sequential")
    try:
        measure_id = sdk.insert_measure(measure_tag="hr", freq=1, freq_units="Hz",
                                        units="bpm", signal_kind="sample")
        device_id = sdk.insert_device(device_tag="seq_dev")
        for i in range(20):
            sdk.write_time_value_pairs(measure_id, device_id,
                                       np.array([BASE + i * SEC], dtype=np.int64),
                                       np.array([float(i)]), period=SEC)
        _, times, values = sdk.get_data(measure_id, BASE - SEC, BASE + 100 * SEC,
                                        device_id=device_id)
        assert values.tolist() == [float(i) for i in range(20)]
        assert times.size == 20
    finally:
        sdk.close()
        shutil.rmtree(location, ignore_errors=True)


# =========================================================================== #
# transfer_devices -- the source device id must always be the map key
# =========================================================================== #
def test_transfer_devices_keys_the_map_by_the_source_device_id():
    """When the destination already holds a DIFFERENT device at the source's device id,
    ``transfer_devices`` inserted the new device correctly but keyed its returned map by
    ``None`` (it reused the loop variable as the "which id to request" flag). Every later
    lookup for that source device then missed: ``extract_device_ids`` resolved it to
    ``dest_device_id=None`` and wrote the device's data under a null device, and the
    label transfer raised ``KeyError``. Pre-existing on main."""
    src, src_location = _new_dataset("dev_src")
    dst, dst_location = _new_dataset("dev_dst")
    try:
        src_device_id = src.insert_device(device_tag="MONITOR_A")
        # Occupy the same device id in the destination with a different device.
        dst.insert_device(device_tag="SOMETHING_ELSE")

        device_map = transfer_devices(src, dst)

        assert None not in device_map
        assert src_device_id in device_map
        dest_device_id = device_map[src_device_id]
        assert dest_device_id is not None
        assert dst.get_device_info(dest_device_id)['tag'] == "MONITOR_A"
    finally:
        src.close()
        dst.close()
        shutil.rmtree(src_location, ignore_errors=True)
        shutil.rmtree(dst_location, ignore_errors=True)


def test_transfer_devices_reuses_a_matching_destination_device():
    """Do-no-harm counterpart: when the destination already holds the SAME device at the
    same id, the id is reused and no duplicate is created."""
    src, src_location = _new_dataset("dev_same_src")
    dst, dst_location = _new_dataset("dev_same_dst")
    try:
        src_device_id = src.insert_device(device_tag="MONITOR_A")
        dst.insert_device(device_tag="MONITOR_A")

        device_map = transfer_devices(src, dst)

        assert device_map == {src_device_id: src_device_id}
        assert len(dst.get_all_devices()) == 1
    finally:
        src.close()
        dst.close()
        shutil.rmtree(src_location, ignore_errors=True)
        shutil.rmtree(dst_location, ignore_errors=True)


# =========================================================================== #
# D4 -- a replay too big to merge must not duplicate SILENTLY
# =========================================================================== #
# Deduplication in this SDK is a side effect of the small-write block merge, which
# ``write_data`` only takes when the write is smaller than one optimal block. The
# behaviour therefore flipped on batch size alone -- a replayed buffer of 131071 values
# was deduplicated and one of 131072 was duplicated, with no error and no warning. These
# tests drive that threshold with a deliberately tiny ``block.block_size`` so they stay
# fast; the size is only ever compared against ``value_data.size``, so 64 exercises
# exactly the same branch as the 131072 default.
SMALL_BLOCK = 64
REPLAY_VALUES = 200          # comfortably more than SMALL_BLOCK -> merge path skipped
FREQ_HZ = 250.0
REPLAY_START_S = 1_700_000_000.0
REPLAY_WINDOW = (int(1_700_000_000 * SEC), int(1_700_000_100 * SEC))


def _write_replay_segment(sdk, measure_id, device_id, start_seconds, offset=0.0):
    sdk.write_segment(measure_id, device_id,
                      np.arange(REPLAY_VALUES, dtype=np.float64) + offset,
                      start_seconds, freq=FREQ_HZ, time_units="s", freq_units="Hz")


def _replay_dataset(prefix, **create_kwargs):
    location = tempfile.mkdtemp(prefix=f"atrium_prewrite_{prefix}_")
    shutil.rmtree(location, ignore_errors=True)
    sdk = AtriumSDK.create_dataset(dataset_location=location, database_type="sqlite",
                                   **create_kwargs)
    sdk.block.block_size = SMALL_BLOCK
    measure_id = sdk.insert_measure(measure_tag="ecg", freq=FREQ_HZ, freq_units="Hz", units="mV")
    device_id = sdk.insert_device(device_tag="d4_dev")
    return sdk, location, measure_id, device_id


def test_large_replay_duplication_is_quiet_and_resolved_on_read(caplog):
    """RE-POINTED (was ``test_large_replay_reports_the_duplication_it_cannot_prevent``,
    which asserted a WARNING on every such write).

    A batch of one optimal block or more never merges, so replaying it stores both copies.
    That is accepted: write speed is the priority and duplicates are expected at live
    ingest. Warning on every large overlapping write is therefore noise on a normal
    workload, so the report is now DEBUG-level and the resolution is the READ-side
    ``allow_duplicates=False``.

    What this pins: no warning is emitted; the debug report, if you ask for it, is precise
    and points at the read-side parameter; and that parameter actually resolves the data."""
    sdk, location, measure_id, device_id = _replay_dataset("d4_quiet")
    try:
        _write_replay_segment(sdk, measure_id, device_id, REPLAY_START_S)

        with caplog.at_level(logging.WARNING, logger="atriumdb.atrium_sdk"):
            _write_replay_segment(sdk, measure_id, device_id, REPLAY_START_S, offset=1000.0)
        assert "overlaps data already stored" not in caplog.text, \
            "an ordinary large overlapping write must not warn"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="atriumdb.atrium_sdk"):
            _write_replay_segment(sdk, measure_id, device_id, REPLAY_START_S, offset=2000.0)
        assert "overlaps data already stored" in caplog.text
        assert "allow_duplicates=False" in caplog.text

        # And the read-side parameter it points at really does resolve them.
        _, dup_times, _ = sdk.get_data(measure_id, *REPLAY_WINDOW, device_id=device_id)
        _, times, values = sdk.get_data(measure_id, *REPLAY_WINDOW, device_id=device_id,
                                        allow_duplicates=False)
        assert dup_times.size == 3 * REPLAY_VALUES
        assert times.size == REPLAY_VALUES
        assert np.unique(times).size == times.size
        # Newest write wins under the default 'overwrite'/'ignore' policy.
        assert np.array_equal(values, np.arange(REPLAY_VALUES, dtype=np.float64) + 2000.0)
    finally:
        sdk.close()
        shutil.rmtree(location, ignore_errors=True)


def test_large_replay_is_refused_under_the_error_overwrite_setting():
    """``overwrite='error'`` already refused an overlapping SMALL write via the merge
    conflict policy. It was unenforceable for a large one, because a large write never
    reached the merge. The same setting now refuses both, before anything is written."""
    sdk, location, measure_id, device_id = _replay_dataset("d4_error", overwrite="error")
    try:
        _write_replay_segment(sdk, measure_id, device_id, REPLAY_START_S)
        with pytest.raises(ValueError, match="overlaps data already stored"):
            _write_replay_segment(sdk, measure_id, device_id, REPLAY_START_S, offset=1000.0)

        # The refused write left the first copy intact and added nothing.
        _, _, values = sdk.get_data(measure_id, *REPLAY_WINDOW, device_id=device_id)
        assert values.size == REPLAY_VALUES
    finally:
        sdk.close()
        shutil.rmtree(location, ignore_errors=True)


def test_contiguous_bulk_append_is_never_reported_as_an_overlap(caplog):
    """Do-no-harm counterpart, and the reason the check uses block-index bounds rather
    than the gap-tolerance-padded interval rows: ordinary bulk ingest appends batch after
    contiguous batch, and must stay both silent and complete."""
    sdk, location, measure_id, device_id = _replay_dataset("d4_append")
    try:
        with caplog.at_level(logging.WARNING, logger="atriumdb.atrium_sdk"):
            for batch in range(3):
                _write_replay_segment(sdk, measure_id, device_id,
                                      REPLAY_START_S + batch * (REPLAY_VALUES / FREQ_HZ))

        assert "overlaps data already stored" not in caplog.text
        _, _, values = sdk.get_data(measure_id, *REPLAY_WINDOW, device_id=device_id)
        assert values.size == 3 * REPLAY_VALUES
    finally:
        sdk.close()
        shutil.rmtree(location, ignore_errors=True)


def test_small_replay_is_still_deduplicated_by_the_merge_path():
    """Do-no-harm counterpart: the size-dependent behaviour is now REPORTED, not changed.
    A replay below block_size must still deduplicate exactly as it did before."""
    sdk, location, measure_id, device_id = _replay_dataset("d4_small")
    try:
        for offset in (0.0, 1000.0):
            sdk.write_segment(measure_id, device_id,
                              np.arange(SMALL_BLOCK // 2, dtype=np.float64) + offset,
                              REPLAY_START_S, freq=FREQ_HZ, time_units="s", freq_units="Hz")

        _, _, values = sdk.get_data(measure_id, *REPLAY_WINDOW, device_id=device_id)
        assert values.size == SMALL_BLOCK // 2, "small replay must still deduplicate"
        assert values[0] == 1000.0, "the newer write's values still win"
    finally:
        sdk.close()
        shutil.rmtree(location, ignore_errors=True)


# =========================================================================== #
# D5 -- labels must survive transfer_data intact, with or without a measure
# =========================================================================== #
def _seed_label_source(sdk):
    """A source dataset holding one alarm measure, one device/patient mapping and two
    labels: one with no measure at all (the pattern tutorial.rst teaches) and one bound
    to a measure the transfer will not include."""
    measure_id = sdk.insert_measure(measure_tag="alarm", freq=1, freq_units="Hz",
                                    units="string", signal_kind="event", value_type="string")
    other_id = sdk.insert_measure(measure_tag="hr", freq=1, freq_units="Hz", units="bpm",
                                  signal_kind="sample")
    device_id = sdk.insert_device(device_tag="lbl_dev")
    sdk.insert_patient(patient_id=1, mrn="MRN0001", first_name="Jane", last_name="Doe")
    sdk.insert_device_patient_data([(device_id, 1, BASE, BASE + 1000 * SEC)])
    sdk.write_time_value_pairs(measure_id, device_id,
                               BASE + np.arange(4, dtype=np.int64) * SEC,
                               np.array(["APNEA", "ASYSTOLE", "APNEA", "VTACH"], dtype=object),
                               period=SEC)
    sdk.insert_labels(labels=[("Seizure noted on review", device_id, None, "chart review",
                               BASE + SEC, BASE + 2 * SEC)],
                      source_type="device_id")
    sdk.insert_labels(labels=[("Artifact on HR", device_id, other_id, "chart review",
                               BASE + SEC, BASE + 2 * SEC)],
                      source_type="device_id")
    return device_id


def _label_definition(device_id, label_names):
    return DatasetDefinition(measures=["alarm"],
                             device_ids={device_id: [{"start": BASE, "end": BASE + 100 * SEC}]},
                             labels=list(label_names))


def test_transfer_carries_a_label_that_has_no_measure():
    """``insert_labels`` documents ``measure_id`` as optional and tutorial.rst teaches
    exactly that shape, so real datasets are full of measure-less labels. The transfer
    looked the source measure id up in ``measure_id_map`` unconditionally and died with a
    bare ``KeyError: None``, taking the whole transfer with it.

    The same list also built its ``insert_labels`` tuples as
    ``(name, device, measure, start, end, source)`` when the signature is
    ``(name, device, measure, SOURCE, start, end)`` -- so every transferred label landed
    one field out of step: its start time was read as the label source id, its end time
    became the start, and end_time collapsed to the source id. Both are checked here."""
    src, src_location = _new_dataset("lbl_src")
    dst, dst_location = _new_dataset("lbl_dst")
    try:
        device_id = _seed_label_source(src)

        transfer_data(src_sdk=src, dest_sdk=dst,
                      definition=_label_definition(device_id, ["Seizure noted on review"]))

        transferred = dst.get_labels()
        assert len(transferred) == 1, "the label must be carried, not dropped"
        label = transferred[0]
        assert label['label_name'] == "Seizure noted on review"
        assert label['measure_id'] is None, "a measure-less label stays measure-less"
        # Times must survive unshifted and un-rotated.
        assert label['start_time_n'] == BASE + SEC
        assert label['end_time_n'] == BASE + 2 * SEC
        # The source is carried by NAME, so the destination resolves a real source row.
        assert label['label_source'] == "chart review"
    finally:
        src.close()
        dst.close()
        shutil.rmtree(src_location, ignore_errors=True)
        shutil.rmtree(dst_location, ignore_errors=True)


def test_transfer_rejects_a_label_bound_to_an_excluded_measure_by_name():
    """The other half of the same lookup: a label pointing at a measure the definition
    does not include has no destination id to map to. It used to raise a bare
    ``KeyError: 2``; it must now say which label and what to do, and it must never be
    dropped quietly -- a label is data."""
    src, src_location = _new_dataset("lbl_ex_src")
    dst, dst_location = _new_dataset("lbl_ex_dst")
    try:
        device_id = _seed_label_source(src)

        with pytest.raises(ValueError) as excinfo:
            transfer_data(src_sdk=src, dest_sdk=dst,
                          definition=_label_definition(device_id, ["Artifact on HR"]))

        message = str(excinfo.value)
        assert "'Artifact on HR'" in message, "the error must name the offending label"
        assert "does not include" in message
        assert "measures" in message, "the error must say how to fix it"
    finally:
        src.close()
        dst.close()
        shutil.rmtree(src_location, ignore_errors=True)
        shutil.rmtree(dst_location, ignore_errors=True)


# =========================================================================== #
# DE-IDENTIFICATION SCOPE
#
# De-identification covers patient-level PHI and time-shifting. It does NOT alter signal
# or label content: a string measure's values and a label's name/text are DATA, and a
# caller permitted to read a signal is permitted to read all of it. A previous change
# defaulted `deidentify=True` to rewriting every string value to "<redacted>" and to
# pseudonymizing bed/unit/institution names; both are reverted here, and the patient-level
# scrub must be untouched by that revert.
# =========================================================================== #


def _raw_all(sdk, query):
    with sdk.sql_handler.connection() as (_conn, cursor):
        cursor.execute(query)
        return cursor.fetchall()


def _seed_deid_source(sdk):
    """One patient with real PHI, one string measure carrying free text, one free-text
    label, and a bed / unit / institution chain on an encounter."""
    measure_id = sdk.insert_measure(measure_tag="vent_mode", freq=1, freq_units="Hz",
                                    units="string", signal_kind="event", value_type="string")
    device_id = sdk.insert_device(device_tag="deid_dev")
    patient_id = sdk.insert_patient(patient_id=1, mrn="MRN0001", first_name="Jane",
                                    last_name="Doe")
    sdk.insert_device_patient_data([(device_id, patient_id, BASE, BASE + 1000 * SEC)])
    sdk.write_time_value_pairs(measure_id, device_id,
                               BASE + np.arange(3, dtype=np.int64) * SEC,
                               np.array(["SIMV", "PRVC", "SIMV"], dtype=object), period=SEC)
    sdk.insert_labels(labels=[("Called Dr. Smith re: Jane Doe", device_id, None,
                               "chart review", BASE + SEC, BASE + 2 * SEC)],
                      source_type="device_id")

    institution_id = sdk.sql_handler.insert_institution(name="General Hospital")
    unit_id = sdk.sql_handler.insert_unit(institution_id=institution_id, name="PICU",
                                          unit_type="icu")
    bed_id = sdk.sql_handler.insert_bed(unit_id=unit_id, name="Bed-12")
    sdk.sql_handler.insert_encounter(patient_id=patient_id, bed_id=bed_id,
                                     start_time=BASE, end_time=BASE + 100 * SEC,
                                     source_id=1, visit_number="VISIT-SECRET-42",
                                     last_updated=BASE + 50 * SEC)
    return device_id


def test_deidentify_leaves_signal_and_label_content_intact():
    """String VALUES, label names, label text and label sources are signal content, not a
    PHI surface, and must survive ``deidentify=True`` intact. Location names likewise:
    "PICU" says where a recording happened, not whose it is."""
    src, src_location = _new_dataset("deid_scope_src")
    dst, dst_location = _new_dataset("deid_scope_dst")
    try:
        device_id = _seed_deid_source(src)
        definition = DatasetDefinition(
            measures=["vent_mode"],
            device_ids={device_id: [{"start": BASE, "end": BASE + 100 * SEC}]},
            labels=["Called Dr. Smith re: Jane Doe"])

        transfer_data(src_sdk=src, dest_sdk=dst, definition=definition, deidentify=True)

        # String measure values: verbatim, not "<redacted>".
        dest_measure = dst.get_measure_id("vent_mode", freq=1, freq_units="Hz", units="string")
        assert sorted(dst.get_measure_string_vocabulary(dest_measure)) == ["PRVC", "SIMV"]
        dest_device = dst.get_device_id("deid_dev")
        times, values = dst.get_string_data(dest_measure, int(BASE), int(BASE) + 100 * SEC,
                                            device_id=dest_device)
        assert times.size == 3
        assert list(values) == ["SIMV", "PRVC", "SIMV"]

        # Label name / text / source: verbatim.
        labels = dst.get_labels()
        assert len(labels) == 1
        assert labels[0]['label_name'] == "Called Dr. Smith re: Jane Doe"
        assert labels[0]['label_source'] == "chart review"

        # Location names: verbatim.
        assert _raw_all(dst, "SELECT name FROM institution")[0][0] == "General Hospital"
        assert _raw_all(dst, "SELECT name FROM unit")[0][0] == "PICU"
        assert _raw_all(dst, "SELECT name FROM bed")[0][0] == "Bed-12"
    finally:
        src.close()
        dst.close()
        shutil.rmtree(src_location, ignore_errors=True)
        shutil.rmtree(dst_location, ignore_errors=True)


def test_deidentify_still_scrubs_patient_level_phi():
    """Do-no-harm counterpart of the test above: narrowing the SCOPE of de-identification
    must not weaken it where it does apply. Patient identifiers, the patient id remap, the
    ``visit_number`` scramble and the log_hl7_adt exclusion all still hold."""
    src, src_location = _new_dataset("deid_phi_src")
    dst, dst_location = _new_dataset("deid_phi_dst")
    try:
        device_id = _seed_deid_source(src)
        definition = DatasetDefinition(
            measures=["vent_mode"],
            device_ids={device_id: [{"start": BASE, "end": BASE + 100 * SEC}]})

        transfer_data(src_sdk=src, dest_sdk=dst, definition=definition, deidentify=True)

        # Patient id remapped, and no MRN / name carried across.
        dest_patients = dst.get_all_patients()
        assert dest_patients, "the patient row must exist at the destination"
        assert 1 not in dest_patients, "the source patient_id must not survive de-id"
        for info in dest_patients.values():
            assert not info.get('mrn'), f"MRN leaked under de-id: {info.get('mrn')!r}"
            assert not info.get('first_name')
            assert not info.get('last_name')

        # visit_number scrambled to a random int.
        visit = _raw_all(dst, "SELECT visit_number FROM encounter")[0][0]
        assert str(visit) != "VISIT-SECRET-42"
        assert str(int(visit)) == str(visit)

        # log_hl7_adt is never transferred.
        assert _raw_all(dst, "SELECT COUNT(*) FROM log_hl7_adt")[0][0] == 0
    finally:
        src.close()
        dst.close()
        shutil.rmtree(src_location, ignore_errors=True)
        shutil.rmtree(dst_location, ignore_errors=True)
