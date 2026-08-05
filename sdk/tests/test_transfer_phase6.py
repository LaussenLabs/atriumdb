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
"""Phase 6 transfer tests (design §24.3).

Covered:
  * transfer_measures carries signal_kind / value_type to the destination.
  * a string measure round-trips through transfer (dict-safe) for a FRESH dest and a
    MERGE dest that already has a dictionary for that measure.
  * encounter + device_encounter transfer with visit_number scrambled to an int and all
    times shifted under time_shift.
  * keep_identified opts a location name back to identified while others are pseudonymized.
  * log_hl7_adt is never transferred, even with deidentify=False.
  * a time_shift transfer leaves no un-shifted time on the new tables.
  * an existing numeric round-trip still holds (numeric path unchanged).
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from atriumdb import AtriumSDK, DatasetDefinition
from atriumdb.transfer.adb.dataset import transfer_data

SEC = 10 ** 9


# --------------------------------------------------------------------------- helpers
def _fresh_sdk(name):
    loc = Path(__file__).parent / "test_datasets" / f"sqlite_phase6_{name}"
    shutil.rmtree(loc, ignore_errors=True)
    return AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite"), loc


def _raw_all(sdk, query):
    with sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute(query)
        return cursor.fetchall()


def _count(sdk, table):
    return _raw_all(sdk, f"SELECT COUNT(*) FROM {table}")[0][0]


def _build_source(sdk, *, numeric_tag="hr", string_tag="alarm", device_tag="dev_1",
                  visit_number="VISIT-777", string_values=None):
    """Build a small synthetic source dataset: one numeric + one string measure with data,
    one patient/device, a bed/unit/institution chain, an encounter + device_encounter, and a
    log_hl7_adt row. Returns a dict of the ids and the raw data written."""
    string_values = ["ASYSTOLE", "V-TACH", "ASYSTOLE", "BRADY"] if string_values is None else string_values

    numeric_id = sdk.insert_measure(numeric_tag, 1, freq_units="Hz", units="bpm")
    string_id = sdk.insert_measure(measure_tag=string_tag, freq=1.0, freq_units="Hz",
                                   units="string", signal_kind="event", value_type="string")
    device_id = sdk.insert_device(device_tag)

    # Numeric data.
    n = 20
    numeric_times = np.arange(n, dtype=np.int64) * SEC + 1000 * SEC
    numeric_values = np.arange(n, dtype=np.int64) * 3
    sdk.write_data_easy(numeric_id, device_id, numeric_times, numeric_values, 1, freq_units="Hz")

    # String data (aperiodic).
    string_times = (np.arange(len(string_values), dtype=np.int64) * 5 * SEC) + 2000 * SEC
    sdk.write_time_value_pairs(string_id, device_id, string_times,
                               np.array(string_values, dtype=object))

    # Patient + device_patient mapping (so a patient-scoped definition resolves to the device).
    patient_id = sdk.insert_patient(mrn="123456", gender="M", dob=-1000 * SEC,
                                    first_name="Jane", last_name="Doe",
                                    first_seen=900 * SEC, last_updated=900 * SEC)
    sdk.insert_device_patient_data([(device_id, patient_id, 900 * SEC, 3000 * SEC)])

    # Location chain: institution -> unit -> bed.
    institution_id = sdk.sql_handler.insert_institution(name="General Hospital")
    unit_id = sdk.sql_handler.insert_unit(institution_id=institution_id, name="PICU",
                                          unit_type="ward")
    bed_id = sdk.sql_handler.insert_bed(unit_id=unit_id, name="Bed-12")

    # Encounter + device_encounter.
    enc_start = 1000 * SEC
    enc_end = 1000 * SEC + n * SEC
    enc_last_updated = 1500 * SEC
    encounter_id = sdk.sql_handler.insert_encounter(
        patient_id=patient_id, bed_id=bed_id, start_time=enc_start, end_time=enc_end,
        source_id=1, visit_number=visit_number, last_updated=enc_last_updated)
    sdk.sql_handler.insert_device_encounter(
        device_id=device_id, encounter_id=encounter_id, start_time=enc_start,
        end_time=enc_end, source_id=1)

    # log_hl7_adt row -- must NEVER be transferred.
    with sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute(
            "INSERT INTO log_hl7_adt (event_type, event_time, mrn, visit_num, location, source_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("A01", enc_start, "123456", visit_number, "General Hospital PICU Bed-12", 1))

    return {
        "numeric_id": numeric_id, "string_id": string_id, "device_id": device_id,
        "device_tag": device_tag, "patient_id": patient_id, "bed_id": bed_id,
        "unit_id": unit_id, "institution_id": institution_id, "encounter_id": encounter_id,
        "numeric_tag": numeric_tag, "string_tag": string_tag,
        "numeric_times": numeric_times, "numeric_values": numeric_values,
        "string_times": string_times, "string_values": string_values,
        "visit_number": visit_number, "enc_start": enc_start, "enc_end": enc_end,
        "enc_last_updated": enc_last_updated,
    }


def _definition(ids):
    return DatasetDefinition(measures=[ids["numeric_tag"], ids["string_tag"]],
                             patient_ids={ids["patient_id"]: "all"})


# --------------------------------------------------------------------------- tests
def test_transfer_measures_carries_signal_kind_and_value_type():
    src, _ = _fresh_sdk("meta_src")
    dest, _ = _fresh_sdk("meta_dest")
    ids = _build_source(src)

    transfer_data(src, dest, _definition(ids), deidentify=False)

    dest_numeric = dest.get_measure_info(dest.get_measure_id(ids["numeric_tag"], freq=1, freq_units="Hz", units="bpm"))
    dest_string = dest.get_measure_info(dest.get_measure_id(ids["string_tag"], freq=1, freq_units="Hz", units="string"))

    assert dest_string["value_type"] == "string"
    assert dest_string["signal_kind"] == "event"
    assert dest_numeric["value_type"] == "numeric"


def test_numeric_round_trip_unchanged():
    src, _ = _fresh_sdk("num_src")
    dest, _ = _fresh_sdk("num_dest")
    ids = _build_source(src)

    transfer_data(src, dest, _definition(ids), deidentify=False)

    _, r_times, r_values = dest.get_data(
        ids["numeric_id"], int(ids["numeric_times"][0]), int(ids["numeric_times"][-1]) + SEC,
        device_id=ids["device_id"])
    assert np.array_equal(r_times, ids["numeric_times"])
    assert np.array_equal(r_values, ids["numeric_values"])


def test_string_measure_round_trip_fresh_dest():
    src, _ = _fresh_sdk("str_src")
    dest, _ = _fresh_sdk("str_dest")
    ids = _build_source(src)

    transfer_data(src, dest, _definition(ids), deidentify=False)

    _, r_values = dest.get_string_data(
        measure_id=None, start_time_n=int(ids["string_times"][0]),
        end_time_n=int(ids["string_times"][-1]) + SEC, device_tag=ids["device_tag"],
        measure_tag=ids["string_tag"], freq=1, freq_units="Hz", units="string")
    assert list(r_values) == ids["string_values"]


def test_string_measure_round_trip_merge_dest():
    """Destination already has a dictionary for the measure (different vocabulary): the
    transfer must union + remap so the source strings still decode correctly."""
    src, _ = _fresh_sdk("merge_src")
    dest, _ = _fresh_sdk("merge_dest")
    ids = _build_source(src)

    # Pre-populate the destination measure (same id/tag/freq/units) with a DIFFERENT
    # vocabulary written on a separate device. This claims the low dictionary codes so a
    # naive block copy of the source codes would decode to the wrong strings.
    dest.insert_measure(measure_tag=ids["string_tag"], freq=1.0, freq_units="Hz",
                        units="string", signal_kind="event", value_type="string",
                        measure_id=ids["string_id"])
    # Use a high device id so it does not collide with the source device id (which the
    # transfer preserves into a fresh dest).
    pre_device = dest.insert_device("pre_device", device_id=999)
    pre_times = np.arange(3, dtype=np.int64) * SEC + 50 * SEC
    dest.write_time_value_pairs(ids["string_id"], pre_device, pre_times,
                                np.array(["PRE_A", "PRE_B", "PRE_A"], dtype=object))

    transfer_data(src, dest, _definition(ids), deidentify=False)

    # The transferred device's strings decode to the ORIGINAL source strings.
    _, r_values = dest.get_string_data(
        ids["string_id"], int(ids["string_times"][0]), int(ids["string_times"][-1]) + SEC,
        device_tag=ids["device_tag"])
    assert list(r_values) == ids["string_values"]

    # The pre-existing vocabulary is untouched.
    _, pre_values = dest.get_string_data(
        ids["string_id"], int(pre_times[0]), int(pre_times[-1]) + SEC, device_id=pre_device)
    assert list(pre_values) == ["PRE_A", "PRE_B", "PRE_A"]


def test_encounter_family_transfer_deid_scrambles_visit_and_shifts_times():
    src, _ = _fresh_sdk("enc_src")
    dest, _ = _fresh_sdk("enc_dest")
    ids = _build_source(src)

    shift = 10_000 * SEC
    transfer_data(src, dest, _definition(ids), deidentify=True, time_shift=shift, time_units="ns")

    dest_encs = _raw_all(dest, "SELECT patient_id, bed_id, start_time, end_time, source_id, "
                               "visit_number, last_updated FROM encounter")
    assert len(dest_encs) == 1
    (d_patient, d_bed, d_start, d_end, d_source, d_visit, d_last) = dest_encs[0]

    # Times all shifted.
    assert d_start == ids["enc_start"] + shift
    assert d_end == ids["enc_end"] + shift
    assert d_last == ids["enc_last_updated"] + shift

    # visit_number scrambled to an int, and different from the original.
    assert d_visit is not None
    assert int(d_visit) != ids["visit_number"]  # original was a non-numeric string
    assert str(int(d_visit)) == str(d_visit)

    # device_encounter transferred with shifted times.
    dest_de = _raw_all(dest, "SELECT start_time, end_time FROM device_encounter")
    assert len(dest_de) == 1
    assert dest_de[0][0] == ids["enc_start"] + shift
    assert dest_de[0][1] == ids["enc_end"] + shift

    # bed / unit / institution transferred (referential integrity) with pseudonymized names.
    assert _count(dest, "bed") == 1
    assert _count(dest, "unit") == 1
    assert _count(dest, "institution") == 1
    inst_name = _raw_all(dest, "SELECT name FROM institution")[0][0]
    bed_name = _raw_all(dest, "SELECT name FROM bed")[0][0]
    unit_name = _raw_all(dest, "SELECT name FROM unit")[0][0]
    assert inst_name != "General Hospital"
    assert bed_name != "Bed-12"
    assert unit_name != "PICU"


def test_keep_identified_opts_location_name_back_to_identified():
    src, _ = _fresh_sdk("keep_src")
    dest, _ = _fresh_sdk("keep_dest")
    ids = _build_source(src)

    # Keep the institution fully identified, and the encounter visit_number identified;
    # bed and unit names should still be pseudonymized.
    transfer_data(src, dest, _definition(ids), deidentify=True,
                  keep_identified={"institution": "all", "encounter": ["visit_number"]})

    inst_name = _raw_all(dest, "SELECT name FROM institution")[0][0]
    bed_name = _raw_all(dest, "SELECT name FROM bed")[0][0]
    unit_name = _raw_all(dest, "SELECT name FROM unit")[0][0]
    visit = _raw_all(dest, "SELECT visit_number FROM encounter")[0][0]

    assert inst_name == "General Hospital"          # kept identified
    assert str(visit) == ids["visit_number"]        # kept identified (not scrambled)
    assert bed_name != "Bed-12"                     # still pseudonymized
    assert unit_name != "PICU"                      # still pseudonymized


def test_log_hl7_adt_never_transferred_even_identified():
    src, _ = _fresh_sdk("hl7_src")
    dest, _ = _fresh_sdk("hl7_dest")
    ids = _build_source(src)

    assert _count(src, "log_hl7_adt") == 1

    transfer_data(src, dest, _definition(ids), deidentify=False)

    assert _count(dest, "log_hl7_adt") == 0


def test_time_shift_leaves_no_unshifted_time_on_new_tables():
    src, _ = _fresh_sdk("shift_src")
    dest, _ = _fresh_sdk("shift_dest")
    ids = _build_source(src)

    shift = 7 * 24 * 3600 * SEC  # one week
    transfer_data(src, dest, _definition(ids), deidentify=False, time_shift=shift, time_units="ns")

    # Collect every time-bearing value on the encounter family at the destination and assert
    # none of them equals an original (un-shifted) source value.
    src_times = set()
    for row in _raw_all(src, "SELECT start_time, end_time, last_updated FROM encounter"):
        src_times.update(v for v in row if v is not None)
    for row in _raw_all(src, "SELECT start_time, end_time FROM device_encounter"):
        src_times.update(v for v in row if v is not None)

    dest_times = []
    for row in _raw_all(dest, "SELECT start_time, end_time, last_updated FROM encounter"):
        dest_times.extend(v for v in row if v is not None)
    for row in _raw_all(dest, "SELECT start_time, end_time FROM device_encounter"):
        dest_times.extend(v for v in row if v is not None)

    assert dest_times, "expected encounter-family rows at the destination"
    for t in dest_times:
        assert t not in src_times, "found an un-shifted time on a destination encounter table"
    # And every dest time equals a source time + shift.
    for t in dest_times:
        assert (t - shift) in src_times


def test_include_encounters_false_skips_family():
    src, _ = _fresh_sdk("noenc_src")
    dest, _ = _fresh_sdk("noenc_dest")
    ids = _build_source(src)

    transfer_data(src, dest, _definition(ids), deidentify=False, include_encounters=False)

    assert _count(dest, "encounter") == 0
    assert _count(dest, "device_encounter") == 0
