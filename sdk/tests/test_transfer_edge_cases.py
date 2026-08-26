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

"""Transfer edge-case tests.

  * string-dictionary MERGE with an OVERLAPPING vocabulary (codes must not collide/corrupt),
    plus many-distinct-values, interval-index presence at dest, and time-shift on string times;
  * visit_number scramble CONSISTENCY (same source -> same int) and COLLISION-FREEDOM
    (different sources -> different ints) across MANY encounters;
  * de-id leakage: after a de-identified transfer, no real location name / visit_number /
    patient identifier may survive on the encounter family;
  * keep_identified: "all", field lists, unknown table/field errors, no-op when not de-id;
  * device_encounter for a device outside the transfer set is skipped cleanly;
  * NULL end_time survives; negative and positive time shifts are complete;
  * bed dedup (two encounters sharing a bed -> one dest bed);
  * log_hl7_adt is never transferred (deidentify False AND True);
  * Numeric-only transfer without encounters.

Fixtures avoid destination device-id collisions.
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
    loc = Path(__file__).parent / "test_datasets" / f"sqlite_transfer_edges_{name}"
    shutil.rmtree(loc, ignore_errors=True)
    return AtriumSDK.create_dataset(dataset_location=loc, database_type="sqlite"), loc


def _raw_all(sdk, query, args=()):
    with sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute(query, args)
        return cursor.fetchall()


def _count(sdk, table):
    return _raw_all(sdk, f"SELECT COUNT(*) FROM {table}")[0][0]


def _base_measures_and_patient(sdk, *, numeric_tag="hr", string_tag="alarm",
                               device_tag="dev_A", string_values=None):
    """One numeric + one string measure, one patient, one in-set device with data +
    a device_patient mapping so a patient-scoped definition resolves to that device."""
    string_values = ["ASYSTOLE", "V-TACH", "ASYSTOLE", "BRADY"] if string_values is None else string_values

    numeric_id = sdk.insert_measure(numeric_tag, 1, freq_units="Hz", units="bpm")
    string_id = sdk.insert_measure(measure_tag=string_tag, freq=1.0, freq_units="Hz",
                                   units="string", signal_kind="event", value_type="string")
    device_id = sdk.insert_device(device_tag)

    n = 20
    numeric_times = np.arange(n, dtype=np.int64) * SEC + 1000 * SEC
    numeric_values = np.arange(n, dtype=np.int64) * 3
    sdk.write_data_easy(numeric_id, device_id, numeric_times, numeric_values, 1, freq_units="Hz")

    string_times = (np.arange(len(string_values), dtype=np.int64) * 5 * SEC) + 2000 * SEC
    sdk.write_time_value_pairs(string_id, device_id, string_times,
                               np.array(string_values, dtype=object))

    patient_id = sdk.insert_patient(mrn="900900", gender="F", dob=-1234 * SEC,
                                    first_name="Alice", last_name="Zephyr",
                                    first_seen=900 * SEC, last_updated=900 * SEC)
    sdk.insert_device_patient_data([(device_id, patient_id, 900 * SEC, 5000 * SEC)])

    return {
        "numeric_id": numeric_id, "string_id": string_id, "device_id": device_id,
        "device_tag": device_tag, "patient_id": patient_id,
        "numeric_tag": numeric_tag, "string_tag": string_tag,
        "numeric_times": numeric_times, "numeric_values": numeric_values,
        "string_times": string_times, "string_values": string_values,
    }


def _chain(sdk, inst_name, unit_name, bed_name):
    inst = sdk.sql_handler.insert_institution(name=inst_name)
    unit = sdk.sql_handler.insert_unit(institution_id=inst, name=unit_name, unit_type="ward")
    bed = sdk.sql_handler.insert_bed(unit_id=unit, name=bed_name)
    return inst, unit, bed


def _definition(ids):
    return DatasetDefinition(measures=[ids["numeric_tag"], ids["string_tag"]],
                             patient_ids={ids["patient_id"]: "all"})


# =========================================================================== STRING DICT
def test_string_dict_merge_overlapping_vocabulary():
    """The headline §24.2#1 obligation: destination already has a dictionary for the measure
    whose vocabulary OVERLAPS the source. Both the pre-existing strings and the freshly
    transferred strings must decode correctly -- codes must be unioned/remapped, not
    collided or corrupted."""
    src, _ = _fresh_sdk("ovlp_src")
    dest, _ = _fresh_sdk("ovlp_dest")
    src_strings = ["ASYSTOLE", "V-TACH", "ASYSTOLE", "BRADY", "V-TACH"]
    ids = _base_measures_and_patient(src, string_values=src_strings)

    # Pre-populate dest measure (same id/tag/freq/units) with an OVERLAPPING vocabulary on a
    # separate high device id. "V-TACH" and "ASYSTOLE" overlap the source; "PRE_ONLY" does not.
    dest.insert_measure(measure_tag=ids["string_tag"], freq=1.0, freq_units="Hz",
                        units="string", signal_kind="event", value_type="string",
                        measure_id=ids["string_id"])
    pre_device = dest.insert_device("pre_device", device_id=999)
    pre_strings = ["V-TACH", "PRE_ONLY", "ASYSTOLE", "PRE_ONLY"]
    pre_times = np.arange(len(pre_strings), dtype=np.int64) * SEC + 50 * SEC
    dest.write_time_value_pairs(ids["string_id"], pre_device, pre_times,
                                np.array(pre_strings, dtype=object))

    transfer_data(src, dest, _definition(ids), deidentify=False)

    _, r_src = dest.get_string_data(
        ids["string_id"], int(ids["string_times"][0]), int(ids["string_times"][-1]) + SEC,
        device_tag=ids["device_tag"])
    assert list(r_src) == src_strings, "transferred strings corrupted after dict merge"

    _, r_pre = dest.get_string_data(
        ids["string_id"], int(pre_times[0]), int(pre_times[-1]) + SEC, device_id=pre_device)
    assert list(r_pre) == pre_strings, "pre-existing vocabulary corrupted by the merge"


def test_string_measure_many_distinct_values():
    """Many distinct strings (larger than any small code space) must all round-trip."""
    src, _ = _fresh_sdk("many_src")
    dest, _ = _fresh_sdk("many_dest")
    many = [f"CODE_{i:04d}" for i in range(200)]
    ids = _base_measures_and_patient(src, string_values=many)

    transfer_data(src, dest, _definition(ids), deidentify=False)

    _, r = dest.get_string_data(
        ids["string_id"], int(ids["string_times"][0]), int(ids["string_times"][-1]) + SEC,
        device_tag=ids["device_tag"])
    assert list(r) == many


def test_string_measure_interval_index_present_at_dest():
    """String measures skip the numeric tsc block/interval path (they `continue`). The
    interval index must still be populated at the destination (needed for within:encounter
    / iteration), otherwise the strings are effectively invisible to interval queries."""
    src, _ = _fresh_sdk("sint_src")
    dest, _ = _fresh_sdk("sint_dest")
    ids = _base_measures_and_patient(src)

    transfer_data(src, dest, _definition(ids), deidentify=False)

    intervals = dest.get_interval_array(
        measure_id=ids["string_id"], device_id=ids["device_id"],
        start=int(ids["string_times"][0]), end=int(ids["string_times"][-1]) + SEC)
    assert intervals is not None and len(intervals) > 0, \
        "string measure has no interval index at dest"
    # The interval union must cover the first and last string timestamps.
    assert intervals[0][0] <= int(ids["string_times"][0])
    assert intervals[-1][1] >= int(ids["string_times"][-1])


def test_string_measure_time_shift_applied():
    """time_shift must move string-measure timestamps at the destination."""
    src, _ = _fresh_sdk("sshift_src")
    dest, _ = _fresh_sdk("sshift_dest")
    ids = _base_measures_and_patient(src)

    shift = 12_345 * SEC
    transfer_data(src, dest, _definition(ids), deidentify=False, time_shift=shift, time_units="ns")

    r_times, r_vals = dest.get_string_data(
        ids["string_id"], int(ids["string_times"][0]) + shift,
        int(ids["string_times"][-1]) + shift + SEC, device_tag=ids["device_tag"])
    assert list(r_vals) == ids["string_values"]
    assert np.array_equal(np.asarray(r_times, dtype=np.int64),
                          ids["string_times"] + shift)


# =========================================================================== VISIT SCRAMBLE
def _build_multi_encounter(sdk, ids, visit_numbers, start_times, bed_names):
    """Insert several encounters for the one patient, each on its own bed, with the given
    visit numbers / start times. Returns list of (start_time, visit_number)."""
    recs = []
    for vn, st, bn in zip(visit_numbers, start_times, bed_names):
        _, _, bed = _chain(sdk, f"Inst_{bn}", f"Unit_{bn}", bn)
        sdk.sql_handler.insert_encounter(
            patient_id=ids["patient_id"], bed_id=bed, start_time=st, end_time=st + 10 * SEC,
            source_id=1, visit_number=vn, last_updated=st + 5 * SEC)
        recs.append((st, vn))
    return recs


def test_visit_number_scramble_consistent_and_collision_free():
    """Same source visit_number -> same scrambled int; different source visit_numbers ->
    different scrambled ints. Tested across several encounters in ONE transfer."""
    src, _ = _fresh_sdk("visit_src")
    dest, _ = _fresh_sdk("visit_dest")
    ids = _base_measures_and_patient(src)

    # A/B share "V1"; C -> "V2"; D -> "V3". Distinct start_times let us correlate rows.
    visits = ["V1", "V1", "V2", "V3"]
    starts = [1000 * SEC, 1100 * SEC, 1200 * SEC, 1300 * SEC]
    beds = ["BedA", "BedB", "BedC", "BedD"]
    _build_multi_encounter(src, ids, visits, starts, beds)

    shift = 500 * SEC
    transfer_data(src, dest, _definition(ids), deidentify=True, time_shift=shift, time_units="ns")

    rows = _raw_all(dest, "SELECT start_time, visit_number FROM encounter")
    # Map dest start_time back to the original source start_time via the known shift.
    by_src_start = {st - shift: vn for (st, vn) in rows}
    assert set(by_src_start.keys()) == set(starts), "unexpected encounter set at dest"

    v_a = by_src_start[1000 * SEC]
    v_b = by_src_start[1100 * SEC]
    v_c = by_src_start[1200 * SEC]
    v_d = by_src_start[1300 * SEC]

    # All scrambled to ints, none equal to any original (non-numeric) visit string.
    for v in (v_a, v_b, v_c, v_d):
        assert v is not None
        assert str(int(v)) == str(v)
        assert str(v) not in {"V1", "V2", "V3"}

    # Consistency: the two "V1" encounters map to the SAME scrambled int.
    assert v_a == v_b, "same source visit_number produced different scrambled ints"
    # Collision-freedom: distinct source visit_numbers map to distinct scrambled ints.
    assert len({v_a, v_c, v_d}) == 3, "distinct visit_numbers collided after scramble"


# =========================================================================== DE-ID LEAKAGE
def test_deid_no_identifier_leakage_across_encounter_family():
    """RE-POINTED for the corrected de-identification scope.

    De-identification covers patient-level PHI. On the encounter family that is exactly
    one field: ``visit_number``, a direct visit identifier that joins back to the source
    record system. It must not survive a de-identified transfer.

    Location NAMES (bed / unit / institution) are no longer pseudonymized -- they say where
    a recording happened, not whose it is -- so this test now asserts they transfer
    verbatim, where it previously required them to be scrambled."""
    src, _ = _fresh_sdk("leak_src")
    dest, _ = _fresh_sdk("leak_dest")
    ids = _base_measures_and_patient(src)
    _, _, bed = _chain(src, "General Hospital", "PICU", "Bed-12")
    src.sql_handler.insert_encounter(
        patient_id=ids["patient_id"], bed_id=bed, start_time=1000 * SEC,
        end_time=1200 * SEC, source_id=1, visit_number="VISIT-SECRET-42",
        last_updated=1100 * SEC)

    transfer_data(src, dest, _definition(ids), deidentify=True, time_shift=999 * SEC)

    # Guard against a vacuous pass: the family must actually be present at the destination.
    assert _count(dest, "encounter") == 1
    assert _count(dest, "institution") == 1
    assert _count(dest, "unit") == 1
    assert _count(dest, "bed") == 1

    # Location names are signal context, not PHI: they transfer verbatim.
    assert _raw_all(dest, "SELECT name FROM institution")[0][0] == "General Hospital"
    assert _raw_all(dest, "SELECT name FROM unit")[0][0] == "PICU"
    assert _raw_all(dest, "SELECT name FROM bed")[0][0] == "Bed-12"

    # The patient-level identifiers must NOT survive.
    dest_visit = _raw_all(dest, "SELECT visit_number FROM encounter")[0][0]
    assert dest_visit is not None, "the encounter must still be present (guard against a vacuous pass)"
    assert str(dest_visit) != "VISIT-SECRET-42", "the real visit_number leaked under de-id"
    assert str(int(dest_visit)) == str(dest_visit), "visit_number is scrambled to a random int"

    # And no patient identifier appears anywhere on the family.
    forbidden = {"VISIT-SECRET-42", "900900"}
    dest_texts = set()
    for tbl, col in [("institution", "name"), ("unit", "name"), ("bed", "name"),
                     ("encounter", "visit_number")]:
        for (v,) in _raw_all(dest, f"SELECT {col} FROM {tbl}"):
            if v is not None:
                dest_texts.add(str(v))
    leaked = forbidden & dest_texts
    assert not leaked, f"real identifiers leaked under de-id: {leaked}"


def test_log_hl7_adt_never_transferred_both_modes():
    """log_hl7_adt is excluded unconditionally -- with deidentify=False AND True."""
    for deid in (False, True):
        src, _ = _fresh_sdk(f"hl7_src_{deid}")
        dest, _ = _fresh_sdk(f"hl7_dest_{deid}")
        ids = _base_measures_and_patient(src)
        _, _, bed = _chain(src, "H", "U", "B")
        src.sql_handler.insert_encounter(
            patient_id=ids["patient_id"], bed_id=bed, start_time=1000 * SEC,
            end_time=1200 * SEC, source_id=1, visit_number="V", last_updated=1100 * SEC)
        with src.sql_handler.connection() as (conn, cursor):
            cursor.execute(
                "INSERT INTO log_hl7_adt (event_type, event_time, mrn, visit_num, location, source_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("A01", 1000 * SEC, "900900", "V", "H U B", 1))
        assert _count(src, "log_hl7_adt") == 1

        transfer_data(src, dest, _definition(ids), deidentify=deid,
                      time_shift=(5 * SEC if deid else None))
        assert _count(dest, "log_hl7_adt") == 0, f"log_hl7_adt transferred (deid={deid})"


# =========================================================================== KEEP_IDENTIFIED
def test_keep_identified_all_keeps_whole_table():
    src, _ = _fresh_sdk("kall_src")
    dest, _ = _fresh_sdk("kall_dest")
    ids = _base_measures_and_patient(src)
    _chain(src, "General Hospital", "PICU", "Bed-12")
    _, _, bed = _chain(src, "General Hospital", "PICU", "Bed-12")
    src.sql_handler.insert_encounter(
        patient_id=ids["patient_id"], bed_id=bed, start_time=1000 * SEC,
        end_time=1200 * SEC, source_id=1, visit_number="V-9", last_updated=1100 * SEC)

    transfer_data(src, dest, _definition(ids), deidentify=True,
                  keep_identified={"bed": "all", "unit": "all", "institution": "all"})

    assert _raw_all(dest, "SELECT name FROM institution")[-1][0] == "General Hospital"
    assert _raw_all(dest, "SELECT name FROM unit")[-1][0] == "PICU"
    assert _raw_all(dest, "SELECT name FROM bed")[-1][0] == "Bed-12"
    # visit_number NOT kept -> still scrambled to an int.
    visit = _raw_all(dest, "SELECT visit_number FROM encounter")[0][0]
    assert str(visit) != "V-9"
    assert str(int(visit)) == str(visit)


def test_keep_identified_unknown_table_raises():
    src, _ = _fresh_sdk("kut_src")
    dest, _ = _fresh_sdk("kut_dest")
    ids = _base_measures_and_patient(src)
    with pytest.raises((ValueError, KeyError)):
        transfer_data(src, dest, _definition(ids), deidentify=True,
                      keep_identified={"not_a_table": ["x"]})


def test_keep_identified_unknown_field_raises():
    src, _ = _fresh_sdk("kuf_src")
    dest, _ = _fresh_sdk("kuf_dest")
    ids = _base_measures_and_patient(src)
    with pytest.raises((ValueError, KeyError)):
        transfer_data(src, dest, _definition(ids), deidentify=True,
                      keep_identified={"encounter": ["not_a_field"]})


def test_keep_identified_noop_when_not_deidentify():
    """When deidentify=False everything is already identified; keep_identified is a no-op and
    names/visit come across verbatim."""
    src, _ = _fresh_sdk("knoop_src")
    dest, _ = _fresh_sdk("knoop_dest")
    ids = _base_measures_and_patient(src)
    _, _, bed = _chain(src, "General Hospital", "PICU", "Bed-12")
    src.sql_handler.insert_encounter(
        patient_id=ids["patient_id"], bed_id=bed, start_time=1000 * SEC,
        end_time=1200 * SEC, source_id=1, visit_number="V-KEEP", last_updated=1100 * SEC)

    transfer_data(src, dest, _definition(ids), deidentify=False, keep_identified={})

    assert _raw_all(dest, "SELECT name FROM institution")[0][0] == "General Hospital"
    assert _raw_all(dest, "SELECT name FROM bed")[0][0] == "Bed-12"
    assert str(_raw_all(dest, "SELECT visit_number FROM encounter")[0][0]) == "V-KEEP"


# =========================================================================== ENCOUNTER FAMILY
def test_device_encounter_for_out_of_set_device_is_skipped():
    """A device_encounter referencing a device NOT in the transfer set must be skipped
    cleanly (referential integrity), while the in-set device_encounter transfers."""
    src, _ = _fresh_sdk("oos_src")
    dest, _ = _fresh_sdk("oos_dest")
    ids = _base_measures_and_patient(src)
    other_device = src.insert_device("out_of_set_dev")  # no device_patient mapping -> not transferred
    _, _, bed = _chain(src, "H", "U", "B")
    enc = src.sql_handler.insert_encounter(
        patient_id=ids["patient_id"], bed_id=bed, start_time=1000 * SEC,
        end_time=1200 * SEC, source_id=1, visit_number="V", last_updated=1100 * SEC)
    src.sql_handler.insert_device_encounter(
        device_id=ids["device_id"], encounter_id=enc, start_time=1000 * SEC,
        end_time=1200 * SEC, source_id=1)
    src.sql_handler.insert_device_encounter(
        device_id=other_device, encounter_id=enc, start_time=1000 * SEC,
        end_time=1200 * SEC, source_id=1)
    assert _count(src, "device_encounter") == 2

    transfer_data(src, dest, _definition(ids), deidentify=False)

    # Only the in-set device_encounter survives.
    dest_de = _raw_all(dest, "SELECT device_id FROM device_encounter")
    assert len(dest_de) == 1, f"expected 1 device_encounter at dest, got {len(dest_de)}"


def test_encounter_null_end_time_transfers_without_crash():
    """NULL end_time must be left NULL (not shifted into a value) and must not crash."""
    src, _ = _fresh_sdk("null_src")
    dest, _ = _fresh_sdk("null_dest")
    ids = _base_measures_and_patient(src)
    _, _, bed = _chain(src, "H", "U", "B")
    enc = src.sql_handler.insert_encounter(
        patient_id=ids["patient_id"], bed_id=bed, start_time=1000 * SEC,
        end_time=None, source_id=1, visit_number="V", last_updated=1100 * SEC)
    src.sql_handler.insert_device_encounter(
        device_id=ids["device_id"], encounter_id=enc, start_time=1000 * SEC,
        end_time=None, source_id=1)

    shift = 42 * SEC
    transfer_data(src, dest, _definition(ids), deidentify=False, time_shift=shift, time_units="ns")

    enc_rows = _raw_all(dest, "SELECT start_time, end_time FROM encounter")
    assert enc_rows[0][0] == 1000 * SEC + shift
    assert enc_rows[0][1] is None, "NULL end_time should remain NULL"
    de_rows = _raw_all(dest, "SELECT start_time, end_time FROM device_encounter")
    assert de_rows[0][0] == 1000 * SEC + shift
    assert de_rows[0][1] is None


def test_negative_time_shift_is_complete():
    src, _ = _fresh_sdk("neg_src")
    dest, _ = _fresh_sdk("neg_dest")
    ids = _base_measures_and_patient(src)
    _, _, bed = _chain(src, "H", "U", "B")
    enc = src.sql_handler.insert_encounter(
        patient_id=ids["patient_id"], bed_id=bed, start_time=5000 * SEC,
        end_time=5200 * SEC, source_id=1, visit_number="V", last_updated=5100 * SEC)
    src.sql_handler.insert_device_encounter(
        device_id=ids["device_id"], encounter_id=enc, start_time=5000 * SEC,
        end_time=5200 * SEC, source_id=1)

    shift = -300 * SEC
    transfer_data(src, dest, _definition(ids), deidentify=False, time_shift=shift, time_units="ns")

    e = _raw_all(dest, "SELECT start_time, end_time, last_updated FROM encounter")[0]
    assert e == (5000 * SEC + shift, 5200 * SEC + shift, 5100 * SEC + shift)
    de = _raw_all(dest, "SELECT start_time, end_time FROM device_encounter")[0]
    assert de == (5000 * SEC + shift, 5200 * SEC + shift)


def test_bed_dedup_two_encounters_share_bed():
    """Two encounters that reference the SAME source bed must produce ONE dest bed (and one
    unit/institution), not duplicates."""
    src, _ = _fresh_sdk("dedup_src")
    dest, _ = _fresh_sdk("dedup_dest")
    ids = _base_measures_and_patient(src)
    _, _, bed = _chain(src, "General Hospital", "PICU", "Bed-12")
    for st, vn in [(1000 * SEC, "V1"), (2000 * SEC, "V2")]:
        src.sql_handler.insert_encounter(
            patient_id=ids["patient_id"], bed_id=bed, start_time=st, end_time=st + 100 * SEC,
            source_id=1, visit_number=vn, last_updated=st + 50 * SEC)

    transfer_data(src, dest, _definition(ids), deidentify=True)

    assert _count(dest, "encounter") == 2
    assert _count(dest, "bed") == 1, "shared bed duplicated at dest"
    assert _count(dest, "unit") == 1, "shared unit duplicated at dest"
    assert _count(dest, "institution") == 1, "shared institution duplicated at dest"


def test_numeric_only_transfer_unaffected_by_default_on_encounters():
    """A device-scoped numeric-only dataset with no encounters must transfer
    cleanly even though include_encounters defaults to True."""
    src, _ = _fresh_sdk("numonly_src")
    dest, _ = _fresh_sdk("numonly_dest")
    numeric_id = src.insert_measure("hr", 1, freq_units="Hz", units="bpm")
    device_id = src.insert_device("dev_only")
    times = np.arange(10, dtype=np.int64) * SEC + 1000 * SEC
    values = np.arange(10, dtype=np.int64)
    src.write_data_easy(numeric_id, device_id, times, values, 1, freq_units="Hz")

    definition = DatasetDefinition(measures=["hr"], device_ids={device_id: "all"})
    transfer_data(src, dest, definition, deidentify=False)

    _, r_t, r_v = dest.get_data(numeric_id, int(times[0]), int(times[-1]) + SEC, device_id=device_id)
    assert np.array_equal(r_t, times)
    assert np.array_equal(r_v, values)
    assert _count(dest, "encounter") == 0
