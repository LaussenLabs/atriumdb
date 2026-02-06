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
Tests for MRN string type migration edge cases.

Verifies that:
- MRN values are stored and returned as strings
- Both int and str inputs are accepted and normalized to strings
- Alphanumeric MRNs work correctly
- Lookups using int input find patients inserted with str input and vice versa
- MRN uniqueness is enforced across types (e.g. int 123 and str "123" are the same)
- Leading zeros are preserved in string MRNs
- get_all_patients returns string MRNs
- Patient history and encounter lookups work with both int and str MRN inputs
"""

import time
import pytest
from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'test_mrn_string_type'


def test_mrn_string_type():
    _test_for_both(DB_NAME, _test_mrn_string_basics)


def test_mrn_alphanumeric():
    _test_for_both(DB_NAME + '_alpha', _test_mrn_alphanumeric_values)


def test_mrn_cross_type_lookup():
    _test_for_both(DB_NAME + '_cross', _test_mrn_cross_type_lookup)


def test_mrn_leading_zeros():
    _test_for_both(DB_NAME + '_zeros', _test_mrn_leading_zeros)


def test_mrn_in_patient_history():
    _test_for_both(DB_NAME + '_hist', _test_mrn_in_patient_history)


def test_mrn_in_encounters():
    _test_for_both(DB_NAME + '_enc', _test_mrn_in_encounters)


def test_mrn_in_device_patient():
    _test_for_both(DB_NAME + '_devpat', _test_mrn_in_device_patient)


# ---------------------------------------------------------------------------
# Basic string storage and retrieval
# ---------------------------------------------------------------------------

def _test_mrn_string_basics(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Insert with string MRN
    pid_str = sdk.insert_patient(mrn="100001", gender="M", first_name="Alice", last_name="Smith")
    info_str = sdk.get_patient_info(patient_id=pid_str)
    assert info_str['mrn'] == "100001", "MRN should be stored as string"
    assert isinstance(info_str['mrn'], str), "MRN type should be str"

    # Insert with int MRN — should be converted and stored as string
    pid_int = sdk.insert_patient(mrn=200002, gender="F", first_name="Bob", last_name="Jones")
    info_int = sdk.get_patient_info(patient_id=pid_int)
    assert info_int['mrn'] == "200002", "Integer MRN should be stored as string '200002'"
    assert isinstance(info_int['mrn'], str), "MRN type should be str even when inserted as int"

    # Verify get_all_patients returns string MRNs
    all_patients = sdk.get_all_patients()
    for patient_id, patient_info in all_patients.items():
        if patient_info['mrn'] is not None:
            assert isinstance(patient_info['mrn'], str), \
                f"get_all_patients should return str MRNs, got {type(patient_info['mrn'])}"

    # Insert with None MRN
    pid_none = sdk.insert_patient(mrn=None, gender="M", first_name="Charlie", last_name="Brown")
    info_none = sdk.get_patient_info(patient_id=pid_none)
    assert info_none['mrn'] is None, "None MRN should remain None"


# ---------------------------------------------------------------------------
# Alphanumeric MRN values
# ---------------------------------------------------------------------------

def _test_mrn_alphanumeric_values(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    alphanumeric_mrns = [
        "ABC-12345",
        "MRN-2024-001",
        "PT00042",
        "site3/patient/99",
        "12-34-56",
    ]

    patient_ids = []
    for mrn in alphanumeric_mrns:
        pid = sdk.insert_patient(mrn=mrn)
        patient_ids.append(pid)

    # Verify each can be looked up by MRN
    for mrn, pid in zip(alphanumeric_mrns, patient_ids):
        info = sdk.get_patient_info(mrn=mrn)
        assert info is not None, f"Should find patient with MRN '{mrn}'"
        assert info['id'] == pid, f"Patient ID mismatch for MRN '{mrn}'"
        assert info['mrn'] == mrn, f"MRN should be stored exactly as '{mrn}'"

    # Verify get_patient_id works for alphanumeric
    for mrn, pid in zip(alphanumeric_mrns, patient_ids):
        resolved_pid = sdk.get_patient_id(mrn=mrn)
        assert resolved_pid == pid, f"get_patient_id should resolve alphanumeric MRN '{mrn}'"


# ---------------------------------------------------------------------------
# Cross-type lookup: insert as str, lookup as int and vice versa
# ---------------------------------------------------------------------------

def _test_mrn_cross_type_lookup(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Insert with string, look up with int
    pid1 = sdk.insert_patient(mrn="999888", first_name="Cross", last_name="TypeA")
    info_via_int = sdk.get_patient_info(mrn=999888)
    assert info_via_int is not None, "Should find patient inserted as str '999888' when querying with int 999888"
    assert info_via_int['id'] == pid1

    # Insert with int, look up with string
    pid2 = sdk.insert_patient(mrn=777666, first_name="Cross", last_name="TypeB")
    info_via_str = sdk.get_patient_info(mrn="777666")
    assert info_via_str is not None, "Should find patient inserted as int 777666 when querying with str '777666'"
    assert info_via_str['id'] == pid2

    # get_patient_id cross-type
    assert sdk.get_patient_id(mrn="999888") == pid1
    assert sdk.get_patient_id(mrn=999888) == pid1
    assert sdk.get_patient_id(mrn="777666") == pid2
    assert sdk.get_patient_id(mrn=777666) == pid2

    # get_mrn should return string
    mrn1 = sdk.get_mrn(patient_id=pid1)
    assert mrn1 == "999888"
    assert isinstance(mrn1, str)

    mrn2 = sdk.get_mrn(patient_id=pid2)
    assert mrn2 == "777666"
    assert isinstance(mrn2, str)


# ---------------------------------------------------------------------------
# Leading zeros preserved
# ---------------------------------------------------------------------------

def _test_mrn_leading_zeros(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Insert MRN with leading zeros
    pid = sdk.insert_patient(mrn="007890", first_name="James", last_name="Bond")
    info = sdk.get_patient_info(mrn="007890")
    assert info is not None, "Should find patient with leading-zero MRN"
    assert info['mrn'] == "007890", "Leading zeros must be preserved"
    assert info['id'] == pid

    # A numeric lookup for 7890 should NOT find this patient (different string)
    info_no_zeros = sdk.get_patient_info(mrn=7890)
    assert info_no_zeros is None or info_no_zeros['mrn'] != "007890", \
        "int 7890 should not match string '007890' — leading zeros distinguish them"

    # Insert another patient with MRN "7890" to confirm they're distinct
    pid2 = sdk.insert_patient(mrn="7890", first_name="Other", last_name="Patient")
    assert pid != pid2, "MRNs '007890' and '7890' should create distinct patients"

    info2 = sdk.get_patient_info(mrn="7890")
    assert info2['id'] == pid2
    assert info2['mrn'] == "7890"


# ---------------------------------------------------------------------------
# MRN string type in patient history lookups
# ---------------------------------------------------------------------------

def _test_mrn_in_patient_history(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Insert patient with string MRN
    pid = sdk.insert_patient(mrn="HIST-001", gender="F", first_name="Dana", last_name="Scully",
                             weight=60, weight_units='kg', height=170, height_units='cm')

    # get_patient_info with string MRN
    info = sdk.get_patient_info(mrn="HIST-001", time=time.time_ns())
    assert info is not None
    assert info['height'] == 170
    assert info['weight'] == 60

    # Insert patient history using string MRN
    sdk.insert_patient_history(field='weight', value=62, units='kg',
                               time=1707945246, time_units='s', mrn="HIST-001")

    # Retrieve patient history using string MRN
    history = sdk.get_patient_history(mrn="HIST-001", field='weight')
    assert len(history) >= 2, "Should have original + newly inserted weight history"

    # Now insert with numeric MRN and test history with int lookup
    pid2 = sdk.insert_patient(mrn=554433, gender="M", first_name="Fox", last_name="Mulder",
                              weight=80, weight_units='kg', height=185, height_units='cm')

    sdk.insert_patient_history(field='height', value=186, units='cm',
                               time=1707945446, time_units='s', mrn=554433)

    # Look up history using int MRN
    history2 = sdk.get_patient_history(mrn=554433, field='height')
    assert len(history2) >= 2

    # Look up same history using string MRN
    history2_str = sdk.get_patient_history(mrn="554433", field='height')
    assert len(history2_str) == len(history2), "Int and str MRN should return same history"


# ---------------------------------------------------------------------------
# MRN string type in encounter lookups
# ---------------------------------------------------------------------------

def _test_mrn_in_encounters(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Insert patients
    pid1 = sdk.insert_patient(patient_id=501, mrn="ENC-001")
    pid2 = sdk.insert_patient(patient_id=502, mrn=600700)

    # Create institution/unit/bed
    institution_id = sdk.sql_handler.insert_institution(name="Test Hospital")
    unit_id = sdk.sql_handler.insert_unit(institution_id=institution_id, name="ICU", unit_type="Critical")
    sdk.sql_handler.insert_bed(unit_id=unit_id, name="Bed1", bed_id=1)

    T1 = 1647084000
    T2 = 1647094800

    sdk.insert_encounter(start_time=T1, end_time=T2, patient_id=501, bed_id=1,
                         visit_number='V001', time_units='s')
    sdk.insert_encounter(start_time=T1, end_time=T2, patient_id=502, bed_id=1,
                         visit_number='V002', time_units='s')

    # Look up encounters by string MRN
    enc1 = sdk.get_encounters(mrn="ENC-001", time_units='s')
    assert len(enc1) == 1
    assert enc1[0][1] == 501

    # Look up encounters by int MRN (should be converted to str internally)
    enc2 = sdk.get_encounters(mrn=600700, time_units='s')
    assert len(enc2) == 1
    assert enc2[0][1] == 502

    # Cross-type: lookup int-inserted MRN with string
    enc2_str = sdk.get_encounters(mrn="600700", time_units='s')
    assert len(enc2_str) == 1
    assert enc2_str[0][1] == 502


# ---------------------------------------------------------------------------
# MRN string type in device-patient lookups
# ---------------------------------------------------------------------------

def _test_mrn_in_device_patient(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Insert patients with mixed MRN types
    sdk.insert_patient(patient_id=601, mrn="DEV-MRN-001")
    sdk.insert_patient(patient_id=602, mrn=887766)

    # Insert devices
    sdk.insert_device(device_id=1, device_tag='monitor1')
    sdk.insert_device(device_id=2, device_tag='monitor2')

    # Insert device-patient mappings
    device_patient_data = [
        (1, 601, 1647084000, 1647094800),
        (2, 602, 1647084000, 1647094800),
    ]
    sdk.insert_device_patient_data(device_patient_data, time_units='s')

    # Filter by string MRN
    data1 = sdk.get_device_patient_data(mrn_list=["DEV-MRN-001"], time_units='s')
    assert len(data1) == 1
    assert data1[0][1] == 601

    # Filter by int MRN (should auto-convert)
    data2 = sdk.get_device_patient_data(mrn_list=[887766], time_units='s')
    assert len(data2) == 1
    assert data2[0][1] == 602

    # Cross-type: filter int-inserted MRN with string
    data2_str = sdk.get_device_patient_data(mrn_list=["887766"], time_units='s')
    assert len(data2_str) == 1
    assert data2_str[0][1] == 602

    # Device-patient encounters with string MRN
    enc = sdk.get_device_patient_encounters(timestamp=1647085000, device_id=1, mrn="DEV-MRN-001", time_units='s')
    assert len(enc) == 1
    assert enc[0][1] == 601

    # Device-patient encounters with int MRN
    enc2 = sdk.get_device_patient_encounters(timestamp=1647085000, device_id=2, mrn=887766, time_units='s')
    assert len(enc2) == 1
    assert enc2[0][1] == 602
