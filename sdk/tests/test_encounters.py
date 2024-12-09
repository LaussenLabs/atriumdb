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
import pytest
import time
import warnings
from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'encounters'

def test_encounters():
    _test_for_both(DB_NAME, _test_encounters)

def _test_encounters(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params
    )

    # Insert mock patients
    patients = [
        {'patient_id': 101, 'mrn': 1001},
        {'patient_id': 102, 'mrn': 1002},
        {'patient_id': 103, 'mrn': 1003},
    ]
    for p in patients:
        sdk.insert_patient(patient_id=p['patient_id'], mrn=p['mrn'])

    # Create a mock institution and unit for testing beds
    institution_id = sdk.sql_handler.insert_institution(name="Test Institution")
    unit_id = sdk.sql_handler.insert_unit(institution_id=institution_id, name="Test Unit", unit_type="Test Type")

    # Insert beds with known IDs
    beds = [
        {'bed_id': 10, 'bed_name': 'BedA'},
        {'bed_id': 20, 'bed_name': 'BedB'},
    ]
    for b in beds:
        sdk.sql_handler.insert_bed(unit_id=unit_id, name=b['bed_name'], bed_id=b.get('bed_id'))

    # Times (in seconds):
    T1 = 1647084000
    T2 = 1647094800
    T3 = 1647105600
    T4 = 1647116400

    # Define encounter data (using 's' as time units for simplicity)
    # 1. A completed encounter for patient 101 in BedA (from T1 to T2)
    # 2. An ongoing encounter for patient 102 in BedB (from T2 to now, no end time)
    # 3. Another completed encounter for patient 103 in BedA (from T3 to T4)
    sdk.insert_encounter(start_time=T1, end_time=T2, patient_id=101, bed_id=10, visit_number='VISIT101', time_units='s')
    sdk.insert_encounter(start_time=T2, end_time=None, patient_id=102, bed_id=20, visit_number='VISIT102', time_units='s')
    sdk.insert_encounter(start_time=T3, end_time=T4, patient_id=103, bed_id=10, visit_number='VISIT103', time_units='s')

    # Test get_encounters() without filters - should return all encounters
    all_encounters = sdk.get_encounters(time_units='s')

    # Sort by (patient_id, start_time) to have a deterministic order.
    all_encounters_sorted = sorted(all_encounters, key=lambda x: (x[1], x[3]))
    assert len(all_encounters_sorted) == 3, "Expected three encounters"

    # Encounter 1 (patient 101, T1-T2 in BedA)
    assert all_encounters_sorted[0][1] == 101
    assert all_encounters_sorted[0][2] == 10
    assert all_encounters_sorted[0][3] == float(T1)
    assert all_encounters_sorted[0][4] == float(T2)
    assert all_encounters_sorted[0][6] == 'VISIT101'

    # Encounter 2 (patient 102, T2-now in BedB)
    assert all_encounters_sorted[1][1] == 102
    assert all_encounters_sorted[1][2] == 20
    assert all_encounters_sorted[1][3] == float(T2)
    assert all_encounters_sorted[1][4] is None
    assert all_encounters_sorted[1][6] == 'VISIT102'

    # Encounter 3 (patient 103, T3-T4 in BedA)
    assert all_encounters_sorted[2][1] == 103
    assert all_encounters_sorted[2][2] == 10
    assert all_encounters_sorted[2][3] == float(T3)
    assert all_encounters_sorted[2][4] == float(T4)
    assert all_encounters_sorted[2][6] == 'VISIT103'

    # Test filtering by patient_id
    patient_101_encounters = sdk.get_encounters(patient_id=101, time_units='s')
    assert len(patient_101_encounters) == 1
    assert patient_101_encounters[0][1] == 101

    # Test filtering by mrn (patient 102 has mrn=1002)
    patient_102_encounters = sdk.get_encounters(mrn=1002, time_units='s')
    assert len(patient_102_encounters) == 1
    assert patient_102_encounters[0][1] == 102

    # Test filtering by bed_id
    bedA_encounters = sdk.get_encounters(bed_id=10, time_units='s')
    # Should have patient 101 (T1-T2) and 103 (T3-T4) in BedA
    assert len(bedA_encounters) == 2
    p_ids = sorted([e[1] for e in bedA_encounters])
    assert p_ids == [101, 103]

    # Test filtering by bed_name
    bedB_encounters = sdk.get_encounters(bed_name='BedB', time_units='s')
    assert len(bedB_encounters) == 1
    assert bedB_encounters[0][1] == 102

    # Test filtering by timestamp (encounters active at a specific time)
    # active means start_time <= timestamp < end_time
    # So only patient 102 should be active at T2
    active_at_T2 = sdk.get_encounters(timestamp=T2, time_units='s')

    assert len(active_at_T2) == 1
    assert active_at_T2[0][1] == 102

    # Test filtering by time range
    # From T1 to T2.5
    time_range_encounters = sdk.get_encounters(start_time=T1, end_time=(T2 + T3) / 2, time_units='s')
    assert len(time_range_encounters) == 2
    p_ids = [e[1] for e in time_range_encounters]
    assert 101 in p_ids, "Patient 101 encounter should be found in the time range"
    assert 102 in p_ids, "Patient 102 encounter should also be found in the time range"

    # Test different time_units (e.g. 'ms')
    patient_101_ms = sdk.get_encounters(patient_id=101, time_units='ms')
    assert len(patient_101_ms) == 1
    assert patient_101_ms[0][3] == T1 * 1000.0
    assert patient_101_ms[0][4] == T2 * 1000.0

    # Test no encounters found
    no_encounters = sdk.get_encounters(patient_id=999, time_units='s')
    assert no_encounters == [], "No encounters expected for non-existent patient"

    # Test error handling for invalid time units in insert_encounter()
    with pytest.raises(ValueError):
        sdk.insert_encounter(start_time=T1, end_time=T2, patient_id=101, bed_id=10, time_units='invalid_unit')

    # Test error handling for invalid time units in get_encounters()
    with pytest.raises(ValueError):
        sdk.get_encounters(time_units='invalid_unit')
