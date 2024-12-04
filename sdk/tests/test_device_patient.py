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

DB_NAME = 'device_patient'


def test_device_patient():
    _test_for_both(DB_NAME, _test_device_patient)


def _test_device_patient(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    current_time_s = time.time_ns() / 1e9

    # Prepare test data for device-patient mappings
    device_patient_data = [
        (1, 100, 1647084000, 1647094800),  # Device 1 associated with patient 100 from T1 to T2
        (2, 200, 1647094800, 1647105600),  # Device 2 associated with patient 200 from T2 to T3
        (3, 400, 1647084000, current_time_s),  # Device 3 associated with patient 400 starting at T1, no end time
        (1, 300, 1647105600, 1647116400),  # Device 1 associated with patient 300 from T3 to T4
        # Overlapping device-patient mapping to test warning
        (1, 500, 1647105600, 1647116400),  # Device 1 also associated with patient 500 from T3 to T4
    ]
    time_units = 's'  # Seconds

    # Prepare patient data with MRNs
    patient_data = [
        {'patient_id': 100, 'mrn': 12345},
        {'patient_id': 200, 'mrn': 23456},
        {'patient_id': 300, 'mrn': 34567},
        {'patient_id': 400, 'mrn': 45678},
        {'patient_id': 500, 'mrn': 56789},
    ]

    for patient_dict in patient_data:
        sdk.insert_patient(patient_id=patient_dict['patient_id'], mrn=patient_dict['mrn'])

    # Prepare device data with tags
    device_data = [
        {'device_id': 1, 'device_tag': 'device1'},
        {'device_id': 2, 'device_tag': 'device2'},
        {'device_id': 3, 'device_tag': 'device3'},
    ]

    for device_dict in device_data:
        sdk.insert_device(device_id=device_dict['device_id'], device_tag=device_dict['device_tag'])

    # Insert the device-patient data
    sdk.insert_device_patient_data(device_patient_data, time_units=time_units)

    # Test get_device_patient_data without filters
    all_data = sdk.get_device_patient_data(time_units=time_units)
    # Expected data with current end time for open-ended associations
    expected_all_data = [
        (1, 100, 1647084000.0, 1647094800.0),
        (1, 300, 1647105600.0, 1647116400.0),
        (1, 500, 1647105600.0, 1647116400.0),
        (2, 200, 1647094800.0, 1647105600.0),
        (3, 400, 1647084000.0, current_time_s),
    ]
    assert len(all_data) == len(expected_all_data), f"Assertion failed at line __line__"
    for expected, actual in zip(sorted(expected_all_data), sorted(all_data)):
        assert expected[0] == actual[0], f"Assertion failed at line __line__"  # device_id
        assert expected[1] == actual[1], f"Assertion failed at line __line__"  # patient_id
        assert expected[2] == actual[2], f"Assertion failed at line __line__"  # start_time
        # Allow a small difference in end_time due to current time
        if expected[3] == current_time_s:
            assert abs(expected[3] - actual[3]) < 5, f"Assertion failed at line __line__"  # Allow up to 5 seconds difference
        else:
            assert expected[3] == actual[3], f"Assertion failed at line __line__"

    # Test get_device_patient_data with device_id_list filter
    device_1_data = sdk.get_device_patient_data(device_id_list=[1], time_units=time_units)
    expected_device_1_data = [
        (1, 100, 1647084000.0, 1647094800.0),
        (1, 300, 1647105600.0, 1647116400.0),
        (1, 500, 1647105600.0, 1647116400.0),
    ]
    assert sorted(device_1_data) == sorted(expected_device_1_data), f"Assertion failed at line __line__"

    # Test get_device_patient_data with patient_id_list filter
    patient_200_data = sdk.get_device_patient_data(patient_id_list=[200], time_units=time_units)
    expected_patient_200_data = [
        (2, 200, 1647094800.0, 1647105600.0),
    ]
    assert patient_200_data == expected_patient_200_data, f"Assertion failed at line __line__"

    # Test get_device_patient_data with mrn_list filter
    mrn_list = [12345]
    data_by_mrn = sdk.get_device_patient_data(mrn_list=mrn_list, time_units=time_units)
    expected_data_by_mrn = [
        (1, 100, 1647084000.0, 1647094800.0),
    ]
    assert data_by_mrn == expected_data_by_mrn, f"Assertion failed at line __line__"

    # Test get_device_patient_data with start_time and end_time filters
    data_in_time_range = sdk.get_device_patient_data(
        start_time=1647084000, end_time=1647090000, time_units=time_units)
    expected_time_filtered_data = [
        (1, 100, 1647084000.0, 1647090000.0),
        (3, 400, 1647084000.0, 1647090000.0),
    ]
    for expected, actual in zip(sorted(expected_time_filtered_data), sorted(data_in_time_range)):
        assert expected[0] == actual[0], f"Assertion failed at line __line__"  # device_id
        assert expected[1] == actual[1], f"Assertion failed at line __line__"  # patient_id
        assert expected[2] == actual[2], f"Assertion failed at line __line__"  # start_time
        assert abs(expected[3] - actual[3]) < 1, f"Assertion failed at line __line__"  # Allow up to 1 second difference

    # Test get_device_patient_data with different time units
    data_in_ms = sdk.get_device_patient_data(time_units='ms')
    expected_data_in_ms = [
        (1, 100, 1647084000.0 * 1e3, 1647094800.0 * 1e3),
        (1, 300, 1647105600.0 * 1e3, 1647116400.0 * 1e3),
        (1, 500, 1647105600.0 * 1e3, 1647116400.0 * 1e3),
        (2, 200, 1647094800.0 * 1e3, 1647105600.0 * 1e3),
        (3, 400, 1647084000.0 * 1e3, current_time_s * 1e3),
    ]
    for expected, actual in zip(sorted(expected_data_in_ms), sorted(data_in_ms)):
        assert expected[0] == actual[0], f"Assertion failed at line __line__"
        assert expected[1] == actual[1], f"Assertion failed at line __line__"
        assert expected[2] == actual[2], f"Assertion failed at line __line__"
        if expected[3] == current_time_s * 1e3:
            assert abs(expected[3] - actual[3]) < 5000, f"Assertion failed at line __line__"  # Allow up to 5 seconds difference in ms
        else:
            assert expected[3] == actual[3], f"Assertion failed at line __line__"

    # Test get_device_patient_encounters with device_tag and time
    encounters = sdk.get_device_patient_encounters(timestamp=1647085000, device_tag='device1', time_units=time_units)
    expected_encounters = [
        (1, 100, 1647084000.0, 1647094800.0),
    ]
    assert encounters == expected_encounters, f"Assertion failed at line __line__"

    # Test get_device_patient_encounters with device_id, mrn, and time
    encounters = sdk.get_device_patient_encounters(timestamp=1647085000, device_id=1, mrn=12345, time_units=time_units)
    assert encounters == expected_encounters, f"Assertion failed at line __line__"

    # Test get_device_patient_encounters with no encounter found
    encounters_none = sdk.get_device_patient_encounters(timestamp=1647120000, device_id=2, time_units=time_units)
    assert encounters_none == [], f"Assertion failed at line __line__"

    # Test error handling for invalid time units
    with pytest.raises(ValueError):
        sdk.insert_device_patient_data(device_patient_data, time_units='invalid_unit')

    with pytest.raises(ValueError):
        sdk.get_device_patient_encounters(timestamp=1647085000, device_id=1, time_units='invalid_unit')
