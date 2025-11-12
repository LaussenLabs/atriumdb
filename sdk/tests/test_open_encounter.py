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

from atriumdb import AtriumSDK, T_TYPE_TIMESTAMP_ARRAY_INT64_NANO, V_TYPE_INT64, V_TYPE_DELTA_INT64
import numpy as np
import time

from tests.test_transfer_info import insert_random_patients
from tests.testing_framework import _test_for_both, create_sibling_sdk

DB_NAME = 'atrium-open-ended-mapping'

SEED = 42


def test_open_ended_device_patient_mapping():
    _test_for_both(DB_NAME, _test_open_ended_mapping)


def _test_open_ended_mapping(db_type, dataset_location, connection_params):
    AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk = AtriumSDK(dataset_location, db_type, connection_params, num_threads=40)

    np.random.seed(SEED)

    # Create test data
    device_id_1 = sdk.insert_device(device_tag="device_open_ended_1")
    device_id_2 = sdk.insert_device(device_tag="device_open_ended_2")
    device_id_3 = sdk.insert_device(device_tag="device_closed_1")

    patient_id_1 = insert_random_patients(sdk, 1)[0]
    patient_id_2 = insert_random_patients(sdk, 1)[0]
    patient_id_3 = insert_random_patients(sdk, 1)[0]

    # Define time periods
    freq_hz = 100
    freq_nano = freq_hz * 1_000_000_000
    period_nano = int(10 ** 18 // freq_nano)

    # Base time for all operations (simulate "past" time)
    base_time = int(time.time() * 1_000_000_000) - (24 * 3600 * 1_000_000_000)  # 24 hours ago

    # Time ranges for data
    early_start = base_time
    early_end = base_time + (3600 * 1_000_000_000)  # 1 hour of data

    middle_start = base_time + (2 * 3600 * 1_000_000_000)  # 2 hours after base
    middle_end = base_time + (4 * 3600 * 1_000_000_000)  # 4 hours after base

    late_start = base_time + (6 * 3600 * 1_000_000_000)  # 6 hours after base
    late_end = base_time + (8 * 3600 * 1_000_000_000)  # 8 hours after base

    future_start = base_time + (10 * 3600 * 1_000_000_000)  # 10 hours after base
    future_end = base_time + (12 * 3600 * 1_000_000_000)  # 12 hours after base

    # Test Case 1: Open-ended mapping from early time
    # Patient 1 is mapped to device 1 from early_start with no end time
    sdk.insert_device_patient_data([(device_id_1, patient_id_1, early_start, None)])

    # Test Case 2: Open-ended mapping from middle time
    # Patient 2 is mapped to device 2 from middle_start with no end time
    sdk.insert_device_patient_data([(device_id_2, patient_id_2, middle_start, None)])

    # Test Case 3: Closed mapping (for comparison)
    # Patient 3 is mapped to device 3 from late_start to late_end (closed interval)
    sdk.insert_device_patient_data([(device_id_3, patient_id_3, late_start, late_end)])

    # Create measures
    measure_tag = "test_signal"
    units = "mV"
    measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq_nano, units=units)

    # Insert data for all three devices across different time periods
    # Device 1: Early, Middle, Late, and Future data
    _insert_test_data(sdk, measure_id, device_id_1, early_start, early_end, period_nano, freq_nano, value_offset=100)
    _insert_test_data(sdk, measure_id, device_id_1, middle_start, middle_end, period_nano, freq_nano, value_offset=200)
    _insert_test_data(sdk, measure_id, device_id_1, late_start, late_end, period_nano, freq_nano, value_offset=300)
    _insert_test_data(sdk, measure_id, device_id_1, future_start, future_end, period_nano, freq_nano, value_offset=400)

    # Device 2: Middle, Late, and Future data (no early data)
    _insert_test_data(sdk, measure_id, device_id_2, middle_start, middle_end, period_nano, freq_nano, value_offset=500)
    _insert_test_data(sdk, measure_id, device_id_2, late_start, late_end, period_nano, freq_nano, value_offset=600)
    _insert_test_data(sdk, measure_id, device_id_2, future_start, future_end, period_nano, freq_nano, value_offset=700)

    # Device 3: Only Late data (within closed mapping window)
    _insert_test_data(sdk, measure_id, device_id_3, late_start, late_end, period_nano, freq_nano, value_offset=800)
    # Also insert data outside the closed mapping window (should not be retrievable by patient_id)
    _insert_test_data(sdk, measure_id, device_id_3, future_start, future_end, period_nano, freq_nano, value_offset=900)

    # Run assertions
    print("\n=== Testing Open-Ended Device-Patient Mappings ===\n")

    # Test 1: Query patient 1 data in early period (should work - within open-ended mapping)
    print("Test 1: Patient 1, Early Period (within open-ended mapping from early_start)")
    _assert_patient_query(sdk, patient_id_1, measure_id, early_start, early_end, period_nano,
                          expected_offset=100, should_have_data=True)

    # Test 2: Query patient 1 data in late period (should work - open-ended extends to present)
    print("Test 2: Patient 1, Late Period (open-ended mapping extends)")
    _assert_patient_query(sdk, patient_id_1, measure_id, late_start, late_end, period_nano,
                          expected_offset=300, should_have_data=True)

    # Test 3: Query patient 1 data in future period (should work - open-ended extends to present)
    print("Test 3: Patient 1, Future Period (open-ended mapping extends)")
    _assert_patient_query(sdk, patient_id_1, measure_id, future_start, future_end, period_nano,
                          expected_offset=400, should_have_data=True)

    # Test 4: Query patient 2 data in early period (should have no data - mapping starts at middle_start)
    print("Test 4: Patient 2, Early Period (before mapping start_time)")
    _assert_patient_query(sdk, patient_id_2, measure_id, early_start, early_end, period_nano,
                          expected_offset=None, should_have_data=False)

    # Test 5: Query patient 2 data in middle period (should work - at mapping start)
    print("Test 5: Patient 2, Middle Period (at open-ended mapping start)")
    _assert_patient_query(sdk, patient_id_2, measure_id, middle_start, middle_end, period_nano,
                          expected_offset=500, should_have_data=True)

    # Test 6: Query patient 2 data in future period (should work - open-ended extends)
    print("Test 6: Patient 2, Future Period (open-ended mapping extends)")
    _assert_patient_query(sdk, patient_id_2, measure_id, future_start, future_end, period_nano,
                          expected_offset=700, should_have_data=True)

    # Test 7: Query patient 3 data in late period (should work - within closed mapping)
    print("Test 7: Patient 3, Late Period (within closed mapping)")
    _assert_patient_query(sdk, patient_id_3, measure_id, late_start, late_end, period_nano,
                          expected_offset=800, should_have_data=True)

    # Test 8: Query patient 3 data in future period (should have no data - outside closed mapping)
    print("Test 8: Patient 3, Future Period (outside closed mapping)")
    _assert_patient_query(sdk, patient_id_3, measure_id, future_start, future_end, period_nano,
                          expected_offset=None, should_have_data=False)

    # Test 9: Verify device queries still work (not using patient_id)
    print("Test 9: Device 3, Future Period (device query without patient_id)")
    _assert_device_query(sdk, device_id_3, measure_id, future_start, future_end, period_nano,
                         expected_offset=900, should_have_data=True)

    # Test with caching
    print("\n=== Testing with Cached SDK ===\n")
    sdk_cached = AtriumSDK(
        dataset_location=dataset_location, metadata_connection_type=db_type, connection_params=connection_params)

    for device_id in sdk_cached.get_all_devices():
        sdk_cached.load_device(device_id)

    # Repeat key tests with cached SDK
    print("Cached Test 1: Patient 1, Future Period")
    _assert_patient_query(sdk_cached, patient_id_1, measure_id, future_start, future_end, period_nano,
                          expected_offset=400, should_have_data=True)

    print("Cached Test 2: Patient 2, Early Period (before mapping)")
    _assert_patient_query(sdk_cached, patient_id_2, measure_id, early_start, early_end, period_nano,
                          expected_offset=None, should_have_data=False)

    print("Cached Test 3: Patient 3, Future Period (outside closed mapping)")
    _assert_patient_query(sdk_cached, patient_id_3, measure_id, future_start, future_end, period_nano,
                          expected_offset=None, should_have_data=False)

    print("\n=== All Open-Ended Mapping Tests Passed ===\n")


def test_multiple_overlapping_mappings():
    _test_for_both(DB_NAME + '-overlapping', _test_overlapping_mappings)


def _test_overlapping_mappings(db_type, dataset_location, connection_params):
    AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk = AtriumSDK(dataset_location, db_type, connection_params, num_threads=40)

    np.random.seed(SEED)

    device_id = sdk.insert_device(device_tag="device_multiple_mappings")
    patient_id_1 = insert_random_patients(sdk, 1)[0]
    patient_id_2 = insert_random_patients(sdk, 1)[0]

    freq_hz = 100
    freq_nano = freq_hz * 1_000_000_000
    period_nano = int(10 ** 18 // freq_nano)

    base_time = int(time.time() * 1_000_000_000) - (24 * 3600 * 1_000_000_000)

    period_1_start = base_time
    period_1_end = base_time + (2 * 3600 * 1_000_000_000)

    period_2_start = base_time + (3 * 3600 * 1_000_000_000)
    period_2_end = base_time + (5 * 3600 * 1_000_000_000)

    period_3_start = base_time + (6 * 3600 * 1_000_000_000)
    period_3_end = base_time + (8 * 3600 * 1_000_000_000)

    # Scenario: Device was with patient 1 (closed), then patient 2 (open-ended)
    # Patient 1: period_1_start to period_1_end (closed)
    sdk.insert_device_patient_data([(device_id, patient_id_1, period_1_start, period_1_end)])

    # Patient 2: period_2_start to None (open-ended)
    sdk.insert_device_patient_data([(device_id, patient_id_2, period_2_start, None)])

    measure_tag = "test_signal_overlap"
    units = "mV"
    measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq_nano, units=units)

    # Insert data for all periods
    _insert_test_data(sdk, measure_id, device_id, period_1_start, period_1_end, period_nano, freq_nano,
                      value_offset=100)
    _insert_test_data(sdk, measure_id, device_id, period_2_start, period_2_end, period_nano, freq_nano,
                      value_offset=200)
    _insert_test_data(sdk, measure_id, device_id, period_3_start, period_3_end, period_nano, freq_nano,
                      value_offset=300)

    print("\n=== Testing Overlapping/Sequential Mappings ===\n")

    # Test patient 1 can access period 1 data
    print("Test 1: Patient 1, Period 1 (within closed mapping)")
    _assert_patient_query(sdk, patient_id_1, measure_id, period_1_start, period_1_end, period_nano,
                          expected_offset=100, should_have_data=True)

    # Test patient 1 cannot access period 2 data
    print("Test 2: Patient 1, Period 2 (after closed mapping ended)")
    _assert_patient_query(sdk, patient_id_1, measure_id, period_2_start, period_2_end, period_nano,
                          expected_offset=None, should_have_data=False)

    # Test patient 2 cannot access period 1 data
    print("Test 3: Patient 2, Period 1 (before open-ended mapping started)")
    _assert_patient_query(sdk, patient_id_2, measure_id, period_1_start, period_1_end, period_nano,
                          expected_offset=None, should_have_data=False)

    # Test patient 2 can access period 2 data
    print("Test 4: Patient 2, Period 2 (at open-ended mapping start)")
    _assert_patient_query(sdk, patient_id_2, measure_id, period_2_start, period_2_end, period_nano,
                          expected_offset=200, should_have_data=True)

    # Test patient 2 can access period 3 data (open-ended extends)
    print("Test 5: Patient 2, Period 3 (open-ended mapping extends)")
    _assert_patient_query(sdk, patient_id_2, measure_id, period_3_start, period_3_end, period_nano,
                          expected_offset=300, should_have_data=True)

    print("\n=== All Overlapping Mapping Tests Passed ===\n")


def _insert_test_data(sdk, measure_id, device_id, start_time, end_time, period_nano, freq_nano, value_offset=0):
    num_samples = int((end_time - start_time) // period_nano)
    time_arr = np.arange(num_samples, dtype=np.int64) * period_nano + start_time

    # Create simple incrementing values with an offset for identification
    values = np.arange(num_samples, dtype=np.int64) + value_offset

    sdk.write_time_value_pairs(
        measure_id=measure_id,
        device_id=device_id,
        times=time_arr,
        values=values,
        freq=freq_nano,
        freq_units="nHz",
        time_units="ns",
    )


def _assert_patient_query(sdk, patient_id, measure_id, start_time, end_time, period_nano,
                          expected_offset, should_have_data):
    headers, read_times, read_values = sdk.get_data(
        measure_id=measure_id,
        start_time_n=start_time,
        end_time_n=end_time,
        patient_id=patient_id
    )


    if should_have_data:
        assert read_values is not None, \
            f"Expected data for patient {patient_id} but read_values is None"
        assert len(read_values) > 0, \
            f"Expected data for patient {patient_id} but read_values is empty (length 0)"

        if expected_offset is not None:
            # Verify the values match expected pattern
            num_samples = int((end_time - start_time) // period_nano)
            expected_values = np.arange(num_samples, dtype=np.int64) + expected_offset

            assert len(read_values) == len(expected_values), \
                f"Expected {len(expected_values)} samples, got {len(read_values)}"

            assert np.array_equal(read_values, expected_values), \
                f"Values don't match. Expected offset {expected_offset}, got different values"

        print(f"Successfully retrieved {len(read_values)} samples")
    else:
        assert read_values is None or len(read_values) == 0, \
            f"Expected no data for patient {patient_id} in range [{start_time}, {end_time}], but got {len(read_values) if read_values is not None else 0} samples"

        print(f"Correctly returned no data (as expected)")


def _assert_device_query(sdk, device_id, measure_id, start_time, end_time, period_nano,
                         expected_offset, should_have_data):
    headers, read_times, read_values = sdk.get_data(
        measure_id=measure_id,
        start_time_n=start_time,
        end_time_n=end_time,
        device_id=device_id
    )

    if should_have_data:
        assert read_values is not None and len(read_values) > 0, \
            f"Expected data for device {device_id} in range [{start_time}, {end_time}], but got none"

        if expected_offset is not None:
            num_samples = int((end_time - start_time) // period_nano)
            expected_values = np.arange(num_samples, dtype=np.int64) + expected_offset

            assert len(read_values) == len(expected_values), \
                f"Expected {len(expected_values)} samples, got {len(read_values)}"

            assert np.array_equal(read_values, expected_values), \
                f"Values don't match. Expected offset {expected_offset}"

        print(f"Successfully retrieved {len(read_values)} samples via device query")
    else:
        assert read_values is None or len(read_values) == 0, \
            f"Expected no data for device {device_id}, but got data"

        print(f"Correctly returned no data (as expected)")

