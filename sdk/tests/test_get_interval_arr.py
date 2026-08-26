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

from atriumdb import AtriumSDK
import numpy as np
from tests.test_transfer_info import insert_random_patients
from typing import List, Tuple
from atriumdb.helpers.block_constants import COMPRESSION_TYPES, TIME_TYPES, VALUE_TYPES

from tests.testing_framework import _test_for_both

DB_NAME = 'atrium-interval'

MAX_RECORDS = 1


def test_get_interval_arr():
    _test_for_both(DB_NAME, _test_get_interval_arr)


def test_get_interval_arr_exact():
    _test_for_both(f"{DB_NAME}-exact", _test_get_interval_arr_exact)


def _test_get_interval_arr(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Write and insert data
    device_tag = "device_tag_1"
    device_id = sdk.insert_device(device_tag)
    freq_hz = 1
    measure_tag = "measure_tag_1"
    measure_id = sdk.insert_measure(measure_tag, freq_hz)

    # Insert random patients
    num_patients = 5
    patient_ids = insert_random_patients(sdk, num_patients)

    # Map patients to devices over different times
    device_patient_data: List[Tuple[int, int, int, int]] = []
    start_time_s = 1234567890
    end_time_s = start_time_s + 3600
    start_time_nano = start_time_s * (10 ** 9)
    end_time_nano = end_time_s * (10 ** 9)
    interval = (end_time_nano - start_time_nano) // num_patients
    # 15-second gap. write_data_easy now applies a smart interval-index gap
    # tolerance (max(10 * period, 200ms) = 10s at 1 Hz), so the gaps separating
    # these segments must exceed that tolerance to remain distinct intervals.
    gap_seconds = 15
    gap_nano = gap_seconds * (10 ** 9)

    expected_intervals = {}
    combined_intervals = []
    for idx, patient_id in enumerate(patient_ids):
        start = start_time_nano + (idx * (interval + gap_nano))
        end = start + interval
        device_patient_data.append((device_id, patient_id, start, end))
        expected_intervals[patient_id] = np.array([[start, end]])
        combined_intervals.append([start, end])

    sdk.insert_device_patient_data(device_patient_data)

    # Generate time_data with gaps
    time_data = []
    for idx in range(num_patients):
        start = start_time_s + (idx * (interval // (10 ** 9) + gap_seconds))
        end = start + (interval // (10 ** 9))
        time_data.extend(np.arange(start, end))

    time_data = np.array(time_data, dtype=np.int64)
    value_data = np.sin(time_data)

    # Write data with gaps
    sdk.write_data_easy(measure_id=measure_id, device_id=device_id, time_data=time_data, value_data=value_data,
                        freq=freq_hz, time_units="s", freq_units="Hz")

    # Test get_interval_array based on device
    start_time_nano = int(combined_intervals[0][0])
    end_time_nano = int(combined_intervals[-1][1])
    interval_arr_device = sdk.get_interval_array(measure_id=measure_id, device_tag=device_tag, start=start_time_nano,
                                                 end=end_time_nano)

    assert interval_arr_device.shape[0] > 0, "No intervals found for the device"
    assert np.array_equal(interval_arr_device, np.array(combined_intervals, dtype=np.int64)), "Unexpected intervals for device"

    # Test get_interval_array based on patient
    for patient_id in patient_ids:
        interval_arr_patient = sdk.get_interval_array(measure_id=measure_id, patient_id=patient_id,
                                                      start=start_time_nano, end=end_time_nano)

        assert interval_arr_patient.shape[0] > 0, f"No intervals found for patient {patient_id}"
        assert np.array_equal(interval_arr_patient, expected_intervals[patient_id]), f"Unexpected intervals for patient {patient_id}"

    # Test for overlapping intervals
    freq_hz = 1
    period_s = 1
    device_id = sdk.insert_device("overlapping device")
    measure_id = sdk.insert_measure("overlapping signal", freq_hz, freq_units="Hz")

    start_1, end_1 = 0, 60
    times_1 = np.arange(start_1, end_1, period_s, dtype=np.int64)
    values_1 = np.sin(times_1)

    start_2, end_2 = 20, 30
    times_2 = np.arange(start_2, end_2, period_s, dtype=np.int64)
    values_2 = np.sin(times_2)

    sdk.write_data_easy(measure_id=measure_id, device_id=device_id, time_data=times_1, value_data=values_1, freq=freq_hz, time_units="s", freq_units="Hz")
    sdk.write_data_easy(measure_id=measure_id, device_id=device_id, time_data=times_2, value_data=values_2, freq=freq_hz, time_units="s", freq_units="Hz")

    start_time_nano = start_1 * 10**9
    end_time_nano = end_1 * 10**9
    interval_arr = sdk.get_interval_array(measure_id=measure_id, device_id=device_id, start=start_time_nano, end=end_time_nano)
    assert np.array_equal(interval_arr,
                          np.array([[start_time_nano, end_time_nano]], dtype=np.int64))


def _test_get_interval_arr_exact(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    period_ns = 1_000_000_000
    device_id = sdk.insert_device("exact_device")
    measure_id = sdk.insert_measure("exact_waveform", 1, freq_units="Hz")

    times = np.array([0, 1, 2, 10, 11, 12], dtype=np.int64) * period_ns
    values = np.arange(times.size, dtype=np.float64)
    sdk.write_time_value_pairs(
        measure_id, device_id, times, values, period=period_ns, time_units="ns", continuous=True)

    coarse = sdk.get_interval_array(measure_id=measure_id, device_id=device_id)
    exact = sdk.get_interval_array(measure_id=measure_id, device_id=device_id, exact=True)

    assert np.array_equal(coarse, np.array([[0, 13 * period_ns]], dtype=np.int64))
    assert np.array_equal(exact, np.array([[0, 3 * period_ns], [10 * period_ns, 13 * period_ns]], dtype=np.int64))

    merged_exact = sdk.get_interval_array(
        measure_id=measure_id, device_id=device_id, exact=True, gap_tolerance_nano=7 * period_ns)
    assert np.array_equal(merged_exact, np.array([[0, 13 * period_ns]], dtype=np.int64))

    patient_id = sdk.insert_patient(mrn=123456)
    sdk.insert_device_patient_data([(device_id, patient_id, period_ns, 11 * period_ns)])
    patient_exact = sdk.get_interval_array(measure_id=measure_id, patient_id=patient_id, exact=True)
    assert np.array_equal(patient_exact, np.array([[period_ns, 3 * period_ns],
                                                  [10 * period_ns, 11 * period_ns]], dtype=np.int64))

    sample_measure_id = sdk.insert_measure("exact_sample", 1, freq_units="Hz", signal_kind="sample")
    sdk.write_time_value_pairs(
        sample_measure_id, device_id, times, values, period=period_ns, time_units="ns", continuous=True)
    sample_coarse = sdk.get_interval_array(measure_id=sample_measure_id, device_id=device_id)
    sample_exact = sdk.get_interval_array(measure_id=sample_measure_id, device_id=device_id, exact=True)
    assert np.array_equal(sample_exact, sample_coarse)

    compressed_measure_id = sdk.insert_measure("exact_compressed_waveform", 1, freq_units="Hz")
    sdk.write_data(
        compressed_measure_id, device_id,
        np.array([3, 7 * period_ns], dtype=np.int64),
        np.arange(6, dtype=np.int64),
        freq_nhz=1_000_000_000,
        time_0=0,
        raw_time_type=TIME_TYPES['GAP_ARRAY_INT64_INDEX_DURATION_NS'],
        raw_value_type=VALUE_TYPES['INT64'],
        encoded_time_type=TIME_TYPES['GAP_ARRAY_INT64_INDEX_DURATION_NS'],
        encoded_value_type=VALUE_TYPES['DELTA_INT64'],
        t_compression=COMPRESSION_TYPES['ZSTD'],
        t_compression_level=3,
        continuous=True,
    )
    compressed_exact = sdk.get_interval_array(
        measure_id=compressed_measure_id, device_id=device_id, exact=True)
    assert np.array_equal(compressed_exact, np.array([[0, 3 * period_ns],
                                                     [10 * period_ns, 13 * period_ns]], dtype=np.int64))
