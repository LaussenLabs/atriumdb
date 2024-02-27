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

from tests.testing_framework import _test_for_both

DB_NAME = 'atrium-interval'

MAX_RECORDS = 1


def test_get_interval_arr():
    _test_for_both(DB_NAME, _test_get_interval_arr)


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
    gap_nano = 2 * (10 ** 9)  # 2-second gap

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
        start = start_time_s + (idx * (interval // (10 ** 9) + 2))
        end = start + (interval // (10 ** 9))
        time_data.extend(np.arange(start, end))

    time_data = np.array(time_data, dtype=np.int64)
    value_data = np.sin(time_data)

    # Write data with gaps
    sdk.write_data_easy(measure_id=measure_id, device_id=device_id, time_data=time_data, value_data=value_data,
                        freq=freq_hz, time_units="s", freq_units="Hz")

    # Test get_interval_array based on device
    start_time_nano = start_time_s * (10 ** 9)
    end_time_nano = end_time_s * (10 ** 9)
    interval_arr_device = sdk.get_interval_array(measure_id=measure_id, device_tag=device_tag, start=start_time_nano,
                                                 end=end_time_nano)

    assert interval_arr_device.shape[0] > 0, "No intervals found for the device"
    assert np.array_equal(interval_arr_device, np.array(combined_intervals, dtype=np.int64)), "Unexpected intervals for device"

    # Test get_interval_array based on patient
    # Needs Fixing
    for patient_id in patient_ids:
        interval_arr_patient = sdk.get_interval_array(measure_id=measure_id, patient_id=patient_id,
                                                      start=start_time_nano, end=end_time_nano)

        assert interval_arr_patient.shape[0] > 0, f"No intervals found for patient {patient_id}"
        assert np.array_equal(interval_arr_patient, expected_intervals[patient_id]), f"Unexpected intervals for patient {patient_id}"


