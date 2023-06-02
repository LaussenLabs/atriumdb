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

import bisect
import numpy as np
from atriumdb import AtriumSDK
from tests.generate_wfdb import get_records
from tests.test_time_type_switch import convert_gap_data_to_timestamp_arr
from tests.testing_framework import _test_for_both

DB_NAME = 'comprehensive'


def test_comprehensive_workflow():
    _test_for_both(DB_NAME, _test_comprehensive_workflow)


def _test_comprehensive_workflow(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # define some constants
    measure_units = "mV"
    measure_tag = "ECG_II"
    freq_hz = 500
    freq_nhz = freq_hz * (10 ** 9)
    period_ns = (10 ** 9) // freq_hz
    device_tag = "dev-1"

    # insert measure and device
    new_measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq_hz, units=measure_units, freq_units="Hz")
    new_device_id = sdk.insert_device(device_tag=device_tag)

    master_times, master_values = [], []

    # use wfdb_generator to generate multiple sets of waveform data
    for record in get_records(dataset_name='mitdb'):
        value_data = record.p_signal.T[0]

        # fictional start times and gap data
        start_time_nano = 1234567890
        gap_data = [10_000, 24_000_000, 12_000, 138_000_000, 54_403, 34_560_000_000]
        gap_data = np.array(gap_data, dtype=np.int64)
        timestamp_arr = convert_gap_data_to_timestamp_arr(gap_data, value_data.size, period_ns, start_time_nano)

        # Write Data with both time types
        for raw_t_t in [1, 2]:
            sdk.write_data(new_measure_id, new_device_id, gap_data, value_data, freq_nhz, start_time_nano,
                           raw_time_type=raw_t_t, raw_value_type=raw_v_t, encoded_time_type=encoded_t_t,
                           encoded_value_type=encoded_v_t, scale_m=None, scale_b=None)

            if raw_t_t == 1:
                master_times.append(timestamp_arr)
                master_values.append(value_data)

    # flatten master_times and master_values into 1D lists
    master_times = [time for sublist in master_times for time in sublist]
    master_values = [value for sublist in master_values for value in sublist]

    # Test get_data with varying measure_ids, device_ids start_times and end_times
    for start_time, end_time in zip(master_times[:-1], master_times[1:]):
        _, read_time_data, read_value_data = sdk.get_data(
            measure_id=new_measure_id,
            start_time_n=start_time,
            end_time_n=end_time,
            device_id=new_device_id,
            analog=False
        )

        left, right = bisect.bisect_left(master_times, start_time), bisect.bisect_right(master_times, end_time)
        expected_times, expected_values = master_times[left:right], master_values[left:right]

        assert np.array_equal(expected_times, read_time_data)
        assert np.array_equal(expected_values, read_value_data)
