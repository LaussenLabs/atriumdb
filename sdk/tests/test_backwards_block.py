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
import time

import pytest
from atriumdb import AtriumSDK, T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO, V_TYPE_INT64, V_TYPE_DELTA_INT64, \
    V_TYPE_DOUBLE
from tests.testing_framework import _test_for_both, slice_times_values
import numpy as np

DB_NAME = 'backwards_block'


def test_create_dataset():
    _test_for_both(DB_NAME, _test_create_dataset)


def _test_create_dataset(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    freq_hz = 500
    freq_nhz = int(freq_hz * (10 ** 9))
    period_ns = 10 ** 18 // freq_nhz

    x = np.linspace(0, 2 * np.pi, 10)
    sine_wave = np.sin(x)
    scaled_sine_wave = np.round(sine_wave * 1000).astype(np.int64)
    timestamps = np.arange(x.size, dtype=np.int64) * period_ns
    timestamps += 10 ** 12

    # Simple case: 1 negative gap
    start = timestamps[0]
    end = timestamps[-1]

    gap_data = np.array([scaled_sine_wave.size // 2, -(end - start) * 2])
    simple_timestamps = timestamps.copy()
    for index, duration in gap_data.reshape(-1, 2):
        simple_timestamps[index:] += duration

    measure_id = sdk.insert_measure("simple", freq_nhz, "units")

    # Time type 1
    device_id_1 = sdk.insert_device("type 1")
    sdk.write_data_easy(measure_id, device_id_1, simple_timestamps, scaled_sine_wave, freq_nhz)

    # Time type 2
    device_id_2 = sdk.insert_device("type 2")

    # Define the times data types to write_data
    raw_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO
    encoded_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

    # Define the value data types (integer data compresses a lot better)
    if np.issubdtype(scaled_sine_wave.dtype, np.integer):
        raw_v_t = V_TYPE_INT64
        encoded_v_t = V_TYPE_DELTA_INT64
    else:
        raw_v_t = V_TYPE_DOUBLE
        encoded_v_t = V_TYPE_DOUBLE

    # Write the data to the sdk
    sdk.write_data(
        measure_id, device_id_2, gap_data, scaled_sine_wave, freq_nhz, int(simple_timestamps[0]),
        raw_time_type=raw_t_t, raw_value_type=raw_v_t, encoded_time_type=encoded_t_t, encoded_value_type=encoded_v_t)

    # Query entire written data
    min_time = np.min(simple_timestamps)
    max_time = np.max(simple_timestamps)
    # expected_times, expected_values = slice_times_values(simple_timestamps, scaled_sine_wave, min_time, max_time + period_ns)
    expected_times, expected_values = slice_times_values(simple_timestamps, scaled_sine_wave, 0,
                                                         10**62)

    for device_id in [device_id_1, device_id_2]:
        # headers, r_times, r_values = sdk.get_data(measure_id, min_time, max_time + period_ns, device_id)
        headers, r_times, r_values = sdk.get_data(measure_id, 0, 2**62, device_id, sort=True, allow_duplicates=False)

        print()
        print(r_times)
        print(expected_times)
        print(np.array_equal(r_times, expected_times))
        print(np.array_equal(r_values, expected_values))

        # assert np.array_equal(r_times, expected_times)
        # assert np.array_equal(r_values, expected_values)
