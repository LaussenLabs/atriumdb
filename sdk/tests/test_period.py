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
import numpy as np
from atriumdb import AtriumSDK, DatasetDefinition
from tests.testing_framework import _test_for_both

DB_NAME = 'period_test'


def test_period_features():
    _test_for_both(DB_NAME, _period_simple)
    _test_for_both(DB_NAME, _period_weird)


def _period_simple(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Add data with odd period
    period_ms = 2_000
    period_ns = int(10**6 * period_ms)
    freq_nhz = 10**18 // period_ns
    start_time_ms = 0
    start_time_ns = start_time_ms * 10**6

    # Create measure
    measure_id = sdk.insert_measure(measure_tag="test", period=period_ms, units="anything", time_units="ms")
    device_id = sdk.insert_device("test_device")
    values = np.arange(20)

    sdk.write_segment(measure_id, device_id, values, start_time=start_time_ms, period=period_ms, time_units="ms")

    expected_end_time_ms = start_time_ms + values.size * period_ms
    expected_times_ms = np.arange(start_time_ms, expected_end_time_ms, period_ms)

    headers, actual_times, actual_values = sdk.get_data(
        measure_id, start_time_ms, expected_end_time_ms, device_id=device_id, time_units="ms")

    # NumPy-based equality tests that handle floating point precision
    # Test values equality
    np.testing.assert_array_equal(actual_values, values,
                                  err_msg="Retrieved values don't match original values")

    # Test times equality with floating point tolerance
    np.testing.assert_allclose(actual_times, expected_times_ms,
                               rtol=1e-10, atol=1e-6,
                               err_msg="Retrieved times don't match expected times")

    # iterate over it
    dataset_definition = DatasetDefinition(measures=['test'], device_tags={'test_device': "all"})

    window_size = period_ms * 4
    window_slide = period_ms * 2

    window_size_ns = window_size * 10**6
    window_slide_ns = window_slide * 10**6

    iterator = sdk.get_iterator(dataset_definition, window_size, window_slide, iterator_type='lightmapped', time_units="ms")
    measure_tuple = ("test", freq_nhz / 10**9, 'anything')

    # Loop through all windows
    window_count = 0
    for window in iterator:
        signal_dict = window.signals[measure_tuple]

        # Calculate expected data for this window
        expected_window_start = start_time_ns + (window_count * window_slide_ns)
        expected_window_end = expected_window_start + window_size_ns
        expected_window_times = np.arange(expected_window_start, expected_window_end, period_ns)

        # Find the corresponding values for this time range
        # Convert times back to indices in the original values array
        start_idx = int((expected_window_start - start_time_ns) // period_ns)

        # Create expected values array with full window size, padding with NaN where needed
        expected_window_values = np.full(len(expected_window_times), np.nan)

        # Fill in the available values
        for i in range(len(expected_window_times)):
            value_idx = start_idx + i
            if 0 <= value_idx < len(values):
                expected_window_values[i] = values[value_idx]

        # Test window times equality with floating point tolerance
        np.testing.assert_allclose(signal_dict['times'], expected_window_times,
                                   rtol=1e-10, atol=1e-6,
                                   err_msg=f"Window {window_count} times don't match expected times")

        # Test window values equality (handle NaN values properly)
        np.testing.assert_array_equal(signal_dict['values'], expected_window_values,
                                      err_msg=f"Window {window_count} values don't match expected values")

        window_count += 1

def _period_weird(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Add data with odd period
    period_ms = 2_001
    period_ns = int(10**6 * period_ms)
    freq_nhz = 10**18 // period_ns
    start_time_ms = 0
    start_time_ns = start_time_ms * 10**6

    # Create measure
    measure_id = sdk.insert_measure(measure_tag="test", period=period_ms, units="anything", time_units="ms")
    device_id = sdk.insert_device("test_device")
    values = np.arange(20)

    sdk.write_segment(measure_id, device_id, values, start_time=start_time_ms, period=period_ms, time_units="ms")

    expected_end_time_ms = start_time_ms + values.size * period_ms
    expected_times_ms = np.arange(start_time_ms, expected_end_time_ms, period_ms)

    headers, actual_times, actual_values = sdk.get_data(
        measure_id, start_time_ms, expected_end_time_ms, device_id=device_id, time_units="ms")

    # NumPy-based equality tests that handle floating point precision
    # Test values equality
    np.testing.assert_array_equal(actual_values, values,
                                  err_msg="Retrieved values don't match original values")

    # Test times equality with floating point tolerance
    np.testing.assert_allclose(actual_times, expected_times_ms,
                               rtol=1e-10, atol=1e-6,
                               err_msg="Retrieved times don't match expected times")

    # iterate over it
    dataset_definition = DatasetDefinition(measures=['test'], device_tags={'test_device': "all"})

    window_size = period_ms * 4
    window_slide = period_ms * 2

    window_size_ns = window_size * 10**6
    window_slide_ns = window_slide * 10**6

    iterator = sdk.get_iterator(dataset_definition, window_size, window_slide, iterator_type='lightmapped', time_units="ms")
    measure_tuple = ("test", freq_nhz / 10**9, 'anything')

    # Loop through all windows
    window_count = 0
    for window in iterator:
        signal_dict = window.signals[measure_tuple]

        # Calculate expected data for this window
        expected_window_start = start_time_ns + (window_count * window_slide_ns)
        expected_window_end = expected_window_start + window_size_ns
        expected_window_times = np.arange(expected_window_start, expected_window_end, period_ns)

        # Find the corresponding values for this time range
        # Convert times back to indices in the original values array
        start_idx = int((expected_window_start - start_time_ns) // period_ns)

        # Create expected values array with full window size, padding with NaN where needed
        expected_window_values = np.full(len(expected_window_times), np.nan)

        # Fill in the available values
        for i in range(len(expected_window_times)):
            value_idx = start_idx + i
            if 0 <= value_idx < len(values):
                expected_window_values[i] = values[value_idx]

        # Test window times equality with floating point tolerance
        np.testing.assert_allclose(signal_dict['times'], expected_window_times,
                                   rtol=1e-10, atol=1e-6,
                                   err_msg=f"Window {window_count} times don't match expected times")

        # Test window values equality (handle NaN values properly)
        np.testing.assert_array_equal(signal_dict['values'], expected_window_values,
                                      err_msg=f"Window {window_count} values don't match expected values")

        window_count += 1