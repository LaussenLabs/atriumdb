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
from collections import defaultdict
import shutil
from pathlib import Path
import numpy as np
import pytest
from atriumdb import AtriumSDK, DatasetDefinition
from atriumdb.adb_functions import rebuild_intervals_from_existing_blocks
from tests.test_mit_bih import write_mit_bih_to_dataset
from tests.testing_framework import _test_for_both

DB_NAME = 'rebuild'
TEST_DIR = Path(__file__).parent
EXAMPLE_DATA_DIR = TEST_DIR / "example_data"


def calculate_continuous_intervals(times, period_ns):
    """
    Calculate continuous intervals from a sorted array of times using numpy vectorization.

    Args:
        times: numpy array of timestamps
        period_ns: period in nanoseconds between consecutive samples

    Returns:
        numpy array of shape (n, 2) where each row is [start_time, end_time + period_ns]
    """
    if len(times) == 0:
        return np.array([]).reshape(0, 2)

    if len(times) == 1:
        return np.array([[times[0], times[0] + period_ns]], dtype=np.int64)

    # Calculate differences between consecutive times
    diffs = np.diff(times)

    # Find where gaps occur (difference > period_ns)
    gap_indices = np.where(diffs > period_ns)[0]

    # Start times are: first time + times right after each gap
    start_times = np.concatenate([[times[0]], times[gap_indices + 1]])

    # End times are: times right before each gap + last time, then add period_ns
    end_times = np.concatenate([times[gap_indices], [times[-1]]]) + period_ns

    # Stack into intervals array
    intervals = np.column_stack([start_times, end_times])

    return intervals.astype(np.int64)


def test_rebuild():
    _test_for_both(DB_NAME, _test_rebuild)


def _test_rebuild(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # larger test
    write_mit_bih_to_dataset(sdk, max_records=2, seed=42)

    # Calculate expected intervals and warn if original intervals differ
    calculated_intervals_dict = {}
    for measure_id in sdk.get_all_measures():
        for device_id in sdk.get_all_devices():
            interval_array = sdk.get_interval_array(measure_id, device_id)
            freq_nhz = sdk.get_measure_info(measure_id)['freq_nhz']
            period_ns = 10 ** 18 // freq_nhz
            _, times, values = sdk.get_data(measure_id, 0, 2 ** 62, device_id=device_id)

            if len(times) > 0:
                # Calculate what the intervals should be from the times
                calculated_intervals = calculate_continuous_intervals(times, period_ns)

                # Warn if the original interval array doesn't match the calculated intervals
                if interval_array is not None and len(interval_array) > 0:
                    try:
                        np.testing.assert_array_equal(interval_array, calculated_intervals)
                    except AssertionError:
                        import warnings
                        warnings.warn(
                            f"Original interval array doesn't match calculated intervals for measure {measure_id}, device {device_id}"
                        )

                # Store the calculated intervals
                calculated_intervals_dict[(measure_id, device_id)] = calculated_intervals.copy()

    # Rebuild intervals from existing blocks
    rebuild_intervals_from_existing_blocks(sdk, interval_gap_tolerance_nano=0)

    # Verify that all interval arrays match the calculated intervals after rebuilding
    for measure_id in sdk.get_all_measures():
        for device_id in sdk.get_all_devices():
            rebuilt_interval_array = sdk.get_interval_array(measure_id, device_id)
            freq_nhz = sdk.get_measure_info(measure_id)['freq_nhz']
            period_ns = 10 ** 18 // freq_nhz
            _, times, values = sdk.get_data(measure_id, 0, 2 ** 62, device_id=device_id)

            # Check if this measure/device combination had intervals originally
            key = (measure_id, device_id)

            if len(times) > 0:
                # Calculate what the intervals should be from the times
                calculated_intervals = calculate_continuous_intervals(times, period_ns)

                # Verify the rebuilt intervals match the calculated intervals
                if key in calculated_intervals_dict:
                    assert rebuilt_interval_array is not None, \
                        f"Interval array for measure {measure_id}, device {device_id} is None after rebuild"

                    np.testing.assert_array_equal(
                        rebuilt_interval_array,
                        calculated_intervals,
                        err_msg=f"Rebuilt interval array doesn't match calculated intervals for measure {measure_id}, device {device_id}"
                    )

                    # Also verify it matches the stored calculated intervals
                    np.testing.assert_array_equal(
                        calculated_intervals_dict[key],
                        rebuilt_interval_array,
                        err_msg=f"Rebuilt intervals don't match calculated intervals for measure {measure_id}, device {device_id}"
                    )
            else:
                # If there are no times, there should be no intervals
                assert rebuilt_interval_array is None or len(rebuilt_interval_array) == 0, \
                    f"Unexpected intervals for measure {measure_id}, device {device_id} after rebuild"