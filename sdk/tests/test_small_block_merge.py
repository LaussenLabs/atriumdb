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

import numpy as np
import pytest

from atriumdb import AtriumSDK
from atriumdb.block import create_gap_arr
from tests.testing_framework import _test_for_both

DB_NAME = 'test-merge-small-block'


def test_merge_small_block():
    _test_for_both(DB_NAME, _test_merge_small_block_timestamp)
    _test_for_both(DB_NAME, _test_merge_small_block_gap)


def _test_merge_small_block_timestamp(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    measure_id = sdk.insert_measure(measure_tag="test_measure", freq=1, units="test_units", freq_units="Hz")
    device_id = sdk.insert_device(device_tag="test_device")

    # first make sure when there are no blocks it just inserts the block
    times, values = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64) * 1_000_000_000, np.array([1, 2, 3, 4, 5, 6])
    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=times[0], raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    # now make sure it merges the new block with the old one
    times, values = np.array([7, 8, 9, 10], dtype=np.int64) * 1_000_000_000, np.array([7, 8, 9, 10])
    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=times[0], raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 0, 11_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], r_values)

    # now make sure it merges the new block with the old one even with a big gap between them
    times, values = np.array([15, 16, 17, 18], dtype=np.int64) * 1_000_000_000, np.array([15, 16, 17, 18])
    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=times[0], raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 0, 19_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18], r_values)

    # now make sure it merges the new block inside the old block
    times, values = np.array([11, 12, 13, 14], dtype=np.int64) * 1_000_000_000, np.array([11, 12, 13, 14])
    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=times[0], raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 0, 19_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], r_values)

    # add in a block that doesn't get merged because merge_blocks is false
    times, values = np.array([20, 21, 22, 24], dtype=np.int64) * 1_000_000_000, np.array([20, 21, 22, 24])
    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=times[0], raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0, merge_blocks=False)

    headers, r_times, r_values = sdk.get_data(measure_id, 19_000_000_000, 25_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(np.array([20, 21, 22, 24], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal([20, 21, 22, 24], r_values)

    # add in a block the overlaps with the newest small block
    times, values = np.array([19, 20, 21], dtype=np.int64) * 1_000_000_000, np.array([19, 20, 21])
    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=times[0], raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 19_000_000_000, 25_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(np.array([19, 20, 21, 22, 24], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal([19, 20, 21, 22, 24], r_values)

    # add some values using a gap array as the time type to test the conversion of gap array to timestamp array since the old block is time type 1
    times, values = np.array([26, 27, 29, 30, 32], dtype=np.int64) * 1_000_000_000, np.array([26, 27, 29, 30, 32], dtype=np.int64)
    # make a gap array out of the times
    times = create_gap_arr(times, 1, 1_000_000_000)

    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=26_000_000_000, raw_time_type=2,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 19_000_000_000, 33_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(np.array([19, 20, 21, 22, 24, 26, 27, 29, 30, 32], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal([19, 20, 21, 22, 24, 26, 27, 29, 30, 32], r_values)


    times, values = np.arange(0, 150_000, dtype=np.int64) * 1_000_000_000, np.arange(0, 150_000, dtype=np.int64)

    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=0, raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 0, 152_000_000_000_000, device_id, allow_duplicates=False)

    assert len(headers) == 3
    assert np.array_equal(times, r_times)
    assert np.array_equal(values, r_values)

    # here we are testing when there's a full block on the end not to merge blocks and just to insert a new block
    times, values = np.array([151_001, 151_002, 151_003], dtype=np.int64) * 1_000_000_000, np.array([151_001, 151_002, 151_003], dtype=np.int64)
    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=times[0], raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 151_000_000_000_000, 152_000_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(times, r_times)
    assert np.array_equal(values, r_values)

    # here we are testing that when the scale factors are different it just inserts a new block and doesn't merge with the old
    times, values = np.array([151_006, 151_007, 151_008], dtype=np.int64) * 1_000_000_000, np.array([151_006, 151_007, 151_008], dtype=np.int64)
    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=times[0], raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=1, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 151_000_000_000_000, 152_000_000_000_000, device_id)

    assert len(headers) == 2
    assert np.array_equal(np.array([151_001, 151_002, 151_003, 151_006, 151_007, 151_008], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal(np.array([151_001, 151_002, 151_003, 151_006, 151_007, 151_008], dtype=np.int64), r_values)

    # make sure that when the encoded time type is different then the block it wants to merge with blocks arnt merged
    times, values = np.array([151_009, 151_010, 151_011], dtype=np.int64) * 1_000_000_000, np.array([151_009, 151_010, 151_011], dtype=np.int64)
    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=times[0], raw_time_type=1,
                       raw_value_type=1, encoded_time_type=1, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 151_000_000_000_000, 152_000_000_000_000, device_id)

    assert len(headers) == 3
    assert np.array_equal(np.array([151_001, 151_002, 151_003, 151_006, 151_007, 151_008, 151_009, 151_010, 151_011], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal(np.array([151_001, 151_002, 151_003, 151_006, 151_007, 151_008, 151_009, 151_010, 151_011], dtype=np.int64), r_values)

    # make sure that if the encoded value types don't match blocks aren't merged
    times, values = np.array([151_012, 151_013, 151_014], dtype=np.int64) * 1_000_000_000, np.array([151_012.1, 151_013.2, 151_014.3], dtype=np.float64)
    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=times[0], raw_time_type=1,
                       raw_value_type=2, encoded_time_type=1, encoded_value_type=2, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 151_000_000_000_000, 152_000_000_000_000, device_id)

    assert len(headers) == 4
    assert np.array_equal(np.array([151_001, 151_002, 151_003, 151_006, 151_007, 151_008, 151_009, 151_010, 151_011, 151_012, 151_013, 151_014], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal(np.array([151_001, 151_002, 151_003, 151_006, 151_007, 151_008, 151_009, 151_010, 151_011, 151_012.1, 151_013.2, 151_014.3], dtype=np.float64), r_values)

    # make sure that when the raw value types are different blocks aren't merged
    times, values = np.array([151_015, 151_016, 151_017], dtype=np.int64) * 1_000_000_000, np.array([151_015, 151_016, 151_017], dtype=np.float64)

    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=times[0], raw_time_type=1,
                       raw_value_type=1, encoded_time_type=1, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 151_000_000_000_000, 152_000_000_000_000, device_id)

    assert len(headers) == 5
    assert np.array_equal(np.array([151_001, 151_002, 151_003, 151_006, 151_007, 151_008, 151_009, 151_010, 151_011, 151_012, 151_013, 151_014, 151_015, 151_016, 151_017], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal(np.array([151_001, 151_002, 151_003, 151_006, 151_007, 151_008, 151_009, 151_010, 151_011, 151_012.1, 151_013.2, 151_014.3, 151_015, 151_016, 151_017], dtype=np.float64), r_values)


def _test_merge_small_block_gap(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    measure_id = sdk.insert_measure(measure_tag="test_measure_gaps", freq=1, units="test_units", freq_units="Hz")
    device_id = sdk.insert_device(device_tag="test_device")

    # first make sure when there are no blocks it just inserts the block
    times, values = np.array([1, 2, 4, 5, 6], dtype=np.int64) * 1_000_000_000, np.array([1, 2, 4, 5, 6], dtype=np.int64)

    sdk.write_data(measure_id, device_id, create_gap_arr(times, 1, 1_000_000_000), values, 1_000_000_000,
                   time_0=times[0], raw_time_type=2,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    # now make sure it merges the new block with the old one
    times, values = np.array([7, 9, 10], dtype=np.int64) * 1_000_000_000, np.array([7, 9, 10], dtype=np.int64)
    sdk.write_data(measure_id, device_id, create_gap_arr(times, 1, 1_000_000_000), values, 1_000_000_000,
                   time_0=times[0], raw_time_type=2,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 0, 11_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(np.array([1, 2, 4, 5, 6, 7, 9, 10], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal([1, 2, 4, 5, 6, 7, 9, 10], r_values)

    # now make sure it merges the new block with the old one even with a big gap between them
    times, values = np.array([15, 17, 18], dtype=np.int64) * 1_000_000_000, np.array([15, 17, 18], dtype=np.int64)
    sdk.write_data(measure_id, device_id, create_gap_arr(times, 1, 1_000_000_000), values, 1_000_000_000,
                   time_0=times[0], raw_time_type=2,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 0, 19_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(np.array([1, 2, 4, 5, 6, 7, 9, 10, 15, 17, 18], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal([1, 2, 4, 5, 6, 7, 9, 10, 15, 17, 18], r_values)

    # now make sure it merges the new block inside the old block
    times, values = np.array([11, 13, 14], dtype=np.int64) * 1_000_000_000, np.array([11, 13, 14], dtype=np.int64)
    sdk.write_data(measure_id, device_id, create_gap_arr(times, 1, 1_000_000_000), values, 1_000_000_000,
                   time_0=times[0], raw_time_type=2,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 0, 19_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(np.array([1, 2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18], dtype=np.int64) * 1_000_000_000,
                          r_times)
    assert np.array_equal([1, 2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18], r_values)

    # add in a block that doesn't get merged because merge_blocks is false
    times, values = np.array([20, 21, 22, 24], dtype=np.int64) * 1_000_000_000, np.array([20, 21, 22, 24],
                                                                                         dtype=np.int64)
    sdk.write_data(measure_id, device_id, create_gap_arr(times, 1, 1_000_000_000), values, 1_000_000_000,
                   time_0=times[0], raw_time_type=2,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0,
                   merge_blocks=False)

    headers, r_times, r_values = sdk.get_data(measure_id, 19_000_000_000, 25_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(np.array([20, 21, 22, 24], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal([20, 21, 22, 24], r_values)

    # add in a block the overlaps with the newest small block
    times, values = np.array([19, 21], dtype=np.int64) * 1_000_000_000, np.array([19, 21], dtype=np.int64)
    sdk.write_data(measure_id, device_id, create_gap_arr(times, 1, 1_000_000_000), values, 1_000_000_000,
                   time_0=times[0], raw_time_type=2,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 19_000_000_000, 25_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(np.array([19, 20, 21, 22, 24], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal([19, 20, 21, 22, 24], r_values)

    # add some values using a timestamp array as the time type to test the conversion of timestamp array to gap array since the old block is time type 2
    times, values = np.array([26, 27, 30, 32], dtype=np.int64) * 1_000_000_000, np.array([26, 27, 30, 32],
                                                                                         dtype=np.int64)

    sdk.write_data(measure_id, device_id, times, values, 1_000_000_000, time_0=26_000_000_000, raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=0, scale_b=0)

    headers, r_times, r_values = sdk.get_data(measure_id, 19_000_000_000, 33_000_000_000, device_id)

    assert len(headers) == 1
    assert np.array_equal(np.array([19, 20, 21, 22, 24, 26, 27, 30, 32], dtype=np.int64) * 1_000_000_000, r_times)
    assert np.array_equal([19, 20, 21, 22, 24, 26, 27, 30, 32], r_values)