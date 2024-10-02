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
import random
from atriumdb import AtriumSDK, create_gap_arr, merge_gap_data
from atriumdb.adb_functions import create_timestamps_from_gap_data
from tests.generate_wfdb import get_records
from tests.test_mit_bih import create_gaps, get_record_data_for_ingest
from tests.test_transfer_info import insert_random_patients
from tests.testing_framework import _test_for_both

DB_NAME = 'gap_data_merge'


def test_gap_data_merge():
    seed = 42
    max_records = 4
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    num_records = 0

    for (record, annotation), (d_record, d_annotation) in zip(get_records(dataset_name='mitdb'),
                                                              get_records(dataset_name='mitdb', physical=False)):
        if max_records and num_records >= max_records:
            return
        num_records += 1

        freq_nano = 500 * 1_000_000_000
        period_nano = int(10 ** 18 // freq_nano)

        time_arr = np.arange(record.sig_len, dtype=np.int64) * period_nano

        gap_data_2d = create_gaps(time_arr.size, period_nano)
        for gap_index, gap_duration in gap_data_2d:
            time_arr[gap_index:] += gap_duration

        start_time = int(time_arr[0])

        if record.n_sig > 1:
            for i in range(len(record.sig_name)):
                measure_tag, scale_b, scale_m, units, value_data = get_record_data_for_ingest(d_record, record, i)
                _test_gap_data_merging(freq_nano, gap_data_2d, start_time, time_arr, value_data, test_riffle=True)
        else:
            measure_tag, scale_b, scale_m, units, value_data = get_record_data_for_ingest(d_record, record, None)
            _test_gap_data_merging(freq_nano, gap_data_2d, start_time, time_arr, value_data, test_riffle=True)



def _test_gap_data_merging(freq_nano, gap_data_2d, start_time, time_arr, value_data, test_riffle=False):
    created_gap_array = create_gap_arr(time_arr, 1, freq_nano)
    assert np.array_equal(gap_data_2d.flatten(), created_gap_array)
    # clean split
    split_index = value_data.size // 2
    values_1, times_1 = value_data[:split_index].copy(), time_arr[:split_index].copy()
    values_2, times_2 = value_data[split_index:].copy(), time_arr[split_index:].copy()
    start_1, start_2 = start_time, int(time_arr[split_index])
    gap_array_1 = create_gap_arr(times_1, 1, freq_nano)
    gap_array_2 = create_gap_arr(times_2, 1, freq_nano)
    merged_values, merged_gap_array, merged_start_time = merge_gap_data(
        values_1, gap_array_1, start_1, values_2, gap_array_2, start_2, freq_nano)

    assert np.array_equal(value_data, merged_values)
    assert np.array_equal(gap_data_2d.flatten(), merged_gap_array)

    # clean split 1 value
    split_index = value_data.size - 1
    values_1, times_1 = value_data[:split_index].copy(), time_arr[:split_index].copy()
    values_2, times_2 = value_data[split_index:].copy(), time_arr[split_index:].copy()
    start_1, start_2 = start_time, int(time_arr[split_index])
    gap_array_1 = create_gap_arr(times_1, 1, freq_nano)
    gap_array_2 = create_gap_arr(times_2, 1, freq_nano)
    merged_values, merged_gap_array, merged_start_time = merge_gap_data(
        values_1, gap_array_1, start_1, values_2, gap_array_2, start_2, freq_nano)

    assert np.array_equal(value_data, merged_values)
    assert np.array_equal(gap_data_2d.flatten(), merged_gap_array)

    # clean split on a gap
    if created_gap_array.size >= 2:
        split_index = created_gap_array[-2]
    else:
        split_index = value_data.size // 2
    values_1, times_1 = value_data[:split_index].copy(), time_arr[:split_index].copy()
    values_2, times_2 = value_data[split_index:].copy(), time_arr[split_index:].copy()
    start_1, start_2 = start_time, int(time_arr[split_index])
    gap_array_1 = create_gap_arr(times_1, 1, freq_nano)
    gap_array_2 = create_gap_arr(times_2, 1, freq_nano)
    merged_values, merged_gap_array, merged_start_time = merge_gap_data(
        values_1, gap_array_1, start_1, values_2, gap_array_2, start_2, freq_nano)

    assert np.array_equal(value_data, merged_values)
    assert np.array_equal(gap_data_2d.flatten(), merged_gap_array)

    # Take a chunk of out of the middle
    split_index_1, split_index_2 = value_data.size // 3, 2 * value_data.size // 3
    values_1, times_1 = value_data[split_index_1:split_index_2].copy(), time_arr[split_index_1:split_index_2].copy()
    values_2 = np.concatenate([value_data[:split_index_1].copy(), value_data[split_index_2:].copy()],
                              dtype=value_data.dtype)
    times_2 = np.concatenate([time_arr[:split_index_1].copy(), time_arr[split_index_2:].copy()],
                             dtype=time_arr.dtype)
    start_1, start_2 = int(times_1[0]), int(times_2[0])
    gap_array_1 = create_gap_arr(times_1, 1, freq_nano)
    gap_array_2 = create_gap_arr(times_2, 1, freq_nano)
    merged_values, merged_gap_array, merged_start_time = merge_gap_data(
        values_1, gap_array_1, start_1, values_2, gap_array_2, start_2, freq_nano)
    merged_timestamps = create_timestamps_from_gap_data(merged_values.size, merged_gap_array, start_time, freq_nano)
    sorted_indices = np.argsort(merged_timestamps)
    merged_timestamps = merged_timestamps[sorted_indices]
    merged_values = merged_values[sorted_indices]
    assert np.array_equal(value_data, merged_values)
    assert np.array_equal(time_arr, merged_timestamps)

    # Riffle Shuffle
    # Works but takes forever
    if test_riffle:
        values_1, times_1 = value_data[::2].copy(), time_arr[::2].copy()
        values_2, times_2 = value_data[1::2].copy(), time_arr[1::2].copy()
        start_1, start_2 = int(times_1[0]), int(times_2[0])
        gap_array_1 = create_gap_arr(times_1, 1, freq_nano)
        gap_array_2 = create_gap_arr(times_2, 1, freq_nano)
        merged_values, merged_gap_array, merged_start_time = merge_gap_data(
            values_1, gap_array_1, start_1, values_2, gap_array_2, start_2, freq_nano)
        merged_timestamps = create_timestamps_from_gap_data(merged_values.size, merged_gap_array, start_time, freq_nano)
        sorted_indices = np.argsort(merged_timestamps)
        merged_timestamps = merged_timestamps[sorted_indices]
        merged_values = merged_values[sorted_indices]
        assert np.array_equal(value_data, merged_values)
        assert np.array_equal(time_arr, merged_timestamps)