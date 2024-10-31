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
    V_TYPE_DOUBLE, T_TYPE_TIMESTAMP_ARRAY_INT64_NANO, create_gap_arr
from atriumdb.adb_functions import get_block_and_interval_data, merge_gap_data, \
    merge_timestamp_data, create_timestamps_from_gap_data, _calc_end_time_from_gap_data, find_intervals, \
    allowed_interval_index_modes, convert_to_nanoseconds, convert_to_nanohz, reencode_dataset
from atriumdb.helpers.block_calculations import freq_nhz_to_period_ns
from atriumdb.helpers.settings import OVERWRITE_SETTING_NAME
from atriumdb.intervals import Intervals
from tests.testing_framework import _test_for_both, slice_times_values
import numpy as np

DB_NAME = 'backwards_block'


def test_backwards_block():
    _test_for_both(DB_NAME, _test_backwards_block)
    _test_for_both(DB_NAME, _test_backwards_block_prospective)


def _test_backwards_block(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk.block.block_size = 10

    freq_hz = 500
    freq_nhz = int(freq_hz * (10 ** 9))
    period_ns = 10 ** 18 // freq_nhz

    x = np.linspace(0, 2 * np.pi, 100)
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

    # Write the correct data to the sdk
    corrected_device_id = sdk.insert_device("corrected")
    expected_times, expected_values = slice_times_values(
        simple_timestamps, scaled_sine_wave, None, None)
    old_write_data_easy(sdk, measure_id, corrected_device_id, expected_times, expected_values, freq_nhz)

    # Time type 1
    device_id_1 = sdk.insert_device("type 1")
    old_write_data_easy(sdk, measure_id, device_id_1, simple_timestamps, scaled_sine_wave, freq_nhz)

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
    old_write_data(sdk,
                   measure_id, device_id_2, gap_data, scaled_sine_wave, freq_nhz, int(simple_timestamps[0]),
                   raw_time_type=raw_t_t, raw_value_type=raw_v_t, encoded_time_type=encoded_t_t,
                   encoded_value_type=encoded_v_t)

    # Fix backward block data
    # fix_block_boundaries(sdk, gap_tolerance_nano=0)
    reencode_dataset(sdk, values_per_block=10, blocks_per_file=2, interval_gap_tolerance_nano=0)

    # Query entire written data
    min_time = np.min(simple_timestamps)
    max_time = np.max(simple_timestamps)
    duration = max_time - min_time

    for start_time, end_time in [[min_time, max_time + period_ns],
                                 [min_time - (duration // 2), max_time - (duration // 2)]]:
        expected_times, expected_values = slice_times_values(simple_timestamps, scaled_sine_wave, start_time, end_time)
        _, presorted_times, presorted_values = sdk.get_data(
            measure_id, start_time, end_time, corrected_device_id, allow_duplicates=False)

        assert np.array_equal(presorted_times, expected_times)
        assert np.array_equal(presorted_values, expected_values)

        for device_id in [device_id_1, device_id_2]:
            headers, r_times, r_values = sdk.get_data(
                measure_id, start_time, end_time, device_id, allow_duplicates=False)

            # print()
            # print(r_times)
            # print(expected_times)
            # print(np.array_equal(r_times, expected_times))
            # print(np.array_equal(r_values, expected_values))

            assert np.array_equal(r_times, expected_times)
            assert np.array_equal(r_values, expected_values)


def _test_backwards_block_prospective(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk.block.block_size = 10

    freq_hz = 500
    freq_nhz = int(freq_hz * (10 ** 9))
    period_ns = 10 ** 18 // freq_nhz

    x = np.linspace(0, 2 * np.pi, 100)
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

    # Write the correct data to the sdk
    corrected_device_id = sdk.insert_device("corrected")
    expected_times, expected_values = slice_times_values(
        simple_timestamps, scaled_sine_wave, None, None)
    sdk.write_data_easy(measure_id, corrected_device_id, expected_times, expected_values, freq_nhz)

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
    sdk.write_data(measure_id, device_id_2, gap_data, scaled_sine_wave, freq_nhz, int(simple_timestamps[0]),
                   raw_time_type=raw_t_t, raw_value_type=raw_v_t, encoded_time_type=encoded_t_t,
                   encoded_value_type=encoded_v_t)

    # Query entire written data
    min_time = np.min(simple_timestamps)
    max_time = np.max(simple_timestamps)
    duration = max_time - min_time

    for start_time, end_time in [[min_time, max_time + period_ns],
                                 [min_time - (duration // 2), max_time - (duration // 2)]]:
        expected_times, expected_values = slice_times_values(simple_timestamps, scaled_sine_wave, start_time, end_time)
        _, presorted_times, presorted_values = sdk.get_data(
            measure_id, start_time, end_time, corrected_device_id, allow_duplicates=False)

        assert np.array_equal(presorted_times, expected_times)
        assert np.array_equal(presorted_values, expected_values)

        for device_id in [device_id_1, device_id_2]:
            headers, r_times, r_values = sdk.get_data(
                measure_id, start_time, end_time, device_id, allow_duplicates=False)

            assert np.array_equal(r_times, expected_times)
            assert np.array_equal(r_values, expected_values)


def old_write_data(sdk, measure_id: int, device_id: int, time_data: np.ndarray, value_data: np.ndarray, freq_nhz: int,
                   time_0: int, raw_time_type: int = None, raw_value_type: int = None, encoded_time_type: int = None,
                   encoded_value_type: int = None, scale_m: float = None, scale_b: float = None,
                   interval_index_mode: str = None, gap_tolerance: int = 0, merge_blocks: bool = True):
    if sdk.metadata_connection_type == "api":
        raise NotImplementedError("API mode is not supported for writing data.")

    # Ensure there is data to be written
    assert value_data.size > 0, "Cannot write no data."

    # Ensure time data is of integer type
    assert np.issubdtype(time_data.dtype, np.integer), "Time information must be encoded as an integer."

    # Set default interval index and ensure valid type.
    interval_index_mode = "merge" if interval_index_mode is None else interval_index_mode
    assert interval_index_mode in allowed_interval_index_modes, \
        f"interval_index must be one of {allowed_interval_index_modes}"

    # Force Python Integers
    freq_nhz = int(freq_nhz)
    time_0 = int(time_0)

    # Calculate new intervals
    write_intervals = find_intervals(freq_nhz, raw_time_type, time_data, time_0, int(value_data.size))

    # check overwrite setting
    if OVERWRITE_SETTING_NAME not in sdk.settings_dict:
        raise ValueError("Overwrite behavior not set. Please set it in the sql settings table")
    overwrite_setting = sdk.settings_dict[OVERWRITE_SETTING_NAME]

    # Initialize variables for handling overwriting
    overwrite_file_dict, old_block_ids, old_file_list = None, None, None

    # if overwrite is ignore there is no reason to calculate this stuff
    if overwrite_setting != 'ignore':
        write_intervals_o = Intervals(write_intervals)

        # Get current intervals
        current_intervals = sdk.get_interval_array(
            measure_id, device_id=device_id, gap_tolerance_nano=0,
            start=int(write_intervals[0][0]), end=int(write_intervals[-1][-1]))

        current_intervals_o = Intervals(current_intervals)

        # Check if there is an overlap between current and new intervals
        if current_intervals_o.intersection(write_intervals_o).duration() > 0:

            # Handle overwriting based on the overwrite_setting
            if overwrite_setting == 'overwrite':
                overwrite_file_dict, old_block_ids, old_file_list = sdk._overwrite_delete_data(
                    measure_id, device_id, time_data, time_0, raw_time_type, value_data.size, freq_nhz)
            elif overwrite_setting == 'error':
                raise ValueError("Data to be written overlaps already ingested data.")
            else:
                raise ValueError(f"Overwrite setting {overwrite_setting} not recognized.")

    # default for block merging code (needed to check if we need to delete old stuff at the end)
    old_block = None
    num_full_blocks = value_data.size // sdk.block.block_size

    # only attempt to merge the data with another block if there isn't a full block worth of data
    if num_full_blocks == 0 and merge_blocks:
        # if the times are a gap array find the end time of the array so we can find the closest block
        if raw_time_type == T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:
            # need to subtract one period since the function gives end_time+1 period
            end_time = _calc_end_time_from_gap_data(values_size=value_data.size, gap_array=time_data,
                                                    start_time=time_0, freq_nhz=freq_nhz) - freq_nhz_to_period_ns(
                freq_nhz)
        elif raw_time_type == T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            end_time = time_data[-1]
        else:
            raise NotImplementedError(
                f"Merging small blocks with other blocks is not supported for time type {raw_time_type}")

        # find the closest block to the data we are trying to insert
        old_block, end_block = sdk.sql_handler.select_closest_block(measure_id, device_id, time_0, end_time)

        # if the new block goes on the end and the current end block is full then don't try to merge blocks
        if old_block is not None and not (end_block and (old_block[8] > sdk.block.block_size)):
            # get the file info for the block we are going to merge these values into
            file_info = sdk.sql_handler.select_file(file_id=old_block[3])
            # Read the encoded data from the files
            encoded_bytes_old = sdk.file_api.read_file_list([old_block[1:6]],
                                                            filename_dict={file_info[0]: file_info[1]})

            # decode the headers before they are edited by decode blocks so we know the original time type
            header = sdk.block.decode_headers(encoded_bytes_old, np.array([0], dtype=np.uint64))

            # make sure the time and value types of the block your merging with match
            if header[0].t_encoded_type != encoded_time_type:
                raise ValueError(f"The time type ({encoded_time_type}) you are trying to encode the times as "
                                 f"doesn't match the encoded time type ({header[0].t_encoded_type}) of the block "
                                 f"you are trying to merge with. Either change the encoded time type to match or"
                                 f" set merge_blocks to false.")
            elif header[0].v_encoded_type != encoded_value_type:
                raise ValueError(f"The value type ({encoded_value_type}) you are trying to encode the values as "
                                 f"doesn't match the encoded value type ({header[0].v_encoded_type}) of the block "
                                 f"you are trying to merge with. Either change the encoded value type to match or"
                                 f" set merge_blocks to false.")
            elif header[0].v_raw_type != raw_value_type:
                raise ValueError(f"The raw value type ({raw_value_type}) doesn't match the raw value type "
                                 f"({header[0].v_raw_type}) of the block you are trying to merge with. Either "
                                 f"change the raw value type to match or set merge_blocks to false.")

            # make sure the scale factors match. If they don't then don't merge the blocks
            if header[0].scale_m == scale_m and header[0].scale_b == scale_b:
                # if the original time type of the old block is not the same as the time type of the data we are
                # trying to save, we need to make them the same
                if header[0].t_raw_type != raw_time_type:
                    # if the new time data is a gap array make it into a timestamp array to match the old times
                    if raw_time_type == T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:
                        try:
                            time_data = create_timestamps_from_gap_data(values_size=value_data.size,
                                                                        gap_array=time_data,
                                                                        start_time=time_0, freq_nhz=freq_nhz)
                            raw_time_type = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
                        except ValueError:
                            raise ValueError(f"You are trying to merge a gap array into a block that has the data "
                                             f"saved as a timestamp array and integer timestamps cannot be created "
                                             f"for your gap data with a frequency of {freq_nhz}. Either set "
                                             f"merge_blocks to false or pass in the times as a timestamp array.")
                    # if the new time data is a gap array convert it to a time array to match the old times
                    elif raw_time_type == T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
                        time_data = create_gap_arr(time_data, 1, freq_nhz)
                        raw_time_type = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

                # Decode the data and get the values and the times we are going to merge this data with
                r_time, r_value, _ = sdk.block.decode_blocks(encoded_bytes_old, num_bytes_list=[old_block[5]],
                                                             analog=False, time_type=header[0].t_raw_type)

                # if raw value type is int and it's not int64 then cast it to int64 so it doesn't fail during merge
                if raw_value_type == 1 and value_data.dtype != np.int64:
                    value_data = value_data.astype(np.int64)

                # merge the blocks
                if raw_time_type == T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
                    time_data, value_data = merge_timestamp_data(r_value, r_time, value_data, time_data)
                    time_0 = time_data[0]
                else:
                    value_data, time_data, time_0 = merge_gap_data(r_value, r_time, header[0].start_n, value_data,
                                                                   time_data, time_0, freq_nhz)
            else:
                # if the scale factors are not the same don't merge and set old block to none, so we don't delete it
                old_block = None
        else:
            # if this is an end block and the closest block is full don't merge and set old block to none, so we don't delete it
            old_block = None

    # check if the write data will make at least one full block and if there will be a small block at the end
    num_full_blocks = value_data.size // sdk.block.block_size
    if (num_full_blocks > 0 and value_data.size % sdk.block.block_size != 0 and
            (
                    raw_time_type == T_TYPE_TIMESTAMP_ARRAY_INT64_NANO or raw_time_type == T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO)):
        byte_start_array, encoded_bytes, encoded_headers = sdk.block.make_oversized_block(
            encoded_time_type, encoded_value_type, freq_nhz, num_full_blocks, raw_time_type, raw_value_type, scale_b,
            scale_m, time_0, time_data, value_data)

    # if all blocks are perfectly sized or there is less than one optimal block worth of data
    else:
        # Encode the block(s)
        encoded_bytes, encoded_headers, byte_start_array = sdk.block.encode_blocks(
            time_data, value_data, freq_nhz, time_0,
            raw_time_type=raw_time_type,
            raw_value_type=raw_value_type,
            encoded_time_type=encoded_time_type,
            encoded_value_type=encoded_value_type,
            scale_m=scale_m,
            scale_b=scale_b)

    # Write the encoded bytes to disk
    filename = sdk.file_api.write_bytes(measure_id, device_id, encoded_bytes)

    # Use the header data to create rows to be inserted into the block_index and interval_index SQL tables
    block_data, interval_data = get_block_and_interval_data(
        measure_id, device_id, encoded_headers, byte_start_array, write_intervals,
        interval_gap_tolerance=gap_tolerance)

    # if your new data was merged with an older block add the new info to mariadb and delete the old block
    if old_block is not None:
        sdk.sql_handler.insert_merged_block_data(filename, block_data, old_block[0], interval_data,
                                                 interval_index_mode, gap_tolerance)

    # If data was overwritten
    elif overwrite_file_dict is not None:
        # Add new data to SQL insertion data
        overwrite_file_dict[filename] = (block_data, interval_data)

        # Update SQL
        old_file_ids = [file_id for file_id, filename in old_file_list]
        sdk.sql_handler.update_tsc_file_data(overwrite_file_dict, old_block_ids, old_file_ids, gap_tolerance)

        # Delete old files
        # for file_id, filename in old_file_list:
        #     file_path = Path(self.file_api.to_abs_path(filename, measure_id, device_id))
        #     file_path.unlink(missing_ok=True)
    else:
        # Insert SQL rows
        sdk.sql_handler.insert_tsc_file_data(filename, block_data, interval_data, interval_index_mode, gap_tolerance)

    return encoded_bytes, encoded_headers, byte_start_array, filename


def old_write_data_easy(sdk, measure_id: int, device_id: int, time_data: np.ndarray, value_data: np.ndarray, freq: int,
                        scale_m: float = None, scale_b: float = None, time_units: str = None, freq_units: str = None):
    if sdk.metadata_connection_type == "api":
        raise NotImplementedError("API mode is not supported for writing data.")

    # Set default time and frequency units if not provided
    time_units = "ns" if time_units is None else time_units
    freq_units = "nHz" if freq_units is None else freq_units

    # Convert time_data to nanoseconds if a different time unit is used
    if time_units != "ns":
        time_data = convert_to_nanoseconds(time_data, time_units)

    # Convert frequency to nanohertz if a different frequency unit is used
    if freq_units != "nHz":
        freq = convert_to_nanohz(freq, freq_units)

    # Determine the raw time type based on the size of time_data and value_data
    if time_data.size == value_data.size:
        raw_t_t = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
    else:
        raw_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

    # Determine the encoded time type
    encoded_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

    # Determine the raw and encoded value types based on the dtype of value_data
    if np.issubdtype(value_data.dtype, np.integer):
        raw_v_t = V_TYPE_INT64
        encoded_v_t = V_TYPE_DELTA_INT64
    else:
        raw_v_t = V_TYPE_DOUBLE
        encoded_v_t = V_TYPE_DOUBLE

    # Call the write_data method with the determined parameters
    old_write_data(sdk, measure_id, device_id, time_data, value_data, freq, int(time_data[0]), raw_time_type=raw_t_t,
                   raw_value_type=raw_v_t, encoded_time_type=encoded_t_t, encoded_value_type=encoded_v_t,
                   scale_m=scale_m, scale_b=scale_b)
