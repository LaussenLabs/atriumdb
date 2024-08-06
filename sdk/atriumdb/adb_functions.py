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
import math
import warnings
import os

import numpy as np
import time
from urllib.parse import urlsplit, urlunsplit
from numpy.lib.stride_tricks import sliding_window_view
from atriumdb.helpers.block_calculations import freq_nhz_to_period_ns
import logging
import bisect

from atriumdb.helpers.block_calculations import calc_time_by_freq
from atriumdb.helpers.block_constants import TIME_TYPES


ALLOWED_TIME_TYPES = [1, 2]
time_unit_options = {"ns": 1, "s": 10 ** 9, "ms": 10 ** 6, "us": 10 ** 3}
freq_unit_options = {"nHz": 1, "uHz": 10 ** 3, "mHz": 10 ** 6, "Hz": 10 ** 9, "kHz": 10 ** 12, "MHz": 10 ** 15}
allowed_interval_index_modes = ["fast", "merge", "disable"]


def get_block_and_interval_data(measure_id, device_id, metadata, start_bytes, intervals, interval_gap_tolerance=0):
    block_data = []
    for header_i, header in enumerate(metadata):
        block_data.append({
            "measure_id": measure_id,
            "device_id": device_id,
            "start_byte": int(start_bytes[header_i]),
            "num_bytes": header.meta_num_bytes + header.t_num_bytes + header.v_num_bytes,
            "start_time_n": header.start_n,
            "end_time_n": header.end_n,
            "num_values": header.num_vals,
        })

    interval_data = []
    intervals = sorted(intervals)
    for interval in intervals:
        if interval_data and int(interval[0]) - interval_data[-1]["end_time_n"] <= interval_gap_tolerance:
            interval_data[-1]["end_time_n"] = int(interval[1])
        else:
            interval_data.append({
                "measure_id": measure_id,
                "device_id": device_id,
                "start_time_n": int(interval[0]),
                "end_time_n": int(interval[1]),
            })
    return block_data, interval_data


def condense_byte_read_list(block_list):
    result = []

    for row in block_list:
        if len(result) == 0 or result[-1][2] != row[3] or result[-1][3] + result[-1][4] != row[4]:
            # append measure_id, device_id, file_id, start_byte and num_bytes
            result.append([row[1], row[2], row[3], row[4], row[5]])
        else:
            # if the blocks are continuous merge the reads together by adding the size of the next block to the
            # num_bytes field
            result[-1][4] += row[5]

    return result


def find_intervals(freq_nhz, raw_time_type, time_data, data_start_time, num_values):
    period_ns = int((10 ** 18) / freq_nhz)
    if raw_time_type == TIME_TYPES['TIME_ARRAY_INT64_NS']:
        intervals = [[time_data[0], 0]]
        time_deltas = time_data[1:] - time_data[:-1]
        for time_arr_i in range(time_data.size - 1):
            if time_deltas[time_arr_i] > period_ns:
                intervals[-1][-1] = time_data[time_arr_i] + period_ns
                intervals.append([time_data[time_arr_i + 1], 0])

        intervals[-1][-1] = time_data[-1] + period_ns

    elif raw_time_type == TIME_TYPES['START_TIME_NUM_SAMPLES']:
        intervals = [[time_data[0], time_data[0] + ((time_data[1] - 1) * period_ns)]]

        for interval_data_i in range(1, time_data.size // 2):
            start_time = time_data[2 * interval_data_i]
            end_time = time_data[2 * interval_data_i] + ((time_data[(2 * interval_data_i) + 1] - 1) * period_ns)

            if start_time <= intervals[-1][-1] + period_ns:
                intervals[-1][-1] = end_time
            else:
                intervals.append([start_time, end_time])

    elif raw_time_type == TIME_TYPES['GAP_ARRAY_INT64_INDEX_DURATION_NS']:
        intervals = [[data_start_time, data_start_time + calc_time_by_freq(freq_nhz, num_values)]]
        last_id = 0

        for sample_id, duration in time_data.reshape((-1, 2)):
            intervals[-1][-1] = intervals[-1][0] + calc_time_by_freq(freq_nhz, sample_id - last_id)
            last_id = sample_id
            intervals.append([intervals[-1][-1] + duration,
                              intervals[-1][-1] + duration + calc_time_by_freq(freq_nhz, num_values - last_id)])

    else:
        raise ValueError("raw_time_type not one of {}.".format(
            [TIME_TYPES['TIME_ARRAY_INT64_NS'], TIME_TYPES['START_TIME_NUM_SAMPLES']]))

    return intervals


# if you want to just use this to sort data will have to add default vals for start/end time and skip bisect
def sort_data(times, values, headers, start_time, end_time, allow_duplicates=True):
    start_bench = time.perf_counter()
    if len(headers) == 0:
        return times, values

    block_info = np.zeros((len(headers), 4), dtype=np.int64)
    block_info[:] = [[h.start_n, h.end_n, h.num_vals, 0] for h in headers]
    np.cumsum(block_info.T[2], out=block_info.T[3])
    block_info.T[2] = block_info.T[3] - block_info.T[2]

    end_bench = time.perf_counter()
    logging.debug(f"rearrange block data info {(end_bench - start_bench) * 1000} ms")

    # check if the start times are sorted
    start_times_sorted = np.all(np.greater(block_info.T[0][1:], block_info.T[0][:-1]))
    # check if the blocks overlap each other
    start_end_times_dont_intersect = np.all(np.greater_equal(block_info.T[0][1:], block_info.T[1][:-1]))
    if start_end_times_dont_intersect and start_times_sorted:
        logging.debug("Blocks already Sorted and don't intersect.")

        left, right = bisect.bisect_left(times, start_time), bisect.bisect_left(times, end_time)
        times, values = times[left:right], values[left:right]

        # if duplicates are allowed then just return times and values
        if allow_duplicates:
            return times, values
        # if duplicates are not allowed then remove them using np.unique
        sorted_times, sorted_time_indices = np.unique(times, return_index=True)
        return sorted_times, values[sorted_time_indices]

    start_bench = time.perf_counter()
    # if the start times were not sorted, sort them if they were then don't bother running the sort
    if not start_times_sorted:
        # use quicksort as it will be faster with smaller arrays
        sorted_block_i = np.argsort(block_info.T[0], kind='quicksort')
        block_info = block_info[sorted_block_i]
        # recheck if the blocks intersect
        start_end_times_dont_intersect = np.all(np.greater_equal(block_info.T[0][1:], block_info.T[1][:-1]))

    end_bench = time.perf_counter()
    logging.debug(f"sort data by block {(end_bench - start_bench) * 1000} ms")

    if allow_duplicates:
        if start_end_times_dont_intersect:
            # Blocks don't intersect each other.
            start_bench = time.perf_counter()

            # Original Index Creation
            sorted_time_indices = np.concatenate([np.arange(i_start, i_end) for _, _, i_start, i_end in block_info])

            times, values = times[sorted_time_indices], values[sorted_time_indices]
            end_bench = time.perf_counter()
            logging.debug(f"Blocks don't intersect, sort by blocks {(end_bench - start_bench) * 1000} ms")

        else:
            # Blocks do intersect each other, so sort every value.
            start_bench = time.perf_counter()

            # If allowing duplicates use argsort to get the indices so duplicates aren't removed
            # Use mergesort as it will be faster than quicksort on large arrays and since its mostly sorted already
            sorted_time_indices = np.argsort(times, kind='mergesort')

            end_bench = time.perf_counter()
            logging.debug(f"Blocks intersect, sort every value {(end_bench - start_bench) * 1000} ms")

            times, values = times[sorted_time_indices], values[sorted_time_indices]
    else:
        # if duplicates are not allowed then remove them using np.unique
        times, sorted_time_indices = np.unique(times, return_index=True)
        values = values[sorted_time_indices]

    # if the start or end time is in the middle of a block, more times/values will be decoded than are needed so
    # bisect the array on the left and right sides to truncate the data
    left, right = bisect.bisect_left(times, start_time), bisect.bisect_left(times, end_time)
    times, values = times[left:right], values[left:right]

    return times, values


def yield_data(r_times, r_values, window_size, step_size, get_last_window, total_query_index):
    if window_size is not None:
        time_sliding_window_view = sliding_window_view(r_times, window_size)
        value_sliding_window_view = sliding_window_view(r_values, window_size)
        index_arr = np.arange(0, value_sliding_window_view.shape[0], step_size) + total_query_index

        # print("r_values.size, window_size")
        # print(r_values.size, window_size)
        # print()
        # print("index_arr.shape")
        # print(index_arr.shape)
        # print("value_sliding_window_view[::step_size, :].shape")
        # print(value_sliding_window_view[::step_size, :].shape)
        assert index_arr.size == value_sliding_window_view[::step_size, :].shape[0]

        yield (index_arr,
               time_sliding_window_view[::step_size, :],
               value_sliding_window_view[::step_size, :])

        if get_last_window and ((value_sliding_window_view.shape[0] - 1) % step_size != 0):
            yield (np.array([(value_sliding_window_view.shape[0] - 1) + total_query_index], dtype=np.int64),
                   time_sliding_window_view[-1::, :],
                   value_sliding_window_view[-1::, :])
    else:
        yield total_query_index, r_times, r_values


def convert_to_nanoseconds(time_data, time_units):
    # check that a correct unit type was entered
    if time_units not in time_unit_options.keys():
        raise ValueError("Invalid time units. Expected one of: %s" % time_unit_options)

    # convert time data into nanoseconds and round off any trailing digits and convert to integer array
    # copy so as to not alter user's data, which should remain in original units.
    time_data = time_data.copy() * time_unit_options[time_units]

    return np.around(time_data).astype("int64")


def convert_value_to_nanoseconds(time_value, time_units):
    if time_units not in time_unit_options:
        raise ValueError(f"Invalid time units. Expected one of {time_unit_options}.")

    return round(time_value * time_unit_options[time_units])


def convert_to_nanohz(freq, freq_units):
    if freq_units not in freq_unit_options.keys():
        raise ValueError("Invalid frequency units. Expected one of: %s" % freq_unit_options)

    freq *= freq_unit_options[freq_units]

    return round(freq)


def convert_from_nanohz(freq_nhz, freq_units):
    if freq_units not in freq_unit_options.keys():
        raise ValueError("Invalid frequency units. Expected one of: %s" % freq_unit_options)

    freq = freq_nhz / freq_unit_options[freq_units]

    if freq == np.floor(freq):
        freq = int(freq)

    return freq


def parse_metadata_uri(metadata_uri):
    parsed_uri = urlsplit(metadata_uri)

    if not parsed_uri.scheme or not parsed_uri.netloc:
        raise ValueError(f"Invalid metadata_uri format: expected 'sqltype://username:password@host:port[/dbname]', got {metadata_uri}")

    sqltype = parsed_uri.scheme

    if not parsed_uri.username or not parsed_uri.password:
        raise ValueError(f"Invalid metadata_uri format: expected 'sqltype://username:password@host:port[/dbname]', got {metadata_uri}")

    username = parsed_uri.username
    password = parsed_uri.password

    if not parsed_uri.hostname or not parsed_uri.port:
        raise ValueError(f"Invalid metadata_uri format: expected 'sqltype://username:password@host:port[/dbname]', got {metadata_uri}")

    host = parsed_uri.hostname
    port = parsed_uri.port

    dbname = parsed_uri.path.lstrip('/')

    return {
        'sqltype': sqltype,
        'user': username,
        'password': password,
        'host': host,
        'port': int(port),
        'database': dbname if dbname else None
    }


def generate_metadata_uri(metadata):
    sqltype = metadata['sqltype']
    username = metadata['user']
    password = metadata['password']
    host = metadata['host']
    port = metadata['port']
    dbname = metadata.get('database')

    netloc = f"{username}:{password}@{host}:{port}"
    path = f"/{dbname}" if dbname else ""

    return urlunsplit((sqltype, netloc, path, "", ""))


def convert_gap_data_to_timestamps(headers, r_times, r_values, start_time_n=None, end_time_n=None, sort=True, allow_duplicates=True):
    """
    Convert gap data to timestamps.

    :param headers: A list of headers containing information about the data.
    :param r_times: An array of times in nanoseconds with gaps.
    :param r_values: An array of values corresponding to the times in r_times.
    :param start_time_n: The start time in nanoseconds. Only needs to be specified if sorting result.
    :param end_time_n: The end time in nanoseconds. Only needs to be specified if sorting result.
    :param bool sort: Whether to sort the returned data.
    :param bool allow_duplicates: Whether to allow duplicate times in the sorted returned data if they exist. Does
    nothing if sort is false.
    :return: A tuple containing an array of timestamps and an array of values.
    """
    # # Start performance benchmark
    # start_bench = time.perf_counter()

    # Check if all times are integers
    is_int_times = all([(10 ** 18) % h.freq_nhz == 0 for h in headers])

    # Set the data type for the full_timestamps array based on is_int_times
    time_dtype = np.int64 if is_int_times else np.float64

    # Create an empty array for the full timestamps
    full_timestamps = np.zeros(r_values.size, dtype=time_dtype)

    # Initialize the current index and gap variables
    cur_index, cur_gap = 0, 0

    # Fill full_timestamps array with times based on headers and gaps
    if is_int_times:
        # Loop through the headers and calculate the timestamps for integer times
        for block_i, h in enumerate(headers):
            period_ns = freq_nhz_to_period_ns(h.freq_nhz)
            full_timestamps[cur_index:cur_index + h.num_vals] = \
                np.arange(h.start_n, h.start_n + (h.num_vals * period_ns), period_ns)

            # Apply the gaps to the timestamps
            for _ in range(h.num_gaps):
                full_timestamps[cur_index + r_times[cur_gap]:cur_index + h.num_vals] += r_times[cur_gap + 1]
                cur_gap += 2

            # Update the current index
            cur_index += h.num_vals
    else:
        # Loop through the headers and calculate the timestamps for non-integer times
        for block_i, h in enumerate(headers):
            period_ns = float(10 ** 18) / float(h.freq_nhz)
            full_timestamps[cur_index:cur_index + h.num_vals] = \
                np.linspace(h.start_n, h.start_n + (h.num_vals * period_ns), num=h.num_vals, endpoint=False)

            # Apply the gaps to the timestamps
            for _ in range(h.num_gaps):
                full_timestamps[cur_index + r_times[cur_gap]:cur_index + h.num_vals] += r_times[cur_gap + 1]
                cur_gap += 2

            # Update the current index
            cur_index += h.num_vals

    # # End performance benchmark and log the time taken
    # end_bench = time.perf_counter()
    # _LOGGER.debug(f"Expand Gap Data {(end_bench - start_bench) * 1000} ms")

    # Sort the data based on the timestamps if sort is true
    if sort and start_time_n is not None and end_time_n is not None:
        r_times, r_values = sort_data(r_times, r_values, headers, start_time_n, end_time_n, allow_duplicates)

    return full_timestamps, r_values


def get_best_measure_id(sdk, measure_tag, freq, units, freq_units):
    measure_dict = {'tag': measure_tag}
    if freq is not None:
        freq_units = "nHz" if freq_units is None else freq_units
        if freq and freq_units and freq_units != "nHz":
            freq = convert_to_nanohz(freq, freq_units)
        measure_dict['freq_nhz'] = freq
    if units is not None:
        measure_dict['units'] = units
    measure_id_list = get_measure_id_from_generic_measure(sdk, measure_dict, measure_tag_match_rule="best")
    if len(measure_id_list) == 0:
        raise ValueError(f"No matching measure found for: {measure_dict}")
    new_measure_id = measure_id_list[0]
    return new_measure_id


def get_measure_id_from_generic_measure(sdk, measure, measure_tag_match_rule="best"):
    measure_ids = []

    if isinstance(measure, int):
        measure_ids.append(measure)
    elif isinstance(measure, str):
        matching_ids = sdk.get_measure_id_list_from_tag(measure, approx=True)
        if measure_tag_match_rule == "best":
            measure_ids.append(matching_ids[0])
        elif measure_tag_match_rule == "all":
            measure_ids.extend(matching_ids)
    elif isinstance(measure, dict):
        assert 'tag' in measure, "tag not in measure dictionary"
        freq = measure.get('freq_nhz') or measure.get('freq_hz')
        units = measure.get('units')
        freq_units = "nHz" if 'freq_nhz' in measure else "Hz" if 'freq_hz' in measure else None

        if freq and units:
            # Use sdk.get_measure_id for a unique match when both freq and units are specified
            measure_id = sdk.get_measure_id(measure['tag'], freq=freq, units=units, freq_units=freq_units)
            if measure_id is not None:
                measure_ids.append(measure_id)
            else:
                raise ValueError(f"No unique measure found for {measure['tag']} with specified frequency and units.")
        else:
            # Use get_measure_id_list_from_tag when either freq or units is not specified
            matching_ids = sdk.get_measure_id_list_from_tag(measure['tag'], approx=True)
            if measure_tag_match_rule == "best":
                measure_ids.append(matching_ids[0])
            elif measure_tag_match_rule == "all":
                measure_ids.extend(matching_ids)
    else:
        raise ValueError(f"measure type {type(measure)} not supported.")

    if not measure_ids:
        raise ValueError(f"Measure {measure} not found or multiple matches found without a clear selection rule.")

    return measure_ids


def collect_all_descendant_ids(label_set_ids, sql_handler):
    """
    Collects all unique descendant label set IDs for a given list of label set IDs,
    along with their closest requested ancestor ID.

    Args:
    - label_set_ids (List[int]): List of label set IDs to start with.
    - sql_handler (SQLHandler): An instance of SQLHandler for database queries.

    Returns:
    - Tuple[Set[int], Dict[int, int]]: A set of all unique descendant label set IDs and
      a dictionary mapping each unique descendant ID to its closest requested ancestor ID.
    """
    all_descendants = set(label_set_ids)
    closest_ancestor_dict = {id: id for id in label_set_ids}  # Maps each ID to itself initially
    to_process = set(label_set_ids)

    while to_process:
        current_id = to_process.pop()
        child_records = sql_handler.select_label_name_children(label_set_id=current_id)
        for child_id, *_ in child_records:
            if child_id not in all_descendants:
                all_descendants.add(child_id)
                closest_ancestor_dict[child_id] = closest_ancestor_dict.get(current_id, current_id)
                to_process.add(child_id)

    return all_descendants, closest_ancestor_dict


def merge_gap_data(values_1, gap_array_1, start_time_1, values_2, gap_array_2, start_time_2, freq_nhz):
    if not all(isinstance(arr, np.ndarray) for arr in [values_1, gap_array_1, values_2, gap_array_2]):
        raise ValueError(f"All input value and gap arrays must be numpy arrays.")

    if not values_1.dtype == values_2.dtype:
        raise ValueError(f"Values 1 and 2 have different dtypes {values_1.dtype}, {values_2.dtype}. Cannot merge.")

    end_time_1 = _calc_end_time_from_gap_data(values_1.size, gap_array_1, start_time_1, freq_nhz)
    end_time_2 = _calc_end_time_from_gap_data(values_2.size, gap_array_2, start_time_2, freq_nhz)

    overlap = (start_time_1 < end_time_2) and (end_time_1 > start_time_2)

    # If there's no overlap, you can simply concatenate the data.
    if is_gap_data_sorted(gap_array_1, freq_nhz) and is_gap_data_sorted(gap_array_2, freq_nhz) and not overlap:
        return _concatenate_gap_data(
            values_1, gap_array_1, start_time_1, values_2, gap_array_2, start_time_2, freq_nhz)

    # if starts, values and gaps are equal, then just return the 1's
    if np.array_equal(values_1, values_2) and \
            np.array_equal(gap_array_1, gap_array_2) and \
            start_time_1 == start_time_2:
        return values_1, gap_array_1, start_time_1

    # Convert both gap_data into messages
    message_starts_1, message_sizes_1 = reconstruct_messages(
        start_time_1, gap_array_1, freq_nhz, int(values_1.size))

    message_starts_2, message_sizes_2 = reconstruct_messages(
        start_time_2, gap_array_2, freq_nhz, int(values_2.size))

    # Sort both message lists + values, and copy values to not mess with the originals
    values_1, values_2 = values_1.copy(), values_2.copy()
    sort_message_time_values(message_starts_1, message_sizes_1, values_1)
    sort_message_time_values(message_starts_2, message_sizes_2, values_2)

    # Merge lists and Overwrite 2 over 1 if overlapping
    merged_starts, merged_sizes, merged_values = merge_sorted_messages(
        message_starts_1, message_sizes_1, values_1,
        message_starts_2, message_sizes_2, values_2, freq_nhz)

    # Convert back into gap data
    merged_gap_data = create_gap_arr_from_variable_messages(merged_starts, merged_sizes, freq_nhz)

    return merged_values, merged_gap_data, int(merged_starts[0])


def _concatenate_gap_data(values_1, gap_array_1, start_time_1, values_2, gap_array_2, start_time_2, freq_nhz):
    # if start_time_2 < start_time_1, swap 1's with 2's so that 1 is always temporally first.
    if start_time_2 < start_time_1:
        values_1, values_2 = values_2, values_1
        gap_array_1, gap_array_2 = gap_array_2, gap_array_1
        start_time_1, start_time_2 = start_time_2, start_time_1

    # Calculate the gap between blocks
    end_time_1 = _calc_end_time_from_gap_data(values_1.size, gap_array_1, start_time_1, freq_nhz)
    new_gap_index, new_gap_duration = int(values_1.size), start_time_2 - end_time_1

    # Calculate positions depending on if we need to add a new gap between blocks.
    if new_gap_duration == 0:
        merged_gap_size = gap_array_1.size + gap_array_2.size
        gap_array_2_index = gap_array_1.size
    else:
        merged_gap_size = gap_array_1.size + gap_array_2.size + 2
        gap_array_2_index = gap_array_1.size + 2

    merged_gap_array = np.empty(merged_gap_size, dtype=np.int64)
    merged_gap_array[:gap_array_1.size] = gap_array_1

    if new_gap_duration != 0:
        merged_gap_array[gap_array_1.size:gap_array_1.size + 2] = [new_gap_index, new_gap_duration]

    merged_gap_array[gap_array_2_index:] = gap_array_2
    # move the gap_array_2 indices forward.
    merged_gap_array[gap_array_2_index::2] += values_1.size

    # concatenate the values.
    merged_values = np.empty(values_1.size + values_2.size, dtype=values_1.dtype)
    merged_values[:values_1.size] = values_1
    merged_values[values_1.size:] = values_2
    return merged_values, merged_gap_array, start_time_1


def _calc_end_time_from_gap_data(values_size, gap_array, start_time, freq_nhz):
    if (int(values_size) * (10 ** 18)) % freq_nhz != 0:
        warnings.warn(f"Blocking starting on epoch {start_time} doesn't end on an integer number of nanoseconds, "
                      f"merge will be approximate.")
    sample_duration = (int(values_size) * (10 ** 18)) // freq_nhz
    gap_total = int(np.sum(gap_array[1::2]))
    return start_time + sample_duration + gap_total


def create_timestamps_from_gap_data(values_size, gap_array, start_time, freq_nhz):
    if (10 ** 18) % freq_nhz != 0:
        raise ValueError(f"Cannot create perfect timestamps from frequency_nhz = {freq_nhz}")

    period_ns = freq_nhz_to_period_ns(freq_nhz)
    timestamps = np.arange(values_size, dtype=np.int64)
    timestamps *= period_ns
    timestamps += start_time
    for i in range(gap_array.size // 2):
        gap_index, gap_duration = gap_array[2*i], gap_array[(2*i)+1]
        timestamps[gap_index:] += gap_duration

    return timestamps


def is_gap_data_sorted(gap_data, freq_nhz):
    period_ns = freq_nhz_to_period_ns(freq_nhz)
    gap_durations = gap_data[1::2]
    return np.all(gap_durations >= -period_ns)


def create_gap_arr_from_variable_messages(message_start_epoch_array: np.ndarray, message_size_array: np.ndarray,
                                          sample_freq):
    sample_freq = int(sample_freq)
    result_list = []
    current_sample = 0

    for i in range(1, len(message_start_epoch_array)):
        # Compute the time difference between consecutive messages
        delta_t = message_start_epoch_array[i] - message_start_epoch_array[i - 1]

        # Calculate the message period for the current message based on its size
        current_message_size = int(message_size_array[i - 1])
        current_message_period_ns = ((10 ** 18) * current_message_size) // sample_freq

        # Check if the time difference doesn't match the expected message period
        if delta_t != current_message_period_ns:
            # Compute the extra duration (time gap) and the starting index of the gap
            time_gap = delta_t - current_message_period_ns
            gap_start_index = current_sample + current_message_size

            # Add the gap information to the result list
            result_list.extend([gap_start_index, time_gap])

        # Update the current sample index for the next iteration
        current_sample += current_message_size

    # Convert the result list to a NumPy array of integers
    return np.array(result_list, dtype=np.int64)


def reconstruct_messages(start_time_nano_epoch, gap_data_array, sample_freq, num_values):
    message_sizes = np.empty((gap_data_array.size // 2) + 2, dtype=np.int64)
    message_sizes[0] = 0
    message_sizes[-1] = num_values
    message_sizes[1:-1] = gap_data_array[::2]
    message_sizes = np.diff(message_sizes)

    message_starts = np.empty((gap_data_array.size // 2) + 1, dtype=np.int64)

    message_starts[0] = start_time_nano_epoch

    if any(((10 ** 18) * int(m_size)) % sample_freq != 0 for m_size in message_sizes[:-1]):
        warnings.warn("Not all messages durations can be expressed as a perfect nanosecond integer, some rounding has occured")
    message_starts[1:] = [_message_size_to_duration_ns(int(m_size), sample_freq) for m_size in message_sizes[:-1]]
    message_starts[1:] += gap_data_array[1::2]
    message_starts = np.cumsum(message_starts)

    return message_starts, message_sizes


def sort_message_time_values(message_starts: np.ndarray, message_sizes: np.ndarray, value_array: np.ndarray):
    # The start and end index of each message within the value_array
    message_end_indices, message_start_indices = _get_message_indices(message_sizes)

    sorted_message_i_arr = np.argsort(message_starts)
    if np.all(sorted_message_i_arr[:-1] <= sorted_message_i_arr[1:]):
        return

    message_starts[:] = message_starts[sorted_message_i_arr]
    message_sizes[:] = message_sizes[sorted_message_i_arr]

    values_copy = value_array.copy()

    start_idx = 0
    for message_i in range(message_start_indices.size):
        sorted_left = message_start_indices[sorted_message_i_arr][message_i]
        sorted_right = message_end_indices[sorted_message_i_arr][message_i]
        message_size = message_sizes[message_i]

        value_array[start_idx:start_idx+message_size] = values_copy[sorted_left:sorted_right]
        start_idx += message_size


def _get_message_indices(message_sizes):
    message_end_indices = np.cumsum(message_sizes)
    message_start_indices = message_end_indices - message_sizes
    return message_end_indices, message_start_indices


def merge_sorted_messages(message_starts_1, message_sizes_1, values_1,
                          message_starts_2, message_sizes_2, values_2, freq_nhz):

    message_ends_1 = message_starts_1 + np.array(
        [_message_size_to_duration_ns(int(m_size), freq_nhz) for m_size in message_sizes_1], dtype=np.int64)

    message_ends_2 = message_starts_2 + np.array(
        [_message_size_to_duration_ns(int(m_size), freq_nhz) for m_size in message_sizes_2], dtype=np.int64)

    message_end_indices_1, message_start_indices_1 = _get_message_indices(message_sizes_1)
    message_end_indices_2, message_start_indices_2 = _get_message_indices(message_sizes_2)

    merged_starts = []
    merged_ends = []
    merged_values = []

    i = j = 0
    while i < len(message_starts_1) and j < len(message_starts_2):
        if message_starts_1[i] <= message_starts_2[j]:
            # 1 <= 2
            if merged_ends and message_starts_1[i] == merged_ends[-1]:
                # If the messages link perfectly together, update the last entry
                merged_ends[-1] = message_ends_1[i]
                new_merged_values = np.empty(
                    merged_values[-1].size + message_sizes_1[i],
                    dtype=merged_values[-1].dtype)

                new_merged_values[:merged_values[-1].size] = merged_values[-1]
                new_merged_values[merged_values[-1].size:] = \
                    values_1[message_start_indices_1[i]:message_end_indices_1[i]]

                merged_values[-1] = new_merged_values
                i += 1
                continue

            elif merged_ends and message_starts_1[i] < merged_ends[-1]:
                # If there is overlap
                duration_ns_overlap = int(merged_ends[-1] - message_starts_1[i])

                if (duration_ns_overlap * int(freq_nhz)) % (10 ** 18) != 0:
                    warnings.warn("Old data overlaps with new data out of phase.")
                    new_starts, new_ends, new_values = generate_single_messages_from_big_message(
                        message_starts_1[i], message_ends_1[i],
                        message_start_indices_1[i], message_end_indices_1[i], values_1,
                        merged_starts[-1], merged_ends[-1],
                        0, merged_values[-1].size, merged_values[-1],
                        freq_nhz)

                    # delete last entry
                    merged_starts.pop()
                    merged_ends.pop()
                    merged_values.pop()

                    # Add new one
                    merged_starts.extend(new_starts)
                    merged_ends.extend(new_ends)
                    merged_values.extend(new_values)
                    i += 1
                    continue

                num_values_overlap = math.ceil((duration_ns_overlap * int(freq_nhz)) / (10 ** 18))

                # Reduce input values
                message_start_indices_1[i] += num_values_overlap

                # Add to start_time
                message_starts_1[i] += _message_size_to_duration_ns(num_values_overlap, freq_nhz)

            if message_starts_1[i] < message_ends_1[i]:
                # If there is any message left, append it
                merged_starts.append(message_starts_1[i])
                merged_ends.append(message_ends_1[i])
                merged_values.append(values_1[message_start_indices_1[i]:message_end_indices_1[i]])

            i += 1

        else:
            # 2 < 1
            if merged_ends and message_starts_2[j] == merged_ends[-1]:
                # If the messages link perfectly together, update the last entry
                merged_ends[-1] = message_ends_2[j]
                new_merged_values = np.empty(
                    merged_values[-1].size + message_sizes_2[j],
                    dtype=merged_values[-1].dtype)

                new_merged_values[:merged_values[-1].size] = merged_values[-1]
                new_merged_values[merged_values[-1].size:] = \
                    values_2[message_start_indices_2[j]:message_end_indices_2[j]]

                merged_values[-1] = new_merged_values
                j += 1
                continue

            if merged_ends and message_starts_2[j] < merged_ends[-1] and message_ends_2[j] < merged_ends[-1]:
                # If the new message is completely within the most recent merged message
                # Then we simply overwrite the needed values
                left_index_duration = int(message_starts_2[j] - merged_starts[-1])
                if (left_index_duration * int(freq_nhz)) % (10 ** 18) != 0:
                    warnings.warn("New data overlaps with old data out of phase.")
                    new_starts, new_ends, new_values = generate_single_messages_from_big_message(
                        message_starts_2[j], message_ends_2[j],
                        message_start_indices_2[j], message_end_indices_2[j], values_2,
                        merged_starts[-1], merged_ends[-1],
                        0, merged_values[-1].size, merged_values[-1],
                        freq_nhz)

                    # delete last entry
                    merged_starts.pop()
                    merged_ends.pop()
                    merged_values.pop()

                    # Add new one
                    merged_starts.extend(new_starts)
                    merged_ends.extend(new_ends)
                    merged_values.extend(new_values)
                    j += 1
                    continue

                left_index = round((left_index_duration * int(freq_nhz)) / (10 ** 18))
                num_values = message_end_indices_2[j] - message_start_indices_2[j]

                right_index = left_index + num_values
                merged_values[-1][left_index:right_index] = values_2[message_start_indices_2[j]:message_end_indices_2[j]]

                j += 1
                continue

            out_of_phase = False
            while merged_ends and message_starts_2[j] < merged_ends[-1]:
                # If there is overlap
                duration_ns_overlap = int(merged_ends[-1] - message_starts_2[j])
                if (duration_ns_overlap * int(freq_nhz)) % (10 ** 18) != 0:
                    warnings.warn("New data overlaps with old data out of phase.")
                    new_starts, new_ends, new_values = generate_single_messages_from_big_message(
                        message_starts_2[j], message_ends_2[j],
                        message_start_indices_2[j], message_end_indices_2[j], values_2,
                        merged_starts[-1], merged_ends[-1],
                        0, merged_values[-1].size, merged_values[-1],
                        freq_nhz)

                    # delete last entry
                    merged_starts.pop()
                    merged_ends.pop()
                    merged_values.pop()

                    # Add new one
                    merged_starts.extend(new_starts)
                    merged_ends.extend(new_ends)
                    merged_values.extend(new_values)
                    out_of_phase = True
                    break

                num_values_overlap = math.ceil((duration_ns_overlap * int(freq_nhz)) / (10 ** 18))

                remaining_values = merged_values[-1].size - num_values_overlap
                if remaining_values <= 0:
                    merged_starts.pop()
                    merged_ends.pop()
                    merged_values.pop()

                else:
                    # Reduce old values
                    merged_values[-1] = merged_values[-1][:remaining_values]

                    # Reduce old end_time
                    merged_ends[-1] -= _message_size_to_duration_ns(num_values_overlap, freq_nhz)

            if not out_of_phase:
                # Once we know there is no overlap, we can safely append.
                merged_starts.append(message_starts_2[j])
                merged_ends.append(message_ends_2[j])
                merged_values.append(values_2[message_start_indices_2[j]:message_end_indices_2[j]])
            j += 1

    # Add any remaining messages from 1
    while i < len(message_starts_1):
        if merged_ends and message_starts_1[i] == merged_ends[-1]:
            # If the messages link perfectly together, update the last entry
            merged_ends[-1] = message_ends_1[i]
            new_merged_values = np.empty(
                merged_values[-1].size + message_sizes_1[i],
                dtype=merged_values[-1].dtype)

            new_merged_values[:merged_values[-1].size] = merged_values[-1]
            new_merged_values[merged_values[-1].size:] = \
                values_1[message_start_indices_1[i]:message_end_indices_1[i]]

            merged_values[-1] = new_merged_values
            i += 1
            continue

        elif merged_ends and message_starts_1[i] < merged_ends[-1]:
            # If there is overlap
            duration_ns_overlap = int(merged_ends[-1] - message_starts_1[i])
            if (duration_ns_overlap * int(freq_nhz)) % (10 ** 18) != 0:
                warnings.warn("Old data overlaps with new data out of phase.")
                new_starts, new_ends, new_values = generate_single_messages_from_big_message(
                    message_starts_1[i], message_ends_1[i],
                    message_start_indices_1[i], message_end_indices_1[i], values_1,
                    merged_starts[-1], merged_ends[-1],
                    0, merged_values[-1].size, merged_values[-1],
                    freq_nhz)

                # delete last entry
                merged_starts.pop()
                merged_ends.pop()
                merged_values.pop()

                # Add new one
                merged_starts.extend(new_starts)
                merged_ends.extend(new_ends)
                merged_values.extend(new_values)
                i += 1
                continue

            num_values_overlap = math.ceil((duration_ns_overlap * int(freq_nhz)) / (10 ** 18))

            # Reduce input values
            message_start_indices_1[i] += num_values_overlap

            # Add to start_time
            message_starts_1[i] += _message_size_to_duration_ns(num_values_overlap, freq_nhz)

        if message_starts_1[i] < message_ends_1[i]:
            # If there is any message left, append it
            merged_starts.append(message_starts_1[i])
            merged_ends.append(message_ends_1[i])
            merged_values.append(values_1[message_start_indices_1[i]:message_end_indices_1[i]])

        i += 1

    # Add any remaining messages from 2
    while j < len(message_starts_2):
        if merged_ends and message_starts_2[j] == merged_ends[-1]:
            # If the messages link perfectly together, update the last entry
            merged_ends[-1] = message_ends_2[j]
            new_merged_values = np.empty(
                merged_values[-1].size + message_sizes_2[j],
                dtype=merged_values[-1].dtype)

            new_merged_values[:merged_values[-1].size] = merged_values[-1]
            new_merged_values[merged_values[-1].size:] = \
                values_2[message_start_indices_2[j]:message_end_indices_2[j]]

            merged_values[-1] = new_merged_values
            j += 1
            continue

        if merged_ends and message_starts_2[j] < merged_ends[-1] and message_ends_2[j] < merged_ends[-1]:
            # If the new message is completely within the most recent merged message
            # Then we simply overwrite the needed values
            left_index_duration = int(message_starts_2[j] - merged_starts[-1])
            if (left_index_duration * int(freq_nhz)) % (10 ** 18) != 0:
                warnings.warn("New data overlaps with old data out of phase.")
                new_starts, new_ends, new_values = generate_single_messages_from_big_message(
                    message_starts_2[j], message_ends_2[j],
                    message_start_indices_2[j], message_end_indices_2[j], values_2,
                    merged_starts[-1], merged_ends[-1],
                    0, merged_values[-1].size, merged_values[-1],
                    freq_nhz)

                # delete last entry
                merged_starts.pop()
                merged_ends.pop()
                merged_values.pop()

                # Add new one
                merged_starts.extend(new_starts)
                merged_ends.extend(new_ends)
                merged_values.extend(new_values)
                j += 1
                continue

            left_index = round((left_index_duration * int(freq_nhz)) / (10 ** 18))
            num_values = message_end_indices_2[j] - message_start_indices_2[j]
            # right_index_duration = int(merged_ends[-1] - message_ends_2[j])
            # right_index = merged_values[-1].size - math.ceil((right_index_duration * int(freq_nhz)) / (10 ** 18))
            right_index = left_index + num_values
            merged_values[-1][left_index:right_index] = values_2[message_start_indices_2[j]:message_end_indices_2[j]]
            j += 1
            continue

        out_of_phase = False
        while merged_ends and message_starts_2[j] < merged_ends[-1]:
            # If there is overlap
            duration_ns_overlap = int(merged_ends[-1] - message_starts_2[j])
            if (duration_ns_overlap * int(freq_nhz)) % (10 ** 18) != 0:
                warnings.warn("New data overlaps with old data out of phase.")
                new_starts, new_ends, new_values = generate_single_messages_from_big_message(
                    message_starts_2[j], message_ends_2[j],
                    message_start_indices_2[j], message_end_indices_2[j], values_2,
                    merged_starts[-1], merged_ends[-1],
                    0, merged_values[-1].size, merged_values[-1],
                    freq_nhz)

                # delete last entry
                merged_starts.pop()
                merged_ends.pop()
                merged_values.pop()

                # Add new one
                merged_starts.extend(new_starts)
                merged_ends.extend(new_ends)
                merged_values.extend(new_values)
                out_of_phase = True
                break
            num_values_overlap = math.ceil((duration_ns_overlap * int(freq_nhz)) / (10 ** 18))

            remaining_values = merged_values[-1].size - num_values_overlap
            if remaining_values <= 0:
                merged_starts.pop()
                merged_ends.pop()
                merged_values.pop()

            else:
                # Reduce old values
                merged_values[-1] = merged_values[-1][:remaining_values]

                # Reduce old end_time
                merged_ends[-1] -= _message_size_to_duration_ns(num_values_overlap, freq_nhz)

        if not out_of_phase:
            # Once we know there is no overlap, we can safely append.
            merged_starts.append(message_starts_2[j])
            merged_ends.append(message_ends_2[j])
            merged_values.append(values_2[message_start_indices_2[j]:message_end_indices_2[j]])
        j += 1

    merged_starts = np.array(merged_starts, dtype=np.int64)
    merged_sizes = np.array([value_arr.size for value_arr in merged_values], dtype=np.int64)
    merged_values = np.array([], dtype=np.float64) if len(merged_values) == 0 else \
        np.concatenate(merged_values, dtype=merged_values[0].dtype)

    return merged_starts, merged_sizes, merged_values


def _message_size_to_duration_ns(m_size, freq_nhz):
    return ((10 ** 18) * int(m_size)) // freq_nhz


def merge_timestamp_data(values_1, times_1, values_2, times_2):
    # concatenate the time and value arrays
    concatenated_times = np.concatenate((times_1, times_2))
    concatenated_values = np.concatenate((values_1, values_2))

    # remove duplicate times and get indices to sort values in time order
    time_data, index_unique = np.unique(concatenated_times, return_index=True)

    # return the sorted arrays
    return time_data, concatenated_values[index_unique]


def generate_single_messages_from_big_message(message_start_1, message_end_1, start_index_1, end_index_1, message_values_1,
                                              message_start_2, message_end_2, start_index_2, end_index_2, message_values_2,
                                              freq_nhz):
    if (10 ** 18) % freq_nhz != 0:
        warnings.warn("Out of phase data is being merged with a freq that doesn't create perfect nanosecond integer "
                      "timestamps, some rounding will occur in the stored data.")

    # Calculate the period based on frequency
    period_ns = (10 ** 18) // freq_nhz

    # Initialize lists for both data sets
    starts_1, ends_1, values_list_1 = [], [], []
    starts_2, ends_2, values_list_2 = [], [], []

    # Process the first set of parameters
    cur_time = message_start_1
    for index in range(start_index_1, end_index_1):
        starts_1.append(cur_time)
        values_list_1.append(message_values_1[index:index+1])
        cur_time += period_ns
        ends_1.append(cur_time)

    # Process the second set of parameters
    cur_time = message_start_2
    for index in range(start_index_2, end_index_2):
        starts_2.append(cur_time)
        values_list_2.append(message_values_2[index:index+1])
        cur_time += period_ns
        ends_2.append(cur_time)

    # Now merge the two sets in order of ascending starts
    starts, ends, values_list = [], [], []
    index_1, index_2 = 0, 0

    while index_1 < len(starts_1) and index_2 < len(starts_2):
        if starts_1[index_1] <= starts_2[index_2]:
            starts.append(starts_1[index_1])
            ends.append(ends_1[index_1])
            values_list.append(values_list_1[index_1])
            index_1 += 1
        else:
            starts.append(starts_2[index_2])
            ends.append(ends_2[index_2])
            values_list.append(values_list_2[index_2])
            index_2 += 1

    # Append remaining elements from the first list
    while index_1 < len(starts_1):
        starts.append(starts_1[index_1])
        ends.append(ends_1[index_1])
        values_list.append(values_list_1[index_1])
        index_1 += 1

    # Append remaining elements from the second list
    while index_2 < len(starts_2):
        starts.append(starts_2[index_2])
        ends.append(ends_2[index_2])
        values_list.append(values_list_2[index_2])
        index_2 += 1

    return starts, ends, values_list


def get_headers(sdk, measure_id: int = None, start_time: int = None, end_time: int = None, device_id: int = None,
                patient_id=None, time_units: str = None, mrn: int = None):
    """
    The method for querying block headers from the dataset, indexed by signal type (measure_id or measure_tag with freq and units),
    time (start_time_n and end_time_n), and data source (device_id, device_tag, patient_id, or mrn).

    If measure_id is None, measure_tag along with freq and units must not be None, and vice versa.
    Similarly, if device_id is None, device_tag must not be None, and if patient_id is None, mrn must not be None.

    :param AtriumSDK sdk: The AtriumSDK object.
    :param int measure_id: The measure identifier. If None, measure_tag must be provided.
    :param int start_time: The start epoch in nanoseconds of the data you would like to query.
    :param int end_time: The end epoch in nanoseconds. The end time is not inclusive.
    :param int device_id: The device identifier. If None, device_tag must be provided.
    :param int patient_id: The patient identifier. If None, mrn must be provided.
    """

    # check that a correct unit type was entered
    time_units = "ns" if time_units is None else time_units

    if time_units not in time_unit_options.keys():
        raise ValueError("Invalid time units. Expected one of: %s" % time_unit_options)

    # convert start and end time to nanoseconds
    start_time = int(start_time * time_unit_options[time_units])
    end_time = int(end_time * time_unit_options[time_units])

    if patient_id is None and mrn is not None:
        patient_id = sdk.get_patient_id(mrn)

    # If we don't already have the blocks
    # Select all blocks from the block_index (sql table) that match params.
    block_list = sdk.sql_handler.select_blocks(int(measure_id), int(start_time), int(end_time), device_id,
                                               patient_id)

    # Concatenate continuous byte intervals to cut down on total number of reads.
    read_list = condense_byte_read_list(block_list)

    # if no matching block ids
    if len(read_list) == 0:
        return [], np.array([]), np.array([])

    # Map file_ids to filenames and return a dictionary.
    file_id_list = [row[2] for row in read_list]
    filename_dict = sdk.get_filename_dict(file_id_list)

    # Condense the block list for optimized reading
    read_list = condense_byte_read_list(block_list)

    # Read the data from the files using the read list
    encoded_bytes = sdk.file_api.read_file_list(read_list, filename_dict)

    num_bytes_list = [row[5] for row in block_list]
    byte_start_array = np.cumsum(num_bytes_list, dtype=np.uint64)
    byte_start_array = np.concatenate([np.array([0], dtype=np.uint64), byte_start_array[:-1]], axis=None)
    headers = sdk.block.decode_headers(encoded_bytes, byte_start_array)

    return headers


def delete_unreferenced_tsc_files(sdk):
    """
    The method is for removing unreferenced tsc files. When you write a lot of data with merge=True
    eventually a lot of unused tsc files will build up since the function is copy on write. This
    Will use up a lot of space unnecessarily and this function will help you free up that space.

    :param AtriumSDK sdk: The AtriumSDK object.
    """
    # find tsc files in the file_index that have no references to them in the block_index
    files = sdk.sql_handler.find_unreferenced_tsc_files()

    # if there are no tsc files to remove just return
    if len(files) == 0:
        return

    # extract file names from files and make it a set so we can do a set intersection later
    file_names = {file[1] for file in files}
    # extract the ids and put them in a tuple so we can remove them from the sql table later
    file_ids = [(file[0],) for file in files]

    # clean up memory
    del files

    # walk the tsc directory looking for files to delete (os.walk is a generator for memory efficiency)
    for root, _, files in os.walk(sdk.file_api.top_level_dir):
        # check if there is a match between any of the tsc file names to be deleted and files in the current directory
        matches = set(files) & file_names

        # if you find a match remove the file from disk
        for m in matches:
            print(f"Deleting tsc file {m} from disk")
            os.remove(os.path.join(root, m))

    # free up memory
    del file_names

    # remove them from the file_index
    sdk.sql_handler.delete_tsc_files(file_ids)
    print("Completed removal of unreferenced tsc files")
