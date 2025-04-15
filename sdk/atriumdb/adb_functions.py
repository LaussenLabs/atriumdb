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
from tqdm import tqdm
import time
from urllib.parse import urlsplit, urlunsplit
from numpy.lib.stride_tricks import sliding_window_view
from atriumdb.helpers.block_calculations import freq_nhz_to_period_ns
import logging
import bisect

from atriumdb.helpers.block_calculations import calc_time_by_freq
from atriumdb.helpers.block_constants import TIME_TYPES
from atriumdb.intervals.union import intervals_union_list

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


def get_measure_id_from_generic_measure(sdk, measure, measure_tag_match_rule="best", freq_units="Hz"):
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
        dict_freq_units = "nHz" if 'freq_nhz' in measure else "Hz" if 'freq_hz' in measure else None

        if freq and units:
            # Use sdk.get_measure_id for a unique match when both freq and units are specified
            measure_id = sdk.get_measure_id(measure['tag'], freq=freq, units=units, freq_units=dict_freq_units)
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
    elif isinstance(measure, tuple):
        tag, freq, units = measure
        measure_id = sdk.get_measure_id(tag, freq=freq, units=units, freq_units=freq_units)
        if measure_id is not None:
            measure_ids.append(measure_id)
        else:
            raise ValueError(f"No unique measure found for tuple: {measure} assuming freq_units = {freq_units}.")
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


def create_gap_arr_from_variable_messages(message_start_epoch_array, message_size_array, sample_freq):
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
    # Find the smallest start time from both inputs
    min_start = min(np.min(message_starts_1), np.min(message_starts_2))

    # Create relative start times
    rel_starts_1 = message_starts_1 - min_start
    rel_starts_2 = message_starts_2 - min_start

    # Calculate period in nanoseconds for one sample
    period_ns = float((10 ** 18) / int(freq_nhz))

    timestamps_1 = []
    timestamps_2 = []

    # Create timestamps for dataset 1
    for start, size in zip(rel_starts_1, message_sizes_1):
        timestamps_1.append(start + (np.arange(size) * period_ns))

    # Create timestamps for dataset 2
    for start, size in zip(rel_starts_2, message_sizes_2):
        timestamps_2.append(start + (np.arange(size) * period_ns))

    # Convert lists to numpy arrays
    timestamps_1 = np.concatenate(timestamps_1, dtype=np.float64)
    timestamps_2 = np.concatenate(timestamps_2, dtype=np.float64)

    # Combine and sort the timestamp and value arrays
    combined_timestamps = np.concatenate((timestamps_1, timestamps_2))
    combined_values = np.concatenate((values_1, values_2))

    sorted_indices = np.argsort(combined_timestamps)
    sorted_timestamps = combined_timestamps[sorted_indices]
    sorted_values = combined_values[sorted_indices]

    # Convert the sorted timestamps into "sample times"
    sample_times = sorted_timestamps / period_ns

    # Find differences in sample times
    diff_sample_times = np.diff(sample_times)

    # Calculate the max floating point imprecision
    max_timestamp = np.max(sorted_timestamps)
    max_diff_sample_times = np.max(diff_sample_times)

    max_imprecision = np.finfo(np.float64).eps * (max_timestamp / period_ns + max_diff_sample_times)

    # Identify where timestamps are close to 1 (consecutive message elements)
    close_to_one = np.isclose(diff_sample_times, 1, atol=max_imprecision, rtol=0)

    # Identify where timestamps are close to 0 (duplicates)
    close_to_zero = np.isclose(diff_sample_times, 0, atol=max_imprecision, rtol=0)

    # Reconstruct the merged messages, handling duplicates by skipping them
    merged_starts = []
    merged_sizes = []
    merged_values = []

    current_message_start = sorted_timestamps[0]
    current_message_values = [sorted_values[0]]

    for i in range(1, sample_times.size):
        # Skip over duplicates by looking ahead until we find a non-duplicate
        if close_to_zero[i - 1]:
            # If the next value is a duplicate, keep overwriting current value until the last one
            current_message_values[-1] = sorted_values[i]
        else:
            if close_to_one[i - 1]:
                # This value continues the current message
                current_message_values.append(sorted_values[i])
            else:
                # End of the current message, start a new one
                merged_starts.append(current_message_start)
                merged_sizes.append(len(current_message_values))
                merged_values.append(np.array(current_message_values))

                current_message_start = sorted_timestamps[i]
                current_message_values = [sorted_values[i]]

    # Append the last message
    if len(current_message_values) > 0:
        merged_starts.append(current_message_start)
        merged_sizes.append(len(current_message_values))
        merged_values.append(np.array(current_message_values))

    # Convert to final outputs
    merged_starts = np.array(merged_starts, dtype=np.int64) + min_start
    merged_sizes = np.array(merged_sizes, dtype=np.int64)
    merged_values = np.concatenate(merged_values) if merged_values else np.array([], dtype=np.float64)

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

    # walk the tsc directory looking for files to delete (walk is a generator for memory efficiency)
    for root, dir_names, files in sdk.file_api.walk(sdk.file_api.top_level_dir):
        # remove any dirs that are not digits (device_id or measure_id) so walk doesn't traverse those directories
        dir_names[:] = [d for d in dir_names if d.isdigit()]

        # check if there is a match between any of the tsc file names to be deleted and files in the current directory
        matches = set(files) & file_names

        # if you find a match remove the file from disk
        for m in matches:
            print(f"Deleting tsc file {m} from disk")
            sdk.file_api.remove(os.path.join(root, m))

    # free up memory
    del file_names

    # remove them from the file_index
    sdk.sql_handler.delete_files_by_ids(file_ids)
    print("Completed removal of unreferenced tsc files")


def get_all_blocks(sdk, measure_id, device_id):
    query = """
        SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values 
        FROM block_index
        WHERE measure_id = ? AND device_id = ?
        ORDER BY start_time_n ASC;
    """

    with sdk.sql_handler.connection() as (conn, cursor):
        cursor.execute(query, (measure_id, device_id))
        block_query_result = cursor.fetchall()

    return block_query_result


def reencode_dataset(sdk, values_per_block=131072, blocks_per_file=2048, interval_gap_tolerance_nano=0):
    # WARNING:
    # This function does not preserve duplicate values. (uses np.unique)
    # Only works if the frequency (in nanohertz) of each signal divides 10^18.
    original_block_size = sdk.block.block_size
    sdk.block.block_size = values_per_block

    measures = sdk.get_all_measures()
    devices = sdk.get_all_devices()

    measure_data = list(measures.items())
    device_data = list(devices.items())

    for device_id, device_info in tqdm(device_data, desc="Devices"):
        for measure_id, measure_info in tqdm(measure_data, desc="Measures"):
            sorted_block_list_by_time = get_all_blocks(sdk, measure_id, device_id)
            if len(sorted_block_list_by_time) == 0:
                continue

            file_id_list = list(set([row[3] for row in sorted_block_list_by_time]))
            filename_dict = sdk.get_filename_dict(file_id_list)

            total_measure_device_interval_array = []

            for block_group in group_sorted_block_list(
                    sorted_block_list_by_time, num_values_per_group=values_per_block * blocks_per_file):
                if (10 ** 18) % measure_info['freq_nhz'] == 0:
                    # Perfect Precision Time-Value Pairs are Possible
                    period_ns = (10 ** 18) // measure_info['freq_nhz']
                    time_type = 1
                    r_headers, timestamp_arr, r_values = sdk.get_data_from_blocks(block_group, filename_dict, 0,
                                                                                2 ** 62, False, time_type, sort=False,
                                                                                allow_duplicates=True)

                    if timestamp_arr.size == 0:
                        continue

                    # Prepare segments for encode_blocks_from_multiple_segments
                    segments = []
                    block_group_interval_data = []

                    # Group data by scale factor
                    for group_headers, group_times, group_values in group_headers_by_scale_factor(
                            r_headers, timestamp_arr, r_values):

                        # Sort the group data
                        group_times, sorted_time_indices = np.unique(group_times, return_index=True)
                        group_values = group_values[sorted_time_indices]

                        # Get interval data
                        group_interval_data = get_interval_list_from_ordered_timestamps(group_times, period_ns)
                        block_group_interval_data.append(group_interval_data)

                        if np.issubdtype(group_values.dtype, np.integer):
                            raw_value_type = 1
                            encoded_value_type = 3
                        else:
                            raw_value_type = 2
                            encoded_value_type = 2

                        # Create segment dictionary
                        segment = {
                            'times': group_times,
                            'values': group_values,
                            'freq_nhz': measure_info['freq_nhz'],
                            'start_ns': group_times[0],
                            'raw_time_type': 1,
                            'encoded_time_type': 2,
                            'raw_value_type': raw_value_type,
                            'encoded_value_type': encoded_value_type,
                            'scale_m': group_headers[0].scale_m,
                            'scale_b': group_headers[0].scale_b
                        }
                        segments.append(segment)

                    # Now, call encode_blocks_from_multiple_segments
                    encoded_bytes, encoded_headers, byte_start_array = (
                        sdk.block.encode_blocks_from_multiple_segments(segments))

                    # Handle the outputs
                    total_encoded_bytes = encoded_bytes
                    total_encoded_headers = encoded_headers
                    total_byte_start_array = byte_start_array

                    # Process the intervals
                    block_group_interval_data = intervals_union_list(block_group_interval_data)
                else:
                    # Perfect Precision Time-Value Pairs Not Possible: Use Message/Gap Format.
                    raise NotImplementedError("Not Implemented For Nanohertz Frequencies That Don't Divide 10^18")

                # Write the encoded bytes to disk
                filename = sdk.file_api.write_bytes(measure_id, device_id, total_encoded_bytes)

                block_data = []
                for header_i, header in enumerate(total_encoded_headers):
                    block_data.append({
                        "measure_id": measure_id,
                        "device_id": device_id,
                        "start_byte": int(total_byte_start_array[header_i]),
                        "num_bytes": header.meta_num_bytes + header.t_num_bytes + header.v_num_bytes,
                        "start_time_n": header.start_n,
                        "end_time_n": header.end_n,
                        "num_values": header.num_vals,
                    })

                # Write new and delete old SQL blocks
                sdk.sql_handler.insert_and_delete_tsc_file_data(
                    filename, block_data, [block[0] for block in block_group])

                total_measure_device_interval_array.append(block_group_interval_data)

            # Combine collected intervals and replace the old intervals with the collected.
            total_measure_device_interval_array = intervals_union_list(
                total_measure_device_interval_array, gap_tolerance_nano=interval_gap_tolerance_nano)

            sdk.sql_handler.replace_intervals(measure_id, device_id, total_measure_device_interval_array)

            # Remove TSC files from SQL and disk
            sdk.sql_handler.delete_files_by_ids(file_id_list)

            for filename in filename_dict.values():
                file_path = sdk.file_api.to_abs_path(filename, measure_id, device_id)
                sdk.file_api.remove(file_path)

    sdk.block.block_size = original_block_size


def group_sorted_block_list(sorted_block_list, num_values_per_group=8388608):
    next_group = []
    total_group_size = 0

    for block in tqdm(sorted_block_list, desc="Blocks"):
        if total_group_size >= num_values_per_group:
            yield next_group
            next_group = []
            total_group_size = 0

        num_values = block[8]
        next_group.append(block)
        total_group_size += num_values

    if total_group_size > 0:
        yield next_group


def group_headers_by_scale_factor(headers, times_array, values_array):
    if len(headers) == 0:
        return
    start_idx = 0
    current_scale_m, current_scale_b = headers[0].scale_m, headers[0].scale_b
    current_group = [headers[0]]

    for h in headers[1:]:
        if h.scale_m == current_scale_m and h.scale_b == current_scale_b:
            # Same scale factors, add to current group
            current_group.append(h)
        else:
            # Different scale factors, yield current group
            end_idx = start_idx + sum(header.num_vals for header in current_group)
            yield current_group, times_array[start_idx:end_idx], values_array[start_idx:end_idx]

            # Start new group
            start_idx = end_idx
            current_scale_m, current_scale_b = h.scale_m, h.scale_b
            current_group = [h]

    # After loop ends, yield any remaining group
    if current_group:
        end_idx = start_idx + sum(header.num_vals for header in current_group)
        yield current_group, times_array[start_idx:end_idx], values_array[start_idx:end_idx]


def get_interval_list_from_ordered_timestamps(timestamps, period_ns):
    if timestamps.size == 0:
        return []

    diffs = np.diff(timestamps)
    break_indices = np.where(diffs > period_ns)[0]

    # Start indices of each region ([0] + break_indices + 1)
    start_indices = np.insert(break_indices + 1, 0, 0)
    # End indices of each region (break_indices + [len(timestamps) - 1])
    end_indices = np.append(break_indices, timestamps.size - 1)

    start_times = timestamps[start_indices]
    end_times = timestamps[end_indices] + period_ns

    regions = np.column_stack((start_times, end_times))

    return regions