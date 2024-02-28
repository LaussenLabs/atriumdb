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
        if len(result) == 0 or result[-1][1] != row[3] or result[-1][2] + result[-1][3] != row[4]:
            # append measure_id, device_id, file_id, start_byte and num_bytes
            result.append([row[1], row[2], row[3], row[4], row[5]])
        else:
            # if the blocks are continuous merge the reads together by adding the size of the next block to the
            # num_bytes field
            result[-1][3] += row[5]

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
