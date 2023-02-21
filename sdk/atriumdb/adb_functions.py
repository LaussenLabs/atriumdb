import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
import logging

from atriumdb.helpers.block_calculations import calc_time_by_freq
from atriumdb.helpers.block_constants import TIME_TYPES


def get_block_and_interval_data(measure_id, device_id, metadata, start_bytes, intervals):
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
    for interval in intervals:
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
            result.append([row[2], row[3], row[4], row[5]])
        else:
            result[-1][3] += row[5]

    return result


def find_intervals(freq_nhz, raw_time_type, time_data, data_start_time, num_values):
    period_ns = int((10 ** 18) / freq_nhz)
    if raw_time_type == TIME_TYPES['TIME_ARRAY_INT64_NS']:
        intervals = [[time_data[0], 0]]
        time_deltas = time_data[1:] - time_data[:-1]
        for time_arr_i in range(time_data.size - 1):
            if time_deltas[time_arr_i] > period_ns:
                intervals[-1][-1] = time_data[time_arr_i]
                intervals.append([time_data[time_arr_i + 1], 0])

        intervals[-1][-1] = time_data[-1]

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


def merge_interval_lists(list_a, list_b):
    return np.array([[max(first[0], second[0]), min(first[1], second[1])]
                     for first in list_a for second in list_b
                     if max(first[0], second[0]) <= min(first[1], second[1])])


def sort_data(times, values, headers):
    start_bench = time.perf_counter()
    if len(headers) == 0:
        return times, values

    block_info = np.zeros((len(headers), 4), dtype=np.int64)
    block_info[:] = [[h.start_n, h.end_n, h.num_vals, 0] for h in headers]
    np.cumsum(block_info.T[2], out=block_info.T[3])
    block_info.T[2] = block_info.T[3] - block_info.T[2]

    end_bench = time.perf_counter()
    logging.debug(f"rearrange block data info {(end_bench - start_bench) * 1000} ms")

    if np.all(np.greater_equal(block_info.T[0][1:], block_info.T[1][:-1])) and \
            np.all(np.greater(block_info.T[0][1:], block_info.T[0][:-1])):
        logging.debug("Already Sorted.")
        return times, values

    start_bench = time.perf_counter()
    _, sorted_block_i = np.unique(block_info.T[0], return_index=True)
    block_info = block_info[sorted_block_i]

    end_bench = time.perf_counter()
    logging.debug(f"sort data by block {(end_bench - start_bench) * 1000} ms")
    if np.all(np.greater_equal(block_info.T[0][1:], block_info.T[1][:-1])):
        # Blocks don't intersect each other.
        start_bench = time.perf_counter()
        # Original Index Creation
        sorted_time_indices = np.concatenate([np.arange(i_start, i_end) for _, _, i_start, i_end in block_info])

        # New Index Creation.
        # sorted_time_indices = np.arange(times.size)

        # new_times, new_values = np.zeros(times.size, dtype=times.dtype), np.zeros(values.size, dtype=values.dtype)
        # sorted_index = 0
        # for _, _, i_start, i_end in block_info:
        #     dur = i_end - i_start
        #     new_times[sorted_index:sorted_index + dur], new_values[sorted_index:sorted_index + dur] = \
        #         times[i_start:i_end], values[i_start:i_end]
        #     sorted_index += dur
        #
        times, values = times[sorted_time_indices], values[sorted_time_indices]
        end_bench = time.perf_counter()
        logging.debug(f"Sort by blocks {(end_bench - start_bench) * 1000} ms")
        # return new_times, new_values
        return times, values
    else:
        # Blocks do intersect each other, so sort every value.
        start_bench = time.perf_counter()
        sorted_times, sorted_time_indices = np.unique(times, return_index=True)
        end_bench = time.perf_counter()
        logging.debug(f"sort every value {(end_bench - start_bench) * 1000} ms")
        return sorted_times, values[sorted_time_indices]


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
    time_unit_options = {"ns": 1, "s": 10 ** 9, "ms": 10 ** 6, "us": 10 ** 3}
    if time_units not in time_unit_options.keys():
        raise ValueError("Invalid time units. Expected one of: %s" % time_unit_options)

    # convert time data into nanoseconds and round off any trailing digits and convert to integer array
    time_data = time_data.copy() * time_unit_options[time_units]

    return np.around(time_data).astype("int64")


def convert_to_nanohz(freq, freq_units):
    freq_unit_options = {"nHz": 1, "uHz": 10 ** 3, "mHz": 10 ** 6, "Hz": 10 ** 9, "kHz": 10 ** 12, "MHz": 10 ** 15}
    if freq_units not in freq_unit_options.keys():
        raise ValueError("Invalid frequency units. Expected one of: %s" % freq_unit_options)

    freq *= freq_unit_options[freq_units]

    return round(freq)


def convert_from_nanohz(freq_nhz, freq_units):
    freq_unit_options = {"nHz": 1, "uHz": 10 ** 3, "mHz": 10 ** 6, "Hz": 10 ** 9, "kHz": 10 ** 12, "MHz": 10 ** 15}
    if freq_units not in freq_unit_options.keys():
        raise ValueError("Invalid frequency units. Expected one of: %s" % freq_unit_options)

    freq = freq_nhz / freq_unit_options[freq_units]

    if freq == np.floor(freq):
        freq = int(freq)

    return freq
