import os
from ctypes import *

import numpy as np
import time
import warnings

from atriumdb.block_wrapper import *
from atriumdb.helpers.block_calculations import freq_nhz_to_period_ns, calc_gap_block_start, calc_time_by_freq
from atriumdb.helpers.type_helpers import *
from atriumdb.helpers.block_constants import *
import logging
#
# logging.basicConfig(
#     level=logging.debug,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.StreamHandler()
#     ]
# )


class Block:

    def __init__(self, path_to_dll, num_threads: int):
        self.wrapped_dll = None

        self.load_dll(path_to_dll, num_threads)

        self.num_threads = num_threads
        self.block_size = 32768
        self.bytes_per_value_min = 1
        self.delta_order_min = 0
        self.delta_order_max = 5

        self.t_compression = 1
        self.v_compression = 3
        self.t_compression_level = 0
        self.v_compression_level = 19

    def load_dll(self, path_to_dll, num_threads):
        self.wrapped_dll = WrappedBlockDll(os.path.abspath(path_to_dll), num_threads)

    def encode_blocks(self, times, values, freq_nhz: int, start_ns: int,
                      raw_time_type=None, raw_value_type=None, encoded_time_type=None,
                      encoded_value_type=None, scale_m: float = None, scale_b: float = None):
        if raw_time_type is None:
            raw_time_type = t_type_raw_choose(len(times), len(values), freq_nhz)

        if raw_value_type is None:
            raw_value_type = v_type_raw_choose(values[0])

        if encoded_time_type is None:
            encoded_time_type = t_type_encoded_choose(raw_time_type, freq_nhz)

        if encoded_value_type is None:
            encoded_value_type = v_type_encoded_choose(raw_value_type)

        times = np.array(times, dtype=np.int64)
        values = np.array(values, dtype=np.int64) if raw_value_type == VALUE_TYPES['INT64'] else \
            np.array(values, dtype=np.float64)

        num_blocks = (values.size + self.block_size - 1) // self.block_size

        time_info_data = None
        if raw_time_type == TIME_TYPES['START_TIME_NUM_SAMPLES']:
            # times = self.blockify_intervals(times, num_blocks)
            times, time_info_data = self.blockify_intervals(freq_nhz, num_blocks, times, values.size)

        times, headers, options, t_block_start, v_block_start = \
            self._gen_metadata(times, values, freq_nhz, start_ns, num_blocks,
                               raw_time_type, raw_value_type, encoded_time_type, encoded_value_type,
                               scale_m=scale_m, scale_b=scale_b, time_data=time_info_data)

        encoded_bytes, byte_start_array = self.wrapped_dll.encode_blocks_sdk(
            times, values, num_blocks, t_block_start, v_block_start, cast(headers, POINTER(BlockMetadata)), options)

        return encoded_bytes, headers, byte_start_array

    def blockify_intervals(self, freq_nhz, num_blocks, times, value_size):
        period_ns = freq_nhz_to_period_ns(freq_nhz)
        true_intervals = times.reshape((-1, 2)).tolist()
        assert sum([t[1] for t in true_intervals]) == value_size

        blocked_intervals = []
        interval_block_start = []
        cur_blocked_interval = 0

        num_block_intervals = []
        elapsed_block_time = []

        for block_i in range(num_blocks):
            actual_block_size = self.block_size if block_i < num_blocks - 1 else \
                value_size - ((num_blocks - 1) * self.block_size)

            interval_block_start.append(cur_blocked_interval)

            interval_count = 0
            elapsed_time = 0
            while actual_block_size > 0:
                if actual_block_size >= true_intervals[0][1]:
                    actual_block_size -= true_intervals[0][1]
                    blocked_intervals.append(true_intervals.pop(0))

                    if len(true_intervals) > 0:
                        elapsed_time += true_intervals[0][0] - blocked_intervals[-1][0]
                    else:
                        elapsed_time += blocked_intervals[-1][1] * period_ns

                else:
                    blocked_intervals.append([true_intervals[0][0], actual_block_size])
                    true_intervals[0][0] += int(period_ns * actual_block_size)
                    true_intervals[0][1] -= actual_block_size

                    elapsed_time += actual_block_size * period_ns
                    actual_block_size = 0

                interval_count += 1

            cur_blocked_interval += interval_count
            num_block_intervals.append(interval_count)
            elapsed_block_time.append(elapsed_time)

        times = np.array(blocked_intervals, dtype=np.int64).flatten()
        interval_block_start = np.array(interval_block_start, dtype=np.int64) * 16

        return times, (num_block_intervals, elapsed_block_time, interval_block_start)

    def decode_blocks(self, encoded_bytes, byte_start_array, analog=True, times_before=None, values_before=None):
        # times_before, values_before = np.array([99], dtype=np.int64), np.array([42], dtype=np.int64)
        if values_before is None:
            new_values_index = 0
        else:
            new_values_index = values_before.size * values_before.itemsize

        if times_before is None:
            new_times_index = 0
        else:
            new_times_index = times_before.size * times_before.itemsize

        start_bench = time.perf_counter()
        headers = self.decode_headers(encoded_bytes, byte_start_array)
        end_bench = time.perf_counter()
        logging.debug(f"decode headers {(end_bench - start_bench) * 1000} ms")

        start_bench = time.perf_counter()
        t_block_start = np.cumsum([h.t_raw_size for h in headers], dtype=np.uint64)
        t_block_start = np.concatenate([np.array([0], dtype=np.uint64), t_block_start[:-1]], axis=None)

        v_block_start = np.cumsum([h.v_raw_size for h in headers], dtype=np.uint64)
        v_block_start = np.concatenate([np.array([0], dtype=np.uint64), v_block_start[:-1]], axis=None)

        t_byte_start = np.cumsum([h.t_encoded_size if h.t_compression != 1 else 0 for h in headers], dtype=np.uint64)
        t_byte_start = np.concatenate([np.array([0], dtype=np.uint64), t_byte_start[:-1]], axis=None)
        end_bench = time.perf_counter()
        logging.debug(f"arrange intra-block information {(end_bench - start_bench) * 1000} ms")

        start_bench = time.perf_counter()
        if not all([h.t_compression == 1 for h in headers]):
            encoded_bytes = np.concatenate(
                [encoded_bytes,
                 np.zeros(np.sum([h.t_encoded_size if h.t_compression != 1 else 0 for h in headers]), dtype=np.uint8)],
                axis=None)
        end_bench = time.perf_counter()
        logging.debug(f"allocate extra memory {(end_bench - start_bench) * 1000} ms")

        start_bench = time.perf_counter()
        time_data = np.zeros(sum(h.t_raw_size for h in headers) + new_times_index, dtype=np.uint8)
        value_data = np.zeros(sum(h.v_raw_size for h in headers) + new_values_index, dtype=np.uint8)
        end_bench = time.perf_counter()
        logging.debug(f"allocate data memory {(end_bench - start_bench) * 1000} ms")

        start_bench = time.perf_counter()
        self.wrapped_dll.decode_blocks_sdk(
            time_data[new_times_index:], value_data[new_values_index:], encoded_bytes, len(byte_start_array),
            t_block_start, v_block_start, byte_start_array, t_byte_start)
        end_bench = time.perf_counter()
        logging.debug(f"C Decode {(end_bench - start_bench) * 1000} ms")

        start_bench = time.perf_counter()
        time_data = np.frombuffer(time_data, dtype=np.int64)
        period_ns = freq_nhz_to_period_ns(headers[0].freq_nhz)

        if headers[0].t_raw_type == T_TYPE_START_TIME_NUM_SAMPLES:
            time_data = merge_interval_data(time_data, period_ns)
        if headers[0].v_raw_type == V_TYPE_INT64:
            value_data = np.frombuffer(value_data, dtype=np.int64)
        elif headers[0].v_raw_type == V_TYPE_DOUBLE:
            value_data = np.frombuffer(value_data, dtype=np.float64)
        else:
            raise ValueError("Header had an unsupported raw value type, {}.".format(headers[0].v_raw_type))

        # if all([h.t_raw_type == T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO for h in headers]):
        #     # Carry on
        #     pass
        #
        # else:
        #     logging.debug("data wasn't all gap data, that shouldn't be")
        #     logging.debug("block time times:")
        #     logging.debug([h.t_raw_type for h in headers])

        # Apply the scale factors
        end_bench = time.perf_counter()
        logging.debug(f"interpret result bytes {(end_bench - start_bench) * 1000} ms")
        #
        logging.debug("\n")
        logging.debug("Applying Scale Factors")
        logging.debug("------------------------")
        start_bench_scale = time.perf_counter()
        scale_m_array = np.array([h.scale_m for h in headers])
        if analog and not np.all(scale_m_array == 0):
            start_bench = time.perf_counter()
            scale_b_array = np.array([h.scale_b for h in headers])
            end_bench = time.perf_counter()
            logging.debug(f"\tscale: arrange linear constants {(end_bench - start_bench) * 1000} ms")

            start_bench = time.perf_counter()
            value_data = value_data.astype(np.float64, copy=False)
            end_bench = time.perf_counter()
            logging.debug(f"\tscale: cast value data {(end_bench - start_bench) * 1000} ms")

            if np.all(scale_m_array == scale_m_array[0]) and np.all(scale_b_array == scale_b_array[0]):
                start_bench = time.perf_counter()
                value_data[new_values_index:] *= scale_m_array[0]
                end_bench = time.perf_counter()
                logging.debug(f"\tscale: apply slope {(end_bench - start_bench) * 1000} ms")

                start_bench = time.perf_counter()
                value_data[new_values_index:] += scale_b_array[0]
                end_bench = time.perf_counter()
                logging.debug(f"\tscale: apply y-int {(end_bench - start_bench) * 1000} ms")

            else:
                v_data_regions = np.cumsum([h.num_vals for h in headers])
                v_data_regions = np.concatenate([np.array([0], dtype=np.uint64), v_data_regions], axis=None)

                for i in range(scale_m_array.size):
                    if scale_m_array[i] != 0:
                        value_data[new_values_index:][int(v_data_regions[i]):int(v_data_regions[i + 1])] *= \
                            scale_m_array[i]

                    if scale_b_array[i] != 0:
                        value_data[new_values_index:][int(v_data_regions[i]):int(v_data_regions[i + 1])] += \
                            scale_b_array[i]

        # else:
        #     logging.debug("Didn't apply scale factors")
        end_bench_scale = time.perf_counter()
        logging.debug(f"apply scale factors total {(end_bench_scale - start_bench_scale) * 1000} ms")
        logging.debug("\n")

        # if times_before is not None:
        #     time_data[:times_before.size] = times_before
        #
        # if values_before is not None:
        #     value_data[:values_before.size] = values_before

        return time_data, value_data, headers

    def decode_headers(self, encoded_bytes, byte_start_array):
        return [BlockMetadata.from_buffer(encoded_bytes, start_byte) for start_byte in byte_start_array]

    def _gen_metadata(self, times, values, freq_nhz: int, start_ns: int, num_blocks: int,
                      raw_time_type: int, raw_value_type: int, encoded_time_type: int, encoded_value_type: int,
                      scale_m: float = None, scale_b: float = None, time_data=None):

        times = np.copy(times)
        num_block_intervals, elapsed_block_time, interval_block_start = \
            time_data if time_data is not None else ([], [], [])

        headers = (BlockMetadata * num_blocks)()
        options = BlockOptions()

        options.bytes_per_value_min = 0
        options.delta_order_min = 0
        options.delta_order_max = 5

        t_block_start = np.zeros(num_blocks, dtype=np.uint64)
        v_block_start = np.zeros(num_blocks, dtype=np.uint64)

        val_offset = 0
        cur_time, cur_gap = int(start_ns), 0
        for i in range(num_blocks):
            headers[i].t_raw_type = raw_time_type
            headers[i].t_encoded_type = encoded_time_type
            headers[i].v_raw_type = raw_value_type
            headers[i].v_encoded_type = encoded_value_type

            headers[i].t_compression = self.t_compression
            headers[i].v_compression = self.v_compression
            headers[i].t_compression_level = self.t_compression_level
            headers[i].v_compression_level = self.v_compression_level

            headers[i].freq_nhz = freq_nhz

            if val_offset + self.block_size <= len(values):
                headers[i].num_vals = self.block_size
            else:
                headers[i].num_vals = len(values) - val_offset

            # Mean, Median, Mode.. All that Jazz
            val_slice = values[val_offset:val_offset + headers[i].num_vals]

            headers[i].max = c_double(val_slice.max())
            headers[i].min = c_double(val_slice.min())
            headers[i].mean = c_double(val_slice.mean())
            headers[i].std = c_double(val_slice.std())

            headers[i].scale_m = 0 if scale_m is None else c_double(scale_m)
            headers[i].scale_b = 0 if scale_b is None else c_double(scale_b)

            if raw_time_type == TIME_TYPES['TIME_ARRAY_INT64_NS']:
                t_block_start[i] = val_offset * 8
                headers[i].start_n = times[val_offset]
                headers[i].end_n = times[val_offset + headers[i].num_vals - 1]

            elif raw_time_type == TIME_TYPES['START_TIME_NUM_SAMPLES']:
                headers[i].start_n = int(cur_time)

                headers[i].num_gaps = num_block_intervals[i]
                elapsed_time = elapsed_block_time[i]
                t_block_start[i] = interval_block_start[i]

                cur_time += elapsed_time

                headers[i].end_n = int(cur_time) - int(freq_nhz_to_period_ns(freq_nhz))

            else:
                headers[i].start_n, t_block_start[i] = int(cur_time), cur_gap * 16

                if raw_time_type == TIME_TYPES['GAP_ARRAY_INT64_INDEX_DURATION_NS']:
                    mode = "duration"
                elif raw_time_type == TIME_TYPES['GAP_ARRAY_INT64_INDEX_NUM_SAMPLES']:
                    mode = "samples"
                else:
                    raise ValueError("time type {} not supported.".format(raw_time_type))

                headers[i].num_gaps, elapsed_time, period_ns = \
                    calc_gap_block_start(times, headers[i].num_vals, freq_nhz, val_offset, cur_gap, mode)

                cur_time += elapsed_time
                cur_gap += headers[i].num_gaps

                headers[i].end_n = int(cur_time) - period_ns

            v_block_start[i] = val_offset * 8

            val_offset += self.block_size

        return times, headers, options, t_block_start, v_block_start


def create_gap_arr(message_time_arr, samples_per_message, freq_nhz):
    assert ((10 ** 18) * samples_per_message) % freq_nhz == 0
    message_period_ns = ((10 ** 18) * samples_per_message) // freq_nhz
    result_list = []

    for sample_i, delta_t in np.ndenumerate(message_time_arr[1:] - message_time_arr[:-1]):
        if delta_t != message_period_ns:
            result_list.extend([(int(sample_i[0]) + 1) * samples_per_message, delta_t - message_period_ns])

    return np.array(result_list, dtype=np.int64)


def create_gap_arr_fast(message_time_arr, samples_per_message, freq_nhz):
    assert ((10 ** 18) * samples_per_message) % freq_nhz == 0
    message_period_ns = ((10 ** 18) * samples_per_message) // freq_nhz

    time_gaps = np.diff(message_time_arr) - message_period_ns
    non_zero_inds = np.nonzero(time_gaps)[0]
    result_arr = np.zeros((non_zero_inds.size, 2))
    result_arr.T[0] = (non_zero_inds + 1) * samples_per_message
    result_arr.T[1] = time_gaps[non_zero_inds]
    return result_arr.flatten()


def interpret_gap_arr(gap_array, start_time_ns, num_messages, samples_per_message, freq_nhz):
    assert ((10 ** 18) * samples_per_message) % freq_nhz == 0
    message_period_ns = ((10 ** 18) * samples_per_message) // freq_nhz

    time_arr = np.arange(start_time_ns, start_time_ns + (num_messages * message_period_ns), message_period_ns)

    for gap_i, gap_dur in gap_array.reshape((-1, 2)):
        time_arr[gap_i // samples_per_message:] += gap_dur

    return time_arr


def convert_gap_array_to_intervals(start_time, gap_arr: np.ndarray, num_values, freq_nhz):
    gap_arr = gap_arr.reshape((-1, 2))
    no_warnings_yet = True
    intervals = []

    cur_time = start_time
    values_so_far = 0
    for gap_ind, gap_dur in gap_arr:
        interval_num_values = gap_ind - values_so_far

        # If the interval before the gap doesn't represent an integer number of nanoseconds,
        # the value will be truncated. The user will be warned once per get_data statement.
        if no_warnings_yet and (int(interval_num_values) * (10 ** 18)) % freq_nhz != 0:
            warnings.warn("Interval Precision Loss: Rounded to the Nearest Nanosecond")

        end_time = cur_time + calc_time_by_freq(freq_nhz, interval_num_values)
        intervals.append([cur_time, end_time, interval_num_values])

        values_so_far = gap_ind
        cur_time = end_time + gap_dur

    # Last interval
    interval_num_values = num_values - values_so_far
    end_time = cur_time + calc_time_by_freq(freq_nhz, interval_num_values)
    intervals.append([cur_time, end_time, interval_num_values])

    return np.array(intervals, dtype=np.int64)


def convert_intervals_to_gap_array(intervals: np.ndarray):
    intervals = intervals.reshape((-1, 3))

    if len(intervals) < 2:
        return np.array([], dtype=np.int64)

    gap_array = []

    cur_index = 0
    for (start_1, end_1, num_1), (start_2, end_2, num_2) in zip(intervals[:-1], intervals[1:]):
        cur_index += num_1
        if end_1 != start_2:
            gap_array.append([cur_index, start_2 - end_1])

    return np.array(gap_array, dtype=np.int64)


def get_compression_types():
    return COMPRESSION_TYPES


def get_time_types():
    return TIME_TYPES


def get_value_types():
    return VALUE_TYPES


def get_compression_levels(compression_type=None):
    return get_compression_levels_help(compression_type)


def merge_interval_data(blocked_interval_data, period_ns):
    blocked_interval_data = blocked_interval_data.flatten().reshape((-1, 2))
    merged_intervals = []

    for start_t, num_samples in blocked_interval_data:

        if len(merged_intervals) > 0 and \
                merged_intervals[-1][0] + ((merged_intervals[-1][1]) * period_ns) == start_t:

            merged_intervals[-1][1] += num_samples
        else:
            merged_intervals.append([start_t, num_samples])

    return np.array(merged_intervals, dtype=np.int64).flatten()
