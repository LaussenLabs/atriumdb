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
import copy
import time
import warnings

import numpy as np

from atriumdb.block_wrapper import *
from atriumdb.helpers.block_calculations import freq_nhz_to_period_ns, calc_gap_block_start, calc_time_by_freq
from atriumdb.helpers.type_helpers import *
from atriumdb.helpers.block_constants import *
import logging


class Block:

    def __init__(self, path_to_dll, num_threads: int):
        self.wrapped_dll = None

        self.load_dll(path_to_dll, num_threads)

        self.num_threads = num_threads
        self.block_size = 131072  # 2^17
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

        scale_m = 1.0 if scale_m is None else scale_m
        scale_b = 0.0 if scale_b is None else scale_b

        # Raise ValueError if any of the types are None
        if None in (raw_time_type, raw_value_type, encoded_time_type, encoded_value_type):
            raise ValueError(
                "All type parameters (raw_time_type, raw_value_type, encoded_time_type, encoded_value_type) "
                "must be specified.")

        # Convert times and values to numpy arrays with appropriate data types
        times = np.array(times, dtype=np.int64)
        values = np.array(values, dtype=np.int64) if raw_value_type == VALUE_TYPES['INT64'] else \
            np.array(values, dtype=np.float64)

        # Calculate the number of blocks needed to store the data
        num_blocks = max(1, values.size // self.block_size)

        # Initialize time_info_data variable
        time_info_data = None
        # Check if raw time type is START_TIME_NUM_SAMPLES and process accordingly
        if raw_time_type == TIME_TYPES['START_TIME_NUM_SAMPLES']:
            times, time_info_data = self.blockify_intervals(freq_nhz, num_blocks, times, values.size)

        # Generate metadata for the blocks
        times, headers, options, t_block_start, v_block_start = \
            self._gen_metadata(times, values, freq_nhz, start_ns, num_blocks,
                               raw_time_type, raw_value_type, encoded_time_type, encoded_value_type,
                               scale_m=scale_m, scale_b=scale_b, time_data=time_info_data)

        # Encode the blocks using the wrapped_dll's encode_blocks_sdk function
        encoded_bytes, byte_start_array = self.wrapped_dll.encode_blocks_sdk(
            times, values, num_blocks, t_block_start, v_block_start, cast(headers, POINTER(BlockMetadata)), options)

        # Return the encoded bytes, headers, and byte start array
        return encoded_bytes, headers, byte_start_array

    def prepare_encode_blocks_inputs(self, times, values, freq_nhz, start_ns,
                                     raw_time_type, raw_value_type, encoded_time_type, encoded_value_type,
                                     scale_m=None, scale_b=None):
        scale_m = 1.0 if scale_m is None else scale_m
        scale_b = 0.0 if scale_b is None else scale_b

        # Raise ValueError if any of the types are None
        if None in (raw_time_type, raw_value_type, encoded_time_type, encoded_value_type):
            raise ValueError(
                "All type parameters (raw_time_type, raw_value_type, encoded_time_type, encoded_value_type) "
                "must be specified.")

        # Convert times and values to numpy arrays with appropriate data types
        times = np.array(times, dtype=np.int64)
        values = np.array(values, dtype=np.int64) if raw_value_type == VALUE_TYPES['INT64'] else \
            np.array(values, dtype=np.float64)

        # Calculate the number of blocks needed to store the data
        num_blocks = max(1, values.size // self.block_size)

        # Initialize time_info_data variable
        time_info_data = None
        # Check if raw time type is START_TIME_NUM_SAMPLES and process accordingly
        if raw_time_type == TIME_TYPES['START_TIME_NUM_SAMPLES']:
            times, time_info_data = self.blockify_intervals(freq_nhz, num_blocks, times, values.size)

        # Generate metadata for the blocks
        times, headers, options, t_block_start, v_block_start = \
            self._gen_metadata(times, values, freq_nhz, start_ns, num_blocks,
                               raw_time_type, raw_value_type, encoded_time_type, encoded_value_type,
                               scale_m=scale_m, scale_b=scale_b, time_data=time_info_data)

        # Return the prepared inputs
        return times, values, num_blocks, t_block_start, v_block_start, headers, options, time_info_data

    def encode_blocks_from_multiple_segments(self, segments):
        # segments is a list of dictionaries, each containing the parameters for encode_blocks

        if len(segments) == 0:
            return np.array([], dtype=np.uint8), [], np.array([], dtype=np.uint64)

        times_list = []
        values_list = []
        t_block_start_list = []
        v_block_start_list = []
        headers_list = []
        num_blocks_total = 0

        cumulative_time_length = 0
        cumulative_value_length = 0

        options_s = None
        for segment in segments:
            # Extract parameters from segment
            times = segment['times']
            values = segment['values']
            freq_nhz = segment['freq_nhz']
            start_ns = segment['start_ns']
            raw_time_type = segment['raw_time_type']
            raw_value_type = segment['raw_value_type']
            encoded_time_type = segment['encoded_time_type']
            encoded_value_type = segment['encoded_value_type']
            scale_m = segment.get('scale_m', None)
            scale_b = segment.get('scale_b', None)

            # Call helper method
            times_s, values_s, num_blocks_s, t_block_start_s, v_block_start_s, headers_s, options_s, time_info_data_s = \
                self.prepare_encode_blocks_inputs(times, values, freq_nhz, start_ns,
                                                  raw_time_type, raw_value_type,
                                                  encoded_time_type, encoded_value_type,
                                                  scale_m=scale_m, scale_b=scale_b)

            # Adjust t_block_start and v_block_start
            if raw_time_type in [TIME_TYPES['TIME_ARRAY_INT64_NS'], TIME_TYPES['GAP_ARRAY_INT64_INDEX_DURATION_NS']]:
                t_block_start_s += cumulative_time_length * times_s.itemsize
            elif raw_time_type == TIME_TYPES['START_TIME_NUM_SAMPLES']:
                # Adjust interval_block_start_s
                interval_block_start_s = time_info_data_s[2]
                interval_block_start_s += cumulative_time_length * times_s.itemsize
                # Update t_block_start_s
                t_block_start_s = interval_block_start_s
            else:
                raise ValueError(
                    "raw_time_type {} not supported in encode_blocks_from_multiple_segments.".format(raw_time_type))

            v_block_start_s += cumulative_value_length * values_s.itemsize

            # Append to lists
            times_list.append(times_s)
            values_list.append(values_s)
            t_block_start_list.append(t_block_start_s)
            v_block_start_list.append(v_block_start_s)
            headers_list.extend(headers_s)

            # Update cumulative lengths
            cumulative_time_length += len(times_s)
            cumulative_value_length += len(values_s)
            num_blocks_total += num_blocks_s

        # Concatenate times and values
        times_total = np.concatenate(times_list)
        values_total = np.concatenate(values_list)
        t_block_start_total = np.concatenate(t_block_start_list)
        v_block_start_total = np.concatenate(v_block_start_list)
        headers_total = (BlockMetadata * num_blocks_total)(*headers_list)

        # Use options from the last segment (assuming options are the same)
        options_total = options_s

        # Now call wrapped_dll.encode_blocks_sdk
        encoded_bytes, byte_start_array = self.wrapped_dll.encode_blocks_sdk(
            times_total, values_total, num_blocks_total, t_block_start_total, v_block_start_total,
            cast(headers_total, POINTER(BlockMetadata)), options_total)

        return encoded_bytes, headers_total, byte_start_array

    def blockify_intervals(self, freq_nhz, num_blocks, times, value_size):
        # Calculate the period in nanoseconds based on the input frequency
        period_ns = freq_nhz_to_period_ns(freq_nhz)

        # Reshape the input times array into a list of pairs (intervals)
        true_intervals = times.reshape((-1, 2)).tolist()

        # Check if the sum of interval lengths is equal to the input value_size
        assert sum([t[1] for t in true_intervals]) == value_size

        # Initialize variables and lists to store the blocked intervals and additional information
        blocked_intervals = []
        interval_block_start = []
        cur_blocked_interval = 0
        num_block_intervals = []
        elapsed_block_time = []

        # Iterate through the specified number of blocks
        for block_i in range(num_blocks):
            # Calculate the actual size of each block
            actual_block_size = self.block_size if block_i < num_blocks - 1 else \
                value_size - ((num_blocks - 1) * self.block_size)

            # Append the starting index of the current block
            interval_block_start.append(cur_blocked_interval)

            # Initialize variables for the current block
            interval_count = 0
            elapsed_time = 0

            # Iterate through the input intervals and add them to the current block
            while actual_block_size > 0:
                # If the remaining block size is greater than or equal to the first interval's length
                if actual_block_size >= true_intervals[0][1]:
                    # Subtract the interval length from the remaining block size
                    actual_block_size -= true_intervals[0][1]
                    # Add the interval to the blocked intervals and remove it from the true intervals list
                    blocked_intervals.append(true_intervals.pop(0))

                    # Update the elapsed time based on the remaining intervals
                    if len(true_intervals) > 0:
                        elapsed_time += true_intervals[0][0] - blocked_intervals[-1][0]
                    else:
                        elapsed_time += blocked_intervals[-1][1] * period_ns

                # If the remaining block size is less than the first interval's length
                else:
                    # Add a partial interval with the remaining block size
                    blocked_intervals.append([true_intervals[0][0], actual_block_size])
                    # Update the true interval's start time and length based on the added partial interval
                    true_intervals[0][0] += int(period_ns * actual_block_size)
                    true_intervals[0][1] -= actual_block_size

                    # Update the elapsed time and set the remaining block size to 0
                    elapsed_time += actual_block_size * period_ns
                    actual_block_size = 0

                # Increment the interval count for the current block
                interval_count += 1

            # Update the variables for the next block
            cur_blocked_interval += interval_count
            num_block_intervals.append(interval_count)
            elapsed_block_time.append(elapsed_time)

        # Convert the lists to numpy arrays and return the results
        times = np.array(blocked_intervals, dtype=np.int64).flatten()
        interval_block_start = np.array(interval_block_start, dtype=np.int64) * 16

        return times, (num_block_intervals, elapsed_block_time, interval_block_start)

    def decode_block_from_bytes_alone(self, encoded_bytes, analog=True, time_type=1):
        headers, byte_start_array = self.decode_headers_and_byte_start_array(encoded_bytes)

        # trick C dll into decoding the data directly into the time type you want by editing the t_raw_type field in
        # each of the block headers in the encoded_bytes_stream so the python sdk doesn't have to do it
        if time_type == 1:
            for start_byte in byte_start_array:
                # using from_buffer() on the header will allow us to directly modify the encoded_bytes variable through
                # the headers ctypes fields. This is because both the header struct and encoded_bytes variable point at
                # the same bytes in memory so modifying one will modify the other
                header = BlockMetadata.from_buffer(encoded_bytes, start_byte)
                if header.t_raw_type != 1:
                    header.t_raw_type = time_type
                    header.t_raw_size = 8 * header.num_vals
        elif time_type == 2:
            for start_byte in byte_start_array:
                header = BlockMetadata.from_buffer(encoded_bytes, start_byte)
                if header.t_raw_type != 2:
                    header.t_raw_type = time_type
                    header.t_raw_size = 16 * header.num_gaps
        else:
            raise ValueError("Time type must be in [1, 2]")

        # Calculate the start of the time and value blocks for the decoded bytes
        start_bench = time.perf_counter()
        t_block_start = np.cumsum([h.t_raw_size for h in headers], dtype=np.uint64)
        t_block_start = np.concatenate([np.array([0], dtype=np.uint64), t_block_start[:-1]], axis=None)

        v_block_start = np.cumsum([h.v_raw_size for h in headers], dtype=np.uint64)
        v_block_start = np.concatenate([np.array([0], dtype=np.uint64), v_block_start[:-1]], axis=None)

        # Calculate the start of the time bytes within the encoded bytes
        t_byte_start = np.cumsum([h.t_encoded_size if h.t_compression != 1 else 0 for h in headers],
                                 dtype=np.uint64)
        t_byte_start = np.concatenate([np.array([0], dtype=np.uint64), t_byte_start[:-1]], axis=None)
        end_bench = time.perf_counter()
        logging.debug(f"arrange intra-block information {(end_bench - start_bench) * 1000} ms")

        # Allocate extra memory for the encoded bytes if necessary
        start_bench = time.perf_counter()
        if not all([h.t_compression == 1 for h in headers]):
            encoded_bytes = np.concatenate(
                [encoded_bytes,
                 np.zeros(np.sum([h.t_encoded_size if h.t_compression != 1 else 0 for h in headers]),
                          dtype=np.uint8)],
                axis=None)
        end_bench = time.perf_counter()
        logging.debug(f"allocate extra memory {(end_bench - start_bench) * 1000} ms")

        # Allocate memory for the decoded time and value data
        start_bench = time.perf_counter()
        time_data = np.zeros(sum(h.t_raw_size for h in headers), dtype=np.uint8)
        value_data = np.zeros(sum(h.v_raw_size for h in headers), dtype=np.uint8)
        end_bench = time.perf_counter()
        logging.debug(f"allocate data memory {(end_bench - start_bench) * 1000} ms")

        # Call the wrapped C library function to decode the blocks
        start_bench = time.perf_counter()
        self.wrapped_dll.decode_blocks_sdk(
            time_data, value_data, encoded_bytes, len(byte_start_array),
            t_block_start, v_block_start, byte_start_array, t_byte_start)
        end_bench = time.perf_counter()
        logging.debug(f"C Decode {(end_bench - start_bench) * 1000} ms")

        # Convert the decoded time and value data into the appropriate data types
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
        end_bench = time.perf_counter()
        logging.debug(f"interpret result bytes {(end_bench - start_bench) * 1000} ms")

        # Apply the scale factors to the value data if necessary
        logging.debug("\n")
        logging.debug("Applying Scale Factors")
        logging.debug("------------------------")
        start_bench_scale = time.perf_counter()
        scale_m_array = np.array([h.scale_m if h.scale_m != 0 else 1.0 for h in headers])
        scale_b_array = np.array([h.scale_b for h in headers])
        if analog and not (np.all(scale_m_array == 1) and np.all(scale_b_array == 0)):
            start_bench = time.perf_counter()
            scale_b_array = np.array([h.scale_b for h in headers])
            end_bench = time.perf_counter()
            logging.debug(f"\tscale: arrange linear constants {(end_bench - start_bench) * 1000} ms")

            start_bench = time.perf_counter()
            value_data = value_data.astype(np.float64, copy=False)
            end_bench = time.perf_counter()
            logging.debug(f"\tscale: cast value data {(end_bench - start_bench) * 1000} ms")

            # Apply the scale factors to the value data
            if np.all(scale_m_array == scale_m_array[0]) and np.all(scale_b_array == scale_b_array[0]):
                start_bench = time.perf_counter()
                value_data *= scale_m_array[0]
                end_bench = time.perf_counter()
                logging.debug(f"\tscale: apply slope {(end_bench - start_bench) * 1000} ms")

                start_bench = time.perf_counter()
                value_data += scale_b_array[0]
                end_bench = time.perf_counter()
                logging.debug(f"\tscale: apply y-int {(end_bench - start_bench) * 1000} ms")

            else:
                # Apply the scale factors to each region of the value data
                v_data_regions = np.cumsum([h.num_vals for h in headers])
                v_data_regions = np.concatenate([np.array([0], dtype=np.uint64), v_data_regions], axis=None)

                for i in range(scale_m_array.size):
                    if scale_m_array[i] != 0:
                        value_data[int(v_data_regions[i]):int(v_data_regions[i + 1])] *= scale_m_array[i]

                    if scale_b_array[i] != 0:
                        value_data[int(v_data_regions[i]):int(v_data_regions[i + 1])] += scale_b_array[i]

        end_bench_scale = time.perf_counter()
        logging.debug(f"apply scale factors total {(end_bench_scale - start_bench_scale) * 1000} ms")
        logging.debug("\n")

        # Return the decoded time data, value data, and headers
        return time_data, value_data, headers

    def decode_blocks(self, encoded_bytes, num_bytes_list, analog=True, time_type=1, return_nan_gap=False,
                      start_time_n=None, end_time_n=None):
        if time_type not in [1, 2, 'encoded', 'raw']:
            raise ValueError("Time type must be one of [1, 2, 'encoded', 'raw']")

        # Calculate the starting byte positions of each block in the encoded bytes array
        byte_start_array = np.cumsum(num_bytes_list, dtype=np.uint64)
        byte_start_array = np.concatenate([np.array([0], dtype=np.uint64), byte_start_array[:-1]], axis=None)

        min_block_start, max_block_end = None, None
        # Adjust the t_raw_type field in each of the block headers based on the requested time_type
        for start_byte in byte_start_array:
            # Using from_buffer() on the header will allow us to directly modify the encoded_bytes variable through
            # the header's ctypes fields. This is because both the header struct and encoded_bytes variable point at
            # the same bytes in memory so modifying one will modify the other
            header = BlockMetadata.from_buffer(encoded_bytes, start_byte)
            if time_type == 'raw':
                # Do nothing, keep header.t_raw_type as is
                pass
            elif time_type == 'encoded':
                # Set t_raw_type to t_encoded_type
                header.t_raw_type = header.t_encoded_type
                if header.t_encoded_type == 1:
                    header.t_raw_size = 8 * header.num_vals
                elif header.t_encoded_type == 2:
                    header.t_raw_size = 16 * header.num_gaps
                else:
                    raise ValueError(f"Requested header.t_raw_type is {header.t_raw_type} but must be 1 or 2")
            elif time_type == 1:
                header.t_raw_type = 1
                header.t_raw_size = 8 * header.num_vals
            elif time_type == 2:
                header.t_raw_type = 2
                header.t_raw_size = 16 * header.num_gaps

            if header.t_raw_type not in [1, 2]:
                raise ValueError(f"Requested header.t_raw_type is {header.t_raw_type} but must be 1 or 2")

            min_block_start = header.start_n if min_block_start is None else min(min_block_start, header.start_n)
            max_block_end = header.end_n if max_block_end is None else max(max_block_end, header.end_n)

        # Decode the headers from the encoded bytes
        headers = self.decode_headers(encoded_bytes, byte_start_array)

        # Calculate the start of the time and value blocks for the decoded bytes
        t_block_start = np.cumsum([h.t_raw_size for h in headers], dtype=np.uint64)
        t_block_start = np.concatenate([np.array([0], dtype=np.uint64), t_block_start[:-1]], axis=None)

        v_block_start = np.cumsum([h.v_raw_size for h in headers], dtype=np.uint64)
        v_block_start = np.concatenate([np.array([0], dtype=np.uint64), v_block_start[:-1]], axis=None)

        # Calculate the start of the time bytes within the encoded bytes
        t_byte_start = np.cumsum([h.t_encoded_size if h.t_compression != 1 else 0 for h in headers], dtype=np.uint64)
        t_byte_start = np.concatenate([np.array([0], dtype=np.uint64), t_byte_start[:-1]], axis=None)
        t_byte_start += encoded_bytes.size

        # Allocate extra memory for the encoded bytes if necessary
        if not all([h.t_compression == 1 for h in headers]):
            encoded_bytes = np.concatenate(
                [encoded_bytes,
                 np.zeros(np.sum([h.t_encoded_size if h.t_compression != 1 else 0 for h in headers]), dtype=np.uint8)],
                axis=None)

        # Allocate memory for the decoded time and value data
        time_data = np.zeros(sum(h.t_raw_size for h in headers), dtype=np.uint8)
        value_data = np.zeros(sum(h.v_raw_size for h in headers), dtype=np.uint8)

        # Call the wrapped C library function to decode the blocks
        self.wrapped_dll.decode_blocks_sdk(
            time_data, value_data, encoded_bytes, len(byte_start_array),
            t_block_start, v_block_start, byte_start_array, t_byte_start)

        # Convert the decoded time and value data into the appropriate data types
        time_data = np.frombuffer(time_data, dtype=np.int64)
        period_ns = freq_nhz_to_period_ns(headers[0].freq_nhz)

        if headers[0].t_raw_type == T_TYPE_START_TIME_NUM_SAMPLES:
            time_data = merge_interval_data(time_data, period_ns)

        # Change scale_m from 0 to 1
        for i in range(len(headers)):
            if headers[i].scale_m == 0:
                headers[i].scale_m = 1.0

        # Apply the scale factors to the value data if necessary
        scale_m_array = np.array([h.scale_m for h in headers])
        scale_b_array = np.array([h.scale_b for h in headers])

        v_raw_types = set(h.v_raw_type for h in headers)
        if len(v_raw_types) == 1:
            # All headers have the same v_raw_type
            v_raw_type = v_raw_types.pop()
            if v_raw_type == V_TYPE_INT64:
                value_data = np.frombuffer(value_data, dtype=np.int64)
            elif v_raw_type == V_TYPE_DOUBLE:
                value_data = np.frombuffer(value_data, dtype=np.float64)
            else:
                raise ValueError("Header had an unsupported raw value type, {}.".format(v_raw_type))
        else:
            # v_raw_type varies among headers
            values_list = []
            idx = 0
            for h in headers:
                size_in_bytes = h.v_raw_size
                data_bytes = value_data[idx: idx + size_in_bytes]
                if h.v_raw_type == V_TYPE_INT64:
                    data_array = np.frombuffer(data_bytes, dtype=np.int64)
                elif h.v_raw_type == V_TYPE_DOUBLE:
                    data_array = np.frombuffer(data_bytes, dtype=np.float64)
                else:
                    raise ValueError("Header had an unsupported raw value type, {}.".format(h.v_raw_type))
                values_list.append(data_array)
                idx += size_in_bytes
            value_data = np.concatenate(values_list)

        if isinstance(return_nan_gap, np.ndarray) or return_nan_gap:
            if time_type != 1:
                raise ValueError("Returning nan gaps is only supported for time type 1.")
            if period_ns is None:
                raise ValueError("Returning nan gaps requires a period ns to be provided.")
            period_ns = 10**18 / headers[0].freq_nhz
            start_ns = start_time_n if start_time_n is not None else time_data[0]
            end_ns = end_time_n if end_time_n is not None else round(time_data[-1] + period_ns)
            expected_num_values = int(round((end_ns - start_ns) / period_ns))
            if isinstance(return_nan_gap, np.ndarray):
                nan_analog_array = return_nan_gap
                # The code won't break if this requirement isn't met, its just that this is the expected number of
                # values needed to represent all the data. But if this is causing trouble due to rounding / off-by-one
                # errors, you can change or remove this requirement.
                if nan_analog_array.size != expected_num_values:
                    raise ValueError("input array must be of size int(round((end_time_n - start_time_n) / period_ns))")
            else:
                nan_analog_array = np.full(expected_num_values, np.nan, dtype=np.float64)
            self.wrapped_dll.fill_nan_array_with_analog(value_data, nan_analog_array,
                                                        headers, len(headers), time_data, start_ns, float(period_ns))

            return headers, nan_analog_array

        no_scale_bool = np.all(scale_m_array == 1) and np.all(scale_b_array == 0)
        if analog and not no_scale_bool:
            analog_values = np.zeros(sum(h.num_vals for h in headers), dtype=np.float64)
            self.wrapped_dll.convert_value_data_to_analog(value_data, analog_values, headers, len(headers))
            value_data = analog_values

        # Return the decoded time data, value data, and headers
        return time_data, value_data, headers

    def decode_headers(self, encoded_bytes, byte_start_array):
        return [BlockMetadata.from_buffer(encoded_bytes, start_byte) for start_byte in byte_start_array]

    def decode_headers_and_byte_start_array(self, encoded_bytes):
        decoded_headers = []
        byte_start_array = []
        start_byte = 0

        while start_byte < len(encoded_bytes):
            # Decode the BlockMetadata from the current start byte
            block_metadata = BlockMetadata.from_buffer(encoded_bytes, start_byte)

            # Append the decoded metadata to our list
            decoded_headers.append(block_metadata)
            byte_start_array.append(start_byte)

            # Calculate the total size of the current block
            block_size = block_metadata.meta_num_bytes + block_metadata.t_num_bytes + block_metadata.v_num_bytes

            # Update the start_byte for the next block
            start_byte += block_size

        byte_start_array = np.array(byte_start_array, dtype=np.uint64)

        return decoded_headers, byte_start_array

    def _gen_metadata(self, times, values, freq_nhz: int, start_ns: int, num_blocks: int,
                      raw_time_type: int, raw_value_type: int, encoded_time_type: int, encoded_value_type: int,
                      scale_m: float = None, scale_b: float = None, time_data=None):

        # Make a copy of the input times array
        times = np.copy(times)

        # Unpack time_data if it's not None, otherwise use default values
        num_block_intervals, elapsed_block_time, interval_block_start = time_data if time_data is not None else ([], [], [])

        # Initialize an array of BlockMetadata objects
        headers = (BlockMetadata * num_blocks)()

        # Initialize a BlockOptions object and set its properties
        options = BlockOptions()
        options.bytes_per_value_min = 0
        options.delta_order_min = 0
        options.delta_order_max = 5

        # Initialize arrays to store the starting positions of time and value data in each block
        t_block_start = np.zeros(num_blocks, dtype=np.uint64)
        v_block_start = np.zeros(num_blocks, dtype=np.uint64)

        # Initialize variables to keep track of the current position in the time and value data
        val_offset = 0
        cur_time, cur_gap = int(start_ns), 0

        # Loop through each block
        for i in range(num_blocks):
            # Set the raw and encoded time and value types for the current block
            headers[i].t_raw_type = raw_time_type
            headers[i].t_encoded_type = encoded_time_type
            headers[i].v_raw_type = raw_value_type
            headers[i].v_encoded_type = encoded_value_type

            # Set the compression settings for the current block
            headers[i].t_compression = self.t_compression
            headers[i].v_compression = self.v_compression
            headers[i].t_compression_level = self.t_compression_level
            headers[i].v_compression_level = self.v_compression_level

            # Set the frequency for the current block
            headers[i].freq_nhz = freq_nhz

            # Determine the number of values in the current block
            if i < num_blocks - 1:
                headers[i].num_vals = self.block_size
            else:
                headers[i].num_vals = len(values) - val_offset

            # Calculate statistics (min, max, mean, std) for the values in the current block
            val_slice = values[val_offset:val_offset + headers[i].num_vals]
            headers[i].max = c_double(val_slice.max())
            headers[i].min = c_double(val_slice.min())
            headers[i].mean = c_double(val_slice.mean())
            headers[i].std = c_double(val_slice.std())

            # Set the scale factors for the current block
            headers[i].scale_m = 0 if scale_m is None else c_double(scale_m)
            headers[i].scale_b = 0 if scale_b is None else c_double(scale_b)

            # Determine the start and end times for the current block based on the raw time type
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

            # Set the starting position of the value data for the current block
            v_block_start[i] = val_offset * 8

            # Update the value offset for the next block
            val_offset += self.block_size

        # Return the processed times array, headers, options, and starting positions of time and value data
        return times, headers, options, t_block_start, v_block_start


def concat_encoded_arrays(encoded_bytes, encoded_headers, encoded_bytes_1, encoded_headers_1, byte_start_array_1, num_full_blocks):
    # concatenate the encoded byte arrays together
    encoded_bytes = np.concatenate((encoded_bytes_1, encoded_bytes))

    # concatenate the encoded headers
    headers = (BlockMetadata * (num_full_blocks + 1))()
    for i, h in enumerate(encoded_headers_1):
        headers[i] = h
    headers[-1] = encoded_headers[0]
    encoded_headers = headers

    # fix byte start array by appending the start byte of the last oversized block to byte_start_array_1
    # the start byte will be equal to start byte of the block before the oversized block plus the
    # number of header, value and time bytes it has
    byte_start_array = np.concatenate((byte_start_array_1, (byte_start_array_1[-1:] +
                                                            encoded_headers_1[-1].meta_num_bytes +
                                                            encoded_headers_1[-1].t_num_bytes +
                                                            encoded_headers_1[-1].v_num_bytes)))
    return encoded_bytes, encoded_headers, byte_start_array


def create_gap_arr(message_time_arr, samples_per_message, freq_nhz):
    # Check if the product of samples_per_message and 10^18 is divisible by freq_nhz
    assert ((10 ** 18) * samples_per_message) % freq_nhz == 0
    # Calculate the message period in nanoseconds
    message_period_ns = ((10 ** 18) * samples_per_message) // freq_nhz

    # Calculate the time gaps between consecutive message times and subtract the message period
    time_gaps = np.diff(message_time_arr) - message_period_ns
    # Find the indices of the non-zero time gaps
    non_zero_inds = np.nonzero(time_gaps)[0]
    # Initialize an array to store the results
    result_arr = np.zeros((non_zero_inds.size, 2), dtype=np.int64)
    # Fill the first column of the result array with the sample indices
    result_arr.T[0] = (non_zero_inds + 1) * samples_per_message
    # Fill the second column of the result array with the non-zero time gaps
    result_arr.T[1] = time_gaps[non_zero_inds]
    # Flatten the result array and return it
    return result_arr.flatten()


def interpret_gap_arr(gap_array, start_time_ns, num_messages, samples_per_message, freq_nhz):
    # Ensure that the product of samples_per_message and 10^18 is divisible by freq_nhz
    assert ((10 ** 18) * samples_per_message) % freq_nhz == 0
    # Calculate the message period in nanoseconds
    message_period_ns = ((10 ** 18) * samples_per_message) // freq_nhz

    # Create an array of time values for each message, taking into account the message period
    time_arr = np.arange(start_time_ns, start_time_ns + (num_messages * message_period_ns), message_period_ns)

    # Iterate through the gap array and adjust the time values based on the gaps between messages
    for gap_i, gap_dur in gap_array.reshape((-1, 2)):
        time_arr[gap_i // samples_per_message:] += gap_dur

    return time_arr


def convert_gap_array_to_intervals(start_time, gap_arr: np.ndarray, num_values, freq_nhz):
    # Reshape the gap array
    gap_arr = gap_arr.reshape((-1, 2))
    # Initialize a flag for warning about interval precision loss
    no_warnings_yet = True
    # Initialize an empty list for storing intervals
    intervals = []

    # Initialize current time and values counter
    cur_time = start_time
    values_so_far = 0
    # Iterate through the gap array
    for gap_ind, gap_dur in gap_arr:
        # Calculate the number of values in the current interval
        interval_num_values = gap_ind - values_so_far

        # Warn the user if the interval before the gap doesn't represent an integer number of nanoseconds
        if no_warnings_yet and (int(interval_num_values) * (10 ** 18)) % freq_nhz != 0:
            warnings.warn("Interval Precision Loss: Rounded to the Nearest Nanosecond")
            no_warnings_yet = False

        # Calculate the end time of the current interval
        end_time = cur_time + calc_time_by_freq(freq_nhz, interval_num_values)
        # Append the interval to the list
        intervals.append([cur_time, end_time, interval_num_values])

        # Update values counter and current time for the next iteration
        values_so_far = gap_ind
        cur_time = end_time + gap_dur

    # Add the last interval
    interval_num_values = num_values - values_so_far
    end_time = cur_time + calc_time_by_freq(freq_nhz, interval_num_values)
    intervals.append([cur_time, end_time, interval_num_values])

    # Convert the list of intervals to a numpy array
    return np.array(intervals, dtype=np.int64)


def convert_intervals_to_gap_array(intervals: np.ndarray):
    # Reshape the intervals array
    intervals = intervals.reshape((-1, 3))

    # If there are less than two intervals, return an empty gap array
    if len(intervals) < 2:
        return np.array([], dtype=np.int64)

    # Initialize an empty list for storing the gap array
    gap_array = []

    # Initialize a counter for the current index
    cur_index = 0
    # Iterate through pairs of consecutive intervals
    for (start_1, end_1, num_1), (start_2, end_2, num_2) in zip(intervals[:-1], intervals[1:]):
        # Update the current index
        cur_index += num_1
        # If the end time of the first interval is not equal to the start time of the second interval,
        # there is a gap between them, so add it to the gap array
        if end_1 != start_2:
            gap_array.append([cur_index, start_2 - end_1])

    # Convert the list of gaps to a numpy array
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
    # Reshape the input data into a 2D array with two columns (start time and number of samples)
    blocked_interval_data = blocked_interval_data.flatten().reshape((-1, 2))

    # Initialize an empty list to store the merged intervals
    merged_intervals = []

    # Iterate through each interval in the reshaped input data
    for start_t, num_samples in blocked_interval_data:
        # Check if the current interval is adjacent to the previous interval in the merged_intervals list
        if len(merged_intervals) > 0 and \
                merged_intervals[-1][0] + ((merged_intervals[-1][1]) * period_ns) == start_t:
            # Update the number of samples in the previous interval by adding the number of samples in the current interval
            merged_intervals[-1][1] += num_samples
        else:
            # Add the current interval as a new interval in the merged_intervals list
            merged_intervals.append([start_t, num_samples])

    # Convert the merged_intervals list into a flattened NumPy array with the int64 data type and return it
    return np.array(merged_intervals, dtype=np.int64).flatten()
