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
from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both
import numpy as np

DB_NAME_MESSAGE = 'write_message'
DB_NAME_MESSAGES = 'test_write_messages'
DB_NAME_BUFFER = 'test_write_buffer'
DB_NAME_TVP_BUFFER = 'test_write_tvp_buffer'
DB_NAME_COMPREHENSIVE = 'test_comprehensive'
DB_NAME_MULTI_BUFFER = 'test_multi_buffer'
DB_NAME_VARYING_SCALE = 'test_varying_scale_buffer'
DB_NAME_VARYING_SCALE_TVP = 'test_varying_scale_tvp_buffer'
DB_NAME_VARYING_DTYPE = 'test_varying_dtype_buffer'
DB_NAME_DIRECT_VARYING_SCALE = 'test_direct_varying_scale'


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


def _read_and_sort(sdk, measure_id, device_id, start_time, end_time, time_units='s'):
    """Helper: read data back and sort by time for consistent ordering."""
    _, time_data, value_data = sdk.get_data(
        measure_id=measure_id,
        device_id=device_id,
        start_time_n=start_time,
        end_time_n=end_time,
        time_units=time_units,
        allow_duplicates=False,
    )
    if time_data.size > 0:
        sorted_indices = np.argsort(time_data)
        time_data = time_data[sorted_indices]
        value_data = value_data[sorted_indices]
    return time_data, value_data


def test_write_message():
    _test_for_both(DB_NAME_MESSAGE, _test_write_message)


def _test_write_message(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Insert measure and device
    measure_tag = 'test_measure'
    device_tag = 'test_device'
    freq = 1.0  # 1 Hz
    measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq, freq_units='Hz')
    device_id = sdk.insert_device(device_tag=device_tag)

    # Generate some data
    message_values = np.arange(100)  # Values from 0 to 99
    start_time = 0.0  # Start at time 0

    # Write the message
    sdk.write_segment(measure_id, device_id, message_values, start_time, freq=freq, freq_units='Hz')

    # Check the blocks
    blocks = get_all_blocks(sdk, measure_id, device_id)
    total_values = sum(block[8] for block in blocks)  # block[8] is num_values
    assert total_values == 100

    # Read back the data
    _, time_data, value_data = sdk.get_data(measure_id=measure_id, device_id=device_id,
                                            start_time_n=start_time, end_time_n=start_time + 100 / freq,
                                            time_units='s')
    # Verify that value_data matches message_values
    assert np.array_equal(value_data, message_values)


def test_write_messages():
    _test_for_both(DB_NAME_MESSAGES, _test_write_messages)


def _test_write_messages(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Insert measure and device
    measure_tag = 'test_measure'
    device_tag = 'test_device'
    freq = 1.0  # 1 Hz
    measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq, freq_units='Hz')
    device_id = sdk.insert_device(device_tag=device_tag)

    # Generate some messages
    messages = [np.arange(10), np.arange(10, 20), np.arange(20, 30)]
    start_times = [0.0, 10.0, 20.0]  # Start times in seconds

    # Write the messages
    sdk.write_segments(measure_id, device_id, messages, start_times, freq=freq, freq_units='Hz')

    # Check the blocks
    blocks = get_all_blocks(sdk, measure_id, device_id)
    total_values = sum(block[8] for block in blocks)  # block[8] is num_values
    assert total_values == 30  # We wrote 30 values in total

    # Read back the data
    _, time_data, value_data = sdk.get_data(measure_id=measure_id, device_id=device_id,
                                            start_time_n=0.0, end_time_n=30.0, time_units='s')

    # Concatenate messages to compare
    expected_values = np.concatenate(messages)
    assert np.array_equal(value_data, expected_values)


def test_write_buffer():
    _test_for_both(DB_NAME_BUFFER, _test_write_buffer)


def _test_write_buffer(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Insert measure and device
    measure_tag = 'test_measure'
    device_tag = 'test_device'
    freq = 1.0  # 1 Hz
    measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq, freq_units='Hz')
    device_id = sdk.insert_device(device_tag=device_tag)

    # Set target block size to 10
    target_block_size = 10
    sdk.block.block_size = target_block_size

    # Set max_values_buffered to 20
    max_values_buffered = 20

    # Use write_buffer with max_values_buffered
    with sdk.write_buffer(max_values_per_measure_device=max_values_buffered) as buffer:
        # Write multiple small messages to buffer
        message_values_list = [np.array([i]) for i in range(43)]  # 43 messages of 1 value each
        start_time = 0.0
        for idx, message_values in enumerate(message_values_list):
            current_start_time = start_time + idx  # Each message at a different time
            sdk.write_segment(measure_id, device_id, message_values, current_start_time, freq=freq, freq_units='Hz')

            # Check if buffer has automatically flushed after exceeding max_values_buffered
            if (idx + 1) % max_values_buffered == 0:
                # Buffer should have flushed automatically
                blocks = get_all_blocks(sdk, measure_id, device_id)
                total_values = sum(block[8] for block in blocks)  # block[8] is num_values
                assert total_values == (idx + 1), f"Expected {idx + 1} values, found {total_values}"
                print(f"Buffer flushed automatically after {total_values} values.")

    # Upon exiting the with block, any remaining data should be flushed
    # Check the blocks
    blocks = get_all_blocks(sdk, measure_id, device_id)
    total_values = sum(block[8] for block in blocks)
    assert total_values == 43, f"Expected 43 values, found {total_values}"

    # Each block should be at most target_block_size, and the total should sum to 43.
    actual_num_values = [block[8] for block in blocks]
    assert sum(actual_num_values) == 43, f"Expected total 43 values across blocks, got {sum(actual_num_values)}"
    assert all(nv <= target_block_size for nv in actual_num_values), \
        f"All blocks should be <= {target_block_size} values, got {actual_num_values}"

    # We expect at least 2 unique file_ids (from the 2 auto-flushes + final flush)
    file_ids = [block[3] for block in blocks]  # block[3] is file_id
    unique_file_ids = list(set(file_ids))
    assert len(unique_file_ids) >= 2, f"Expected at least 2 unique file_ids, found {len(unique_file_ids)}"

    # Read back the data
    time_data, value_data = _read_and_sort(sdk, measure_id, device_id, 0.0, 43.0, time_units='s')

    # Verify that value_data matches the concatenated message_values_list
    expected_values = np.concatenate(message_values_list)
    assert np.array_equal(value_data, expected_values), \
        f"Data read does not match data written.\nActual:   {value_data}\nExpected: {expected_values}"


def test_write_time_value_pairs_buffered():
    _test_for_both(DB_NAME_TVP_BUFFER, _test_write_time_value_pairs_buffered)


def _test_write_time_value_pairs_buffered(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Insert measure and device
    measure_tag = 'test_measure'
    device_tag = 'test_device'
    freq = 1.0  # 1 Hz
    period = 1 / freq
    measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq, freq_units='Hz')
    device_id = sdk.insert_device(device_tag=device_tag)

    # Set target block size to 10
    target_block_size = 10
    sdk.block.block_size = target_block_size

    # Set max_values_buffered to 20
    max_values_buffered = 20

    # Use write_buffer with max_values_buffered
    times_written = []
    values_written = []
    total_pairs = 43
    current_time = 0.0

    with sdk.write_buffer(max_values_per_measure_device=max_values_buffered) as buffer:
        for idx in range(total_pairs):
            current_time += period
            time = np.array([current_time])
            value = np.array([idx])
            times_written.append(time[0])
            values_written.append(value[0])
            # No freq/period provided — relies on deferred detection in the buffer
            sdk.write_time_value_pairs(measure_id, device_id, times=time, values=value)

            # Check if buffer has automatically flushed after exceeding max_values_buffered
            if (idx + 1) % max_values_buffered == 0:
                # Buffer should have flushed automatically
                blocks = get_all_blocks(sdk, measure_id, device_id)
                total_values = sum(block[8] for block in blocks)  # block[8] is num_values
                assert total_values == (idx + 1), f"Expected {idx + 1} values, found {total_values}"
                print(f"Buffer flushed automatically after {total_values} values.")

    # Upon exiting the with block, any remaining data should be flushed
    blocks = get_all_blocks(sdk, measure_id, device_id)
    total_values = sum(block[8] for block in blocks)
    assert total_values == total_pairs, f"Expected {total_pairs} values, found {total_values}"

    actual_num_values = [block[8] for block in blocks]
    assert sum(actual_num_values) == total_pairs, \
        f"Expected total {total_pairs} values across blocks, got {sum(actual_num_values)}"
    assert all(nv <= target_block_size for nv in actual_num_values), \
        f"All blocks should be <= {target_block_size} values, got {actual_num_values}"

    file_ids = [block[3] for block in blocks]  # block[3] is file_id
    unique_file_ids = list(set(file_ids))
    assert len(unique_file_ids) >= 2, f"Expected at least 2 unique file_ids, found {len(unique_file_ids)}"

    # Read back the data using the block time range from the database directly,
    # to avoid any unit-conversion mismatch between write and read paths.
    all_start_times = [block[6] for block in blocks]  # block[6] is start_time_n
    all_end_times = [block[7] for block in blocks]    # block[7] is end_time_n
    query_start_ns = min(all_start_times)
    query_end_ns = max(all_end_times) + 1  # +1 to be inclusive of the last sample

    _, time_data, value_data = sdk.get_data(
        measure_id=measure_id,
        device_id=device_id,
        start_time_n=query_start_ns,
        end_time_n=query_end_ns,
    )

    assert time_data.size > 0, \
        f"get_data returned no data. Blocks exist with start_times_n={all_start_times}, end_times_n={all_end_times}"

    # Sort the read data by time
    sorted_indices = np.argsort(time_data)
    time_data = time_data[sorted_indices]
    value_data = value_data[sorted_indices]

    # Prepare expected data (sorted by time)
    times_written = np.array(times_written)
    values_written = np.array(values_written)
    sorted_indices = np.argsort(times_written)
    times_written = times_written[sorted_indices]
    values_written = values_written[sorted_indices]

    assert value_data.size == values_written.size, \
        f"Expected {values_written.size} values, got {value_data.size}"
    assert np.array_equal(value_data, values_written), "Value data read does not match values written."


def test_comprehensive():
    _test_for_both(DB_NAME_COMPREHENSIVE, _test_comprehensive)


def _test_comprehensive(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Insert measure and device
    measure_tag = 'test_measure'
    device_tag = 'test_device'
    freq = 1.0  # 1 Hz
    measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq, freq_units='Hz')
    device_id = sdk.insert_device(device_tag=device_tag)

    # Set target block size to 10
    target_block_size = 10
    sdk.block.block_size = target_block_size

    # Set max_values_buffered for write_buffer
    max_values_buffered = 20

    # --- Part 1: Test write_segment ---

    # Generate data for write_segment
    message_values = np.arange(100)
    start_time = 0.0

    # Write the segment
    sdk.write_segment(measure_id, device_id, message_values, start_time, freq=freq, freq_units='Hz')

    # Check blocks after write_segment
    blocks = get_all_blocks(sdk, measure_id, device_id)
    total_values = sum(block[8] for block in blocks)
    assert total_values == 100, f"Expected 100 values after write_segment, found {total_values}"

    # Read back data and verify
    time_data, value_data = _read_and_sort(sdk, measure_id, device_id, 0.0, 100.0, time_units='s')
    assert value_data.size == 100, f"Expected 100 values in readback, got {value_data.size}"
    assert np.array_equal(value_data, message_values), "Data mismatch in write_segment test."

    # --- Part 2: Test write_segments ---

    # Generate data for write_segments (non-overlapping time range with Part 1)
    messages = [np.arange(100, 110), np.arange(110, 120), np.arange(120, 130)]
    start_times = [100.0, 110.0, 120.0]

    # Write the segments
    sdk.write_segments(measure_id, device_id, messages, start_times, freq=freq, freq_units='Hz')

    # Check blocks after write_segments
    blocks = get_all_blocks(sdk, measure_id, device_id)
    total_values = sum(block[8] for block in blocks)
    assert total_values == 130, f"Expected 130 values after write_segments, found {total_values}"

    # Verify total data integrity: all 130 values should be present.
    # Use the block index to determine the full nanosecond time range for a clean query.
    all_start_times_ns = [block[6] for block in blocks]
    all_end_times_ns = [block[7] for block in blocks]
    query_start_ns = min(all_start_times_ns)
    query_end_ns = max(all_end_times_ns) + 1

    _, time_data_all, value_data_all = sdk.get_data(
        measure_id=measure_id,
        device_id=device_id,
        start_time_n=query_start_ns,
        end_time_n=query_end_ns,
        allow_duplicates=False,
    )
    assert value_data_all.size == 130, f"Expected 130 total values, got {value_data_all.size}"

    # All expected values should be present (regardless of ordering)
    expected_all = np.arange(130)
    assert np.array_equal(np.sort(value_data_all), expected_all), \
        f"Not all expected values present after write_segments."

    # --- Part 3: Test write_buffer with automatic flushing and small write_segment calls ---

    # Use write_buffer with max_values_buffered
    with sdk.write_buffer(max_values_per_measure_device=max_values_buffered) as buffer:
        # Write multiple small messages to buffer
        message_values_list = [np.array([i]) for i in range(130, 173)]  # 43 messages of 1 value each
        start_time = 130.0
        for idx, message_values in enumerate(message_values_list):
            current_start_time = start_time + idx
            sdk.write_segment(measure_id, device_id, message_values, current_start_time, freq=freq, freq_units='Hz')

            # Check for automatic buffer flush
            if (idx + 1) % max_values_buffered == 0:
                # Buffer should have flushed automatically
                blocks = get_all_blocks(sdk, measure_id, device_id)
                total_values = sum(block[8] for block in blocks)
                expected_total = 130 + (idx + 1)
                assert total_values == expected_total, f"Expected {expected_total} values after buffer flush, found {total_values}"

    # Upon exiting the with block, any remaining data should be flushed
    blocks = get_all_blocks(sdk, measure_id, device_id)
    total_values = sum(block[8] for block in blocks)
    assert total_values == 173, f"Expected 173 values after write_buffer, found {total_values}"

    # Verify that all blocks are at most target_block_size
    actual_num_values = [block[8] for block in blocks]
    assert all(nv <= target_block_size for nv in actual_num_values), \
        f"All blocks should be <= {target_block_size} values, got {actual_num_values}"

    # Verify full data integrity: all 173 values present
    all_start_times_ns = [block[6] for block in blocks]
    all_end_times_ns = [block[7] for block in blocks]
    query_start_ns = min(all_start_times_ns)
    query_end_ns = max(all_end_times_ns) + 1

    _, time_data_all, value_data_all = sdk.get_data(
        measure_id=measure_id,
        device_id=device_id,
        start_time_n=query_start_ns,
        end_time_n=query_end_ns,
        allow_duplicates=False,
    )
    assert value_data_all.size == 173, f"Expected 173 total values, got {value_data_all.size}"

    expected_all = np.arange(173)
    assert np.array_equal(np.sort(value_data_all), expected_all), \
        f"Not all expected values present after write_buffer."


def test_multi_buffer():
    _test_for_both(DB_NAME_MULTI_BUFFER, _test_multi_buffer)


def _test_multi_buffer(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Insert multiple measures and devices
    measure_ids = []
    device_ids = []
    num_measures = 2
    num_devices = 2

    freq = 1.0  # 1 Hz
    for i in range(num_measures):
        measure_tag = f'test_measure_{i}'

        measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq, freq_units='Hz')
        measure_ids.append(measure_id)

    for i in range(num_devices):
        device_tag = f'test_device_{i}'
        device_id = sdk.insert_device(device_tag=device_tag)
        device_ids.append(device_id)

    # Set target block size to 10
    target_block_size = 10
    sdk.block.block_size = target_block_size

    max_values_per_measure_device = 20

    max_total_values_buffered = 100

    # Use write_buffer with specified max values
    with sdk.write_buffer(max_values_per_measure_device=max_values_per_measure_device,
                          max_total_values_buffered=max_total_values_buffered) as buffer:

        for measure_id in measure_ids:
            for device_id in device_ids:
                # Write multiple small messages to buffer
                message_values_list = [np.array([i]) for i in range(43)]  # 43 messages of 1 value each
                start_time = 0.0
                for idx, message_values in enumerate(message_values_list):
                    current_start_time = start_time + idx  # Each message at a different time
                    sdk.write_segment(measure_id, device_id, message_values, current_start_time, freq=freq,
                                      freq_units='Hz')

                    # Check if buffer has automatically flushed after exceeding max_values_buffered
                    if (idx + 1) % max_values_per_measure_device == 0:
                        # Buffer should have flushed automatically
                        blocks = get_all_blocks(sdk, measure_id, device_id)
                        total_values = sum(block[8] for block in blocks)  # block[8] is num_values
                        assert total_values == (idx + 1), f"Expected {idx + 1} values, found {total_values}"
                        print(f"Buffer flushed automatically after {total_values} values.")

    # After all writes and flushes, verify each measure-device combination
    for measure_id in measure_ids:
        for device_id in device_ids:
            # Check the blocks
            blocks = get_all_blocks(sdk, measure_id, device_id)
            total_values = sum(block[8] for block in blocks)
            assert total_values == 43, f"Expected 43 values, found {total_values}"

            actual_num_values = [block[8] for block in blocks]
            assert sum(actual_num_values) == 43, \
                f"Expected total 43 values across blocks, got {sum(actual_num_values)}"
            assert all(nv <= target_block_size for nv in actual_num_values), \
                f"All blocks should be <= {target_block_size} values, got {actual_num_values}"

            file_ids = [block[3] for block in blocks]  # block[3] is file_id
            unique_file_ids = list(set(file_ids))
            assert len(unique_file_ids) >= 2, f"Expected at least 2 unique file_ids, found {len(unique_file_ids)}"

            # Read back the data
            time_data, value_data = _read_and_sort(sdk, measure_id, device_id, 0.0, 43.0, time_units='s')

            # Verify that value_data matches the concatenated message_values_list
            expected_values = np.concatenate(message_values_list)
            assert np.array_equal(value_data, expected_values), \
                f"Data read does not match data written.\nActual:   {value_data}\nExpected: {expected_values}"


def test_write_buffer_varying_scale_factors():
    """Buffered segments with differing scale_m / scale_b flush correctly."""
    _test_for_both(DB_NAME_VARYING_SCALE, _test_write_buffer_varying_scale_factors)


def _test_write_buffer_varying_scale_factors(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    freq = 1.0  # 1 Hz
    one_second_ns = 1_000_000_000
    measure_id = sdk.insert_measure(measure_tag='test_measure', freq=freq, freq_units='Hz')
    device_id = sdk.insert_device(device_tag='test_device')

    # Three segments at non-overlapping times. Segments A and C share scale
    # factors; B uses a different pair. The flush must produce digital
    # readback that matches the digital input, and analog readback that
    # applies each segment's own (scale_m * digital + scale_b).
    segment_a = np.arange(0, 10, dtype=np.int64)
    segment_b = np.arange(100, 110, dtype=np.int64)
    segment_c = np.arange(200, 210, dtype=np.int64)

    start_a_s, start_b_s, start_c_s = 0.0, 100.0, 200.0
    seg_len_s = 10  # one value per second, 10 values per segment

    scale_a = (2.0, 10.0)        # (scale_m, scale_b)
    scale_b_pair = (3.0, -5.0)
    scale_c = (2.0, 10.0)        # matches A so the partitioner co-locates them

    with sdk.write_buffer() as buffer:
        sdk.write_segment(measure_id, device_id, segment_a, start_a_s,
                          freq=freq, freq_units='Hz', time_units='s',
                          scale_m=scale_a[0], scale_b=scale_a[1])
        sdk.write_segment(measure_id, device_id, segment_b, start_b_s,
                          freq=freq, freq_units='Hz', time_units='s',
                          scale_m=scale_b_pair[0], scale_b=scale_b_pair[1])
        sdk.write_segment(measure_id, device_id, segment_c, start_c_s,
                          freq=freq, freq_units='Hz', time_units='s',
                          scale_m=scale_c[0], scale_b=scale_c[1])

    # All 30 values should be present.
    blocks = get_all_blocks(sdk, measure_id, device_id)
    total_values = sum(block[8] for block in blocks)
    assert total_values == 30, f"Expected 30 values total, found {total_values}"

    # Pull every value in one query (covering the union of all three writes),
    # then slice by timestamp. We can't query a single segment by range
    # because segments A and C live in the same block group (same scale
    # factors, same partition), and that block group spans 0 s to 210 s.
    full_start_ns = 0
    full_end_ns = (int(start_c_s) + seg_len_s) * one_second_ns

    # Digital readback (analog=False) should round-trip each segment.
    _, dig_times, dig_vals = sdk.get_data(
        measure_id=measure_id, device_id=device_id,
        start_time_n=full_start_ns, end_time_n=full_end_ns, analog=False)
    order = np.argsort(dig_times)
    dig_times, dig_vals = dig_times[order], dig_vals[order]

    def slice_by_start(times, vals, start_s):
        start_ns = int(start_s) * one_second_ns
        end_ns = start_ns + seg_len_s * one_second_ns
        mask = (times >= start_ns) & (times < end_ns)
        return vals[mask]

    assert np.array_equal(slice_by_start(dig_times, dig_vals, start_a_s), segment_a)
    assert np.array_equal(slice_by_start(dig_times, dig_vals, start_b_s), segment_b)
    assert np.array_equal(slice_by_start(dig_times, dig_vals, start_c_s), segment_c)

    # Analog readback should apply each segment's own scale_m / scale_b.
    _, an_times, an_vals = sdk.get_data(
        measure_id=measure_id, device_id=device_id,
        start_time_n=full_start_ns, end_time_n=full_end_ns, analog=True)
    order = np.argsort(an_times)
    an_times, an_vals = an_times[order], an_vals[order]

    expected_a = segment_a * scale_a[0] + scale_a[1]
    expected_b = segment_b * scale_b_pair[0] + scale_b_pair[1]
    expected_c = segment_c * scale_c[0] + scale_c[1]

    assert np.allclose(slice_by_start(an_times, an_vals, start_a_s), expected_a), \
        f"Analog A mismatch."
    assert np.allclose(slice_by_start(an_times, an_vals, start_b_s), expected_b), \
        f"Analog B mismatch."
    assert np.allclose(slice_by_start(an_times, an_vals, start_c_s), expected_c), \
        f"Analog C mismatch."


def test_write_buffer_varying_scale_tvp():
    """Buffered time-value pairs with differing scale_m / scale_b flush correctly."""
    _test_for_both(DB_NAME_VARYING_SCALE_TVP, _test_write_buffer_varying_scale_tvp)


def _test_write_buffer_varying_scale_tvp(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    freq = 1.0  # 1 Hz
    period_ns = 1_000_000_000
    measure_id = sdk.insert_measure(measure_tag='test_measure', freq=freq, freq_units='Hz')
    device_id = sdk.insert_device(device_tag='test_device')

    # Two batches of time-value pairs at disjoint time ranges, each with its
    # own scale_m / scale_b. Both pushed during a single buffer context, so
    # flush sees them as one heterogeneous data_dicts list.
    times_a_ns = np.array([0, 1, 2, 3, 4], dtype=np.int64) * period_ns
    values_a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    scale_a = (0.5, 1.0)

    times_b_ns = np.array([100, 101, 102, 103, 104], dtype=np.int64) * period_ns
    values_b = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    scale_b_pair = (4.0, -2.0)

    with sdk.write_buffer() as buffer:
        sdk.write_time_value_pairs(measure_id, device_id, times=times_a_ns, values=values_a,
                                   freq=freq, freq_units='Hz', time_units='ns',
                                   scale_m=scale_a[0], scale_b=scale_a[1])
        sdk.write_time_value_pairs(measure_id, device_id, times=times_b_ns, values=values_b,
                                   freq=freq, freq_units='Hz', time_units='ns',
                                   scale_m=scale_b_pair[0], scale_b=scale_b_pair[1])

    # All 10 values present after flush.
    blocks = get_all_blocks(sdk, measure_id, device_id)
    total_values = sum(block[8] for block in blocks)
    assert total_values == 10, f"Expected 10 values total, found {total_values}"

    # Read everything once and split by timestamp. A and B live in different
    # partitions (different scale factors), so each forms its own block
    # group; reading by full range and splitting client-side is the
    # consistent pattern across these tests.
    full_start_ns = int(times_a_ns[0])
    full_end_ns = int(times_b_ns[-1]) + period_ns

    _, an_times, an_vals = sdk.get_data(
        measure_id=measure_id, device_id=device_id,
        start_time_n=full_start_ns, end_time_n=full_end_ns, analog=True)
    order = np.argsort(an_times)
    an_times, an_vals = an_times[order], an_vals[order]

    a_lo, a_hi = int(times_a_ns[0]), int(times_a_ns[-1]) + period_ns
    b_lo, b_hi = int(times_b_ns[0]), int(times_b_ns[-1]) + period_ns

    a_mask = (an_times >= a_lo) & (an_times < a_hi)
    b_mask = (an_times >= b_lo) & (an_times < b_hi)

    expected_a = values_a * scale_a[0] + scale_a[1]
    expected_b = values_b * scale_b_pair[0] + scale_b_pair[1]

    assert np.allclose(an_vals[a_mask], expected_a), \
        f"Analog A mismatch.\nGot: {an_vals[a_mask]}\nExpected: {expected_a}"
    assert np.allclose(an_vals[b_mask], expected_b), \
        f"Analog B mismatch.\nGot: {an_vals[b_mask]}\nExpected: {expected_b}"


def test_write_buffer_varying_dtype():
    """Buffered segments with mixed integer and float dtypes flush correctly."""
    _test_for_both(DB_NAME_VARYING_DTYPE, _test_write_buffer_varying_dtype)


def _test_write_buffer_varying_dtype(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    freq = 1.0  # 1 Hz
    one_second_ns = 1_000_000_000
    measure_id = sdk.insert_measure(measure_tag='test_measure', freq=freq, freq_units='Hz')
    device_id = sdk.insert_device(device_tag='test_device')

    int_values = np.arange(0, 10, dtype=np.int64)
    float_values = np.arange(50, 60, dtype=np.float64) + 0.25

    int_start_s = 0.0
    float_start_s = 100.0
    seg_len_s = 10

    with sdk.write_buffer() as buffer:
        sdk.write_segment(measure_id, device_id, int_values, int_start_s,
                          freq=freq, freq_units='Hz', time_units='s')
        sdk.write_segment(measure_id, device_id, float_values, float_start_s,
                          freq=freq, freq_units='Hz', time_units='s')

    blocks = get_all_blocks(sdk, measure_id, device_id)
    total_values = sum(block[8] for block in blocks)
    assert total_values == 20, f"Expected 20 values total, found {total_values}"

    # Pull everything in one go and slice by timestamp.
    full_start_ns = 0
    full_end_ns = (int(float_start_s) + seg_len_s) * one_second_ns
    _, times_back, vals_back = sdk.get_data(
        measure_id=measure_id, device_id=device_id,
        start_time_n=full_start_ns, end_time_n=full_end_ns, analog=False)

    order = np.argsort(times_back)
    times_back, vals_back = times_back[order], vals_back[order]

    int_lo = int(int_start_s) * one_second_ns
    int_hi = int_lo + seg_len_s * one_second_ns
    float_lo = int(float_start_s) * one_second_ns
    float_hi = float_lo + seg_len_s * one_second_ns

    int_back = vals_back[(times_back >= int_lo) & (times_back < int_hi)]
    float_back = vals_back[(times_back >= float_lo) & (times_back < float_hi)]

    # The decoded array is float64 across both partitions because at least
    # one block is float-encoded; the integer values come back as exact whole
    # numbers and convert losslessly back to int64.
    assert np.array_equal(int_back.astype(np.int64), int_values)
    assert np.allclose(float_back, float_values)


def test_direct_write_segments_varying_scale():
    """`write_segments` accepts per-segment scale_m / scale_b lists in the direct (non-buffered) path."""
    _test_for_both(DB_NAME_DIRECT_VARYING_SCALE, _test_direct_write_segments_varying_scale)


def _test_direct_write_segments_varying_scale(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    freq = 1.0  # 1 Hz
    one_second_ns = 1_000_000_000
    measure_id = sdk.insert_measure(measure_tag='test_measure', freq=freq, freq_units='Hz')
    device_id = sdk.insert_device(device_tag='test_device')

    segments = [
        np.arange(0, 5, dtype=np.int64),
        np.arange(20, 25, dtype=np.int64),
        np.arange(40, 45, dtype=np.int64),
    ]
    start_times = [0.0, 100.0, 200.0]
    scale_m_list = [2.0, 3.0, 2.0]
    scale_b_list = [1.0, -1.0, 1.0]
    seg_len_s = 5

    sdk.write_segments(measure_id, device_id, segments, start_times,
                       freq=freq, freq_units='Hz', time_units='s',
                       scale_m=scale_m_list, scale_b=scale_b_list)

    blocks = get_all_blocks(sdk, measure_id, device_id)
    total_values = sum(block[8] for block in blocks)
    assert total_values == 15, f"Expected 15 values total, found {total_values}"

    # Read all values once across the full span and split by timestamp.
    # Segments 0 and 2 share scale (2.0, 1.0) and end up in the same block
    # group spanning 0 s to 205 s; a per-segment range query would return
    # both of them.
    full_start_ns = 0
    full_end_ns = (int(start_times[-1]) + seg_len_s) * one_second_ns

    _, times_back, vals_back = sdk.get_data(
        measure_id=measure_id, device_id=device_id,
        start_time_n=full_start_ns, end_time_n=full_end_ns, analog=True)
    order = np.argsort(times_back)
    times_back, vals_back = times_back[order], vals_back[order]

    for segment, start_s, m, b in zip(segments, start_times, scale_m_list, scale_b_list):
        lo = int(start_s) * one_second_ns
        hi = lo + seg_len_s * one_second_ns
        slice_vals = vals_back[(times_back >= lo) & (times_back < hi)]
        expected = segment * m + b
        assert np.allclose(slice_vals, expected), \
            f"Analog mismatch for segment starting at {start_s}s.\nGot: {slice_vals}\nExpected: {expected}"