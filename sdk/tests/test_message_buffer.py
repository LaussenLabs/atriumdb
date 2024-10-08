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
DB_NAME_COMPREHENSIVE = 'test_comprehensive'

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

    # Check that blocks are split according to target_block_size and flushes
    # We expect flushes at 20 and 40, and the final flush when exiting the context
    expected_num_values = [10, 10, 10, 10, 3]  # Blocks of sizes based on block_size and remaining values
    actual_num_values = [block[8] for block in blocks]
    assert actual_num_values == expected_num_values, f"Expected block sizes {expected_num_values}, got {actual_num_values}"

    # Check file_ids to see if they change upon automatic flushes
    file_ids = [block[3] for block in blocks]  # block[3] is file_id
    unique_file_ids = list(set(file_ids))
    assert len(unique_file_ids) == 3, f"Expected 3 unique file_ids, found {len(unique_file_ids)}"

    # Read back the data
    _, time_data, value_data = sdk.get_data(
        measure_id=measure_id,
        device_id=device_id,
        start_time_n=0.0,
        end_time_n=43.0,
        time_units='s'
    )

    # Verify that value_data matches the concatenated message_values_list
    expected_values = np.concatenate(message_values_list)
    assert np.array_equal(value_data, expected_values), "Data read does not match data written."


def test_write_time_value_pairs_buffered():
    _test_for_both(DB_NAME_BUFFER, _test_write_time_value_pairs_buffered)


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
            sdk.write_time_value_pairs(measure_id, device_id, times=time, values=value)

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
    assert total_values == total_pairs, f"Expected {total_pairs} values, found {total_values}"

    # Check that blocks are split according to target_block_size and flushes
    expected_num_values = [10, 10, 10, 10, 3]  # Blocks of sizes based on block_size and remaining values
    actual_num_values = [block[8] for block in blocks]
    assert actual_num_values == expected_num_values, f"Expected block sizes {expected_num_values}, got {actual_num_values}"

    # Check file_ids to see if they change upon automatic flushes
    file_ids = [block[3] for block in blocks]  # block[3] is file_id
    unique_file_ids = list(set(file_ids))
    assert len(unique_file_ids) == 3, f"Expected 3 unique file_ids, found {len(unique_file_ids)}"

    # Read back the data
    start_time_n = times_written[0]
    end_time_n = times_written[-1] + period

    _, time_data, value_data = sdk.get_data(
        measure_id=measure_id,
        device_id=device_id,
        start_time_n=start_time_n,
        end_time_n=end_time_n,
        time_units='s'
    )

    # Since times are irregular, we need to sort them
    times_written = np.array(times_written)
    values_written = np.array(values_written)

    # Sort the written data
    sorted_indices = np.argsort(times_written)
    times_written = times_written[sorted_indices]
    values_written = values_written[sorted_indices]

    # Sort the read data
    sorted_indices = np.argsort(time_data)
    time_data = time_data[sorted_indices]
    value_data = value_data[sorted_indices]

    # Now compare the times and values
    # Allow for small differences in times due to storage precision
    assert np.allclose(time_data, times_written, atol=1e-6), "Time data read does not match times written."
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

    # --- Part 1: Test write_message ---

    # Generate data for write_message
    message_values = np.arange(100)
    start_time = 0.0

    # Write the message
    sdk.write_segment(measure_id, device_id, message_values, start_time, freq=freq, freq_units='Hz')

    # Check blocks after write_message
    blocks = get_all_blocks(sdk, measure_id, device_id)
    block_sizes = tuple(block[8] for block in blocks)
    assert block_sizes == (10,) * 10

    # Read back data and verify
    _, time_data, value_data = sdk.get_data(
        measure_id=measure_id,
        device_id=device_id,
        start_time_n=start_time,
        end_time_n=start_time + 100 / freq,
        time_units='s'
    )
    assert np.array_equal(value_data, message_values), "Data mismatch in write_message test."

    # --- Part 2: Test write_messages ---

    # Generate data for write_messages
    messages = [np.arange(100, 110), np.arange(110, 120), np.arange(120, 130)]
    start_times = [100.0, 110.0, 120.0]

    # Write the messages
    sdk.write_segments(measure_id, device_id, messages, start_times, freq=freq, freq_units='Hz')

    # Check blocks after write_messages
    blocks = get_all_blocks(sdk, measure_id, device_id)
    total_values = sum(block[8] for block in blocks)
    assert total_values == 130, f"Expected 130 values after write_messages, found {total_values}"

    # Read back data and verify
    expected_values = np.concatenate([message_values] + messages)
    _, time_data, value_data = sdk.get_data(
        measure_id=measure_id,
        device_id=device_id,
        start_time_n=0.0,
        end_time_n=130.0,
        time_units='s'
    )
    assert np.array_equal(value_data, expected_values), "Data mismatch in write_messages test."

    # --- Part 3: Test write_buffer with automatic flushing and small write_message calls ---

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

    # Check that blocks are split correctly
    # Since initial blocks were already created, we focus on the new blocks
    # Expected block sizes for new data: [10, 10, 10, 10, 3]
    # Total blocks should now be previous blocks plus these new blocks
    new_blocks = blocks[len(blocks) - 5:]  # Last 5 blocks are from write_buffer
    expected_num_values = [10, 10, 10, 10, 3]
    actual_num_values = [block[8] for block in new_blocks]
    assert actual_num_values == expected_num_values, f"Expected new block sizes {expected_num_values}, got {actual_num_values}"

    # Check file_ids for new blocks to see if they change upon automatic flushes
    file_ids = [block[3] for block in new_blocks]
    unique_file_ids = list(set(file_ids))
    assert len(unique_file_ids) == 3, f"Expected 3 unique file_ids for new data, found {len(unique_file_ids)}"

    # Read back all data and verify
    _, time_data, value_data = sdk.get_data(
        measure_id=measure_id,
        device_id=device_id,
        start_time_n=0.0,
        end_time_n=173.0,
        time_units='s',
        allow_duplicates=False,
    )

    # Create expected values by concatenating all messages
    expected_values = np.arange(total_values)
    assert np.array_equal(value_data, expected_values), "Data mismatch in write_buffer test."


def test_multi_buffer():
    _test_for_both(DB_NAME_COMPREHENSIVE, _test_multi_buffer)

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

                buffer.flush_sub_buffer((measure_id, device_id))
                # Check the blocks
                blocks = get_all_blocks(sdk, measure_id, device_id)
                total_values = sum(block[8] for block in blocks)
                assert total_values == 43, f"Expected 43 values, found {total_values}"

                # Check that blocks are split according to target_block_size and flushes
                # We expect flushes at 20 and 40, and the final flush when exiting the context
                expected_num_values = [10, 10, 10, 10, 3]  # Blocks of sizes based on block_size and remaining values
                actual_num_values = [block[8] for block in blocks]
                assert actual_num_values == expected_num_values, f"Expected block sizes {expected_num_values}, got {actual_num_values}"

                # Check file_ids to see if they change upon automatic flushes
                file_ids = [block[3] for block in blocks]  # block[3] is file_id
                unique_file_ids = list(set(file_ids))
                assert len(unique_file_ids) == 3, f"Expected 3 unique file_ids, found {len(unique_file_ids)}"

                # Read back the data
                _, time_data, value_data = sdk.get_data(
                    measure_id=measure_id,
                    device_id=device_id,
                    start_time_n=0.0,
                    end_time_n=43.0,
                    time_units='s'
                )

                # Verify that value_data matches the concatenated message_values_list
                expected_values = np.concatenate(message_values_list)
                assert np.array_equal(value_data, expected_values), "Data read does not match data written."