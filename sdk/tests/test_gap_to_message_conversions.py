import pytest

import numpy as np

from atriumdb.adb_functions import create_gap_arr_from_variable_messages, reconstruct_messages, \
    sort_message_time_values, merge_sorted_messages


def test_gap_data_to_message_time_conversion():
    test_cases = [
        ([1000000000_000_000_000, 1000002000_000_000_000, 1000004000_000_000_000],
         [200, 200, 200],
         1e9),
        ([1000000000_000_000_000, 1000003000_000_000_000, 1000006000_000_000_000],
         [100, 300, 200],
         1e9),
        ([1000003000_000_000_000, 1000000000_000_000_000, 1000006000_000_000_000],
         [300, 100, 200],
         1e9),  # Out of Order
        ([1000000000_000_000_000], [100], 1e9),  # Single message test case
    ]

    for message_start_epoch_array, message_size_array, freq in test_cases:
        message_start_epoch_array = np.array(message_start_epoch_array)
        message_size_array = np.array(message_size_array)
        start_time_nano_epoch = int(message_start_epoch_array[0])
        num_values = int(np.sum(message_size_array))

        gap_data_array = create_gap_arr_from_variable_messages(message_start_epoch_array, message_size_array, freq)
        rebuilt_starts, rebuilt_sizes = reconstruct_messages(start_time_nano_epoch, gap_data_array, freq, num_values)

        np.testing.assert_array_equal(message_start_epoch_array, rebuilt_starts)
        np.testing.assert_array_equal(message_size_array, rebuilt_sizes)


@pytest.mark.parametrize("message_starts, message_sizes, value_array, expected_output", [
    (np.array([0]), np.array([5]), np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])),
    (np.array([5, 0]), np.array([3, 5]), np.array([6, 7, 8, 1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5, 6, 7, 8])),
    (np.array([0, 5]), np.array([5, 3]), np.array([1, 2, 3, 4, 5, 6, 7, 8]), np.array([1, 2, 3, 4, 5, 6, 7, 8])),
    (np.array([0, 0]), np.array([3, 3]), np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3, 4, 5, 6])),
    (np.array([3, 0]), np.array([2, 3]), np.array([4, 5, 1, 2, 3]), np.array([1, 2, 3, 4, 5])),
    (np.array([5, 0, 10]), np.array([5, 5, 2]), np.array([6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 11, 12]), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),
])
def test_sort_message_time_values(message_starts, message_sizes, value_array, expected_output):
    sort_message_time_values(message_starts, message_sizes, value_array)
    np.testing.assert_array_equal(value_array, expected_output)


def test_merge_sorted_messages_no_overlap():
    message_starts_1 = np.array([0, 10**9], dtype=np.int64)
    message_sizes_1 = np.array([1, 1], dtype=np.int64)
    values_1 = np.array([0.1, 0.2], dtype=np.float64)

    message_starts_2 = np.array([2 * 10**9, 3 * 10**9], dtype=np.int64)
    message_sizes_2 = np.array([1, 1], dtype=np.int64)
    values_2 = np.array([0.3, 0.4], dtype=np.float64)

    freq_nhz = 1e9

    merged_starts_expected = np.array([0], dtype=np.int64)
    merged_sizes_expected = np.array([4], dtype=np.int64)
    merged_values_expected = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    merged_starts, merged_sizes, merged_values = merge_sorted_messages(
        message_starts_1, message_sizes_1, values_1,
        message_starts_2, message_sizes_2, values_2,
        freq_nhz)

    np.testing.assert_array_equal(merged_starts, merged_starts_expected)
    np.testing.assert_array_equal(merged_sizes, merged_sizes_expected)
    np.testing.assert_array_equal(merged_values, merged_values_expected)


def test_merge_sorted_messages_with_overlap():
    message_starts_1 = np.array([0, 5 * 10**9], dtype=np.int64)
    message_sizes_1 = np.array([2, 2], dtype=np.int64)
    values_1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    message_starts_2 = np.array([5 * 10**9, 15 * 10**9], dtype=np.int64)
    message_sizes_2 = np.array([2, 2], dtype=np.int64)
    values_2 = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)

    freq_nhz = 1e9

    merged_starts_expected = np.array([0, 5 * 10**9, 15 * 10**9], dtype=np.int64)
    merged_sizes_expected = np.array([2, 2, 2], dtype=np.int64)  # Adjusted due to overlap
    merged_values_expected = np.array([0.1, 0.2, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)  # Adjusted due to overlap

    merged_starts, merged_sizes, merged_values = merge_sorted_messages(
        message_starts_1, message_sizes_1, values_1,
        message_starts_2, message_sizes_2, values_2,
        freq_nhz)

    np.testing.assert_array_equal(merged_starts, merged_starts_expected)
    np.testing.assert_array_equal(merged_sizes, merged_sizes_expected)
    np.testing.assert_array_equal(merged_values, merged_values_expected)


if __name__ == "__main__":
    pytest.main()
