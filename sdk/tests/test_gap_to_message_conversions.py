import pytest

import numpy as np

from atriumdb import create_gap_arr
from atriumdb.adb_functions import create_gap_arr_from_variable_messages, reconstruct_messages, \
    sort_message_time_values, merge_sorted_messages, merge_gap_data

import pytest
import numpy as np


@pytest.mark.parametrize(
    "times_1, values_1, times_2, values_2, expected_merged_values",
    [
        # Simple Test Cases
        ([20, 21, 22, 24], [20, 21, 22, 24], [19, 21], [19, 21], [19, 20, 21, 22, 24]),
        ([25, 26], [25, 26], [27, 28], [27, 28], [25, 26, 27, 28]),
        ([10, 11], [10, 11], [13, 14], [13, 14], [10, 11, 13, 14]),
        ([15, 16, 17], [15, 16, 17], [16, 17, 18], [16, 17, 18], [15, 16, 17, 18]),

        # Mutually Exclusive Subsets of a range
        ([1, 3, 5], [1, 3, 5], [2, 4], [2, 4], [1, 2, 3, 4, 5]),

        # Single-Element Arrays
        ([30], [30], [31], [31], [30, 31]),

        # Consecutive Times with Repeated Elements
        ([40, 40, 41], [40, 40, 41], [41, 42, 42], [41, 42, 42], [40, 41, 42]),

        # Large Gaps Between Times
        ([100, 200], [100, 200], [300, 400], [300, 400], [100, 200, 300, 400]),

        # Reverse Ordered Times
        ([50, 49], [50, 49], [48, 47], [48, 47], [47, 48, 49, 50]),

        # Identical Times and Values
        ([60, 61], [60, 61], [60, 61], [60, 61], [60, 61]),

        # Overlapping times; values from 2 overwrite values from 1
        ([10, 11, 12, 13], [1, 2, 3, 4], [12, 13, 14], [100, 200, 300], [1, 2, 100, 200, 300]),
        ([20, 21, 22, 23], [5, 6, 7, 8], [21, 22], [500, 600], [5, 500, 600, 8]),

        # Times from 1 inside 2; with overwriting
        ([30, 31], [9, 10], [29, 30, 31, 32], [900, 1000, 1100, 1200], [900, 1000, 1100, 1200]),

        # Times from 2 inside 1; with overwriting
        ([40, 41, 42, 43], [11, 12, 13, 14], [41, 42], [4000, 5000], [11, 4000, 5000, 14]),

        # Larger series with gaps, smaller series inside with overwriting
        ([50, 52, 54, 56], [15, 16, 17, 18], [51, 52, 53], [7000, 8000, 9000], [15, 7000, 8000, 9000, 17, 18]),

        # Weaving two arrays together
        ([60, 63, 66, 69], [19, 20, 21, 22], [62, 63, 64, 67], [10000, 11000, 12000, 13000],
         [19, 10000, 11000, 12000, 21, 13000, 22]),

        # Edge case: Overlapping start and end times with different values
        ([70, 71, 73], [23, 24, 25], [71, 72, 73], [14000, 15000, 16000], [23, 14000, 15000, 16000]),

        # Extended case: 1 large with gaps, 2 small with mixed overlapping
        ([80, 82, 84, 86, 88], [26, 27, 28, 29, 30], [81, 82, 85, 87], [17000, 18000, 19000, 20000],
         [26, 17000, 18000, 28, 19000, 29, 20000, 30]),

        # Case with multiple identical times, ensuring overwriting is consistent
        ([90, 91, 91, 92], [31, 32, 33, 34], [91, 92, 93], [21000, 22000, 23000], [31, 21000, 22000, 23000]),

        # Out of Phase overlap
        ([1, 2, 3, 4], [1, 2, 3, 4], [1.5, 2.5, 3.5, 4.5], [1.5, 2.5, 3.5, 4.5], [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]),
    ]
)
def test_end_to_end_merging(times_1, values_1, times_2, values_2, expected_merged_values):
    freq_nhz = 1_000_000_000
    nanoseconds_in_a_second = 1_000_000_000
    times_1, values_1 = (np.array(times_1, dtype=np.float64) * nanoseconds_in_a_second).astype(np.int64), np.array(values_1, dtype=np.float64)
    gap_array_1 = create_gap_arr(times_1, 1, freq_nhz)

    times_2, values_2 = (np.array(times_2, dtype=np.float64) * nanoseconds_in_a_second).astype(np.int64), np.array(values_2, dtype=np.float64)
    gap_array_2 = create_gap_arr(times_2, 1, freq_nhz)

    merged_values, merged_gap_array, merged_start_time = merge_gap_data(
        values_1, gap_array_1, int(times_1[0]), values_2, gap_array_2, int(times_2[0]), freq_nhz)

    assert np.allclose(merged_values, np.array(expected_merged_values, dtype=np.float64))


def test_gap_data_to_message_time_conversion():
    test_cases = [
        ([1000000000_000_000_000, 1000002000_000_000_000, 1000004000_000_000_000],
         [200, 200, 200],
         10**9),
        ([1000000000_000_000_000, 1000003000_000_000_000, 1000006000_000_000_000],
         [100, 300, 200],
         10**9),
        ([1000003000_000_000_000, 1000000000_000_000_000, 1000006000_000_000_000],
         [300, 100, 200],
         10**9),  # Out of Order
        ([1000000000_000_000_000], [100], 10**9),  # Single message test case
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

    freq_nhz = 10**9

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

    ####

    message_starts_1 = np.array([4_000, 12_000], dtype=np.int64)
    message_sizes_1 = np.array([4, 4], dtype=np.int64)
    values_1 = np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype=np.float64)

    message_starts_2 = np.array([8_000, 16_000], dtype=np.int64)
    message_sizes_2 = np.array([4, 4], dtype=np.int64)
    values_2 = np.array([4, 5, 6, 7, 12, 13, 14, 15], dtype=np.float64)

    freq_nhz = 10**15

    merged_starts_expected = np.array([4_000], dtype=np.int64)
    merged_sizes_expected = np.array([16], dtype=np.int64)
    merged_values_expected = np.array(list(range(16)), dtype=np.float64)

    merged_starts, merged_sizes, merged_values = merge_sorted_messages(
        message_starts_1, message_sizes_1, values_1,
        message_starts_2, message_sizes_2, values_2,
        freq_nhz)

    np.testing.assert_array_equal(merged_starts, merged_starts_expected)
    np.testing.assert_array_equal(merged_sizes, merged_sizes_expected)
    np.testing.assert_array_equal(merged_values, merged_values_expected)

    ####

    freq_hz = 300

    message_starts_1 = np.array([3, 5], dtype=np.int64) * (10 ** 9)
    message_sizes_1 = np.array([300, 300], dtype=np.int64)
    values_1 = np.concatenate((np.arange(0, 300), np.arange(600, 900)), dtype=np.float64)

    message_starts_2 = np.array([4, 6], dtype=np.int64) * (10 ** 9)
    message_sizes_2 = np.array([300, 300], dtype=np.int64)
    values_2 = np.concatenate((np.arange(300, 600), np.arange(900, 1200)), dtype=np.float64)

    freq_nhz = freq_hz * (10 ** 9)

    merged_starts_expected = np.array([3], dtype=np.int64) * (10 ** 9)
    merged_sizes_expected = np.array([1200], dtype=np.int64)
    merged_values_expected = np.array(list(range(1200)), dtype=np.float64)

    merged_starts, merged_sizes, merged_values = merge_sorted_messages(
        message_starts_1, message_sizes_1, values_1,
        message_starts_2, message_sizes_2, values_2,
        freq_nhz)

    np.testing.assert_array_equal(merged_starts, merged_starts_expected)
    np.testing.assert_array_equal(merged_sizes, merged_sizes_expected)
    np.testing.assert_array_equal(merged_values, merged_values_expected)

    ####

    freq_hz = 1

    message_starts_1 = np.array([1000], dtype=np.int64) * (10 ** 6)
    message_sizes_1 = np.array([4], dtype=np.int64)
    values_1 = np.array([1, 2, 3, 4], dtype=np.float64)

    message_starts_2 = np.array([1001], dtype=np.int64) * (10 ** 6)
    message_sizes_2 = np.array([4], dtype=np.int64)
    values_2 = np.array([5, 6, 7, 8], dtype=np.float64)

    freq_nhz = int(freq_hz * (10 ** 9))

    merged_starts_expected = np.array([1000, 1001, 2000, 2001, 3000, 3001, 4000, 4001], dtype=np.int64) * (10 ** 6)
    merged_sizes_expected = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    merged_values_expected = np.array([1, 5, 2, 6, 3, 7, 4, 8], dtype=np.float64)

    merged_starts, merged_sizes, merged_values = merge_sorted_messages(
        message_starts_1, message_sizes_1, values_1,
        message_starts_2, message_sizes_2, values_2,
        freq_nhz)

    np.testing.assert_array_equal(merged_starts, merged_starts_expected)
    np.testing.assert_array_equal(merged_sizes, merged_sizes_expected)
    np.testing.assert_allclose(merged_values, merged_values_expected)


    ####

    freq_hz = 300

    message_starts_1 = np.array([315532800], dtype=np.int64) * (10 ** 9)
    message_sizes_1 = np.array([300_000], dtype=np.int64)
    values_1 = np.arange(0, 300_000, dtype=np.float64)

    # 10,000 years later
    message_starts_2 = np.array([317270000000], dtype=np.int64) * (10 ** 9)
    message_sizes_2 = np.array([300_000], dtype=np.int64)
    values_2 = np.arange(300_000, 600_000, dtype=np.float64)

    freq_nhz = freq_hz * (10 ** 9)

    merged_starts_expected = np.array([315532800, 317270000000], dtype=np.int64) * (10 ** 9)
    merged_sizes_expected = np.array([300_000, 300_000], dtype=np.int64)
    merged_values_expected = np.arange(0, 600_000, dtype=np.float64)

    merged_starts, merged_sizes, merged_values = merge_sorted_messages(
        message_starts_1, message_sizes_1, values_1,
        message_starts_2, message_sizes_2, values_2,
        freq_nhz)

    np.testing.assert_array_equal(merged_starts, merged_starts_expected)
    np.testing.assert_array_equal(merged_sizes, merged_sizes_expected)
    np.testing.assert_allclose(merged_values, merged_values_expected)

def test_merge_sorted_messages_with_overlap():
    message_starts_1 = np.array([0, 5 * 10**9], dtype=np.int64)
    message_sizes_1 = np.array([2, 2], dtype=np.int64)
    values_1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    message_starts_2 = np.array([5 * 10**9, 15 * 10**9], dtype=np.int64)
    message_sizes_2 = np.array([2, 2], dtype=np.int64)
    values_2 = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)

    freq_nhz = 10**9

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
