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
import numpy as np

from atriumdb.adb_functions import group_sorted_block_list, get_interval_list_from_ordered_timestamps, \
    group_headers_by_scale_factor


def test_group_sorted_block_list():
    blocks = [
        (1, 23, 46, 99, 0,   100, 1000, 2000, 4),  # num_values=4
        (2, 23, 46, 99, 100, 100, 2000, 3000, 3),  # num_values=3
        (3, 23, 46, 99, 200, 100, 3000, 4000, 5),  # num_values=5
        (4, 23, 46, 99, 300, 100, 4000, 5000, 2),  # num_values=2
        (5, 23, 46, 100, 0, 100, 5000, 6000, 6),  # num_values=6
    ]

    num_values_per_group = 10

    # Expected grouping:
    # - First group: blocks 1, 2, 3 (total num_values = 12)
    # - Second group: blocks 4, 5 (total num_values = 8)
    expected_groups = [
        [blocks[0], blocks[1], blocks[2]],
        [blocks[3], blocks[4]],
    ]

    result_groups = list(group_sorted_block_list(blocks, num_values_per_group))

    assert len(result_groups) == len(expected_groups), "Number of groups does not match expected."

    for i, group in enumerate(result_groups):
        assert group == expected_groups[i], f"Group {i+1} does not match expected."

    all_grouped_blocks = [block for group in result_groups for block in group]
    assert set(all_grouped_blocks) == set(blocks), "Some blocks are missing or duplicated in groups."

    for group in result_groups:
        indices = [blocks.index(block) for block in group]
        assert indices == sorted(indices), "Blocks within a group are not in original order."

    total_values_per_group = [sum(block[8] for block in group) for group in result_groups]
    expected_totals = [12, 8]
    assert total_values_per_group == expected_totals, "Total num_values per group do not match expected."


def test_get_time_regions():
    timestamps = np.array([0, 1, 2, 3, 4, 7, 8, 10, 12, 13, 14], dtype=np.int64)
    period_ns = 1
    expected = np.array([[0, 5], [7, 9], [10, 11], [12, 15]], dtype=np.int64)
    assert np.array_equal(get_interval_list_from_ordered_timestamps(timestamps, period_ns), expected)

    # Edge case: Empty array
    timestamps = np.array([], dtype=np.int64)
    expected = np.array([], dtype=np.int64)
    assert np.array_equal(get_interval_list_from_ordered_timestamps(timestamps, period_ns), expected)

    # Edge case: Single timestamp
    timestamps = np.array([100], dtype=np.int64)
    expected = np.array([[100, 101]], dtype=np.int64)
    assert np.array_equal(get_interval_list_from_ordered_timestamps(timestamps, period_ns), expected)

    # All timestamps equally spaced at exactly period_ns
    timestamps = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    period_ns = 1
    expected = np.array([[0, 5]], dtype=np.int64)
    assert np.array_equal(get_interval_list_from_ordered_timestamps(timestamps, period_ns), expected)

    # Timestamps with varying gaps
    timestamps = np.array([0, 2, 4, 7, 8, 15], dtype=np.int64)
    period_ns = 2
    expected = np.array([[0, 6], [7, 10], [15, 17]], dtype=np.int64)
    assert np.array_equal(get_interval_list_from_ordered_timestamps(timestamps, period_ns), expected)

    # Negative timestamps
    timestamps = np.array([-5, -4, -3, 0, 1, 10], dtype=np.int64)
    period_ns = 1
    expected = np.array([[-5, -2], [0, 2], [10, 11]], dtype=np.int64)
    assert np.array_equal(get_interval_list_from_ordered_timestamps(timestamps, period_ns), expected)

    # Large period_ns compared to time deltas
    timestamps = np.array([0, 100, 200, 300], dtype=np.int64)
    period_ns = 500
    expected = np.array([[0, 800]], dtype=np.int64)
    assert np.array_equal(get_interval_list_from_ordered_timestamps(timestamps, period_ns), expected)

    # Timestamps with duplicates
    timestamps = np.array([0, 0, 1, 1, 2, 3], dtype=np.int64)
    period_ns = 1
    expected = np.array([[0, 4]], dtype=np.int64)
    assert np.array_equal(get_interval_list_from_ordered_timestamps(timestamps, period_ns), expected)

    # Timestamps with large gaps
    timestamps = np.array([0, 1000, 2000, 3000], dtype=np.int64)
    period_ns = 500
    expected = np.array([[0, 500], [1000, 1500], [2000, 2500], [3000, 3500]], dtype=np.int64)
    assert np.array_equal(get_interval_list_from_ordered_timestamps(timestamps, period_ns), expected)


# Mock header class
class MockHeader:
    def __init__(self, scale_m, scale_b, num_vals):
        self.scale_m = scale_m
        self.scale_b = scale_b
        self.num_vals = num_vals

def test_group_headers_by_scale_factor():
    # Test case 1: Consecutive headers with the same scale factors
    headers1 = [
        MockHeader(1.0, 2.0, 3),
        MockHeader(1.0, 2.0, 2),
        MockHeader(1.0, 2.0, 4),
    ]
    times_array1 = list(range(9))  # times from 0 to 8
    values_array1 = list(range(100, 109))  # values from 100 to 108
    expected_groups1 = [
        (headers1, times_array1, values_array1)
    ]

    # Test case 2: Headers with different scale factors
    headers2 = [
        MockHeader(1.0, 2.0, 3),
        MockHeader(1.1, 2.0, 2),
        MockHeader(1.0, 2.0, 4),
    ]
    times_array2 = list(range(9))
    values_array2 = list(range(200, 209))
    expected_groups2 = [
        ([headers2[0]], times_array2[0:3], values_array2[0:3]),
        ([headers2[1]], times_array2[3:5], values_array2[3:5]),
        ([headers2[2]], times_array2[5:9], values_array2[5:9]),
    ]

    # Test case 3: Empty headers list
    headers3 = []
    times_array3 = []
    values_array3 = []
    expected_groups3 = []

    # Test case 4: Non-consecutive headers with same scale factors
    headers4 = [
        MockHeader(1.0, 2.0, 3),
        MockHeader(1.1, 2.0, 2),
        MockHeader(1.0, 2.0, 4),
        MockHeader(1.0, 2.0, 1),
    ]
    times_array4 = list(range(10))
    values_array4 = list(range(300, 310))
    expected_groups4 = [
        ([headers4[0]], times_array4[0:3], values_array4[0:3]),
        ([headers4[1]], times_array4[3:5], values_array4[3:5]),
        ([headers4[2], headers4[3]], times_array4[5:10], values_array4[5:10]),
    ]

    # Test case 5: All headers have different scale factors
    headers5 = [
        MockHeader(1.0, 2.0, 1),
        MockHeader(1.1, 2.1, 2),
        MockHeader(1.2, 2.2, 3),
    ]
    times_array5 = list(range(6))
    values_array5 = list(range(400, 406))
    expected_groups5 = [
        ([headers5[0]], times_array5[0:1], values_array5[0:1]),
        ([headers5[1]], times_array5[1:3], values_array5[1:3]),
        ([headers5[2]], times_array5[3:6], values_array5[3:6]),
    ]

    # Test case 6: Only one header
    headers6 = [MockHeader(1.0, 2.0, 3)]
    times_array6 = list(range(3))
    values_array6 = list(range(500, 503))
    expected_groups6 = [
        (headers6, times_array6, values_array6)
    ]

    # Test case 7: Headers with zero num_vals
    headers7 = [
        MockHeader(1.0, 2.0, 0),
        MockHeader(1.0, 2.0, 0),
    ]
    times_array7 = []
    values_array7 = []
    expected_groups7 = [
        (headers7, times_array7, values_array7)
    ]

    # Collect all test cases
    test_cases = [
        (headers1, times_array1, values_array1, expected_groups1),
        (headers2, times_array2, values_array2, expected_groups2),
        (headers3, times_array3, values_array3, expected_groups3),
        (headers4, times_array4, values_array4, expected_groups4),
        (headers5, times_array5, values_array5, expected_groups5),
        (headers6, times_array6, values_array6, expected_groups6),
        (headers7, times_array7, values_array7, expected_groups7),
    ]

    # Run tests
    for idx, (headers, times_array, values_array, expected_groups) in enumerate(test_cases, 1):
        result = list(group_headers_by_scale_factor(headers, times_array, values_array))
        assert len(result) == len(expected_groups), f"Test case {idx}: Number of groups does not match expected."

        for i, (group, times_slice, values_slice) in enumerate(result):
            expected_group, expected_times, expected_values = expected_groups[i]
            assert group == expected_group, f"Test case {idx}, group {i+1}: Headers do not match expected."
            assert times_slice == expected_times, f"Test case {idx}, group {i+1}: Times slice does not match expected."
            assert values_slice == expected_values, f"Test case {idx}, group {i+1}: Values slice does not match expected."

