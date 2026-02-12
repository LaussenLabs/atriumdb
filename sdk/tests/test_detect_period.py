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

from atriumdb.adb_functions import detect_period


class TestDetectPeriod:

    def test_basic_consistent_period(self):
        """Test a perfectly consistent integer sequence."""
        times = np.array([10, 20, 30, 40, 50])
        assert detect_period(times) == 10
        assert isinstance(detect_period(times), int)

    def test_float_period(self):
        """Test a sequence with floating point timestamps."""
        times = np.array([0.0, 0.5, 1.0, 1.5])
        assert detect_period(times) == 0.5
        assert isinstance(detect_period(times), float)

    def test_threshold_not_met_returns_best_effort(self):
        """Test that it returns the mode delta with a warning if threshold is not met."""
        # 4 deltas: [1, 2, 3, 1]. Mode is 1 (count=2). 2/4 = 0.5.
        # If we set threshold to 0.6, it should warn but still return 1 (the mode).
        times = np.array([0, 1, 3, 6, 7])
        with pytest.warns(UserWarning, match="Automatic period detection: no single time delta"):
            result = detect_period(times, threshold_ratio=0.6)
            assert result == 1

    def test_duplicates_trigger_warning_and_recursion(self):
        """Test that duplicate values trigger a warning and still find the correct period."""
        # Deltas: [0, 10, 0, 10, 0]. Mode is 0.
        # After removing duplicates: [0, 10, 20, 30], Delta is 10.
        times = np.array([0, 0, 10, 10, 20, 20, 30])

        with pytest.warns(UserWarning, match="Dominant delta is 0"):
            result = detect_period(times)
            assert result == 10

    def test_empty_array_returns_default(self):
        """Test that an empty array returns default 1_000_000_000."""
        times = np.array([])
        with pytest.warns(UserWarning, match="Cannot detect period from fewer than 2 timestamps"):
            result = detect_period(times)
            assert result == 1_000_000_000
            assert isinstance(result, int)

    def test_empty_int_array_returns_default(self):
        """Test that an empty int64 array returns default 1_000_000_000."""
        times = np.array([], dtype=np.int64)
        with pytest.warns(UserWarning, match="Cannot detect period from fewer than 2 timestamps"):
            result = detect_period(times)
            assert result == 1_000_000_000
            assert isinstance(result, int)

    def test_single_element_returns_default(self):
        """Test that a single integer element returns default 1_000_000_000."""
        times = np.array([10])
        with pytest.warns(UserWarning, match="Cannot detect period from fewer than 2 timestamps"):
            result = detect_period(times)
            assert result == 1_000_000_000
            assert isinstance(result, int)

    def test_single_element_float_returns_default(self):
        """Test that a single float element returns default 1_000_000_000."""
        times = np.array([10.5])
        with pytest.warns(UserWarning, match="Cannot detect period from fewer than 2 timestamps"):
            result = detect_period(times)
            assert result == 1_000_000_000
            assert isinstance(result, int)

    def test_noisy_data_passing_threshold(self):
        """Test that the function ignores outliers if the mode is strong enough."""
        # Deltas: [1, 1, 1, 5, 1]. Mode 1 count = 4. Total = 5. Ratio = 0.8.
        times = np.array([0, 1, 2, 3, 8, 9])
        assert detect_period(times, threshold_ratio=0.7) == 1

    def test_all_duplicates_after_recursion(self):
        """Test case where even after taking unique values, there aren't enough points."""
        # Becomes [0, 10] -> only 1 delta.
        times = np.array([0, 0, 0, 10, 10, 10])
        with pytest.warns(UserWarning):
            # The recursive call gets np.array([0, 10]), len is 2, 1 delta of 10.
            # 1/1 = 1.0, which is > 0.3. Should return 10.
            assert detect_period(times) == 10

    def test_two_identical_values(self):
        """Test with exactly two identical timestamps."""
        times = np.array([5, 5])
        # After unique: [5] -> insufficient data -> default
        with pytest.warns(UserWarning, match="Cannot detect period from fewer than 2 timestamps"):
            result = detect_period(times)
            assert result == 1_000_000_000

    def test_negative_period(self):
        """Test with timestamps in descending order (negative delta)."""
        times = np.array([50, 40, 30, 20, 10])
        result = detect_period(times)
        assert result == -10
        assert isinstance(result, int)

    def test_mixed_positive_negative_deltas(self):
        """Test with mixed positive and negative deltas."""
        # Deltas: [10, -5, 10, -5, 10]. Mode is 10 (count=3). 3/5 = 0.6.
        times = np.array([0, 10, 5, 15, 10, 20])
        result = detect_period(times, threshold_ratio=0.5)
        assert result == 10

    def test_very_small_float_period(self):
        """Test with very small floating point periods."""
        times = np.array([0.0, 0.001, 0.002, 0.003, 0.004])
        result = detect_period(times)
        assert result == 0.001
        assert isinstance(result, float)

    def test_large_integer_period(self):
        """Test with large integer periods (nanosecond scale)."""
        times = np.array([1000000000, 2000000000, 3000000000, 4000000000])
        result = detect_period(times)
        assert result == 1000000000
        assert isinstance(result, int)

    def test_exactly_at_threshold(self):
        """Test when the mode delta is exactly at the threshold."""
        # Deltas: [10, 10, 20]. Mode is 10 (count=2). 2/3 = 0.6667.
        times = np.array([0, 10, 20, 40])
        # Default threshold is 0.3, so this should pass without warning
        result = detect_period(times, threshold_ratio=0.3)
        assert result == 10

    def test_zero_delta_in_middle(self):
        """Test case where there's a zero delta in the middle of the sequence."""
        # Deltas: [10, 0, 10]. Mode could be 10 (count=2) or 0 (count=1).
        # Since 0 is not the mode, no duplicate warning should be raised.
        times = np.array([0, 10, 10, 20])
        result = detect_period(times)
        # Mode is 10 (count=2 out of 3), which is 0.6667 > 0.3 threshold
        assert result == 10

    def test_all_zeros_delta(self):
        """Test where all deltas are zero (all same value)."""
        times = np.array([5, 5, 5, 5])
        # Should get two warnings: first about dominant delta being 0, then about insufficient data
        with pytest.warns(UserWarning) as warning_list:
            result = detect_period(times)
            assert result == 1_000_000_000

        # Verify we got both expected warnings
        warning_messages = [str(w.message) for w in warning_list]
        assert any("Dominant delta is 0" in msg for msg in warning_messages)
        assert any("Cannot detect period from fewer than 2 timestamps" in msg for msg in warning_messages)

    def test_mostly_zeros_with_one_outlier(self):
        """Test where most deltas are zero with one outlier."""
        # Deltas: [0, 0, 0, 10]. Mode is 0 (count=3). 3/4 = 0.75 > 0.3.
        times = np.array([5, 5, 5, 5, 15])
        with pytest.warns(UserWarning, match="Dominant delta is 0"):
            result = detect_period(times)
            # After unique: [5, 15], delta is 10
            assert result == 10