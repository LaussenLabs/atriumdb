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

    def test_threshold_failure(self):
        """Test that it returns -1 if no single period meets the threshold_ratio."""
        # 4 deltas: [1, 2, 3, 1]. Mode is 1 (count=2). 2/4 = 0.5.
        # If we set threshold to 0.6, it should fail.
        times = np.array([0, 1, 3, 6, 7])
        assert detect_period(times, threshold_ratio=0.6) == -1

    def test_duplicates_trigger_warning_and_recursion(self):
        """Test that duplicate values trigger a warning and still find the correct period."""
        # Deltas: [0, 10, 0, 10, 0]. Mode is 0.
        # After removing duplicates: [0, 10, 20, 30], Delta is 10.
        times = np.array([0, 0, 10, 10, 20, 20, 30])

        with pytest.warns(UserWarning, match="Dominant delta is 0"):
            result = detect_period(times)
            assert result == 10

    @pytest.mark.parametrize("invalid_input", [
        np.array([]),  # Empty
        np.array([10]),  # Single element
    ])
    def test_insufficient_data(self, invalid_input):
        """Test that arrays with fewer than 2 elements return -1."""
        assert detect_period(invalid_input) == -1

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
            # The recursive call gets np.array([0, 10]), len is 2, but only 1 delta.
            # 1/1 = 1.0, which is > 0.3. Should return 10.
            assert detect_period(times) == 10