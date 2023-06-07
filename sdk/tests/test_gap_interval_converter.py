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

import numpy as np
import unittest

from atriumdb.block import convert_gap_array_to_intervals, convert_intervals_to_gap_array


class MyTestCase(unittest.TestCase):
    def test_simple(self):
        nano = 10 ** 9
        start_time = 420 * nano
        freq_nhz = 500 * (10 ** 6)
        num_values = 100

        gap_arr = \
            np.array([[3, 40 * nano], [12, 100 * nano], [98, 20 * nano]], dtype=np.int64)

        expected_interval_array = \
            np.array([[start_time, 426 * nano, 3],
                      [466 * nano, 484 * nano, 9],
                      [584 * nano, 756 * nano, 86],
                      [776 * nano, 780 * nano, 2]], dtype=np.int64)

        calculated_interval_array = convert_gap_array_to_intervals(start_time, gap_arr, num_values, freq_nhz)

        self.assertTrue(np.array_equal(expected_interval_array, calculated_interval_array))

        re_calculated_gap_array = convert_intervals_to_gap_array(calculated_interval_array)
        self.assertTrue(np.array_equal(re_calculated_gap_array, gap_arr))

        re_calculated_gap_array = convert_intervals_to_gap_array(expected_interval_array)
        self.assertTrue(np.array_equal(re_calculated_gap_array, gap_arr))


if __name__ == '__main__':
    unittest.main()
