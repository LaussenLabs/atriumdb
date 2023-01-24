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
