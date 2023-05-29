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

import unittest

from atriumdb.interval_overlap import is_interval_overlap, is_interval_overlap_2


class TestIsIntervalOverlap(unittest.TestCase):
    def test_is_interval_overlap(self):
        self.assertEqual(is_interval_overlap([(1, 3), (6, 10), (3, 5), (15, 20), (12, 13)]), False)
        self.assertEqual(is_interval_overlap([(1, 3), (6, 10), (3, 5), (15, 20), (12, 13), (19, 22)]), True)
        self.assertRaises(ValueError, is_interval_overlap, [])

    def test_is_interval_overlap_2(self):
        self.assertEqual(is_interval_overlap_2([(1, 3), (6, 10), (3, 5), (15, 20), (12, 13)]), False)
        self.assertEqual(is_interval_overlap_2([(1, 3), (6, 10), (3, 5), (15, 20), (12, 13), (19, 22)]), True)
        self.assertRaises(ValueError, is_interval_overlap_2, [])


if __name__ == '__main__':
    unittest.main()
