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
