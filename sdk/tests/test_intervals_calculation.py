import numpy as np
import pytest
from atriumdb.adb_functions import find_continuous_intervals

@pytest.mark.parametrize("timestamps, period_ns, gap_tolerance_nano, expected", [
    # Case 1
    (np.array([0, 10, 20, 30, 60, 70, 80, 90, 105, 125, 126, 136, 152, 163], dtype=np.int64),
     10,
     5,
     np.array([
         [0, 40],
         [60, 115],
         [125, 146],
         [152, 173]
     ])),
    # Case 2
    (np.array([0, 11, 22, 33, 44, 55, 70, 81, 92, 103], dtype=np.int64),
     11,
     2,
     np.array([
         [0, 66],
         [70, 114]
     ])),
    # Case 3
    (np.array([0, 20, 40, 60, 100, 120, 140, 160], dtype=np.int64),
     20,
     10,
     np.array([
         [0, 80],
         [100, 180]
     ])),
    # Case 4
    (np.array([5, 15, 25, 50, 75, 100, 105, 110], dtype=np.int64),
     10,
     5,
     np.array([
         [5, 35],
         [50, 60],
         [75, 85],
         [100, 120]
     ])),
    # Case 5
    (np.array([0, 5, 10, 15, 25, 30, 35, 45, 50, 100, 150], dtype=np.int64),
     5,
     5,
     np.array([
         [0, 55],
         [100, 105],
         [150, 155]
     ]))
])
def test_find_continuous_intervals(timestamps, period_ns, gap_tolerance_nano, expected):
    output = find_continuous_intervals(timestamps, period_ns, gap_tolerance_nano)
    assert np.array_equal(output, expected), f"Failed for timestamps: {timestamps}, period_ns: {period_ns}, gap_tolerance_nano: {gap_tolerance_nano}. Output: {output}, Expected: {expected}"
