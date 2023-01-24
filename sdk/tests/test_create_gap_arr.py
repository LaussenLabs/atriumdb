import numpy as np

from atriumdb import create_gap_arr
from atriumdb.block import create_gap_arr_fast

from time import perf_counter


def test_create_gap_arr():
    period_ns = 2_000_000
    freq_nhz = (10 ** 18) // period_ns
    time_arr = np.arange(1_000_000_000, 1_281_000_000_000, period_ns)
    expected_gap_arr = np.array([[4, 100_000_000_000], [130, 450_000_000_000], [671, 123_000_000_000]])

    for index, gap in expected_gap_arr:
        time_arr[index:] += gap

    assert period_ns == ((10 ** 18) * 1) // freq_nhz

    start_1 = perf_counter()
    assert np.array_equal(np.array(create_gap_arr(time_arr, 1, freq_nhz)), expected_gap_arr.flatten())
    end_1 = perf_counter()

    start_2 = perf_counter()
    assert np.array_equal(create_gap_arr_fast(time_arr, 1, freq_nhz), expected_gap_arr.flatten())
    end_2 = perf_counter()

    print("Slow Method", end_1 - start_1)
    print("Fast Method", end_2 - start_2)
