import numpy as np

from atriumdb.intervals.compact import reverse_compact_list
from atriumdb.intervals.intersection import list_intersection


class Intervals:
    def __init__(self, interval_list):
        if isinstance(interval_list, np.ndarray):
            if not np.issubdtype(interval_list.dtype, np.integer):
                raise TypeError(f"dtype must be integers, found {interval_list.dtype}.")
            self.interval_arr = interval_list

        elif isinstance(interval_list, list):
            if not all([all([isinstance(n, int) or isinstance(n, np.integer) for n in sub_li])
                        for sub_li in interval_list]):
                raise TypeError("All values of interval_list must be integers.")
            self.interval_arr = np.array(interval_list, dtype=np.int64)

        else:
            raise TypeError("interval_list must be a list or numpy array")

    def intersection(self, other):
        intersection_interval = list_intersection(self.interval_arr, other.interval_arr)
        return self.__class__(np.array(reverse_compact_list(intersection_interval), dtype=np.int64))

    def difference(self, other):
        pass

    def duration(self):
        if len(self) == 0:
            return 0
        return np.sum(self.interval_arr.T[1] - self.interval_arr.T[0])

    def is_empty(self):
        return self.interval_arr.size == 0

    def __len__(self):
        return len(self.interval_arr)

    def __str__(self):
        return str(self.interval_arr)

    def __eq__(self, other):
        return np.array_equal(self.interval_arr, other.interval_arr)
