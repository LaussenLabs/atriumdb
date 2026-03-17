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

from atriumdb.intervals.compact import reverse_compact_list
from atriumdb.intervals.difference import list_difference
from atriumdb.intervals.intersection import list_intersection
from atriumdb.intervals.union import intervals_union_list


class Intervals:
    """A class representing a collection of non-overlapping half-open time intervals ``[start, end)``.

    Each interval is stored as a row in a 2D NumPy array with columns ``[start, end]``.
    All values are integers (typically nanosecond-precision epoch timestamps).

    The class provides set-like operations (intersection, union, difference) as well
    as convenience methods for querying duration, emptiness, and containment.

    :param interval_list: A 2D structure of ``[start, end]`` pairs. Accepts a NumPy
        array with an integer dtype or a Python list of two-element integer lists.
    :type interval_list: numpy.ndarray or list

    :raises TypeError: If *interval_list* is not a list or NumPy array, or if the
        values are not integers.

    Example
    -------
    >>> from atriumdb.intervals import Intervals
    >>> iv = Intervals([[0, 10], [20, 30]])
    >>> len(iv)
    2
    >>> iv.duration()
    20
    """

    def __init__(self, interval_list):
        if isinstance(interval_list, np.ndarray):
            if interval_list.size == 0:
                self.interval_arr = np.array([], dtype=np.int64).reshape(0, 2)
                return
            if not np.issubdtype(interval_list.dtype, np.integer):
                raise TypeError(f"dtype must be integers, found {interval_list.dtype}.")
            self.interval_arr = interval_list.reshape(-1, 2) if interval_list.ndim == 1 else interval_list

        elif isinstance(interval_list, list):
            if len(interval_list) == 0:
                self.interval_arr = np.array([], dtype=np.int64).reshape(0, 2)
                return
            if not all(all(isinstance(n, (int, np.integer)) for n in sub_li)
                       for sub_li in interval_list):
                raise TypeError("All values of interval_list must be integers.")
            self.interval_arr = np.array(interval_list, dtype=np.int64)

        else:
            raise TypeError("interval_list must be a list or numpy array")

    # ---- Set operations ----

    def intersection(self, other):
        """Return a new :class:`Intervals` containing only the time ranges present in both *self* and *other*.

        :param other: The intervals to intersect with.
        :type other: Intervals
        :return: The intersection of the two interval sets.
        :rtype: Intervals

        Example
        -------
        >>> a = Intervals([[0, 10], [20, 30]])
        >>> b = Intervals([[5, 25]])
        >>> a.intersection(b)
        Intervals([[5, 10], [20, 25]])
        """
        intersection_interval = list_intersection(self.interval_arr.tolist(), other.interval_arr.tolist())
        return self.__class__(np.array(reverse_compact_list(intersection_interval), dtype=np.int64))

    def difference(self, other):
        """Return a new :class:`Intervals` containing the time ranges present in *self* but not in *other*.

        :param other: The intervals to subtract.
        :type other: Intervals
        :return: The set difference (*self* minus *other*).
        :rtype: Intervals

        Example
        -------
        >>> a = Intervals([[0, 10], [20, 30]])
        >>> b = Intervals([[5, 25]])
        >>> a.difference(b)
        Intervals([[0, 5], [25, 30]])
        """
        diff = list_difference(self.interval_arr.tolist(), other.interval_arr.tolist())
        if len(diff) == 0:
            return self.__class__(np.array([], dtype=np.int64).reshape(0, 2))
        return self.__class__(np.array(diff, dtype=np.int64))

    def union(self, other, gap_tolerance_nano=0):
        """Return a new :class:`Intervals` containing all time ranges present in either *self* or *other*.

        Overlapping or adjacent intervals are merged. An optional *gap_tolerance_nano*
        allows merging intervals separated by a gap smaller than the tolerance.

        :param other: The intervals to unite with.
        :type other: Intervals
        :param int gap_tolerance_nano: Maximum gap (in nanoseconds) between two
            intervals that should still be merged. Defaults to ``0``.
        :return: The union of the two interval sets.
        :rtype: Intervals

        Example
        -------
        >>> a = Intervals([[0, 10]])
        >>> b = Intervals([[10, 20]])
        >>> a.union(b)
        Intervals([[0, 20]])
        """
        combined = intervals_union_list(
            [self.interval_arr, other.interval_arr],
            gap_tolerance_nano=gap_tolerance_nano,
        )
        if combined.size == 0:
            return self.__class__(np.array([], dtype=np.int64).reshape(0, 2))
        return self.__class__(combined)

    # ---- Queries ----

    def duration(self):
        """Return the total duration covered by all intervals.

        :return: Sum of ``(end - start)`` for every interval. Returns ``0`` if empty.
        :rtype: int

        Example
        -------
        >>> Intervals([[0, 10], [20, 25]]).duration()
        15
        """
        if len(self) == 0:
            return 0
        return int(np.sum(self.interval_arr[:, 1] - self.interval_arr[:, 0]))

    def is_empty(self):
        """Return ``True`` if there are no intervals.

        :rtype: bool
        """
        return self.interval_arr.size == 0

    def contains(self, timestamp):
        """Return ``True`` if *timestamp* falls within any interval (half-open: ``[start, end)``).

        :param int timestamp: The point in time to test.
        :rtype: bool

        Example
        -------
        >>> iv = Intervals([[0, 10], [20, 30]])
        >>> iv.contains(5)
        True
        >>> iv.contains(15)
        False
        """
        if self.is_empty():
            return False
        starts = self.interval_arr[:, 0]
        ends = self.interval_arr[:, 1]
        return bool(np.any((starts <= timestamp) & (timestamp < ends)))

    def gaps(self):
        """Return a new :class:`Intervals` representing the gaps between consecutive intervals.

        :return: An :class:`Intervals` object where each interval is a gap between
            two consecutive intervals in *self*. Empty if there are fewer than two
            intervals.
        :rtype: Intervals

        Example
        -------
        >>> Intervals([[0, 10], [20, 30], [50, 60]]).gaps()
        Intervals([[10, 20], [30, 50]])
        """
        if len(self) < 2:
            return self.__class__(np.array([], dtype=np.int64).reshape(0, 2))
        gap_starts = self.interval_arr[:-1, 1]
        gap_ends = self.interval_arr[1:, 0]
        gap_arr = np.column_stack([gap_starts, gap_ends])
        return self.__class__(gap_arr)

    # ---- Dunder methods ----

    def __len__(self):
        return self.interval_arr.shape[0] if self.interval_arr.ndim == 2 else 0

    def __str__(self):
        return str(self.interval_arr)

    def __repr__(self):
        if self.is_empty():
            return "Intervals([])"
        return f"Intervals({self.interval_arr.tolist()})"

    def __eq__(self, other):
        if not isinstance(other, Intervals):
            return NotImplemented
        return np.array_equal(self.interval_arr, other.interval_arr)

    def __contains__(self, timestamp):
        """Support the ``in`` operator for point-in-interval checks.

        >>> 5 in Intervals([[0, 10]])
        True
        """
        return self.contains(timestamp)

    def __iter__(self):
        """Iterate over individual ``(start, end)`` tuples.

        >>> list(Intervals([[0, 10], [20, 30]]))
        [(0, 10), (20, 30)]
        """
        for row in self.interval_arr:
            yield (int(row[0]), int(row[1]))

    def __and__(self, other):
        """``a & b`` is equivalent to ``a.intersection(b)``."""
        return self.intersection(other)

    def __or__(self, other):
        """``a | b`` is equivalent to ``a.union(b)``."""
        return self.union(other)

    def __sub__(self, other):
        """``a - b`` is equivalent to ``a.difference(b)``."""
        return self.difference(other)