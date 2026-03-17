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
import pytest

from atriumdb.intervals import Intervals


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestIntervalsInit:
    def test_from_list(self):
        iv = Intervals([[0, 10], [20, 30]])
        assert len(iv) == 2
        assert iv.interval_arr.dtype == np.int64

    def test_from_numpy(self):
        arr = np.array([[0, 10], [20, 30]], dtype=np.int64)
        iv = Intervals(arr)
        assert len(iv) == 2

    def test_empty_list(self):
        iv = Intervals([])
        assert iv.is_empty()
        assert len(iv) == 0

    def test_empty_numpy(self):
        iv = Intervals(np.array([], dtype=np.int64))
        assert iv.is_empty()
        assert len(iv) == 0

    def test_single_interval_list(self):
        iv = Intervals([[5, 15]])
        assert len(iv) == 1

    def test_rejects_float_list(self):
        with pytest.raises(TypeError):
            Intervals([[0.5, 1.5]])

    def test_rejects_float_numpy(self):
        with pytest.raises(TypeError):
            Intervals(np.array([[0.0, 1.0]]))

    def test_rejects_string(self):
        with pytest.raises(TypeError):
            Intervals("not an interval")

    def test_rejects_dict(self):
        with pytest.raises(TypeError):
            Intervals({"start": 0, "end": 10})

    def test_numpy_integer_subtypes(self):
        for dtype in [np.int32, np.int64, np.uint32]:
            iv = Intervals(np.array([[0, 10]], dtype=dtype))
            assert len(iv) == 1


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

class TestDuration:
    def test_single_interval(self):
        assert Intervals([[0, 10]]).duration() == 10

    def test_multiple_intervals(self):
        assert Intervals([[0, 10], [20, 25]]).duration() == 15

    def test_empty(self):
        assert Intervals([]).duration() == 0

    def test_zero_width_intervals(self):
        # An interval [5, 5) has zero duration
        assert Intervals([[5, 5]]).duration() == 0

    def test_large_timestamps(self):
        start = 1_700_000_000_000_000_000  # ~2023 in nanoseconds
        end = start + 10_000_000_000       # 10 seconds later
        assert Intervals([[start, end]]).duration() == 10_000_000_000


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------

class TestIsEmpty:
    def test_empty(self):
        assert Intervals([]).is_empty()

    def test_not_empty(self):
        assert not Intervals([[0, 1]]).is_empty()


# ---------------------------------------------------------------------------
# Intersection
# ---------------------------------------------------------------------------

class TestIntersection:
    def test_overlapping(self):
        a = Intervals([[0, 10], [20, 30]])
        b = Intervals([[5, 25]])
        result = a.intersection(b)
        expected = Intervals([[5, 10], [20, 25]])
        assert result == expected

    def test_no_overlap(self):
        a = Intervals([[0, 5]])
        b = Intervals([[10, 15]])
        result = a.intersection(b)
        assert result.is_empty()

    def test_identical(self):
        a = Intervals([[0, 10]])
        b = Intervals([[0, 10]])
        assert a.intersection(b) == a

    def test_one_empty(self):
        a = Intervals([[0, 10]])
        b = Intervals([])
        result = a.intersection(b)
        assert result.is_empty()

    def test_both_empty(self):
        result = Intervals([]).intersection(Intervals([]))
        assert result.is_empty()

    def test_partial_overlap_multiple(self):
        a = Intervals([[0, 10], [20, 30], [40, 50]])
        b = Intervals([[5, 45]])
        result = a.intersection(b)
        expected = Intervals([[5, 10], [20, 30], [40, 45]])
        assert result == expected

    def test_operator(self):
        a = Intervals([[0, 10]])
        b = Intervals([[5, 15]])
        assert (a & b) == a.intersection(b)


# ---------------------------------------------------------------------------
# Difference
# ---------------------------------------------------------------------------

class TestDifference:
    def test_basic(self):
        a = Intervals([[0, 10], [20, 30]])
        b = Intervals([[5, 25]])
        result = a.difference(b)
        expected = Intervals([[0, 5], [25, 30]])
        assert result == expected

    def test_no_overlap(self):
        a = Intervals([[0, 5]])
        b = Intervals([[10, 15]])
        result = a.difference(b)
        assert result == a

    def test_complete_removal(self):
        a = Intervals([[0, 10]])
        b = Intervals([[0, 10]])
        result = a.difference(b)
        assert result.is_empty()

    def test_subtract_empty(self):
        a = Intervals([[0, 10]])
        b = Intervals([])
        result = a.difference(b)
        assert result == a

    def test_empty_minus_something(self):
        a = Intervals([])
        b = Intervals([[0, 10]])
        result = a.difference(b)
        assert result.is_empty()

    def test_middle_punch(self):
        a = Intervals([[0, 10]])
        b = Intervals([[3, 7]])
        result = a.difference(b)
        expected = Intervals([[0, 3], [7, 10]])
        assert result == expected

    def test_multiple_punches(self):
        a = Intervals([[0, 20]])
        b = Intervals([[2, 5], [8, 12], [15, 18]])
        result = a.difference(b)
        expected = Intervals([[0, 2], [5, 8], [12, 15], [18, 20]])
        assert result == expected

    def test_operator(self):
        a = Intervals([[0, 10]])
        b = Intervals([[5, 15]])
        assert (a - b) == a.difference(b)


# ---------------------------------------------------------------------------
# Union
# ---------------------------------------------------------------------------

class TestUnion:
    def test_overlapping(self):
        a = Intervals([[0, 10]])
        b = Intervals([[5, 15]])
        result = a.union(b)
        expected = Intervals([[0, 15]])
        assert result == expected

    def test_adjacent(self):
        a = Intervals([[0, 10]])
        b = Intervals([[10, 20]])
        result = a.union(b)
        expected = Intervals([[0, 20]])
        assert result == expected

    def test_disjoint(self):
        a = Intervals([[0, 5]])
        b = Intervals([[10, 15]])
        result = a.union(b)
        expected = Intervals([[0, 5], [10, 15]])
        assert result == expected

    def test_with_gap_tolerance(self):
        a = Intervals([[0, 10]])
        b = Intervals([[12, 20]])
        # Without tolerance, two intervals
        assert len(a.union(b)) == 2
        # With tolerance of 2, they merge
        result = a.union(b, gap_tolerance_nano=2)
        expected = Intervals([[0, 20]])
        assert result == expected

    def test_one_empty(self):
        a = Intervals([[0, 10]])
        b = Intervals([])
        result = a.union(b)
        assert result == a

    def test_both_empty(self):
        result = Intervals([]).union(Intervals([]))
        assert result.is_empty()

    def test_operator(self):
        a = Intervals([[0, 10]])
        b = Intervals([[5, 15]])
        assert (a | b) == a.union(b)


# ---------------------------------------------------------------------------
# contains
# ---------------------------------------------------------------------------

class TestContains:
    def test_inside(self):
        iv = Intervals([[0, 10], [20, 30]])
        assert iv.contains(0)
        assert iv.contains(5)
        assert iv.contains(25)

    def test_at_boundary(self):
        iv = Intervals([[0, 10]])
        assert iv.contains(0)       # start is inclusive
        assert not iv.contains(10)  # end is exclusive

    def test_outside(self):
        iv = Intervals([[0, 10]])
        assert not iv.contains(-1)
        assert not iv.contains(15)

    def test_between_intervals(self):
        iv = Intervals([[0, 10], [20, 30]])
        assert not iv.contains(15)

    def test_empty(self):
        assert not Intervals([]).contains(5)

    def test_in_operator(self):
        iv = Intervals([[0, 10]])
        assert 5 in iv
        assert 10 not in iv


# ---------------------------------------------------------------------------
# gaps
# ---------------------------------------------------------------------------

class TestGaps:
    def test_basic(self):
        iv = Intervals([[0, 10], [20, 30], [50, 60]])
        result = iv.gaps()
        expected = Intervals([[10, 20], [30, 50]])
        assert result == expected

    def test_single_interval(self):
        assert Intervals([[0, 10]]).gaps().is_empty()

    def test_empty(self):
        assert Intervals([]).gaps().is_empty()

    def test_adjacent_no_gaps(self):
        iv = Intervals([[0, 10], [10, 20]])
        result = iv.gaps()
        # [10, 10) is a zero-width gap
        expected = Intervals([[10, 10]])
        assert result == expected


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestLen:
    def test_zero(self):
        assert len(Intervals([])) == 0

    def test_one(self):
        assert len(Intervals([[0, 1]])) == 1

    def test_multiple(self):
        assert len(Intervals([[0, 1], [2, 3], [4, 5]])) == 3


# ---------------------------------------------------------------------------
# __eq__
# ---------------------------------------------------------------------------

class TestEquality:
    def test_equal(self):
        assert Intervals([[0, 10]]) == Intervals([[0, 10]])

    def test_not_equal(self):
        assert not (Intervals([[0, 10]]) == Intervals([[0, 11]]))

    def test_different_lengths(self):
        assert not (Intervals([[0, 10]]) == Intervals([[0, 10], [20, 30]]))

    def test_non_intervals_returns_not_implemented(self):
        iv = Intervals([[0, 10]])
        assert iv.__eq__("not intervals") is NotImplemented


# ---------------------------------------------------------------------------
# __repr__ and __str__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr(self):
        iv = Intervals([[0, 10], [20, 30]])
        assert repr(iv) == "Intervals([[0, 10], [20, 30]])"

    def test_repr_empty(self):
        assert repr(Intervals([])) == "Intervals([])"

    def test_str(self):
        iv = Intervals([[0, 10]])
        # __str__ delegates to the numpy array
        assert "0" in str(iv)
        assert "10" in str(iv)


# ---------------------------------------------------------------------------
# __iter__
# ---------------------------------------------------------------------------

class TestIter:
    def test_iterate(self):
        iv = Intervals([[0, 10], [20, 30]])
        result = list(iv)
        assert result == [(0, 10), (20, 30)]

    def test_iterate_empty(self):
        assert list(Intervals([])) == []

    def test_unpack(self):
        iv = Intervals([[0, 10], [20, 30]])
        for start, end in iv:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert end > start


# ---------------------------------------------------------------------------
# Combined operations / identity laws
# ---------------------------------------------------------------------------

class TestSetIdentities:
    """Verify basic set algebra identities hold."""

    def test_intersection_commutative(self):
        a = Intervals([[0, 10], [20, 30]])
        b = Intervals([[5, 25]])
        assert a.intersection(b) == b.intersection(a)

    def test_union_commutative(self):
        a = Intervals([[0, 10]])
        b = Intervals([[5, 15]])
        assert a.union(b) == b.union(a)

    def test_difference_and_intersection_partition(self):
        """A = (A - B) | (A & B)"""
        a = Intervals([[0, 20]])
        b = Intervals([[5, 15]])
        diff = a.difference(b)
        inter = a.intersection(b)
        reconstructed = diff.union(inter)
        assert reconstructed == a

    def test_union_with_self(self):
        a = Intervals([[0, 10], [20, 30]])
        assert a.union(a) == a

    def test_intersection_with_self(self):
        a = Intervals([[0, 10], [20, 30]])
        assert a.intersection(a) == a

    def test_difference_with_self(self):
        a = Intervals([[0, 10], [20, 30]])
        assert a.difference(a).is_empty()

    def test_duration_of_partition(self):
        """duration(A - B) + duration(A & B) == duration(A) when B ⊂ A"""
        a = Intervals([[0, 100]])
        b = Intervals([[20, 50]])
        assert a.difference(b).duration() + a.intersection(b).duration() == a.duration()