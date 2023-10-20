import pytest

from atriumdb.intervals.difference import list_difference


def test_list_difference():
    assert list_difference([[0, 2], [3, 8], [10, 20]], [[2, 5], [6, 12], [20, 22]]) == [[0, 2], [5, 6], [12, 20]]
    assert list_difference([[0, 5]], [[2, 4]]) == [[0, 2], [4, 5]]
    assert list_difference([[1, 3], [4, 6], [7, 9]], [[2, 5], [8, 10]]) == [[1, 2], [5, 6], [7, 8]]
    assert list_difference([[0, 2]], [[3, 5]]) == [[0, 2]]
    assert list_difference([[1, 2]], [[2, 3]]) == [[1, 2]]
    assert list_difference([[0, 10]], [[1, 9]]) == [[0, 1], [9, 10]]
    assert list_difference([], [[0, 1], [2, 3]]) == []
    assert list_difference([[0, 1], [2, 3]], []) == [[0, 1], [2, 3]]
    assert list_difference([[1, 2]], [[1, 2]]) == []
    assert list_difference([[0, 1]], [[0, 1], [2, 3]]) == []
    assert list_difference([[1, 2]], [[0, 1]]) == [[1, 2]]
    assert list_difference([[0, 5]], [[1, 2], [3, 4]]) == [[0, 1], [2, 3], [4, 5]]
    assert list_difference([[0, 1], [3, 4], [6, 7]], [[1, 3], [4, 6]]) == [[0, 1], [3, 4], [6, 7]]
