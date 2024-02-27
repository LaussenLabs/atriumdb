import pytest

from atriumdb.windowing.windowing_functions import find_closest_measurement


@pytest.mark.parametrize("time,measurements,expected", [
    # Basic functionality
    (1483268400000000000, [('id1', 'patient1', 'weight', 3.3, 'kg', 1483264800000000000), ('id2', 'patient1', 'weight', 3.4, 'kg', 1483268400000000000)], ('id2', 'patient1', 'weight', 3.4, 'kg', 1483268400000000000)),
    # Empty list
    (1483268400000000000, [], None),
    # Single element
    (1483264800000000000, [('id1', 'patient1', 'weight', 3.3, 'kg', 1483264800000000000)], ('id1', 'patient1', 'weight', 3.3, 'kg', 1483264800000000000)),
    # Measurement exactly at the given time
    (1483268400000000001,
     [('id1', 'patient1', 'weight', 3.3, 'kg', 1483264800000000000), ('id2', 'patient1', 'weight', 3.4, 'kg', 1483268400000000000), ('id3', 'patient1', 'weight', 3.5, 'kg', 1483268400000000001)],
     ('id3', 'patient1', 'weight', 3.5, 'kg', 1483268400000000001)),
    # Measurement before and after the given time
    (1483268400000000001,
     [('id1', 'patient1', 'weight', 3.3, 'kg', 1483264800000000000), ('id2', 'patient1', 'weight', 3.4, 'kg', 1483268400000000000)],
     ('id2', 'patient1', 'weight', 3.4, 'kg', 1483268400000000000)),
    # All measurements are after the given time
    (1483264800000000000,
     [('id2', 'patient1', 'weight', 3.4, 'kg', 1483268400000000001), ('id3', 'patient1', 'weight', 3.5, 'kg', 1483268400000000002)],
     None),
    # All measurements are before the given time
    (1483268400000000002,
     [('id1', 'patient1', 'weight', 3.3, 'kg', 1483264800000000000), ('id2', 'patient1', 'weight', 3.4, 'kg', 1483268400000000001)],
     ('id2', 'patient1', 'weight', 3.4, 'kg', 1483268400000000001)),
    # Large list
    (100, [('id' + str(i), 'patient1', 'weight', i, 'kg', i) for i in range(1000)], ('id100', 'patient1', 'weight', 100, 'kg', 100)),
])
def test_find_closest_measurement(time, measurements, expected):
    assert find_closest_measurement(time, measurements) == expected
