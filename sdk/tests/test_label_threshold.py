import numpy as np

from atriumdb.windowing.windowing_functions import get_threshold_labels


def test_threshold_labels():
    # Simple 3x3x4 matrix
    sliced_labels = np.array([
        [[1, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 0]],
        [[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]],
        [[1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 1, 1]]
    ])

    expected = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 1]
    ])

    result = get_threshold_labels(sliced_labels)
    assert np.array_equal(result, expected), f"Expected:\n{expected}\nGot:\n{result}"