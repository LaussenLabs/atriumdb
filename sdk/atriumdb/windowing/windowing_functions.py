import numpy as np
from bisect import bisect_right


def get_threshold_labels(sliced_labels, label_threshold=0.5):
    # Calculate the percentage of 1s for each label in each window
    percentages = np.mean(sliced_labels, axis=-1)
    # Apply threshold
    return (percentages > label_threshold).astype(int)


def find_closest_measurement(time, measurements):
    """
    Find the measurement with the time value closest, but less than or equal to the given time,
    directly using bisect on the list of tuples.

    :param time: The time (epoch timestamp) to find the closest measurement for.
    :param measurements: A list of tuples containing the measurement value, units, and epoch timestamp.
    :return: The tuple from the measurements list with the closest time less than or equal to the given time.
    """
    # Use bisect_right with a key function that extracts the timestamp part of the tuple
    idx = bisect_right(measurements, time, key=lambda x: x[5])

    if idx > 0:
        return measurements[idx - 1]
    else:
        return None
