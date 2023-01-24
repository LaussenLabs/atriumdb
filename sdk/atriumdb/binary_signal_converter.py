import numpy as np


def indices_to_signal(array_size, indices):
    """
    A function that takes an array size an array of indices and converts it into a binary signal.
    """
    signal = np.zeros(array_size, dtype=np.int64)
    signal[indices] = 1
    return signal


def signal_to_indices(signal):
    """
    A function that takes a binary signal and converts it into a list of indices.
    """
    indices = np.where(signal == 1)[0]
    return indices
