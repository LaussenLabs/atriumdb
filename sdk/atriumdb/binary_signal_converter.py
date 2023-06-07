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
