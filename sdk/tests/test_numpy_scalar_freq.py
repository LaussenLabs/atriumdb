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

"""Frequency arguments may arrive as numpy scalars, not just Python ints.

Nanosecond duration math multiplies the sample count by 10**18, which exceeds int64 for
even modest writes. Under NumPy 2 (NEP 50) an oversized Python int combined with an
np.int64 raises OverflowError instead of promoting, so every such expression must
narrow the frequency to a Python int first. NumPy 1 promoted silently, which is why
this class of defect stayed invisible until the test image moved to NumPy 2.
"""

import numpy as np
import pytest

from atriumdb.adb_functions import (
    _calc_end_time_from_gap_data,
    _message_size_freq_to_duration_ns,
    _message_size_to_duration_ns,
    reconstruct_messages,
)

FREQ_NHZ = 500_000_000_000
# Large enough that num_values * 10**18 overflows int64 (int64 max ~9.22e18).
BIG_NUM_VALUES = 650_000


def _both_freq_types():
    return [pytest.param(FREQ_NHZ, id="python_int"),
            pytest.param(np.int64(FREQ_NHZ), id="np_int64")]


@pytest.mark.parametrize("freq", _both_freq_types())
def test_calc_end_time_from_gap_data_accepts_numpy_freq(freq):
    gap_array = np.array([1200, 12_984_000_000, 2711, 17_880_000_000], dtype=np.int64)
    result = _calc_end_time_from_gap_data(
        values_size=BIG_NUM_VALUES, gap_array=gap_array, start_time=0, freq_nhz=freq)
    assert result == 1_330_864_000_000


@pytest.mark.parametrize("freq", _both_freq_types())
def test_message_size_duration_helpers_accept_numpy_freq(freq):
    expected = ((10 ** 18) * BIG_NUM_VALUES) // FREQ_NHZ
    assert _message_size_freq_to_duration_ns(BIG_NUM_VALUES, freq) == expected
    assert _message_size_to_duration_ns(BIG_NUM_VALUES, freq) == expected


@pytest.mark.parametrize("freq", _both_freq_types())
def test_reconstruct_messages_accepts_numpy_freq(freq):
    gap_data_array = np.array([BIG_NUM_VALUES // 2, 1_000_000_000], dtype=np.int64)
    starts, sizes = reconstruct_messages(
        start_time_nano_epoch=0, gap_data_array=gap_data_array,
        freq_nhz=freq, num_values=BIG_NUM_VALUES)
    assert starts.shape == sizes.shape
    assert int(sizes.sum()) == BIG_NUM_VALUES


def test_numpy_and_python_freq_agree_exactly():
    """The coercion must not change any result -- only widen accepted input types."""
    gap_array = np.array([1200, 12_984_000_000, 2711, 17_880_000_000], dtype=np.int64)
    py = _calc_end_time_from_gap_data(
        values_size=BIG_NUM_VALUES, gap_array=gap_array, start_time=0, freq_nhz=FREQ_NHZ)
    npy = _calc_end_time_from_gap_data(
        values_size=BIG_NUM_VALUES, gap_array=gap_array, start_time=0,
        freq_nhz=np.int64(FREQ_NHZ))
    assert py == npy
