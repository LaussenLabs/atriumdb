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


def intervals_union(a, b):
    # Combine all intervals
    intervals = np.vstack((a, b))

    # Sort intervals by their start
    intervals = intervals[np.argsort(intervals[:, 0])]

    merged = [intervals[0].tolist()]
    for current in intervals:
        # if the list of merged intervals is empty or if the current interval does not overlap with the previous, append it
        if not merged or merged[-1][1] < current[0]:
            merged.append(current.tolist())
        # otherwise, there is overlap, so we merge the current and previous intervals.
        else:
            merged[-1][1] = max(merged[-1][1], current[1])

    return np.array(merged, dtype=np.int64)


def intervals_union_list(interval_list):
    interval_list = [interval for interval in interval_list if len(interval) > 0]
    if len(interval_list) == 0:
        return np.array([], dtype=np.int64)

    # Combine all intervals from the list
    intervals = np.vstack(interval_list)

    if intervals.size == 0:
        return np.array([], dtype=np.int64)

    # Sort intervals by their start
    intervals = intervals[np.argsort(intervals[:, 0])]

    merged = [intervals[0].tolist()]
    for current in intervals:
        # if the list of merged intervals is empty or if the current interval does not overlap with the previous, append it
        if not merged or merged[-1][1] < current[0]:
            merged.append(current.tolist())
        # otherwise, there is overlap, so we merge the current and previous intervals.
        else:
            merged[-1][1] = max(merged[-1][1], current[1])

    return np.array(merged, dtype=np.int64)
