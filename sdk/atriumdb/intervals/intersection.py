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

def list_intersection(first, second):
    result = []
    i, j = 0, 0

    while i < len(first) and j < len(second):
        inter_list = [max(first[i][0], second[j][0]), min(first[i][1], second[j][1])]
        if inter_list[0] < inter_list[1]:
            if result and result[-1][-1] >= inter_list[0]:
                result[-1][-1] = inter_list[1]
            else:
                result.append(inter_list)

        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1

    return result
