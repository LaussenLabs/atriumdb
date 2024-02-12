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

gap_modes = ["samples", "duration"]


def calc_gap_block_start(gap_data, num_vals, freq_nhz, val_offset, cur_gap, mode):
    """
    One concern that needs to be checked is if the blocks are divided over a gap. The way I think it should work
    is that the start_time (last blocks cur_time/start_n) should be without the gap size and then a gap and index 0
    will be saved, not ideal, but easier than designing around this edge case.
    :param gap_data: A list/array of ints, the even indexed ints represent the index of the end of the gap and the odd
    indexed ints represent the duration of the gap either in nanosecond or number of samples (see mode)
    :param num_vals:
    :param freq_nhz:
    :param val_offset:
    :param cur_gap:
    :param mode: The method of gap duration representation.
    :return num_gaps, elapsed_time, period_ns:
    """
    num_gaps, elapsed_time = 0, 0
    period_ns = freq_nhz_to_period_ns(freq_nhz)

    while 2 * cur_gap < gap_data.size and gap_data[2 * cur_gap] < val_offset + num_vals:
        if mode == "samples":
            elapsed_time += gap_data[(2 * cur_gap) + 1] * period_ns
            elapsed_time += calc_time_by_freq(freq_nhz, gap_data[(2 * cur_gap) + 1])
        elif mode == "duration":
            elapsed_time += gap_data[(2 * cur_gap) + 1]
        else:
            raise ValueError("mode {} not a valid mode, use one of {}.".format(mode, gap_modes))

        # Change the gap data to be with respect to block start, not data start.
        gap_data[2 * cur_gap] -= val_offset

        num_gaps += 1
        cur_gap += 1

    # elapsed_time += period_ns * num_vals
    elapsed_time += calc_time_by_freq(freq_nhz, num_vals)

    return num_gaps, elapsed_time, period_ns


def freq_nhz_to_period_ns(freq_nhz):
    """

    :param freq_nhz:
    :return period_ns: The period in nano seconds.
    """
    return int((10 ** 18) // freq_nhz)


def calc_time_by_freq(freq_nhz, num_samples):
    return (int(num_samples) * (10 ** 18)) // freq_nhz
