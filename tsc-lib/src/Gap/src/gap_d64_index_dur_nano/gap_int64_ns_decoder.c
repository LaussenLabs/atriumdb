/*
    AtriumDB is a timeseries database software designed to best handle the unique features and
    challenges that arise from clinical waveform data.
        Copyright (C) 2023  The Hospital for Sick Children

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <https:www.gnu.org/licenses/>.
*/

//
// Created by Will Dixon on 2021-05-25.
//
#include <freq_period_converter.h>
#include <stdint.h>


void gap_int64_ns_gap_array_decode(const int64_t * gap_array, int64_t * time_data, uint64_t num_values, uint64_t num_gaps,
                                   int64_t start_time_ns, uint64_t freq_nhz)
{
    // This function requires `time_data` to be initialized to all zeros (like by calloc).

    // Calculate the period of what would be continuous data.
    int64_t period_ns = (int64_t)uint64_nhz_freq_to_uint64_ns_period(freq_nhz);

    // Set the start time.
    time_data[0] = start_time_ns;

    // Place the jumps.
    uint64_t i;
    for(i=0; i<num_gaps; i++){
        // gap_array[2 * i] contains the index of the jump, and gap_array[(2 * i) + 1] contains the magnitude.
        time_data[gap_array[2 * i]] = gap_array[(2 * i) + 1];
    }

    // Place the continuous timestamps.
    for(i=0; i<num_values-1; i++){
        time_data[i+1] += time_data[i] + period_ns;
    }
}
