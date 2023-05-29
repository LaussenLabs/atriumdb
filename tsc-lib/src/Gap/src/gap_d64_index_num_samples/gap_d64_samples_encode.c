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
// Created by Will Dixon on 2021-06-09.
//
#include <freq_period_converter.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

uint64_t gap_int64_samples_encode(const int64_t * time_data, int64_t * gap_array, uint64_t num_values,
                                  uint64_t freq_nhz)
{
    // This function isn't going to work correctly if the period corresponding to the frequency isn't
    // an integer number of nanoseconds.

    if(num_values > INT64_MAX){
        printf("num_values %" PRIu64 " unable to be represented in nanosecond gap format "
               "(max: %" PRIu64 ") on line %d, in file %s",
               num_values, INT64_MAX, __LINE__, __FILE__);
        exit(1);
    }

    // Calculate the period of what would be continuous data.
    int64_t period_ns = (int64_t)uint64_nhz_freq_to_uint64_ns_period(freq_nhz);

    // Count how many gaps are in the data, and record them in the gap array.
    uint64_t num_gaps = 0;
    uint64_t i;
    int64_t delta;
    for(i=0; i<num_values-1; i++){
        delta = time_data[i+1] - time_data[i];
        if(delta != period_ns){
            gap_array[num_gaps * 2] = (int64_t)(i + 1);
            gap_array[(num_gaps * 2) + 1] = (delta - period_ns) / period_ns;
            num_gaps++;
        }
    }

    // Return the number of gaps.
    return num_gaps;
}