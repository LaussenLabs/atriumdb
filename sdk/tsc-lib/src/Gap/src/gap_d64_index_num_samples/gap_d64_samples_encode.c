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
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

uint64_t gap_int64_samples_encode(const int64_t * time_data, int64_t * gap_array, uint64_t num_values,
                                  uint64_t period_ns)
{
    if(num_values > INT64_MAX){
        printf("num_values %" PRIu64 " unable to be represented in nanosecond gap format "
                                     "(max: %" PRIu64 ") on line %d, in file %s",
                num_values, INT64_MAX, __LINE__, __FILE__);
        exit(1);
    }
    int64_t period_ns_signed = (int64_t)period_ns;

    // Count how many gaps are in the data, and record them in the gap array.
    uint64_t num_gaps = 0;
    uint64_t i;
    int64_t delta;
    for(i=0; i<num_values-1; i++){
        delta = time_data[i+1] - time_data[i];
        if(delta != period_ns_signed){
            gap_array[num_gaps * 2] = (int64_t)(i + 1);
            gap_array[(num_gaps * 2) + 1] = (delta - period_ns_signed) / period_ns_signed;
            num_gaps++;
        }
    }

    return num_gaps;
}