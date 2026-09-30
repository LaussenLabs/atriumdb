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
// Created by Will Dixon on 2021-06-10.
//

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <distinct.h>

#include <entropy_p.h>

double entropy_get_d64(const int64_t *arr, uint64_t arr_len, void * buffer, size_t buffer_size)
{
    uint64_t num_distinct = count_num_distinct_d64(arr, arr_len);
    if(buffer_size < (num_distinct * 2 * sizeof(int64_t))){
        printf("Buffer too small, must be at least %" PRIu64 " bytes. On line %d in file %s\n",
               num_distinct * 2 * sizeof(int64_t), __LINE__, __FILE__);
        exit(1);
    }

    // Portion out the buffer memory.
    int64_t *distinct_elements = buffer;
    uint64_t *distinct_count = &(((uint64_t *)buffer)[num_distinct]);

    // Tally the distinct values.
    count_arr_distinct_d64(arr, arr_len, num_distinct, distinct_elements, distinct_count);

    // Return the Shannon Entropy.
    return entropy(distinct_count, num_distinct, arr_len);
}