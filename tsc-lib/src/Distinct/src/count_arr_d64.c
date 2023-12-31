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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void count_arr_distinct_d64(const int64_t *arr, uint64_t arr_len, uint64_t num_distinct,
                                int64_t *distinct_elements, uint64_t *distinct_count)
{
    uint64_t cur_distinct_elements = 0;

    // Initialize counts to zero.
    uint64_t i;
    for(i=0; i<num_distinct; i++){
        distinct_count[i] = 0;
    }

    uint64_t j;
    for(i=0; i<arr_len; i++){
        for(j=0; j<cur_distinct_elements; j++){
            // Add to the count for that number.
            if(arr[i] == distinct_elements[j]){
                distinct_count[j]++;
                break;
            }
        }
        // Check to make sure I don't misunderstand my own logic. Remove once thoroughly tested.
        if(j == num_distinct){
            printf("I guess j does go that high. Your Logic is Flawed Mr. Spock. "
                   "On line %d in file %s\n", __LINE__, __FILE__);
            exit(1);
        }
        // Cheeky `if` Logic: Remove as soon as someone doesn't fully understand it.
        if(!distinct_count[j]){
            // if a spot for that number doesn't exist, make one!
            distinct_elements[cur_distinct_elements] = arr[i];
            distinct_count[cur_distinct_elements]++;
            cur_distinct_elements++;
        }
    }
}