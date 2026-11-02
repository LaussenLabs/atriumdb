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

uint64_t count_num_distinct_d64(const int64_t *arr, uint64_t arr_len)
{
    uint64_t num_distinct_elements = 0;

    uint64_t i;
    uint64_t j;
    for(i=0; i<arr_len; i++){
        for(j=0; j<i; j++){
            if(arr[i] == arr[j]){
                break;
            }
        }
        if(i == j){
            num_distinct_elements++;
        }
    }
    return num_distinct_elements;
}