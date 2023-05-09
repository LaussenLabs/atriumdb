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
// Created by Will Dixon on 2021-06-13.
//
#include <stdint.h>


typedef struct bounds_d64{
    int64_t max;
    int64_t min;
}bounds_d64_t;

uint8_t get_bytes_per_value(bounds_d64_t bounds, uint8_t min_size);
bounds_d64_t get_bounds(int64_t * arr, uint64_t num_values);

uint8_t size_find_best_d64(int64_t * arr, uint64_t num_values, uint8_t min_size)
{
    bounds_d64_t bounds = get_bounds(arr, num_values);
    uint8_t smallest_size = get_bytes_per_value(bounds, min_size);
    return smallest_size;
}


bounds_d64_t get_bounds(int64_t * arr, uint64_t num_values)
{
    bounds_d64_t result;
    result.max = INT64_MIN;
    result.min = INT64_MAX;

    uint64_t i;
    for(i=0; i<num_values; i++){
        if(arr[i] > result.max){
            result.max = arr[i];
        }
        if(arr[i] < result.min){
            result.min = arr[i];
        }
    }
    return result;
}

uint8_t get_bytes_per_value(bounds_d64_t bounds, uint8_t min_size)
{
    if (min_size <= 1 && -128 <= bounds.min && bounds.max <= 127)
        return 1;
    if (min_size <= 2 && -32768 <= bounds.min && bounds.max <= 32767)
        return 2;
    if (min_size <= 4 && -2147483648 <= bounds.min && bounds.max <= 2147483647)
        return 4;
    return 8;
}
