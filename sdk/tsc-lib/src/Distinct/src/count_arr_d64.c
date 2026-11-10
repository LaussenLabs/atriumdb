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
#include <string.h>

#include <distinct.h>

// A power of two at least twice arr_len, so the table is at most half full.
uint64_t distinct_table_size_d64(uint64_t arr_len)
{
    uint64_t table_size = 16;
    while(table_size < 2 * arr_len){
        table_size <<= 1;
    }
    return table_size;
}

// MurmurHash3 64-bit finalizer.
static uint64_t hash_d64(int64_t value)
{
    uint64_t x = (uint64_t)value;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

uint64_t count_arr_distinct_d64(const int64_t *arr, uint64_t arr_len, int64_t *distinct_elements,
                                uint64_t *distinct_count, uint64_t *table)
{
    uint64_t num_distinct = 0;
    uint64_t mask = distinct_table_size_d64(arr_len) - 1;
    memset(table, 0, (mask + 1) * sizeof(uint64_t));

    // Open addressing with linear probing. Each slot holds an index into distinct_elements + 1,
    // so 0 marks an empty slot and any int64 can be a key.
    uint64_t i, j;
    for(i=0; i<arr_len; i++){
        j = hash_d64(arr[i]) & mask;
        while(table[j] && distinct_elements[table[j] - 1] != arr[i]){
            j = (j + 1) & mask;
        }
        if(!table[j]){
            distinct_elements[num_distinct] = arr[i];
            distinct_count[num_distinct] = 0;
            num_distinct++;
            table[j] = num_distinct;
        }
        distinct_count[table[j] - 1]++;
    }
    return num_distinct;
}
