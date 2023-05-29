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

void size_decode_d64(int64_t * arr, uint64_t num_values, uint8_t bytes_per_value)
{
    int64_t i;

    switch (bytes_per_value) {
        case 1:
            for(i=(int64_t)num_values-1; i>=0; i--){
                arr[i] = (int64_t)(((int8_t *)arr)[i]);
            }
            break;

        case 2:
            for(i=(int64_t)num_values-1; i>=0; i--){
                arr[i] = (int64_t)(((int16_t *)arr)[i]);
            }
            break;

        case 4:
            for(i=(int64_t)num_values-1; i>=0; i--){
                arr[i] = (int64_t)(((int32_t *)arr)[i]);
            }
            break;

        default:
            break;
    }
}