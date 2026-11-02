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

#ifndef MINIMAL_SDK_SIZE_H
#define MINIMAL_SDK_SIZE_H

uint8_t size_encode_d64(int64_t * arr, uint64_t num_values, uint8_t min_size);
void size_decode_d64(int64_t * arr, uint64_t num_values, uint8_t bytes_per_value);

uint8_t size_find_best_d64(int64_t * arr, uint64_t num_values, uint8_t min_size);

#endif //MINIMAL_SDK_SIZE_H
