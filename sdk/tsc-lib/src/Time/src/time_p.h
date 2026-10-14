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
// Created by Will Dixon on 2021-05-14.
//

#ifndef MINIMAL_SDK_TIME_P_H
#define MINIMAL_SDK_TIME_P_H

#include <block_header.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

/* Time Array Nanosecond Int64 */
size_t time_timestamps_int64_nano_get_size(block_metadata_t * block_metadata);
size_t time_int64_nano_data_encode(const int64_t * time_data, void * encoded_time,
                                   block_metadata_t * block_metadata);
size_t time_int64_nano_data_decode(int64_t *time_data, const void *encoded_time,
                                   block_metadata_t * block_metadata);

/* Gap Array Index-Duration Nano */
size_t time_gap_int64_nano_get_size(const void *time_data, block_metadata_t *block_metadata);
size_t time_gap_int64_nano_encode(const int64_t * time_data, void * encoded_time,
                                  block_metadata_t * block_metadata);
size_t time_gap_int64_nano_decode(int64_t * time_data, const void * time_bytes,
                                  block_metadata_t * block_metadata);

/* Gap Array Index-Num Samples */
size_t time_gap_d64_num_samples_get_size(const void *time_data, block_metadata_t *block_metadata);
size_t time_gap_d64_num_samples_encode(const int64_t * time_data, void * encoded_time,
                                       block_metadata_t * block_metadata);
size_t time_gap_d64_num_samples_decode(int64_t * time_data, const void * time_bytes,
                                       block_metadata_t * block_metadata);


/* Helpers */
size_t gap_array_size(uint64_t num_gaps);


#endif //MINIMAL_SDK_TIME_P_H
