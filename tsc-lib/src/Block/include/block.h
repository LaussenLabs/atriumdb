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
// Created by Will Dixon on 2021-06-14.
//

#ifndef MINIMAL_SDK_BLOCK_H
#define MINIMAL_SDK_BLOCK_H

#include <block_header.h>
#include <stddef.h>

size_t block_get_buffer_size(const void *time_data, uint64_t num_blocks, const uint64_t *t_block_start,
                             uint64_t *byte_start, block_metadata_t *headers);
size_t encode_blocks(const void *time_data, const void *value_data, void *encoded_bytes, uint64_t num_blocks,
                     const uint64_t *t_block_start, const uint64_t *v_block_start, uint64_t *byte_start,
                     block_metadata_t *headers, const block_options_t *options, uint16_t num_threads);
void decode_blocks(void *time_data, void *value_data, void *encoded_bytes, uint64_t num_blocks,
                   const uint64_t *t_block_start, const uint64_t *v_block_start, const uint64_t *byte_start,
                   const uint64_t *t_byte_start, uint16_t num_threads);

void convert_value_data_to_analog(const void *value_data, double *analog_values, const block_metadata_t *headers,
                                  const uint64_t *analog_block_start_index_array, uint64_t num_blocks);

#endif //MINIMAL_SDK_BLOCK_H
