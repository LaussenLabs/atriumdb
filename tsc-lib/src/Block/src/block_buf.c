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

#include <block_header.h>
#include <stddef.h>

#include <time_block.h>
#include <value.h>
#include <compression.h>

#include <stdio.h>


size_t block_get_buffer_size(const void *time_data, uint64_t num_blocks, const uint64_t *t_block_start,
                             uint64_t *byte_start, block_metadata_t *headers)
{
    size_t result = 0;
    size_t time_size, value_size, total_size;
    size_t t_comp_size, v_comp_size, offset;

    uint8_t *time_data_u8 = (uint8_t *)time_data;

    uint64_t i;
    for(i=0; i<num_blocks; i++){
        time_size = time_encode_buffer_get_size(&(time_data_u8[t_block_start[i]]), &(headers[i]));
        value_size = value_encode_buffer_get_size(&(headers[i]));
        v_comp_size = get_compress_buffer_size(value_size, headers[i].v_compression);
        t_comp_size = get_compress_buffer_size(time_size, headers[i].t_compression);

        time_size = (v_comp_size + 7) & ~7;
        value_size = (t_comp_size + 7) & ~7;

        v_comp_size = (v_comp_size + 7) & ~7;
        t_comp_size = (t_comp_size + 7) & ~7;

        offset = (sizeof(block_metadata_t) + 7) & ~7;

        // Time Byte Position
        headers[i].t_start_byte = offset;
        offset += time_size;

        // Time Compression Buffer Size
        headers[i].t_num_bytes = t_comp_size;
        offset += t_comp_size;

        // Value Byte Position
        headers[i].v_start_byte = offset;

        // Value Compression Buffer Size
        headers[i].v_num_bytes = v_comp_size;

        // Overall Memory Position for Block
        byte_start[i] = result;

        total_size = 2 * (((sizeof(block_metadata_t) + 7) & ~7) + time_size + value_size + t_comp_size + v_comp_size);

        result += total_size;
    }
    return result;
}