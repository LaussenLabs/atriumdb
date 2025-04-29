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
// Created by Will Dixon on 2021-06-15.
//

#include <block_header.h>
#include <stddef.h>
#include <string.h>

#include <time_block.h>
#include <value.h>
#include <compression.h>
#include <omp.h>

size_t encode_blocks(const void *time_data, const void *value_data, void *encoded_bytes, uint64_t num_blocks,
                     const uint64_t *t_block_start, const uint64_t *v_block_start, uint64_t *byte_start,
                     block_metadata_t *headers, const block_options_t *options, uint16_t num_threads)
{
    // Convert to 8-bit pointers.
    uint8_t *encoded_bytes_u8 = (uint8_t *)encoded_bytes;
    uint8_t *time_data_u8 = (uint8_t *)time_data;
    uint8_t *value_data_u8 = (uint8_t *)value_data;

    // Define some temp pointers.
    uint8_t *cur_encoded_bytes;

    uint64_t i;
    uint8_t j;

    omp_set_dynamic(1);
    omp_set_num_threads(num_threads);

    #pragma omp parallel for default(none) private(i, j, cur_encoded_bytes) \
    shared(time_data_u8, value_data_u8, encoded_bytes_u8, headers, options, t_block_start, v_block_start, byte_start, num_blocks) \
    collapse(2)
    for(i=0; i<num_blocks; i++){
        // Encode Time and Value in Parallel
        for(j=0; j<2; j++){
            cur_encoded_bytes = &(encoded_bytes_u8[byte_start[i]]);
            if(j){
                // Encode Time
                time_encode(&(time_data_u8[t_block_start[i]]), &(cur_encoded_bytes[headers[i].t_start_byte]), &(headers[i]));

                // Compress
                if(headers[i].t_compression != 1){
                    headers[i].t_num_bytes = my_compress(
                            &(cur_encoded_bytes[headers[i].t_start_byte]),
                            &(cur_encoded_bytes[headers[i].t_start_byte + headers[i].t_encoded_size]),
                            headers[i].t_encoded_size,
                            headers[i].t_num_bytes,
                            headers[i].t_compression,
                            headers[i].t_compression_level);
                } else{
                    headers[i].t_num_bytes = headers[i].t_encoded_size;
                }
            } else{
                // Encode Values
                value_encode(&(value_data_u8[v_block_start[i]]), &(cur_encoded_bytes[headers[i].v_start_byte]), &(headers[i]), options);

                // Compress
                if(headers[i].v_compression != 1){
                    headers[i].v_num_bytes = my_compress(
                            &(cur_encoded_bytes[headers[i].v_start_byte]),
                            &(cur_encoded_bytes[headers[i].v_start_byte + headers[i].v_encoded_size]),
                            headers[i].v_encoded_size,
                            headers[i].v_num_bytes,
                            headers[i].v_compression,
                            headers[i].v_compression_level);
                } else{
                    headers[i].v_num_bytes = headers[i].v_encoded_size;
                }
            }
        }
    }

    // Put it all in order.

    // The absolute position in the result byte array
    size_t offset = 0;

    // The start position of the current block
    size_t last_offset;
    uint64_t time_position, value_position;
    for(i=0; i<num_blocks; i++){
        last_offset = offset;
        cur_encoded_bytes = &(encoded_bytes_u8[byte_start[i]]);

        // Place the Time
        offset += sizeof(block_metadata_t);

        // The location of the data depends on whether it was compressed
        if(headers[i].t_compression != 1){
            time_position = headers[i].t_start_byte + headers[i].t_encoded_size;
        } else{
            time_position = headers[i].t_start_byte;
        }

        memcpy(&(encoded_bytes_u8[offset]), &(cur_encoded_bytes[time_position]), headers[i].t_num_bytes);
        offset += headers[i].t_num_bytes;

        // Set the correct relative start byte.
        headers[i].t_start_byte = sizeof(block_metadata_t);

        // Place the Values
        // The location of the data depends on whether it was compressed
        if(headers[i].v_compression != 1){
            value_position = headers[i].v_start_byte + headers[i].v_encoded_size;
        } else{
            value_position = headers[i].v_start_byte;
        }

        memcpy(&(encoded_bytes_u8[offset]), &(cur_encoded_bytes[value_position]), headers[i].v_num_bytes);
        offset += headers[i].v_num_bytes;

        // Set the correct relative start byte.
        headers[i].v_start_byte = headers[i].t_start_byte + headers[i].t_num_bytes;

        // Place the Header
        if (headers[i].tsc_version_num == 0) {
            headers[i].tsc_version_num = TSC_VERSION_NUM;
        }
        if (headers[i].tsc_version_ext == 0) {
            headers[i].tsc_version_ext = TSC_VERSION_EXT;
        }
        if (headers[i].num_channels == 0) {
            headers[i].num_channels = TSC_NUM_CHANNELS;
        }
        headers[i].meta_num_bytes = sizeof(block_metadata_t);
        memcpy(&(encoded_bytes_u8[last_offset]), &(headers[i]), sizeof(block_metadata_t));

        // Remember the absolute block position.
        byte_start[i] = last_offset;
    }
    return offset;
}