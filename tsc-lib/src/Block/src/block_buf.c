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
        t_comp_size = get_compress_buffer_size(time_size, headers[i].t_compression);  // Borrow this space.

        offset = sizeof(block_metadata_t);

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

        total_size = sizeof(block_metadata_t) + time_size + value_size + t_comp_size + v_comp_size;

        result += total_size;
    }
    return result;
}