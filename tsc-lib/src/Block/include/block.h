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

#endif //MINIMAL_SDK_BLOCK_H
