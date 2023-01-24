//
// Created by Will Dixon on 2021-06-13.
//

#ifndef MINIMAL_SDK_VALUE_H
#define MINIMAL_SDK_VALUE_H
#include <stddef.h>

#include <block_header.h>

size_t value_encode_buffer_get_size(block_metadata_t *block_metadata);
size_t value_encode(const void *value_data, void *encoded_value, block_metadata_t *block_metadata,
                    const block_options_t *block_options);
void value_decode(void *value_data, block_metadata_t *block_metadata);

#endif //MINIMAL_SDK_VALUE_H
