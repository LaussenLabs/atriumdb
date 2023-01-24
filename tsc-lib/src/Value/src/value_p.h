//
// Created by Will Dixon on 2021-06-13.
//

#ifndef MINIMAL_SDK_VALUE_P_H
#define MINIMAL_SDK_VALUE_P_H

#include <stdint.h>
#include <stddef.h>

#include <block_header.h>

/* Int64 Value */
size_t value_d64_get_size(block_metadata_t *block_metadata);
size_t value_d64_encode(const int64_t *value_data, void *encoded_value, block_metadata_t *block_metadata,
                        const block_options_t *block_options);
void value_d64_decode(void *value_data, block_metadata_t *block_metadata);

/* Double Value */
size_t value_f64_get_size(block_metadata_t *block_metadata);
size_t value_f64_encode(const double * value_data, void *encoded_value, block_metadata_t * block_metadata);
void value_f64_decode(void *value_data, block_metadata_t *block_metadata);

/* Int64 Delta */
size_t value_delta_d64_get_size(block_metadata_t *block_metadata);
size_t value_delta_d64_encode(const int64_t *value_data, void *encoded_value, block_metadata_t * block_metadata);
void value_delta_d64_decode(void *value_data, block_metadata_t *block_metadata);

/* Double XOR */
size_t value_xor_f64_get_size(block_metadata_t *block_metadata);
size_t value_xor_f64_encode(const double * value_data, void *encoded_value, block_metadata_t * block_metadata);
void value_xor_f64_decode(void *value_data, block_metadata_t *block_metadata);


#endif //MINIMAL_SDK_VALUE_P_H
