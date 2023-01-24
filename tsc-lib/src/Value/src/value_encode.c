//
// Created by Will Dixon on 2021-06-14.
//
#include <value_p.h>

#include <stdio.h>
#include <stdlib.h>


size_t value_encode(const void *value_data, void *encoded_value, block_metadata_t *block_metadata,
                    const block_options_t *block_options)
{
    switch (block_metadata->v_raw_type) {

        case V_TYPE_INT64:
            /* Int64 Value */
            block_metadata->v_raw_size = value_d64_get_size(block_metadata);
            block_metadata->v_encoded_size = value_d64_encode(value_data, encoded_value, block_metadata,
                                                             block_options);
            break;

        case V_TYPE_DOUBLE:
            /* Double Value */
            block_metadata->v_raw_size = value_f64_get_size(block_metadata);
            block_metadata->v_encoded_size = value_f64_encode(value_data, encoded_value, block_metadata);
            break;

        case V_TYPE_DELTA_INT64:
            /* Int64 Delta */
            block_metadata->v_raw_size = value_d64_get_size(block_metadata);
            block_metadata->v_encoded_size = value_delta_d64_encode(value_data, encoded_value, block_metadata);
            break;

        case V_TYPE_XOR_DOUBLE:
            /* Int64 Delta */
            block_metadata->v_raw_size = value_xor_f64_get_size(block_metadata);
            block_metadata->v_encoded_size = value_xor_f64_encode(value_data, encoded_value, block_metadata);
            break;

        default:
            printf("value type %u not supported. On line %d in file %s\n",
                   block_metadata->v_raw_type, __LINE__, __FILE__);
            exit(1);
    }
    return block_metadata->v_encoded_size;
}