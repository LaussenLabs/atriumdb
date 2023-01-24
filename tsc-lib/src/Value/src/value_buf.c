//
// Created by Will Dixon on 2021-06-13.
//
#include <value_p.h>

#include <stdio.h>
#include <stdlib.h>


size_t value_encode_buffer_get_size(block_metadata_t *block_metadata)
{
    switch (block_metadata->v_encoded_type) {

        case V_TYPE_INT64:
            /* Int64 Value */
            return value_d64_get_size(block_metadata);

        case V_TYPE_DOUBLE:
            /* Double Value */
            return value_f64_get_size(block_metadata);

        case V_TYPE_DELTA_INT64:
            /* Int64 Delta */
            return value_delta_d64_get_size(block_metadata);

        case V_TYPE_XOR_DOUBLE:
            /* Int64 Delta */
            return value_xor_f64_get_size(block_metadata);

        default:
            printf("value type %u not supported. On line %d in file %s\n",
                   block_metadata->v_encoded_type, __LINE__, __FILE__);
            exit(1);
    }
}