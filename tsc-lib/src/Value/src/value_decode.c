//
// Created by Will Dixon on 2021-06-14.
//
#include <value_p.h>

#include <stdio.h>
#include <stdlib.h>


void value_decode(void *value_data, block_metadata_t *block_metadata)
{
    switch (block_metadata->v_encoded_type) {

        case V_TYPE_INT64:
            /* Int64 Value */
            value_d64_decode(value_data, block_metadata);
            break;

        case V_TYPE_DOUBLE:
            /* Double Value */
            value_f64_decode(value_data, block_metadata);
            break;

        case V_TYPE_DELTA_INT64:
            /* Int64 Delta */
            value_delta_d64_decode(value_data, block_metadata);
            break;

        case V_TYPE_XOR_DOUBLE:
            /* Int64 Delta */
            value_xor_f64_decode(value_data, block_metadata);
            break;

        default:
            printf("value type %u not supported. On line %d in file %s\n",
                   block_metadata->v_encoded_type, __LINE__, __FILE__);
            exit(1);
    }
}
