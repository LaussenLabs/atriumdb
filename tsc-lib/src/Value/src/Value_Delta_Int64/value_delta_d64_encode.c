//
// Created by Will Dixon on 2021-06-14.
//
#include <value_p.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t value_delta_d64_encode(const int64_t *value_data, void *encoded_value, block_metadata_t * block_metadata)
{
    switch (block_metadata->v_encoded_type) {
        case V_TYPE_DELTA_INT64:
            /* Int64 Delta */
            memcpy(encoded_value, value_data, block_metadata->num_vals * sizeof(int64_t));
            return block_metadata->num_vals * sizeof(int64_t);

        default:
            printf("value type %u not supported. On line %d in file %s\n",
                   block_metadata->v_encoded_type, __LINE__, __FILE__);
            exit(1);
    }
}