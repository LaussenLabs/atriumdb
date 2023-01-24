//
// Created by Will Dixon on 2021-06-14.
//
#include <value_p.h>

#include <stdio.h>
#include <stdlib.h>


void value_d64_decode(void *value_data, block_metadata_t *block_metadata)
{
    switch (block_metadata->v_raw_type) {
        case V_TYPE_INT64:
            /* Int64 Value */
            break;

        default:
            printf("value type %u not supported. On line %d in file %s\n",
                   block_metadata->v_raw_type, __LINE__, __FILE__);
            exit(1);
    }
}
