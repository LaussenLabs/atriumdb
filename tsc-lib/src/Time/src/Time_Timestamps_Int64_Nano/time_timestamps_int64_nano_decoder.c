//
// Created by Will Dixon on 2021-05-29.
//

#include <time_p.h>
#include <string.h>
#include <gap.h>


size_t time_int64_nano_data_decode(int64_t *time_data, const void *encoded_time,
                                   block_metadata_t * block_metadata)
{
    switch (block_metadata->t_raw_type) {
        case T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            memcpy(time_data, encoded_time, block_metadata->num_vals * sizeof(int64_t));
            return block_metadata->num_vals * sizeof(int64_t);

        default:
            printf("time encoded type %u not supported. On line %d in file %s\n",
                   block_metadata->t_raw_type, __LINE__, __FILE__);
            exit(1);
    }
}