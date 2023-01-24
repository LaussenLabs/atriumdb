//
// Created by Will Dixon on 2021-05-14.
//

#include <block_header.h>
#include <time_p.h>

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>


size_t time_encode_buffer_get_size(const void * time_data, block_metadata_t * block_metadata)
{
    switch (block_metadata->t_encoded_type) {
        case T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            /* signed 64 bit integer nanoseconds */
            return time_timestamps_int64_nano_get_size(block_metadata);

        case T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:
            /* Gap Array Index-Duration Nano */
            // Leaving this blank copies the behavior of the case below.
            // Which is what we want.

        case T_TYPE_GAP_ARRAY_INT64_INDEX_NUM_SAMPLES:

        case T_TYPE_START_TIME_NUM_SAMPLES:
            /* Gap Array Index-Samples */
            return time_gap_int64_nano_get_size(time_data, block_metadata);

        default:
            printf("time type %u not supported. On line %d in file %s\n", block_metadata->t_encoded_type, __LINE__, __FILE__);
            exit(1);
    }
}