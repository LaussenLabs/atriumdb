//
// Created by Will Dixon on 2021-05-29.
//

#include <block_header.h>
#include <time_p.h>

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>


void time_decode(void * time_data, const void *encoded_time, block_metadata_t * block_metadata)
{
    switch (block_metadata->t_encoded_type) {
        case T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            /* signed 64 bit integer nanoseconds */
            time_int64_nano_data_decode((int64_t *)time_data, encoded_time, block_metadata);
            break;

        case T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:
            /* Gap Array Index-Duration Nano */
            time_gap_int64_nano_decode((int64_t *)time_data, encoded_time, block_metadata);
            break;

        case T_TYPE_GAP_ARRAY_INT64_INDEX_NUM_SAMPLES:

        case T_TYPE_START_TIME_NUM_SAMPLES:
            /* Gap Array Index-Samples */
            time_gap_d64_num_samples_decode((int64_t *)time_data, encoded_time, block_metadata);
            break;

        default:
            printf("time type %u not supported. On line %d in file %s\n", block_metadata->t_encoded_type, __LINE__, __FILE__);
            exit(1);
    }
}