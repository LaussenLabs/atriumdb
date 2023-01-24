//
// Created by Will Dixon on 2021-05-10.
//
#include <block_header.h>
#include <time_p.h>

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>


size_t time_encode(const void * time_data, void *encoded_time, block_metadata_t * block_metadata)
{
    switch (block_metadata->t_raw_type) {
        case T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            /* signed 64 bit integer nanoseconds */
            block_metadata->t_raw_size = time_timestamps_int64_nano_get_size(block_metadata);
            block_metadata->t_encoded_size = time_int64_nano_data_encode(
                    (int64_t *)time_data, encoded_time, block_metadata);
            break;

        case T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:
            /* Gap Array Index-Duration Nano */
            block_metadata->t_raw_size = time_gap_int64_nano_get_size(NULL, block_metadata);
            block_metadata->t_encoded_size = time_gap_int64_nano_encode(
                    (int64_t *)time_data, encoded_time, block_metadata);
            break;

        case T_TYPE_GAP_ARRAY_INT64_INDEX_NUM_SAMPLES:
            /* Gap Array Index-Samples */

        case T_TYPE_START_TIME_NUM_SAMPLES:
            /* Gap Array Index-Samples */
            block_metadata->t_raw_size = time_gap_d64_num_samples_get_size(NULL, block_metadata);
            block_metadata->t_encoded_size = time_gap_d64_num_samples_encode(
                    (int64_t *)time_data, encoded_time, block_metadata);
            break;

        default:
            printf("time type %u not supported. On line %d in file %s\n", block_metadata->t_raw_type, __LINE__, __FILE__);
            exit(1);
    }
    return block_metadata->t_encoded_size;
}
