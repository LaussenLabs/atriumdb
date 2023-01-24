//
// Created by Will Dixon on 2021-05-30.
//

#include <time_p.h>
#include <gap.h>


size_t time_gap_int64_nano_get_size(const void *time_data, block_metadata_t *block_metadata)
{
    switch (block_metadata->t_raw_type) {
        case T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            return gap_int64_ns_gap_array_get_size(
                    (int64_t *)time_data, block_metadata->num_vals, block_metadata->freq_nhz);

        case T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:

        case T_TYPE_GAP_ARRAY_INT64_INDEX_NUM_SAMPLES:

        case T_TYPE_START_TIME_NUM_SAMPLES:
            /* Convert Time Array to Gap Array */
            return gap_array_size(block_metadata->num_gaps);

        default:
            printf("time type %u not supported. On line %d in file %s\n",
                   block_metadata->t_raw_type, __LINE__, __FILE__);
            exit(1);
    }
}