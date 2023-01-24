//
// Created by Will Dixon on 2021-06-09.
//
#include <time_p.h>
#include <gap.h>
#include <string.h>


size_t time_gap_d64_num_samples_decode(int64_t * time_data, const void * time_bytes,
                                       block_metadata_t * block_metadata)
{
    switch (block_metadata->t_raw_type) {
        case T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            gap_int64_samples_decode((int64_t *)time_bytes, time_data, block_metadata->num_vals,
                                          block_metadata->num_gaps, block_metadata->start_n, block_metadata->freq_nhz);

            // Return size of timestamp array.
            return block_metadata->num_vals * sizeof(int64_t);

        case T_TYPE_GAP_ARRAY_INT64_INDEX_NUM_SAMPLES:

        case T_TYPE_START_TIME_NUM_SAMPLES:
            /* Convert Time Array to Gap Array */
            // Copy Gap Array.
            memcpy(time_data, time_bytes, gap_array_size(block_metadata->num_gaps));

            // Return size of gap array.
            return gap_array_size(block_metadata->num_gaps);

        default:
            printf("time encoded type %u not supported. On line %d in file %s\n",
                   block_metadata->t_raw_type, __LINE__, __FILE__);
            exit(1);
    }
}