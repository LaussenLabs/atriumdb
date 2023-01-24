//
// Created by Will Dixon on 2021-05-31.
//

#include <time_p.h>
#include <string.h>


size_t time_gap_int64_nano_encode(const int64_t * time_data, void * encoded_time,
                                  block_metadata_t * block_metadata)
{
    switch (block_metadata->t_encoded_type) {
        case T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:
            /* No Conversion */
            memcpy(encoded_time, time_data, gap_array_size(block_metadata->num_gaps));

            // Return size of gap array.
            return gap_array_size(block_metadata->num_gaps);

        default:
            printf("time encoded type %u not supported. On line %d in file %s\n",
                   block_metadata->t_encoded_type, __LINE__, __FILE__);
            exit(1);
    }
}
