//
// Created by Will Dixon on 2021-05-30.
//

#include <time_p.h>


size_t time_timestamps_int64_nano_get_size(block_metadata_t * block_metadata)
{
    return block_metadata->num_vals * sizeof(int64_t);
}
