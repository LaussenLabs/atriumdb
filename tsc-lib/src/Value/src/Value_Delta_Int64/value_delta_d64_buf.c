//
// Created by Will Dixon on 2021-06-14.
//
#include <value_p.h>


size_t value_delta_d64_get_size(block_metadata_t *block_metadata)
{
    return block_metadata->num_vals * sizeof(int64_t) * 2;
}
