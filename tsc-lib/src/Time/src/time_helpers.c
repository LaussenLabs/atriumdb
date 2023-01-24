//
// Created by Will Dixon on 2021-05-31.
//

#include <time_p.h>


size_t gap_array_size(uint64_t num_gaps)
{
    return num_gaps * 2 * sizeof(int64_t);
}