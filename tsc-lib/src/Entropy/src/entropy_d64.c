//
// Created by Will Dixon on 2021-06-10.
//

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <distinct.h>

#include <entropy_p.h>

double entropy_get_d64(const int64_t *arr, uint64_t arr_len, void * buffer, size_t buffer_size)
{
    uint64_t num_distinct = count_num_distinct_d64(arr, arr_len);
    if(buffer_size < (num_distinct * 2 * sizeof(int64_t))){
        printf("Buffer too small, must be at least %" PRIu64 " bytes. On line %d in file %s\n",
               num_distinct * 2 * sizeof(int64_t), __LINE__, __FILE__);
        exit(1);
    }

    // Portion out the buffer memory.
    int64_t *distinct_elements = buffer;
    uint64_t *distinct_count = &(((uint64_t *)buffer)[num_distinct]);

    // Tally the distinct values.
    count_arr_distinct_d64(arr, arr_len, num_distinct, distinct_elements, distinct_count);

    // Return the Shannon Entropy.
    return entropy(distinct_count, num_distinct, arr_len);
}