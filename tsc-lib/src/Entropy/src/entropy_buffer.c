//
// Created by Will Dixon on 2021-06-10.
//
#include <stdlib.h>
#include <stdint.h>
#include <distinct.h>


size_t entropy_buffer_size_d64(const int64_t *arr, uint64_t arr_len)
{
    uint64_t num_distinct_elements = count_num_distinct_d64(arr, arr_len);

    // 2 Arrays of n number of int64's
    return 2 * num_distinct_elements * sizeof(int64_t);
}