//
// Created by Will Dixon on 2021-06-10.
//

#ifndef MINIMAL_SDK_DISTINCT_H
#define MINIMAL_SDK_DISTINCT_H
#include <stdint.h>

/* These functions don't ask for more memory, and they preserve the order of the array */
uint64_t count_num_distinct_d64(const int64_t *arr, uint64_t arr_len);
void count_arr_distinct_d64(const int64_t *arr, uint64_t arr_len, uint64_t num_distinct,
                                int64_t *distinct_elements, uint64_t *distinct_count);

#endif //MINIMAL_SDK_DISTINCT_H
