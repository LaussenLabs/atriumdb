//
// Created by Will Dixon on 2021-06-10.
//

#ifndef MINIMAL_SDK_ENTROPY_H
#define MINIMAL_SDK_ENTROPY_H

#include <stdint.h>
#include <stdlib.h>

/* Int64 */
size_t entropy_buffer_size_d64(const int64_t *arr, uint64_t arr_len);
double entropy_get_d64(const int64_t *arr, uint64_t arr_len, void * buffer, size_t buffer_size);

#endif //MINIMAL_SDK_ENTROPY_H
