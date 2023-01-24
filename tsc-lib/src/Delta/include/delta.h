//
// Created by Will Dixon on 2021-06-11.
//

#ifndef MINIMAL_SDK_DELTA_H
#define MINIMAL_SDK_DELTA_H

#include <stdint.h>
#include <stddef.h>

/* Int64 Codec */
uint8_t delta_lowest_entropy_encode_d64(const int64_t *value_arr, int64_t *delta_arr, uint64_t num_values,
                                        void *entropy_buffer, size_t buffer_size, uint8_t min_order, uint8_t max_order);
void delta_decode_d64(int64_t *delta_arr, uint64_t num_values, uint8_t order);

/* Int64 Transformation */
void delta_transform_d64(int64_t *arr, uint64_t start, uint64_t num_vals);
void delta_inverse_transform_d64(int64_t *arr, uint64_t start, uint64_t num_vals);

#endif //MINIMAL_SDK_DELTA_H
