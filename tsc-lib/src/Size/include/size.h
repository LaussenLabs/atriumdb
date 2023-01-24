//
// Created by Will Dixon on 2021-06-13.
//

#ifndef MINIMAL_SDK_SIZE_H
#define MINIMAL_SDK_SIZE_H

uint8_t size_encode_d64(int64_t * arr, uint64_t num_values, uint8_t min_size);
void size_decode_d64(int64_t * arr, uint64_t num_values, uint8_t bytes_per_value);

uint8_t size_find_best_d64(int64_t * arr, uint64_t num_values, uint8_t min_size);

#endif //MINIMAL_SDK_SIZE_H
