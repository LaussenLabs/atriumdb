//
// Created by Will Dixon on 2021-06-13.
//
#include <stdint.h>

#include <size.h>

uint8_t size_encode_d64(int64_t * arr, uint64_t num_values, uint8_t min_size)
{
    uint8_t best_size = size_find_best_d64(arr, num_values, min_size);
    uint64_t i;

    switch (best_size) {
        case 1:
            for(i=0; i<num_values; i++){
                ((int8_t *)arr)[i] = (int8_t)arr[i];
            }
            break;

        case 2:
            for(i=0; i<num_values; i++){
                ((int16_t *)arr)[i] = (int16_t)arr[i];
            }
            break;

        case 4:
            for(i=0; i<num_values; i++){
                ((int32_t *)arr)[i] = (int32_t)arr[i];
            }
            break;

        default:
            break;
    }
    return best_size;
}