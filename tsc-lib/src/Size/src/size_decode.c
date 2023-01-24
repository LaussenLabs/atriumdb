//
// Created by Will Dixon on 2021-06-13.
//
#include <stdint.h>

void size_decode_d64(int64_t * arr, uint64_t num_values, uint8_t bytes_per_value)
{
    int64_t i;

    switch (bytes_per_value) {
        case 1:
            for(i=(int64_t)num_values-1; i>=0; i--){
                arr[i] = (int64_t)(((int8_t *)arr)[i]);
            }
            break;

        case 2:
            for(i=(int64_t)num_values-1; i>=0; i--){
                arr[i] = (int64_t)(((int16_t *)arr)[i]);
            }
            break;

        case 4:
            for(i=(int64_t)num_values-1; i>=0; i--){
                arr[i] = (int64_t)(((int32_t *)arr)[i]);
            }
            break;

        default:
            break;
    }
}