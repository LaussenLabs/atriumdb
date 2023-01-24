//
// Created by Will Dixon on 2021-06-12.
//

#include <stdint.h>

#include <delta.h>

void delta_decode_d64(int64_t *delta_arr, uint64_t num_values, uint8_t order)
{
    uint8_t i, start;
    for(i=0; i<order; i++){
        start = (order - 1) - i;
        delta_inverse_transform_d64(delta_arr, start, num_values-start);
    }
}