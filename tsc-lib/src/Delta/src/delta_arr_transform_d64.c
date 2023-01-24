//
// Created by Will Dixon on 2021-06-12.
//

#include <stdint.h>

void delta_transform_d64(int64_t *arr, uint64_t start, uint64_t num_vals)
{
    uint64_t i;
    for(i=start + num_vals - 1; i>start; i--){
        arr[i] = arr[i] - arr[i-1];
    }
}


void delta_inverse_transform_d64(int64_t *arr, uint64_t start, uint64_t num_vals)
{
    uint64_t i;
    for(i=start; i<start + num_vals - 1; i++){
        arr[i+1] = arr[i+1] + arr[i];
    }
}