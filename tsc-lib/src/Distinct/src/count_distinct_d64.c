//
// Created by Will Dixon on 2021-06-10.
//
#include <stdint.h>

uint64_t count_num_distinct_d64(const int64_t *arr, uint64_t arr_len)
{
    uint64_t num_distinct_elements = 0;

    uint64_t i;
    uint64_t j;
    for(i=0; i<arr_len; i++){
        for(j=0; j<i; j++){
            if(arr[i] == arr[j]){
                break;
            }
        }
        if(i == j){
            num_distinct_elements++;
        }
    }
    return num_distinct_elements;
}