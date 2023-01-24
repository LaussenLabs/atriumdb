//
// Created by Will Dixon on 2021-06-10.
//
#include <math.h>
#include <stdint.h>

double entropy(const uint64_t * hist, uint64_t hist_len, uint64_t arr_len)
{
    uint64_t i;
    double H=0;
    for(i=0; i<hist_len; i++){
        H-=(double)hist[i]/arr_len*log2((double)hist[i]/arr_len);
    }
    return H;
}