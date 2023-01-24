//
// Created by Will Dixon on 2021-06-10.
//

#ifndef MINIMAL_SDK_ENTROPY_P_H
#define MINIMAL_SDK_ENTROPY_P_H

#include <stdint.h>

double entropy(const uint64_t * hist, uint64_t hist_len, uint64_t arr_len);

#endif //MINIMAL_SDK_ENTROPY_P_H
