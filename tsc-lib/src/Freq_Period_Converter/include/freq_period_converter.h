//
// Created by Will Dixon on 2021-05-17.
//

#ifndef MINIMAL_SDK_FREQ_PERIOD_CONVERTER_H
#define MINIMAL_SDK_FREQ_PERIOD_CONVERTER_H

#include <stdint.h>

uint64_t uint64_nhz_freq_to_uint64_ns_period(uint64_t freq_nhz);

uint64_t uint64_ns_period_to_uint64_nhz_freq(uint64_t period_ns);

#endif //MINIMAL_SDK_FREQ_PERIOD_CONVERTER_H
