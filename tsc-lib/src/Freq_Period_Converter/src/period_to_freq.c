//
// Created by Will Dixon on 2021-05-17.
//
#include <freq_period_converter_p.h>

uint64_t uint64_ns_period_to_uint64_nhz_freq(uint64_t period_ns)
{
    return ((uint64_t)NANO * (uint64_t)NANO) / period_ns;
}
