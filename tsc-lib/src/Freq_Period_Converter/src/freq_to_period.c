//
// Created by Will Dixon on 2021-05-17.
//
#include <freq_period_converter_p.h>

uint64_t uint64_nhz_freq_to_uint64_ns_period(uint64_t freq_nhz)
{
    return ((uint64_t)NANO * (uint64_t)NANO) / freq_nhz;
}