//
// Created by Will Dixon on 2021-05-17.
//
#include "gtest/gtest.h"

extern "C" {
#include <freq_period_converter.h>
#include <stdint.h>
}

#define NANO 1000000000

TEST(FreqPeriodConverterTestSuite, NhzToNsTest){
    // Two Arrays of corresponding freq and periods both in nano scale.
    uint64_t freq_nhz_array[] = {500000000000, 256000000000, 1, 1000000000000000000};
    uint64_t period_nhz_array[] = {2000000, 3906250, 1000000000000000000, 1};

    // Automatically calculate the length of the two arrays so they are easy to add to.
    int freq_size = sizeof(freq_nhz_array) / sizeof(freq_nhz_array[0]);
    int period_size = sizeof(period_nhz_array) / sizeof(period_nhz_array[0]);

    // If the two arrays aren't of the same length there's a problem.
    ASSERT_EQ(freq_size, period_size);

    // Check that both freq and period come back correctly when converted.
    int i;
    for(i=0; i<freq_size; i++){
        EXPECT_EQ(uint64_nhz_freq_to_uint64_ns_period(freq_nhz_array[i]), period_nhz_array[i]);
        EXPECT_EQ(uint64_ns_period_to_uint64_nhz_freq(period_nhz_array[i]), freq_nhz_array[i]);
    }
}