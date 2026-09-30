//
// Created by Will Dixon on 2021-06-11.
//
#include "gtest/gtest.h"

extern "C" {
#include <entropy.h>
#include <inttypes.h>
#include <stddef.h>
}


TEST(EntropyTestSuite, SimpleD64Test){
    int64_t arr[] = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    uint64_t arr_len = sizeof(arr) / sizeof(arr[0]);
    uint64_t num_distinct = 4;

    // Ask for buffer requirements
    size_t entropy_buffer_size = entropy_buffer_size_d64((int64_t *)arr, arr_len);
    void *buffer = malloc(entropy_buffer_size);

    ASSERT_EQ(entropy_buffer_size, num_distinct * 2 * sizeof(int64_t));

    double entropy = entropy_get_d64((int64_t *)arr, arr_len, buffer, entropy_buffer_size);

    ASSERT_NEAR(entropy, 1.84644, 0.001);

    free(buffer);
}
