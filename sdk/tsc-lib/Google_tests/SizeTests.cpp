//
// Created by Will Dixon on 2021-06-13.
//
#include "gtest/gtest.h"

extern "C" {
#include <size.h>
#include <inttypes.h>
#include <string.h>
}


TEST(SizeTestSuite, SimpleSizeD64Test){
    int64_t arr[] = {4,34,54,2,1,-3,-43,-543,-23,2,34,54,-34,-6547,31413,33,-453,-2,-4,-5,-534,-3,-4,-33,342};
    uint64_t arr_len = sizeof(arr)/ sizeof(arr[0]);
    int64_t arr_copy[arr_len];
    memcpy(arr_copy, arr, sizeof(arr));

    uint8_t bytes_per_value;
    uint8_t min_sizes[] = {1, 2, 4, 8};
    uint8_t expected_sizes[] = {2, 2, 4, 8};
    uint8_t num_min_sizes = sizeof(min_sizes) / sizeof(min_sizes[0]);

    uint8_t size_i, min_size;
    uint64_t i;
    for(size_i=0; size_i<num_min_sizes; size_i++){
        min_size = min_sizes[size_i];

        // Play with min_size
        bytes_per_value = size_encode_d64(arr_copy, arr_len, min_size);
        ASSERT_EQ(bytes_per_value, expected_sizes[size_i]);

        size_decode_d64(arr_copy, arr_len, bytes_per_value);
        for(i=0; i<arr_len; i++){
            EXPECT_EQ(arr[i], arr_copy[i]);
        }
    }
}
