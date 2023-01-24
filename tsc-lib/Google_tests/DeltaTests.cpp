//
// Created by Will Dixon on 2021-06-12.
//
#include "gtest/gtest.h"

extern "C" {
#include <delta.h>
#include <inttypes.h>
}

static void print_arr_d64(int64_t *arr, uint64_t arr_len);

static int64_t pow_d64(int64_t base, int64_t exponent);

TEST(DeltaTestSuite, SimpleTransD64Test){
    int64_t arr[] = {4,34,54,2,1,-3,-43,-543,-23,2,34,54,-34,-6547,33413,33,-453,-2,-4,-5,-534,-3,-4,-33,342};
    //int64_t arr[] = {0, 2, 4, 8, 12, 15};
    uint64_t arr_len = sizeof(arr) / sizeof(arr[0]);
    int64_t delta[arr_len];

    uint64_t min_order = 1, max_order = 10;

    int64_t i, j;
    for(i=min_order; i<=max_order; i++){
        memcpy(delta, arr, sizeof(arr));

        for(j=0; j<i; j++){
            delta_transform_d64((int64_t *)delta, j, arr_len-j);
        }

        delta_decode_d64((int64_t *)delta, arr_len, i);

        for(j=0; j<arr_len; j++){
            EXPECT_EQ(arr[j], delta[j]);
        }
    }
}

TEST(DeltaTestSuite, MultiOrderD64Test)
{
    int64_t num_values = 100;
    int64_t arr[num_values];
    int64_t delta[num_values];
    int64_t entropy_buff[num_values * 2];
    int8_t max_order = 6;

    int64_t i, order, best_order;
    for(order=0; order < max_order; order++){

        // Create Array
        for(i=0; i<num_values; i++){
            arr[i] = pow_d64(i, order);
        }

        // Encode
        best_order = delta_lowest_entropy_encode_d64(arr, delta, num_values,
                (void *)entropy_buff, sizeof(entropy_buff), 0, 7);

        // Expect the chosen order to be greater than or equal to the polynomial order.
        ASSERT_GE(best_order, order);

        // Decode
        delta_decode_d64(delta, num_values, best_order);

        for(i=0; i<num_values; i++){
            EXPECT_EQ(arr[i], delta[i]);
        }
    }
}

static void print_arr_d64(int64_t *arr, uint64_t arr_len)
{
    uint64_t i;
    for(i=0; i<arr_len; i++){
        printf("%" PRId64 " ", arr[i]);
    }
    printf("\n");
}

static int64_t pow_d64(int64_t base, int64_t exponent)
{
    int64_t result = 1;
    int64_t i;
    for(i=0; i<exponent; i++){
        result *= base;
    }
    return result;
}