//
// Created by Will Dixon on 2021-06-10.
//
#include "gtest/gtest.h"

extern "C" {
#include <distinct.h>
#include <inttypes.h>
}

#define NANO 1000000000

static void print_arr_d64(int64_t *arr, uint64_t arr_len);
static void print_arr_u64(uint64_t *arr, uint64_t arr_len);

TEST(DistinctTestSuite, SimpleD64Test){
    // Test Array
    int64_t arr[] = {1, 2, 3, 4, 4, 5, 3, 2, 1, 0, 34};

    int64_t expected_d_elements[] = {1, 2, 3, 4, 5, 0, 34};
    int64_t expected_d_count[] = {2, 2, 2, 2, 1, 1, 1};

    uint64_t arr_len = sizeof(arr) / sizeof(arr[0]);
    uint64_t expected_answer = 7;

    uint64_t num_distinct = count_num_distinct_d64(arr, arr_len);

    ASSERT_EQ(num_distinct, expected_answer);

    auto *distinct_elements = (int64_t *)malloc(num_distinct * sizeof(int64_t));
    auto *distinct_count = (uint64_t *)malloc(num_distinct * sizeof(uint64_t));

    count_arr_distinct_d64(arr, arr_len, num_distinct, distinct_elements, distinct_count);

    uint64_t i;
    for(i=0; i<num_distinct; i++){
        EXPECT_EQ(distinct_elements[i], expected_d_elements[i]);
        EXPECT_EQ(distinct_count[i], expected_d_count[i]);
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


static void print_arr_u64(uint64_t *arr, uint64_t arr_len)
{
    uint64_t i;
    for(i=0; i<arr_len; i++){
        printf("%" PRIu64 " ", arr[i]);
    }
    printf("\n");
}

