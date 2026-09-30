//
// Created by Will Dixon on 2021-06-14.
//
#include "gtest/gtest.h"

extern "C" {
#include <value.h>
#include <inttypes.h>
}

static void print_arr_d64(int64_t *arr, uint64_t arr_len);
static int64_t pow_d64(int64_t base, int64_t exponent);


TEST(ValueTestSuite, SimpleD64Test){
    int64_t arr[] = {4,34,54,2,1,-3,-43,-543,-23,2,34,54,-34,-6547,33413,33,-453,-2,-4,-5,-534,-3,-4,-33,342};
    //int64_t arr[] = {0, 2, 4, 8, 12, 15};
    block_metadata_t header;
    block_options_t options;

    options.bytes_per_value_min = 0;
    options.delta_order_min = 0;
    options.delta_order_max = 10;

    header.num_vals = sizeof(arr) / sizeof(arr[0]);

    header.v_raw_type = V_TYPE_INT64;
    header.v_encoded_type = V_TYPE_DELTA_INT64;

    size_t value_buffer_size = value_encode_buffer_get_size(&header);
    void *value_buf = malloc(value_buffer_size);

    value_encode(arr, value_buf, &header, &options);

    value_decode(value_buf, &header);

    uint64_t i;
    for(i=0; i<header.num_vals; i++){
        EXPECT_EQ(arr[i], ((int64_t *)value_buf)[i]);
    }

    free(value_buf);
}

TEST(ValueTestSuite, ComprehensiveD64Test)
{
    int64_t num_values = 100;
    int64_t arr[num_values];
    int8_t max_order = 6;

    block_metadata_t header;
    block_options_t options;

    header.num_vals = num_values;

    options.bytes_per_value_min = 0;
    options.delta_order_min = 0;
    options.delta_order_max = 10;

    header.v_raw_type = V_TYPE_INT64;
    header.v_encoded_type = V_TYPE_DELTA_INT64;

    int64_t i, order;
    size_t value_buffer_size;
    void *value_buf;
    for(order=0; order < max_order; order++){

        // Create Array
        for(i=0; i<num_values; i++){
            arr[i] = pow_d64(i, order);
        }

        value_buffer_size = value_encode_buffer_get_size(&header);
        value_buf = malloc(value_buffer_size);

        value_encode(arr, value_buf, &header, &options);

        value_decode(value_buf, &header);

        // Expect the chosen order to be greater than or equal to the polynomial order.
        ASSERT_GE(header.order, order);

        for(i=0; i<(int64_t)header.num_vals; i++){
            EXPECT_EQ(arr[i], ((int64_t *)value_buf)[i]);
        }

        free(value_buf);
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