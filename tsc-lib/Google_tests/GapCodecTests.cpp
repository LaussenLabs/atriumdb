//
// Created by Will Dixon on 2021-05-25.
//
#include "gtest/gtest.h"
#define NANO 1000000000

extern "C" {
#include <gap.h>
#include <stdint.h>
}

void test_gap_conversion_duration(int64_t *time_array, uint64_t num_values, int64_t *gap_array,
                                  uint64_t num_gaps, uint64_t freq_nhz);

void test_gap_conversion_samples(int64_t *time_array, uint64_t num_values, int64_t *gap_array,
                                  uint64_t num_gaps, uint64_t freq_nhz);


TEST(GapCodecTestSuite, SimpleInt64NsTest){
    uint64_t freq_nhz = 1 * NANO;  // 1Hz

    // Generate Sample Time Array.
    int64_t input_time_data[] = {(int64_t)0, (int64_t)1 * NANO, (int64_t)2 * NANO, (int64_t)4 * NANO,
                                 (int64_t)5 * NANO, (int64_t)6 * NANO, (int64_t)10 * NANO};

    // Compute the number of values in the Sample Time Array.
    uint64_t num_values = sizeof(input_time_data) / sizeof(input_time_data[0]);

    // Create Corresponding Gap Array.
    int64_t input_gap_array[] = {3, (int64_t)1 * NANO, 6, (int64_t)3 * NANO};

    // Computer the number of gaps in the Gap Array.
    uint64_t num_gaps = sizeof(input_gap_array) / (sizeof(input_gap_array[0]) * 2);

    test_gap_conversion_duration(input_time_data, num_values, input_gap_array, num_gaps, freq_nhz);


}


TEST(GapCodecTestSuite, SimpleInt64SampleTest){
    uint64_t freq_nhz = 1 * NANO;  // 1Hz

    // Generate Sample Time Array.
    int64_t input_time_data[] = {(int64_t)0, (int64_t)1 * NANO, (int64_t)2 * NANO, (int64_t)4 * NANO,
                                 (int64_t)5 * NANO, (int64_t)6 * NANO, (int64_t)10 * NANO};

    // Compute the number of values in the Sample Time Array.
    uint64_t num_values = sizeof(input_time_data) / sizeof(input_time_data[0]);

    // Create Corresponding Gap Array.
    int64_t input_gap_array[] = {3, 1 , 6, 3};

    // Computer the number of gaps in the Gap Array.
    uint64_t num_gaps = sizeof(input_gap_array) / (sizeof(input_gap_array[0]) * 2);

    test_gap_conversion_samples(input_time_data, num_values, input_gap_array, num_gaps, freq_nhz);
}


void test_gap_conversion_duration(int64_t *time_array, uint64_t num_values, int64_t *gap_array,
                                  uint64_t num_gaps, uint64_t freq_nhz)
{
    // Ask the module how large a buffer it needs for encoding.
    size_t encode_buffer_size = gap_int64_ns_gap_array_get_size(time_array, num_values, freq_nhz);

    // We expect that to be equal to the size of our answer array.
    ASSERT_EQ(encode_buffer_size, num_gaps * 2 * sizeof(int64_t));

    // Allocate buffer memory for the encode.
    void * encode_buffer = malloc(encode_buffer_size);

    // Use the whole buffer as the output gap array.
    auto * output_gap_array = (int64_t *)encode_buffer;

    // Call the encode function.
    uint64_t num_gaps_calculated = gap_int64_ns_time_array_encode(time_array, output_gap_array, num_values,
                                                                  freq_nhz);

    // Check the number of gaps.
    ASSERT_EQ(num_gaps, num_gaps_calculated);

    // Check that the gap arrays are identical
    uint64_t i;
    for(i=0; i<(num_gaps * 2); i++){
        EXPECT_EQ(output_gap_array[i], gap_array[i]);
    }

    // Start the Decode Test!

    // Calculate Result Array size
    size_t decode_buffer_size = num_values * sizeof(int64_t);

    // Allocate buffer memory for the decode.
    void * decode_buffer = calloc(decode_buffer_size, 1); // The time decoder needs data initialized at 0.
    auto * output_time_array = (int64_t *)decode_buffer;

    // Decode
    int64_t start_time = time_array[0];
    gap_int64_ns_gap_array_decode(gap_array, output_time_array, num_values, num_gaps, start_time,
                                  freq_nhz);

    // Check that the time arrays are identical.
    for(i=0; i<num_values; i++){
        EXPECT_EQ(output_time_array[i], time_array[i]) << i;
    }

    // Free Memory
    free(encode_buffer);
    free(decode_buffer);
}


void test_gap_conversion_samples(int64_t *time_array, uint64_t num_values, int64_t *gap_array,
                                  uint64_t num_gaps, uint64_t freq_nhz)
{
    // Ask the module how large a buffer it needs for encoding.
    size_t encode_buffer_size = gap_int64_ns_gap_array_get_size(time_array, num_values, freq_nhz);

    // We expect that to be equal to the size of our answer array.
    ASSERT_EQ(encode_buffer_size, num_gaps * 2 * sizeof(int64_t));

    // Allocate buffer memory for the encode.
    void * encode_buffer = malloc(encode_buffer_size);

    // Use the whole buffer as the output gap array.
    auto * output_gap_array = (int64_t *)encode_buffer;

    // Call the encode function.
    uint64_t num_gaps_calculated = gap_int64_samples_encode(time_array, output_gap_array, num_values,
                                                                  freq_nhz);

    // Check the number of gaps.
    ASSERT_EQ(num_gaps, num_gaps_calculated);

    // Check that the gap arrays are identical
    uint64_t i;
    for(i=0; i<(num_gaps * 2); i++){
        EXPECT_EQ(output_gap_array[i], gap_array[i]);
    }

    // Start the Decode Test!

    // Calculate Result Array size
    size_t decode_buffer_size = num_values * sizeof(int64_t);

    // Allocate buffer memory for the decode.
    void * decode_buffer = calloc(decode_buffer_size, 1); // The time decoder needs data initialized at 0.
    auto * output_time_array = (int64_t *)decode_buffer;

    // Decode
    int64_t start_time = time_array[0];
    gap_int64_samples_decode(gap_array, output_time_array, num_values, num_gaps, start_time,
                                  freq_nhz);

    // Check that the time arrays are identical.
    for(i=0; i<num_values; i++){
        EXPECT_EQ(output_time_array[i], time_array[i]) << i;
    }

    // Free Memory
    free(encode_buffer);
    free(decode_buffer);
}
