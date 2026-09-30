//
// Created by Will Dixon on 2021-06-08.
//
#include "gtest/gtest.h"
#define NANO 1000000000

extern "C" {
#include <time_block.h>
#include <gap.h>
#include <inttypes.h>
}


void test_time_codec_d64_d64(int64_t *time_data, uint64_t num_vals, int64_t *gap_array, uint64_t num_gaps,
                             uint64_t freq_nhz, int64_t start_n, uint8_t t_raw_type, uint8_t t_encoded_type,
                             size_t gap_array_size);


TEST(TimeCodecTestSuite, SimplePhillipsTest)
{
    // Create a Test Time Array

    // Declare the metadata
    uint64_t period = (2 * NANO) / 1000;  // 2 ms

    uint64_t num_vals = 10000;
    uint64_t freq_nhz = ((uint64_t)NANO * (uint64_t)NANO) / period;
    int64_t start_n = (int64_t)34 * NANO;

    // declare raw, encoded types
    uint8_t t_raw_type = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO;
    uint8_t t_encoded_type = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO;

    // Define a Gap Array
    uint64_t num_gaps = 20;

    int64_t gap_array[num_gaps * 2];

    uint64_t i;
    for(i=0; i<num_gaps; i++){
        gap_array[2 * i] = (int64_t)((i + 1) * (num_vals / (num_gaps + 4)));
        gap_array[(2 * i) + 1] = (int64_t)NANO * (int64_t)((i * 13 % 71) + 3);  // 13 and 71 are primes for diverse gaps.
    }

    // Convert Gap Array into a Timestamp Array
    auto * time_nano = (int64_t *)calloc(num_vals, sizeof(int64_t));

    // This function tested in GapCodecTests.cpp
    gap_int64_ns_gap_array_decode(gap_array, time_nano, num_vals, num_gaps,
                                  start_n, freq_nhz);

    // Do The thing.
    test_time_codec_d64_d64(time_nano, num_vals, gap_array, num_gaps,
                            freq_nhz, start_n, t_raw_type, t_encoded_type, sizeof(gap_array));

    free(time_nano);
}


TEST(TimeCodecTestSuite, SimpleNatusTest)
{
    // Create a Test Time Array

    // Declare the metadata
    uint64_t period = (2 * NANO) / 1000;  // 2 ms

    uint64_t num_vals = 10000;
    uint64_t freq_nhz = ((uint64_t)NANO * (uint64_t)NANO) / period;
    int64_t start_n = (int64_t)34 * NANO;

    // declare raw, encoded types
    uint8_t t_raw_type = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO;
    uint8_t t_encoded_type = T_TYPE_GAP_ARRAY_INT64_INDEX_NUM_SAMPLES;

    // Define a Gap Array
    uint64_t num_gaps = 20;

    int64_t gap_array[num_gaps * 2];

    uint64_t i;
    for(i=0; i<num_gaps; i++){
        gap_array[2 * i] = (int64_t)((i + 1) * (num_vals / (num_gaps + 4)));
        gap_array[(2 * i) + 1] = (int64_t)((i * 13 % 71) + 3);  // 13 and 71 are primes for diverse gaps.
    }

    // Convert Gap Array into a Timestamp Array
    auto * time_nano = (int64_t *)calloc(num_vals, sizeof(int64_t));

    // This function tested in GapCodecTests.cpp
    gap_int64_samples_decode(gap_array, time_nano, num_vals, num_gaps,
                                  start_n, freq_nhz);

    // Do The thing.
    test_time_codec_d64_d64(time_nano, num_vals, gap_array, num_gaps,
                            freq_nhz, start_n, t_raw_type, t_encoded_type, sizeof(gap_array));

    free(time_nano);
}


void test_time_codec_d64_d64(int64_t *time_data, uint64_t num_vals, int64_t *gap_array, uint64_t num_gaps,
                             uint64_t freq_nhz, int64_t start_n, uint8_t t_raw_type, uint8_t t_encoded_type,
                             size_t gap_array_size)
{
    // Init Header Struct
    block_metadata_t header;
    header.t_raw_type = t_raw_type;
    header.t_encoded_type = t_encoded_type;

    header.num_vals = num_vals;
    header.freq_nhz = freq_nhz;
    header.start_n = start_n;

    // Get Buffer Size
    size_t encoded_size = time_encode_buffer_get_size(time_data, &header);
    ASSERT_EQ(encoded_size, gap_array_size);

    // Encode
    auto * encoded_time = (int64_t *)calloc(encoded_size / sizeof(int64_t), sizeof(int64_t));
    time_encode(time_data, encoded_time, &header);

    // Check for correctly calculated sizes
    ASSERT_EQ(sizeof(int64_t) * header.num_vals, header.t_raw_size);
    ASSERT_EQ(gap_array_size, header.t_encoded_size);
    ASSERT_EQ(num_gaps, header.num_gaps);

    uint64_t i;
    for(i=0; i<num_gaps; i++){
        EXPECT_EQ(gap_array[2 * i], encoded_time[2 * i]);
        EXPECT_EQ(gap_array[(2 * i) + 1], encoded_time[(2 * i) + 1]);
    }

    // Decode
    auto * time_result = (int64_t *)calloc(header.t_raw_size / sizeof(int64_t), sizeof(int64_t));
    time_decode(time_result, encoded_time, &header);

    for(i=0; i<header.num_vals; i++){
        EXPECT_EQ(time_result[i], time_data[i]);
    }

    free(encoded_time);
    free(time_result);
}
