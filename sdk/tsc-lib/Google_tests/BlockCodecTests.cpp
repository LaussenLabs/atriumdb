//
// Created by Will Dixon on 2021-06-18.
//
#include "gtest/gtest.h"
#define NANO 1000000000

extern "C" {
#include <block.h>

#include <time_block.h>
#include <value.h>
#include <gap.h>
#include <inttypes.h>
#include <string.h>
}

static void print_arr_d64(int64_t *arr, uint64_t arr_len);

static int64_t pow_mod_d64(int64_t base, int64_t exponent, int64_t mod);

TEST(BlockTestSuite, SimpleTest){
    uint8_t num_threads=8;

    uint64_t period = (2 * NANO) / 1000;  // 2 ms

    block_metadata_t header;
    block_options_t options;

    header.num_vals = 100000;
    header.freq_nhz = ((uint64_t)NANO * (uint64_t)NANO) / period;
    header.start_n = (int64_t)34 * NANO;

    // Define a Gap Array
    uint64_t num_gaps = header.num_vals / 250;

    int64_t gap_array[num_gaps * 2];

    uint64_t i;
    for(i=0; i<num_gaps; i++){
        gap_array[2 * i] = (int64_t)((i + 1) * (header.num_vals / (num_gaps + 4)));
        gap_array[(2 * i) + 1] = (int64_t)NANO * (int64_t)((i * 13 % 71) + 3);  // 13 and 71 are primes for diverse gaps.
    }

    // Convert Gap Array into a Timestamp Array
    auto * time_nano = (int64_t *)calloc(header.num_vals, sizeof(int64_t));

    // This function tested in GapCodecTests.cpp
    gap_int64_ns_gap_array_decode(gap_array, time_nano, header.num_vals, num_gaps,
                                  header.start_n, header.freq_nhz);

    // Create the values array.
    int64_t *arr = (int64_t *)calloc(header.num_vals, sizeof(int64_t));
    int8_t order = 3;
    int64_t mod = ((int64_t)1)<<30;

    for(i=0; i<header.num_vals; i++){
        arr[i] = pow_mod_d64((int64_t)i, order, mod);
    }

    options.bytes_per_value_min = 0;
    options.delta_order_min = 0;
    options.delta_order_max = 5;

    // declare raw, encoded types
    header.t_raw_type = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO;
    header.t_encoded_type = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO;
    header.v_raw_type = V_TYPE_INT64;
    header.v_encoded_type = V_TYPE_DELTA_INT64;

    header.t_compression = 1;
    header.v_compression = 5;
    header.t_compression_level = 0;
    header.v_compression_level = 9;

    uint64_t block_size = 32000;
    uint64_t num_blocks = (header.num_vals + block_size - 1) / block_size;

    uint64_t *t_block_start = (uint64_t *)malloc(num_blocks * sizeof(uint64_t));
    uint64_t *v_block_start = (uint64_t *)malloc(num_blocks * sizeof(uint64_t));
    uint64_t *byte_start = (uint64_t *)malloc(num_blocks * sizeof(uint64_t));
    block_metadata_t *headers = (block_metadata_t *)malloc(num_blocks * sizeof(block_metadata_t));


    ASSERT_EQ(sizeof(time_nano[0]), sizeof(int64_t)/sizeof(uint8_t));
    ASSERT_EQ(sizeof(arr[0]), sizeof(int64_t)/sizeof(uint8_t));
    uint64_t val_offset = 0;
    for(i=0; i<num_blocks; i++){
        memcpy(&(headers[i]), &header, sizeof(block_metadata_t));
        t_block_start[i] = val_offset * sizeof(time_nano[0]);
        v_block_start[i] = val_offset * sizeof(arr[0]);

        headers[i].start_n = time_nano[val_offset];
        val_offset += block_size;
        if(val_offset <= header.num_vals){
            headers[i].num_vals = block_size;
        }else{
            headers[i].num_vals = header.num_vals - (val_offset - block_size);
        }
        printf("block %u: %u: %u, %u - num_vals %u\n", i, val_offset, t_block_start[i], v_block_start[i], headers[i].num_vals);
    }

    // Check that initial parameters are correct
    for(i=0; i<num_blocks; i++){
        if(i < num_blocks - 1)
            ASSERT_EQ(headers[i].num_vals, block_size);

        ASSERT_GE(header.num_vals * sizeof(int64_t), v_block_start[i] + (headers[i].num_vals * sizeof(int64_t)));
        ASSERT_GE(header.num_vals * sizeof(int64_t), t_block_start[i] + (headers[i].num_vals * sizeof(int64_t)));

        ASSERT_EQ(headers[i].start_n, time_nano[t_block_start[i] / sizeof(time_nano[0])]);
        ASSERT_EQ(headers[i].freq_nhz, header.freq_nhz);

        ASSERT_EQ(headers[i].t_raw_type, header.t_raw_type);
        ASSERT_EQ(headers[i].t_encoded_type, header.t_encoded_type);
        ASSERT_EQ(headers[i].v_encoded_type, header.v_encoded_type);
        ASSERT_EQ(headers[i].t_compression, header.t_compression);
        ASSERT_EQ(headers[i].v_compression, header.v_compression);
        ASSERT_EQ(headers[i].t_compression_level, header.t_compression_level);
        ASSERT_EQ(headers[i].v_compression_level, header.v_compression_level);
    }
    ASSERT_EQ(headers[num_blocks - 1].num_vals, header.num_vals - (block_size*(num_blocks-1)));

    size_t buffer_size = block_get_buffer_size(time_nano, num_blocks, t_block_start,
                                               byte_start, headers);
    void *buffer = malloc(buffer_size);

    // Encode
    size_t byte_data_size = encode_blocks(time_nano, arr, buffer, num_blocks,
                                          t_block_start, v_block_start, byte_start, headers, &options, num_threads);

    uint64_t *t_byte_start = (uint64_t *)malloc(num_blocks * sizeof(uint64_t));

    size_t t_buf_offset = byte_data_size;
    for(i=0; i<num_blocks; i++){
        t_byte_start[i] = t_buf_offset;
        t_buf_offset += headers[i].t_encoded_size;
    }

    int64_t *res_time = (int64_t *) calloc(header.num_vals, sizeof(int64_t));
    int64_t *res_values = (int64_t *) calloc(header.num_vals, sizeof(int64_t));
    void *res_buffer = malloc(t_buf_offset);

    // Copy over buffers to simulate I/O
    memcpy(res_buffer, buffer, byte_data_size);

    decode_blocks(res_time, res_values, buffer, num_blocks, t_block_start, v_block_start, byte_start, t_byte_start, num_threads);

    for(i=0; i<header.num_vals; i++){
        printf("%u ", i);
        ASSERT_EQ(res_time[i], time_nano[i]);
        ASSERT_EQ(arr[i], res_values[i]);
    }

    free(buffer);
    free(t_byte_start);
    free(t_block_start);
    free(v_block_start);
    free(byte_start);
    free(headers);

    free(res_time);
    free(res_values);
    free(res_buffer);

    free(time_nano);
    free(arr);
}


static int64_t pow_mod_d64(int64_t base, int64_t exponent, int64_t mod)
{
    int64_t result = 1;
    int64_t i;
    for(i=0; i<exponent; i++){
        result = (result * base) % mod;
    }
    return result;
}