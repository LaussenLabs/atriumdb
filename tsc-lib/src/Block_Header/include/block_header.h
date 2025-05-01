/*
    AtriumDB is a timeseries database software designed to best handle the unique features and
    challenges that arise from clinical waveform data.
        Copyright (C) 2023  The Hospital for Sick Children

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <https:www.gnu.org/licenses/>.
*/

//
// Created by Will Dixon on 2021-05-09.
//

#ifndef MINIMAL_SDK_BLOCK_HEADER_H
#define MINIMAL_SDK_BLOCK_HEADER_H

#include <stdint.h>

// Current Block Data
#define TSC_VERSION_NUM 2
#define TSC_VERSION_EXT 3
#define TSC_VERSION_EXT_PERIOD 4
#define TSC_NUM_CHANNELS 1


// Time Types
#define T_TYPE_TIMESTAMP_ARRAY_INT64_NANO 1
#define T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO 2
#define T_TYPE_GAP_ARRAY_INT64_INDEX_NUM_SAMPLES 3
#define T_TYPE_START_TIME_NUM_SAMPLES 4

// Value Types
#define V_TYPE_INT64 1
#define V_TYPE_DOUBLE 2

#define V_TYPE_DELTA_INT64 3
#define V_TYPE_XOR_DOUBLE 4


typedef struct block_metadata{
    // General Metadata
    uint8_t tsc_version_num;
    uint8_t tsc_version_ext;
    uint64_t num_vals;
    uint8_t num_channels;
    uint16_t meta_num_bytes;

    // Time Metadata
    uint8_t t_raw_type;
    uint64_t t_raw_size;
    uint8_t t_encoded_type;
    uint64_t t_encoded_size;
    uint8_t t_compression;
    int8_t t_compression_level;
    int64_t start_n;
    int64_t end_n;
    uint64_t freq_nhz;
    uint64_t num_gaps;
    uint64_t t_start_byte;
    uint64_t t_num_bytes;

    // Value Metadata

    /* Multi-Channel Mode OFF */
    //value_metadata_t *value_metadata; // Need to create this type

    /* Single-Channel Mode ON */
    uint8_t v_raw_type;
    uint64_t v_raw_size;
    uint8_t v_encoded_type;
    uint64_t v_encoded_size;
    uint8_t v_compression;
    int8_t v_compression_level;
    uint8_t bytes_per_value;
    uint8_t order;
    double max;
    double min;
    double mean;
    double std;
    double scale_m;
    double scale_b;
    uint64_t v_start_byte;
    uint64_t v_num_bytes;

}block_metadata_t;


typedef struct block_options{
    uint8_t delta_order_min;
    uint8_t delta_order_max;

    uint8_t bytes_per_value_min;
}block_options_t;

#endif //MINIMAL_SDK_BLOCK_HEADER_H
