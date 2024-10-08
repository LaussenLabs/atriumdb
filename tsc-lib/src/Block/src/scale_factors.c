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
// Created by Will Dixon on 2024-01-11.
//
#include <block_header.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>


void convert_value_data_to_analog(const void *value_data, double *analog_values, const block_metadata_t *headers,
                                  const uint64_t *analog_block_start_index_array, uint64_t num_blocks)
{
    uint64_t i;
    uint64_t j;
    char *data_pointer;
    block_metadata_t header;
    uint64_t analog_index;

    // Parallel processing of blocks
    #pragma omp parallel for default(none) private(i, j, data_pointer, header, analog_index) shared(headers, value_data, analog_values, analog_block_start_index_array, num_blocks)
    for(i = 0; i < num_blocks; i++) {
        header = headers[i];
        analog_index = analog_block_start_index_array[i];

        // This only works because int64_t and double are the same size.
        // Will break if some architecture has double != 8 bytes.
        data_pointer = (char*)value_data + (analog_block_start_index_array[i]) * sizeof(int64_t);

        if(header.v_raw_type == V_TYPE_INT64) {
            // Interpret as int64_t
            for(j = 0; j < header.num_vals; j++) {
                int64_t int_value;
                memcpy(&int_value, data_pointer, sizeof(int64_t));
                analog_values[analog_index + j] = (double)int_value;
                data_pointer += sizeof(int64_t);
            }
        }
        else if(header.v_raw_type == V_TYPE_DOUBLE) {
            // Interpret as double
            for(j = 0; j < header.num_vals; j++) {
                double double_value;
                memcpy(&double_value, data_pointer, sizeof(double));
                analog_values[analog_index + j] = double_value;
                data_pointer += sizeof(double);
            }
        } else {
            // Unsupported value type
            printf("ERROR: Header had an unsupported raw value type: %d\n", header.v_raw_type);
            continue;
        }

        // Apply scale factors
        if(header.scale_m != 0) {
            for(j = 0; j < header.num_vals; j++) {
                analog_values[analog_block_start_index_array[i] + j] =
                        (analog_values[analog_block_start_index_array[i] + j] * header.scale_m) + header.scale_b;
            }
        }
    }
}


void fill_nan_array_with_analog(const void *value_data, double *nan_analog_array, const block_metadata_t *headers,
                                const uint64_t *analog_block_start_index_array, uint64_t num_blocks,
                                const int64_t *times, int64_t start_ns, double period_ns, uint64_t nan_array_length)
{
    uint64_t i, j;
    char *data_pointer;
    block_metadata_t header;
    uint64_t analog_index;

    // Parallel processing of blocks
    #pragma omp parallel for default(none) private(i, j, data_pointer, header, analog_index) \
        shared(headers, value_data, nan_analog_array, analog_block_start_index_array, num_blocks, times, \
               start_ns, period_ns, nan_array_length)
    for(i = 0; i < num_blocks; i++) {
        header = headers[i];
        analog_index = analog_block_start_index_array[i];

        // Calculate data_pointer
        data_pointer = (char*)value_data + (analog_index) * sizeof(int64_t);
        const int64_t *block_times = times + analog_index;

        if(header.v_raw_type == V_TYPE_INT64) {
            // Interpret as int64_t
            for(j = 0; j < header.num_vals; j++) {
                int64_t int_value;
                memcpy(&int_value, data_pointer, sizeof(int64_t));
                double analog_value = (double)int_value;
                data_pointer += sizeof(int64_t);

                // Apply scale factors
                analog_value = analog_value * header.scale_m + header.scale_b;

                // Compute index into nan_analog_array
                int64_t time = block_times[j];
                int64_t nan_index = (int64_t)round((double)(time - start_ns) / period_ns);

                if(nan_index >= 0 && nan_index < (int64_t)nan_array_length) {
                    nan_analog_array[nan_index] = analog_value;
                }
            }
        }
        else if(header.v_raw_type == V_TYPE_DOUBLE) {
            // Interpret as double
            for(j = 0; j < header.num_vals; j++) {
                double double_value;
                memcpy(&double_value, data_pointer, sizeof(double));
                double analog_value = double_value;
                data_pointer += sizeof(double);

                // Apply scale factors
                analog_value = analog_value * header.scale_m + header.scale_b;

                // Compute index into nan_analog_array
                int64_t time = block_times[j];
                int64_t nan_index = (int64_t)round((double)(time - start_ns) / period_ns);

                if(nan_index >= 0 && nan_index < (int64_t)nan_array_length) {
                    nan_analog_array[nan_index] = analog_value;
                }
            }
        } else {
            // Unsupported value type
            printf("ERROR: Header had an unsupported raw value type: %d\n", header.v_raw_type);
            continue;
        }
    }
}