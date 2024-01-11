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
#include <omp.h>


void convert_value_data_to_analog(void *value_data, double *analog_values, block_metadata_t *headers,
                                  uint64_t num_blocks)
{
    char *data_pointer = (char*)value_data;
    uint64_t analog_index = 0;

    // move over values and convert type if needed
    for(uint64_t i=0; i<num_blocks; i++) {
        block_metadata_t header = headers[i];

        if(header.v_raw_type == V_TYPE_INT64) {
            // Interpret as int64_t
            for(uint64_t j=0; j<header.num_vals; j++) {
                int64_t int_value;
                memcpy(&int_value, data_pointer, sizeof(int64_t));
                analog_values[analog_index++] = (double)int_value;
                data_pointer += sizeof(int64_t);
            }
        }
        else if(header.v_raw_type == V_TYPE_DOUBLE) {
            // Interpret as double
            for(uint64_t j=0; j<header.num_vals; j++) {
                double double_value;
                memcpy(&double_value, data_pointer, sizeof(double));
                analog_values[analog_index++] = double_value;
                data_pointer += sizeof(double);
            }
        } else {
            // Unsupported value type
            printf("ERROR: Header had an unsupported raw value type: %d\n", header.v_raw_type);
            return;
        }
    }
    
    // Apply scale factors
    uint64_t analog_block_start_index = 0;
    for(uint64_t i=0; i<num_blocks; i++) {
        block_metadata_t header = headers[i];
        
        if(header.scale_m != 0) {
            for(uint64_t j=0; j<header.num_vals; j++) {
                analog_values[analog_block_start_index+j] = 
                        (analog_values[analog_block_start_index+j] * header.scale_m) + header.scale_b;
            } 
        }
        analog_block_start_index += header.num_vals;
    }
}

