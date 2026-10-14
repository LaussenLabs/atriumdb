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
// Created by Will Dixon on 2021-06-13.
//

#ifndef MINIMAL_SDK_VALUE_P_H
#define MINIMAL_SDK_VALUE_P_H

#include <stdint.h>
#include <stddef.h>

#include <block_header.h>

/* Int64 Value */
size_t value_d64_get_size(block_metadata_t *block_metadata);
size_t value_d64_encode(const int64_t *value_data, void *encoded_value, block_metadata_t *block_metadata,
                        const block_options_t *block_options);
void value_d64_decode(void *value_data, block_metadata_t *block_metadata);

/* Double Value */
size_t value_f64_get_size(block_metadata_t *block_metadata);
size_t value_f64_encode(const double * value_data, void *encoded_value, block_metadata_t * block_metadata);
void value_f64_decode(void *value_data, block_metadata_t *block_metadata);

/* Int64 Delta */
size_t value_delta_d64_get_size(block_metadata_t *block_metadata);
size_t value_delta_d64_encode(const int64_t *value_data, void *encoded_value, block_metadata_t * block_metadata);
void value_delta_d64_decode(void *value_data, block_metadata_t *block_metadata);

/* Double XOR */
size_t value_xor_f64_get_size(block_metadata_t *block_metadata);
size_t value_xor_f64_encode(const double * value_data, void *encoded_value, block_metadata_t * block_metadata);
void value_xor_f64_decode(void *value_data, block_metadata_t *block_metadata);


#endif //MINIMAL_SDK_VALUE_P_H
