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
// Created by Will Dixon on 2021-06-14.
//
#include <value_p.h>
#include <delta.h>
#include <size.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t value_d64_encode(const int64_t *value_data, void *encoded_value, block_metadata_t *block_metadata,
                        const block_options_t *block_options)
{
    switch (block_metadata->v_encoded_type) {
        case V_TYPE_INT64:
            /* Int64 Value */
            memcpy(encoded_value, value_data, block_metadata->num_vals * sizeof(int64_t));
            return block_metadata->num_vals * sizeof(int64_t);

        case V_TYPE_DELTA_INT64:
            /* Int64 Delta */
            block_metadata->order = delta_lowest_entropy_encode_d64(
                    value_data, encoded_value, block_metadata->num_vals,
                    &(((int64_t *)encoded_value)[block_metadata->num_vals]),
                    value_delta_d64_get_size(block_metadata),
                    block_options->delta_order_min, block_options->delta_order_max);

            block_metadata->bytes_per_value = size_encode_d64(
                    encoded_value, block_metadata->num_vals, block_options->bytes_per_value_min);

            return (uint64_t)block_metadata->bytes_per_value * block_metadata->num_vals;

        default:
            printf("value type %u not supported. On line %d in file %s\n",
                   block_metadata->v_encoded_type, __LINE__, __FILE__);
            exit(1);
    }
}
