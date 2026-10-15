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

/*
 * Doubles stored as int64 k plus a scale and an offset, where each value is exactly (k - offset) / scale.
 * The caller checks exactness before choosing this type. Encode input and encoded output both start with
 * the scale and offset (two doubles), followed by k. Decoding needs no multiply, so it gives the same bits
 * on every platform.
 */
#include <value_p.h>
#include <delta.h>
#include <size.h>

#include <string.h>

#define SCALE_BYTES (2 * sizeof(double))

size_t value_scaled_d64_get_size(block_metadata_t *block_metadata)
{
    return SCALE_BYTES + value_delta_d64_get_size(block_metadata);
}

size_t value_scaled_d64_encode(const void *value_data, void *encoded_value, block_metadata_t *block_metadata,
                               const block_options_t *block_options)
{
    const int64_t *k = (const int64_t *)((const uint8_t *)value_data + SCALE_BYTES);
    int64_t *delta_arr = (int64_t *)((uint8_t *)encoded_value + SCALE_BYTES);

    memcpy(encoded_value, value_data, SCALE_BYTES);
    block_metadata->order = delta_lowest_entropy_encode_d64(
            k, delta_arr, block_metadata->num_vals, &(delta_arr[block_metadata->num_vals]),
            value_delta_d64_get_size(block_metadata), block_options->delta_order_min, block_options->delta_order_max);
    block_metadata->bytes_per_value = size_encode_d64(
            delta_arr, block_metadata->num_vals, block_options->bytes_per_value_min);

    return SCALE_BYTES + (uint64_t)block_metadata->bytes_per_value * block_metadata->num_vals;
}

void value_scaled_d64_decode(void *value_data, block_metadata_t *block_metadata)
{
    double scale, offset;
    memcpy(&scale, value_data, sizeof(double));
    memcpy(&offset, (uint8_t *)value_data + sizeof(double), sizeof(double));
    memmove(value_data, (uint8_t *)value_data + SCALE_BYTES,
            (uint64_t)block_metadata->bytes_per_value * block_metadata->num_vals);

    size_decode_d64(value_data, block_metadata->num_vals, block_metadata->bytes_per_value);
    delta_decode_d64(value_data, block_metadata->num_vals, block_metadata->order);

    int64_t *k = (int64_t *)value_data;
    double *values = (double *)value_data;
    uint64_t i;
    for(i=0; i<block_metadata->num_vals; i++){
        values[i] = ((double)k[i] - offset) / scale;
    }
}
