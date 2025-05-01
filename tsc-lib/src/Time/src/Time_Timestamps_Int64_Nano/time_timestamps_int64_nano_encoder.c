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
// Created by Will Dixon on 2021-05-14.
//
#include <time_p.h>
#include <gap.h>
#include <string.h>


size_t time_int64_nano_data_encode(const int64_t * time_data, void * encoded_time,
                                   block_metadata_t * block_metadata)
{
    /* pick either raw nhz or converted ns‐period, once */
    int64_t period_arg = block_metadata->freq_nhz;
    if (block_metadata->tsc_version_ext == TSC_VERSION_EXT) {
        period_arg = (int64_t)uint64_nhz_freq_to_uint64_ns_period(block_metadata->freq_nhz);
    } else if (block_metadata->tsc_version_ext != TSC_VERSION_EXT_PERIOD) {
        printf("tsc_version_ext %u not supported. On line %d in file %s\n",
               block_metadata->tsc_version_ext, __LINE__, __FILE__);
        exit(1);
    }

    switch (block_metadata->t_encoded_type) {
        case T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            memcpy(encoded_time, time_data, block_metadata->num_vals * sizeof(int64_t));
            return block_metadata->num_vals * sizeof(int64_t);

        case T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:
            /* Convert Time Array to Gap Array (Duration ns) */
            // Record Number of Gaps.
            block_metadata->num_gaps = gap_int64_ns_time_array_encode(
                    time_data, (int64_t *)encoded_time, block_metadata->num_vals, period_arg);
            // Return size of gap array.
            return gap_array_size(block_metadata->num_gaps);

        case T_TYPE_GAP_ARRAY_INT64_INDEX_NUM_SAMPLES:
            /* Convert Time Array to Gap Array (Num Samples) */
            // Record Number of Gaps.
            block_metadata->num_gaps = gap_int64_samples_encode(
                    time_data, (int64_t *)encoded_time, block_metadata->num_vals, period_arg);

            // Return size of gap array.
            return gap_array_size(block_metadata->num_gaps);

        default:
            printf("time encoded type %u not supported. On line %d in file %s\n",
                   block_metadata->t_encoded_type, __LINE__, __FILE__);
            exit(1);
    }
}

