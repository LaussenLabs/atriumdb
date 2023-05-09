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
// Created by Will Dixon on 2021-05-30.
//

#include <time_p.h>
#include <gap.h>
#include <string.h>


size_t time_gap_int64_nano_decode(int64_t * time_data, const void * time_bytes,
                                  block_metadata_t * block_metadata)
{
    switch (block_metadata->t_raw_type) {
        case T_TYPE_TIMESTAMP_ARRAY_INT64_NANO:
            gap_int64_ns_gap_array_decode((int64_t *)time_bytes, time_data, block_metadata->num_vals,
                                          block_metadata->num_gaps, block_metadata->start_n, block_metadata->freq_nhz);

            // Return size of timestamp array.
            return block_metadata->num_vals * sizeof(int64_t);

        case T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO:
            /* Convert Time Array to Gap Array */
            // Copy Gap Array.
            memcpy(time_data, time_bytes, gap_array_size(block_metadata->num_gaps));

            // Return size of gap array.
            return gap_array_size(block_metadata->num_gaps);

        default:
            printf("time encoded type %u not supported. On line %d in file %s\n",
                   block_metadata->t_raw_type, __LINE__, __FILE__);
            exit(1);
    }
}


