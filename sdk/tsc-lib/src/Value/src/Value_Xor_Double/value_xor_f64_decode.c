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

#include <stdio.h>
#include <stdlib.h>


void value_xor_f64_decode(void *value_data, block_metadata_t *block_metadata)
{
    switch (block_metadata->v_raw_type) {
        case V_TYPE_XOR_DOUBLE:
            /* Double XOR */
            break;

        default:
            printf("value type %u not supported. On line %d in file %s\n",
                   block_metadata->v_raw_type, __LINE__, __FILE__);
            exit(1);
    }
}
