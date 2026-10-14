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

#ifndef MINIMAL_SDK_TIME_BLOCK_H
#define MINIMAL_SDK_TIME_BLOCK_H

#include <block_header.h>
#include <stddef.h>


/* encoder */
size_t time_encode_buffer_get_size(const void * time_data, block_metadata_t * block_metadata);
size_t time_encode(const void * time_data, void *encoded_time, block_metadata_t * block_metadata);

/* decoder */
void time_decode(void * time_data, const void *encoded_time, block_metadata_t * block_metadata);

#endif //MINIMAL_SDK_TIME_BLOCK_H
