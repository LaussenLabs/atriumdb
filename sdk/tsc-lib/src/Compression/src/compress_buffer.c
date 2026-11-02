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
// Created by Will Dixon on 2021-05-13.
//
#include <stdio.h>
#include <stdlib.h>
#include <my_compression.h>

size_t get_compress_buffer_size(size_t size_in, uint8_t compression_type)
{
    switch (compression_type) {
        case 1:
            /* No Compression */
            return 0;

        case 3:
            /* zstd */
            return get_zstd_compressBound(size_in);

        case 4:
            /* lz4 */
            // Same as below

        case 5:
            /* lz4hc */
            return get_lz4_compressBound(size_in);

        default:
            printf("compression type %u not supported.\n", compression_type);
            exit(1);

    }
}