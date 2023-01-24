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