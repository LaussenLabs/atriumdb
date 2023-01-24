//
// Created by Will Dixon on 2021-05-13.
//
#include <stdio.h>
#include <stdlib.h>
#include <my_compression.h>

size_t my_decompress(unsigned char *compressed_in, unsigned char *data_out, size_t size_in, size_t size_out,
                     uint8_t compression_type)
{
    switch (compression_type) {
        case 3:
            /* zstd */
            return my_zstd_decompress(compressed_in, data_out, size_in, size_out);

        case 4:
            /* lz4 */
            // Same as below.

        case 5:
            /* lz4hc */
            return my_lz4_decompress(compressed_in, data_out, size_in, size_out);

        default:
            printf("compression type %u not supported.\n", compression_type);
            exit(1);

    }
}