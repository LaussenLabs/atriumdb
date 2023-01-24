//
// Created by Will Dixon on 2021-05-10.
//

#include <stdio.h>
#include <stdlib.h>
#include <my_compression.h>

size_t my_compress(const unsigned char * data_in, unsigned char * compressed_out, size_t size_in,
                   size_t size_buf, uint8_t compression_type, int8_t compression_level)
{
    switch (compression_type) {
        case 3:
            /* zstd */
            return my_zstd_compress(data_in, compressed_out, size_in, size_buf, compression_level);

        case 4:
            /* lz4 */
            return my_lz4_compress(data_in, compressed_out, size_in, size_buf);

        case 5:
            /* lz4hc */
            return my_lz4hc_compress(data_in, compressed_out, size_in, size_buf, compression_level);

        default:
            printf("compression type %u not supported.\n", compression_type);
            exit(1);

    }
}
