//
// Created by Will Dixon on 2021-05-12.
//

#ifndef MINIMAL_SDK_MY_COMPRESSION_H
#define MINIMAL_SDK_MY_COMPRESSION_H

#include <stddef.h>  /* size_t */
#include <stdint.h>  /* int8_t */

size_t my_zstd_compress(unsigned char * data_in, unsigned char * compressed_out, size_t size_in,
                         size_t size_buf, int8_t compression_level);

size_t my_zstd_decompress(unsigned char *compressed_in, unsigned char *data_out, size_t size_in, size_t size_out);

size_t get_zstd_compressBound(size_t size_in);

size_t my_lz4_compress(unsigned char * data_in, unsigned char * compressed_out, size_t size_in, size_t size_buf);

size_t my_lz4hc_compress(unsigned char * data_in, unsigned char * compressed_out, size_t size_in,
                         size_t size_buf, int8_t compression_level);

size_t my_lz4_decompress(unsigned char *compressed_in, unsigned char *data_out, size_t size_in, size_t size_out);

size_t get_lz4_compressBound(size_t size_in);

#endif //MINIMAL_SDK_MY_COMPRESSION_H
