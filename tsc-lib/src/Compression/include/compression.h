//
// Created by Will Dixon on 2021-05-09.
//

#ifndef MINIMAL_SDK_COMPRESSION_H
#define MINIMAL_SDK_COMPRESSION_H

/* --- Dependency --- */
#include <stddef.h>   /* size_t */
#include <stdint.h>

size_t get_compress_buffer_size(size_t size_in, uint8_t compression_type);

size_t my_compress(const unsigned char * data_in, unsigned char * compressed_out, size_t size_in,
                   size_t size_buf, uint8_t compression_type, int8_t compression_level);

size_t my_decompress(unsigned char *compressed_in, unsigned char *data_out, size_t size_in, size_t size_out,
                     uint8_t compression_type);

#endif //MINIMAL_SDK_COMPRESSION_H
