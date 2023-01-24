//
// Created by Will Dixon on 2021-05-12.
//
#include <my_compression.h>
#include <zstd.h>


size_t my_zstd_compress(unsigned char * data_in, unsigned char * compressed_out, size_t size_in,
                        size_t size_buf, int8_t compression_level){
    return ZSTD_compress(compressed_out, size_buf, data_in, size_in, compression_level);
}


size_t my_zstd_decompress(unsigned char *compressed_in, unsigned char *data_out, size_t size_in, size_t size_out)
{
    return ZSTD_decompress(data_out, size_out, compressed_in, size_in);
}

size_t get_zstd_compressBound(size_t size_in)
{
    return ZSTD_compressBound(size_in);
}
