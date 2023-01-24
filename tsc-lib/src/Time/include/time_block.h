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
