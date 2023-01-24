//
// Created by Will Dixon on 2021-05-14.
//

#ifndef MINIMAL_SDK_TIME_P_H
#define MINIMAL_SDK_TIME_P_H

#include <block_header.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

/* Time Array Nanosecond Int64 */
size_t time_timestamps_int64_nano_get_size(block_metadata_t * block_metadata);
size_t time_int64_nano_data_encode(const int64_t * time_data, void * encoded_time,
                                   block_metadata_t * block_metadata);
size_t time_int64_nano_data_decode(int64_t *time_data, const void *encoded_time,
                                   block_metadata_t * block_metadata);

/* Gap Array Index-Duration Nano */
size_t time_gap_int64_nano_get_size(const void *time_data, block_metadata_t *block_metadata);
size_t time_gap_int64_nano_encode(const int64_t * time_data, void * encoded_time,
                                  block_metadata_t * block_metadata);
size_t time_gap_int64_nano_decode(int64_t * time_data, const void * time_bytes,
                                  block_metadata_t * block_metadata);

/* Gap Array Index-Num Samples */
size_t time_gap_d64_num_samples_get_size(const void *time_data, block_metadata_t *block_metadata);
size_t time_gap_d64_num_samples_encode(const int64_t * time_data, void * encoded_time,
                                       block_metadata_t * block_metadata);
size_t time_gap_d64_num_samples_decode(int64_t * time_data, const void * time_bytes,
                                       block_metadata_t * block_metadata);


/* Helpers */
size_t gap_array_size(uint64_t num_gaps);


#endif //MINIMAL_SDK_TIME_P_H
