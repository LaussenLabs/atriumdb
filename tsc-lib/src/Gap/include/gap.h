//
// Created by Will Dixon on 2021-05-16.
//

#ifndef MINIMAL_SDK_GAP_H
#define MINIMAL_SDK_GAP_H

#include <stddef.h>
#include <stdint.h>

/* Int64 Index-Duration */
size_t gap_int64_ns_gap_array_get_size(const int64_t * time_data, uint64_t num_values, uint64_t freq_nhz);
uint64_t gap_int64_ns_time_array_encode(const int64_t * time_data, int64_t * gap_array, uint64_t num_values,
                                        uint64_t freq_nhz);
void gap_int64_ns_gap_array_decode(const int64_t *gap_array, int64_t *time_data, uint64_t num_values, uint64_t num_gaps,
                                   int64_t start_time_ns, uint64_t freq_nhz);

/* Int64 Index-Num Samples */
// `get_size()` can be reused from the Index-Duration type.
uint64_t gap_int64_samples_encode(const int64_t * time_data, int64_t * gap_array, uint64_t num_values,
                                        uint64_t freq_nhz);
void gap_int64_samples_decode(const int64_t *gap_array, int64_t *time_data, uint64_t num_values, uint64_t num_gaps,
                                   int64_t start_time_ns, uint64_t freq_nhz);

#endif //MINIMAL_SDK_GAP_H
