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
// Created by Will Dixon on 2021-05-16.
//

#ifndef MINIMAL_SDK_GAP_H
#define MINIMAL_SDK_GAP_H

#include <stddef.h>
#include <stdint.h>

/* Int64 Index-Duration */
size_t gap_int64_ns_gap_array_get_size(const int64_t * time_data, uint64_t num_values, uint64_t period_ns);
uint64_t gap_int64_ns_time_array_encode(const int64_t * time_data, int64_t * gap_array, uint64_t num_values,
                                        uint64_t period_ns);
void gap_int64_ns_gap_array_decode(const int64_t *gap_array, int64_t *time_data, uint64_t num_values, uint64_t num_gaps,
                                   int64_t start_time_ns, uint64_t period_ns);

/* Int64 Index-Num Samples */
// `get_size()` can be reused from the Index-Duration type.
uint64_t gap_int64_samples_encode(const int64_t * time_data, int64_t * gap_array, uint64_t num_values,
                                  uint64_t period_ns);
void gap_int64_samples_decode(const int64_t *gap_array, int64_t *time_data, uint64_t num_values, uint64_t num_gaps,
                              int64_t start_time_ns, uint64_t period_ns);

#endif //MINIMAL_SDK_GAP_H
