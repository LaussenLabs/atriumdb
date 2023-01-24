//
// Created by Will Dixon on 2021-06-09.
//
#include <freq_period_converter.h>
#include <stdint.h>

void gap_int64_samples_decode(const int64_t *gap_array, int64_t *time_data, uint64_t num_values, uint64_t num_gaps,
                              int64_t start_time_ns, uint64_t freq_nhz)
{
    // This function isn't going to work correctly if the period corresponding to the frequency isn't
    // an integer number of nanoseconds.

    // This function requires `time_data` to be initialized to all zeros (like by calloc).

    // Calculate the period of what would be continuous data.
    int64_t period_ns = (int64_t)uint64_nhz_freq_to_uint64_ns_period(freq_nhz);

    // Set the start time.
    time_data[0] = start_time_ns;

    // Place the jumps.
    uint64_t i;
    for(i=0; i<num_gaps; i++){
        // gap_array[2 * i] contains the index of the jump, and gap_array[(2 * i) + 1] contains the magnitude.
        time_data[gap_array[2 * i]] = gap_array[(2 * i) + 1] * period_ns;
    }

    // Place the continuous timestamps.
    for(i=0; i<num_values-1; i++){
        time_data[i+1] += time_data[i] + period_ns;
    }
}