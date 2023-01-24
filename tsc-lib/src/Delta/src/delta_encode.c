//
// Created by Will Dixon on 2021-06-11.
//

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <float.h>

#include <entropy.h>
#include <delta.h>

uint8_t delta_lowest_entropy_encode_d64(const int64_t *value_arr, int64_t *delta_arr, uint64_t num_values,
                                        void *entropy_buffer, size_t buffer_size, uint8_t min_order, uint8_t max_order)
{
    // Set the max_order to no more than num_values - 1
    max_order = (max_order >= (num_values-1)) ? num_values - 1 : max_order;
    // Copy the value array into delta_arr
    memcpy(delta_arr, value_arr, num_values * sizeof(int64_t));

    uint8_t best_order = 0;
    double min_entropy = DBL_MAX;

    if(min_order == 0){
        min_entropy = entropy_get_d64(delta_arr, num_values, entropy_buffer, buffer_size);
        min_order = 1;
    }

    uint16_t order, start;
    double entropy;
    for(order=min_order; order<=(uint16_t)max_order; order++){
        // Perform the Delta Transform.
        start = order - 1;
        delta_transform_d64(delta_arr, start, num_values - start);

        // Calculate the Entropy of the new array.
        entropy = entropy_get_d64(delta_arr, num_values, entropy_buffer, buffer_size);

        if(entropy < min_entropy){
            min_entropy = entropy;
            best_order = order;
        }
    }

    // Now get back the delta array corresponding to the best order.
    memcpy(delta_arr, value_arr, num_values * sizeof(int64_t));
    for(order=0; order<best_order; order++){
        delta_transform_d64(delta_arr, order, num_values - order);
    }

    return best_order;
}