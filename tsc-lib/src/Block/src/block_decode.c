//
// Created by Will Dixon on 2021-06-18.
//
#include <block_header.h>
#include <string.h>

#include <time_block.h>
#include <value.h>
#include <compression.h>
#include <omp.h>

void decode_blocks(void *time_data, void *value_data, void *encoded_bytes, uint64_t num_blocks,
                     const uint64_t *t_block_start, const uint64_t *v_block_start, const uint64_t *byte_start,
                     const uint64_t *t_byte_start, uint16_t num_threads)
{
    // Convert to 8-bit pointers.
    uint8_t *encoded_bytes_u8 = (uint8_t *)encoded_bytes;
    uint8_t *time_data_u8 = (uint8_t *)time_data;
    uint8_t *value_data_u8 = (uint8_t *)value_data;

    // Define some temp pointers.
    uint8_t *cur_encoded_bytes;

    uint64_t i;
    uint8_t j;
    block_metadata_t *header;

    //set number of threads to use
    omp_set_dynamic(1);
    omp_set_num_threads(num_threads);

    #pragma omp parallel for default(none) private(i, j, cur_encoded_bytes, header) \
    shared(time_data_u8, value_data_u8, encoded_bytes_u8, t_block_start, v_block_start, byte_start, num_blocks, t_byte_start) \
    collapse(2)
    for(i=0; i<num_blocks; i++){
        for(j=0; j<2; j++){
            cur_encoded_bytes = &(encoded_bytes_u8[byte_start[i]]);
            header = (block_metadata_t *)cur_encoded_bytes;
            if(j){
                // Decode Time
                if(header->t_compression != 1){
                    // Decompress
                    my_decompress(&(cur_encoded_bytes[header->t_start_byte]),
                                  &(encoded_bytes_u8[t_byte_start[i]]),
                                  header->t_num_bytes, header->t_encoded_size,
                                  header->t_compression);

                    time_decode(&(time_data_u8[t_block_start[i]]), &(encoded_bytes_u8[t_byte_start[i]]), header);
                } else{
                    time_decode(&(time_data_u8[t_block_start[i]]), &(cur_encoded_bytes[header->t_start_byte]), header);
                }
            } else{
                // Decode Values
                if(header->v_compression != 1){
                    // Decompress
                    my_decompress(&(cur_encoded_bytes[header->v_start_byte]),
                                  &(value_data_u8[v_block_start[i]]),
                                  header->v_num_bytes, header->v_encoded_size,
                                  header->v_compression);
                } else{
                    // Copy into result array
                    memcpy(&(value_data_u8[v_block_start[i]]),
                           &(cur_encoded_bytes[header->v_start_byte]),
                           header->v_num_bytes);
                }
                value_decode(&(value_data_u8[v_block_start[i]]), header);
            }
        }
    }
}
