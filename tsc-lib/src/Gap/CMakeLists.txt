add_library(Gap include/gap.h src/gap_d64_index_dur_nano/gap_int64_ns_encoder.c src/gap_d64_index_dur_nano/gap_int64_ns_decoder.c src/gap_d64_index_num_samples/gap_d64_samples_decode.c src/gap_d64_index_num_samples/gap_d64_samples_encode.c)

target_include_directories(Gap PUBLIC include)
target_include_directories(Gap PRIVATE src)

target_link_libraries(Gap PRIVATE Freq_Period_Converter)
