add_library(Value include/value.h src/value_buf.c src/value_p.h src/Value_Int64/value_d64_buf.c src/Value_Double/value_f64_buf.c src/Value_Delta_Int64/value_delta_d64_buf.c src/Value_Xor_Double/value_xor_f64_buf.c src/value_encode.c src/Value_Delta_Int64/value_delta_d64_encode.c src/Value_Double/value_f64_encode.c src/Value_Int64/value_d64_encode.c src/Value_Xor_Double/value_xor_f64_encode.c src/value_decode.c src/Value_Delta_Int64/value_delta_d64_decode.c src/Value_Double/value_f64_decode.c src/Value_Int64/value_d64_decode.c src/Value_Xor_Double/value_xor_f64_decode.c)

target_include_directories(Value PUBLIC include)
target_include_directories(Value PRIVATE src)

target_link_libraries(Value PUBLIC Block_Header)
target_link_libraries(Value PRIVATE Delta Size)