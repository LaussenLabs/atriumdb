add_library(Delta include/delta.h src/delta_encode.c src/delta_arr_transform_d64.c src/delta_decode.c)

target_include_directories(Delta PUBLIC include)
target_include_directories(Delta PRIVATE src)

target_link_libraries(Delta Entropy)