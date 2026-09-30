#[SHARED here tells the program to make a .dll/.so file]
add_library(Block SHARED include/block.h src/block_buf.c src/block_encode.c src/block_decode.c src/scale_factors.c)

target_include_directories(Block PUBLIC include)
target_include_directories(Block PRIVATE src)

target_link_libraries(Block PUBLIC Block_Header)
target_link_libraries(Block PRIVATE Value Time Compression)

set_target_properties(Block PROPERTIES PUBLIC_HEADER include/block.h)
set_target_properties(Block PROPERTIES OUTPUT_NAME "TSC")