#[SHARED here tells the program to make a .dll/.so file]
add_library(Block SHARED include/block.h src/block_buf.c src/block_encode.c src/block_decode.c src/scale_factors.c)

target_include_directories(Block PUBLIC include)
target_include_directories(Block PRIVATE src)

target_link_libraries(Block PUBLIC Block_Header)
target_link_libraries(Block PRIVATE Value Time Compression)

#[[The Windows build takes OpenMP from the -fopenmp flag in the top level CMakeLists.txt, which links it statically]]
if(NOT WIN32)
    target_link_libraries(Block PRIVATE OpenMP::OpenMP_C)
endif()

set_target_properties(Block PROPERTIES OUTPUT_NAME "TSC")

#[[The python package loads the library from atriumdb/bin]]
install(TARGETS Block LIBRARY DESTINATION atriumdb/bin RUNTIME DESTINATION atriumdb/bin)

#[[An editable install imports the package from the source tree, so the library is installed there as well]]
if(SKBUILD_STATE STREQUAL "editable")
    set(TSC_SOURCE_PACKAGE_BIN ${CMAKE_SOURCE_DIR}/../atriumdb/bin)
    install(TARGETS Block LIBRARY DESTINATION ${TSC_SOURCE_PACKAGE_BIN} RUNTIME DESTINATION ${TSC_SOURCE_PACKAGE_BIN})
endif()