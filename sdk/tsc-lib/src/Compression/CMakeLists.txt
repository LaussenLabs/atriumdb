add_library(Compression include/compression.h src/compressor.c src/my_lz4.c src/my_compression.h src/my_zstd.c src/decompressor.c src/compress_buffer.c)

target_include_directories(Compression PUBLIC include)
target_include_directories(Compression PRIVATE src)

#[[Search the system paths and CMAKE_PREFIX_PATH. The Windows build links the static archives in bin]]
find_library(LZ4_LIBRARY NAMES lz4 liblz4 lz4_static NAMES_PER_DIR HINTS ${PROJECT_SOURCE_DIR}/bin)
find_library(ZSTD_LIBRARY NAMES zstd libzstd zstd_static NAMES_PER_DIR HINTS ${PROJECT_SOURCE_DIR}/bin)

if(NOT LZ4_LIBRARY OR NOT ZSTD_LIBRARY)
    message(FATAL_ERROR "The lz4 and zstd libraries are required "
            "(Debian: liblz4-dev libzstd-dev, RHEL: lz4-devel libzstd-devel, macOS: brew install lz4 zstd).")
endif()

target_link_libraries(Compression PRIVATE ${LZ4_LIBRARY} ${ZSTD_LIBRARY})
