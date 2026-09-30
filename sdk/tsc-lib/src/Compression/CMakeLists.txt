add_library(Compression include/compression.h src/compressor.c src/my_lz4.c src/my_compression.h src/my_zstd.c src/decompressor.c src/compress_buffer.c)

target_include_directories(Compression PUBLIC include)
target_include_directories(Compression PRIVATE src)
target_include_directories(Compression PRIVATE /usr/local/include)

if(APPLE)

    find_library(LZ4_LIBRARY NAMES lz4 liblz4 HINTS "/usr/local/lib")
    find_library(ZSTD_LIBRARY NAMES zstd libzstd HINTS "/usr/local/lib")

    target_link_directories(Compression PRIVATE /usr/local/lib)
    target_link_libraries(Compression PRIVATE ${LZ4_LIBRARY} ${ZSTD_LIBRARY})

elseif(CYGWIN OR WIN32 OR MINGW OR (CMAKE_SYSTEM MATCHES Windows))

    find_library(LZ4_LIBRARY NAMES lz4 liblz4 HINTS ${PROJECT_SOURCE_DIR}/bin)
    find_library(ZSTD_LIBRARY NAMES zstd libzstd HINTS ${PROJECT_SOURCE_DIR}/bin)

    target_link_directories(Compression PRIVATE ${PROJECT_SOURCE_DIR}/bin)
    target_link_libraries(Compression PRIVATE ${LZ4_LIBRARY} ${ZSTD_LIBRARY})

else()
    target_link_libraries(Compression PRIVATE lz4 zstd)

endif()

