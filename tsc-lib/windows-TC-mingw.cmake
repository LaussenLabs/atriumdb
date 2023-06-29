# the name of the target operating system
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_VERSION 10.0)

# which compilers to use for C and C++
set(CMAKE_C_COMPILER   $ENV{MINGW_HOME}/bin/x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER $ENV{MINGW_HOME}/bin/x86_64-w64-mingw32-g++)

# where is the target environment located
set(CMAKE_FIND_ROOT_PATH $ENV{MINGW_HOME}
        ${CMAKE_SOURCE_DIR}/bin)

# link the lz4 and zstd libraries
set(ZSTD_LIBRARY ${CMAKE_SOURCE_DIR}/bin/libzstd_static.lib)
set(LZ4_LIBRARY ${CMAKE_SOURCE_DIR}/bin/liblz4_static.lib)

# adjust the default behavior of the FIND_XXX() commands:
# search programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# search headers and libraries in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)