cmake_minimum_required(VERSION 3.13)
project(Minimal_SDK C)

set(CMAKE_C_STANDARD 99)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


if(MSVC)
    #[[/W3 means show level 1-3 warnings (production quality warnings
    /WX treats all compiler warnings as errors]]
    add_compile_options(/W3 /WX)
elseif(MINGW OR CYGWIN OR WIN32 OR (CMAKE_SYSTEM MATCHES Windows))
    add_compile_options(-W -Wall -fpic)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -static -fopenmp")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -static -fopenmp")

else()
    #[[linux]]
    #[[Position Independent Code (compiler option -fPIC)]]
    add_compile_options(-W -Wall -fpic)
    #[[sets environment variable to its self plus -03 -fopenmp
    -static makes libraries static so you dont have to include the dll files in the distribution]]
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -fopenmp")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -fopenmp")

endif()

add_subdirectory(src)
add_subdirectory(Google_tests)
add_subdirectory(Functional_Tests)