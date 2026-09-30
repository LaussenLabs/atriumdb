cmake_minimum_required(VERSION 3.15...3.31)
project(tsc-lib C)

set(CMAKE_C_STANDARD 99)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(TSC_BUILD_TESTS "Build the unit and functional tests" OFF)

if(APPLE)
    #[[Apple's compiler ships without an OpenMP runtime, so look for Homebrew's keg-only libomp]]
    list(APPEND CMAKE_SYSTEM_PREFIX_PATH /opt/homebrew/opt/libomp /usr/local/opt/libomp)
endif()

find_package(OpenMP COMPONENTS C)
if(NOT OpenMP_C_FOUND)
    message(FATAL_ERROR "An OpenMP runtime is required (macOS: brew install libomp).")
endif()

if(MSVC)
    #[[/W3 means show level 1-3 warnings (production quality warnings
    /WX treats all compiler warnings as errors]]
    add_compile_options(/W3 /WX)
elseif(MINGW OR CYGWIN OR WIN32 OR (CMAKE_SYSTEM MATCHES Windows))
    add_compile_options(-W -Wall -fpic)
    #[[Older binutils leave ASLR and DEP off unless asked]]
    add_link_options(-Wl,--dynamicbase,--nxcompat,--high-entropy-va)
    #[[-static with -fopenmp links the OpenMP runtime in, so the dll has no dependencies to distribute]]
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -static -fopenmp")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -static -fopenmp")

else()
    #[[linux]]
    #[[Position Independent Code (compiler option -fPIC)]]
    add_compile_options(-W -Wall -fpic -fstack-protector-strong)
    if(NOT APPLE)
        #[[Full RELRO: resolve every symbol at load time, then make the GOT read-only]]
        add_link_options(-Wl,-z,relro,-z,now)
    endif()
    #[[sets environment variable to its self plus -03
    -static makes libraries static so you dont have to include the dll files in the distribution]]
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")

endif()

add_subdirectory(src)

if(TSC_BUILD_TESTS)
    add_subdirectory(Google_tests)
    add_subdirectory(Functional_Tests)
endif()
