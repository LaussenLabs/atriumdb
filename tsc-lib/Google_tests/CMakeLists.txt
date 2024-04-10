#[need to use cygwin compiler to make google tests work in windows]
project(Google_tests)
add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Use link_directories before the target that requires these directories
link_directories(${Minimal_SDK_BINARY_DIR}/src/Block ${Minimal_SDK_SOURCE_DIR}/bin)

add_executable(Google_Tests_run CompressionTests.cpp FreqPeriodConverterTest.cpp GapCodecTests.cpp TimeCodecTests.cpp
        DistinctTests.cpp EntropyTests.cpp DeltaTests.cpp SizeTests.cpp ValuesCodecTests.cpp BlockCodecTests.cpp)

# Since link_directories() has been called, these directories will be considered during linking.
target_link_libraries(Google_Tests_run Compression Freq_Period_Converter Gap Time Distinct Entropy
        Delta Size Value Block)

target_link_libraries(Google_Tests_run gtest gtest_main pthread)
