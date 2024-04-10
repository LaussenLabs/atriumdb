#!/bin/bash

cd /atriumdb/tsc-lib

#****windows binary****
#remove old build files
rm -rf cmake-build-release/** &&

cd cmake-build-release &&

# Load windows toolchain file and setup build files
cmake .. -DCMAKE_TOOLCHAIN_FILE=../windows-TC-mingw.cmake -DCMAKE_BUILD_TYPE='Release' &&

# Build binary
cmake --build . --target Block &&

# Make the output directory if it doesn't exist and navigate back to the parent directory
mkdir -p ../../sdk/bin &&
cp src/Block/libTSC.dll ../../sdk/bin/libTSC.dll &&

cd .. &&

#****linux binary****
rm -rf cmake-build-release/** &&

# Create the build directory again after cleaning it and navigate into it
mkdir -p cmake-build-release &&
cd cmake-build-release &&

# Setup cmake flags and build files for Linux
cmake .. -DCMAKE_BUILD_TYPE='Release' &&
cmake --build . --target Block &&

# Copy linux binary to output directory
cp src/Block/libTSC.so ../../sdk/bin/libTSC.so &&

# Return to the original top directory
cd ..