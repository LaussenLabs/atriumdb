#!/bin/bash

#change into project directory
cd /tsc-lib;

#****windows binary****
#remove old build files from build dir
rm -rf cmake-build-debug/**;

#load windows toolchain file and setup build files
cmake -Bcmake-build-debug -DCMAKE_TOOLCHAIN_FILE=windows-TC-mingw.cmake -DCMAKE_BUILD_TYPE='Debug';

#build binary
cmake --build cmake-build-debug --target Block;

#copy binary to output directory
cp /tsc-lib/cmake-build-debug/src/Block/libTSC.dll /tsc-lib/built_bin/libTSC_debug.dll;

#****linux binary****
rm -rf cmake-build-debug/**;

#setup cmake flags and build files 
cmake -Bcmake-build-debug -H. -DCMAKE_BUILD_TYPE='Debug';
cmake --build cmake-build-debug --target Block;

#copy binary to output directory
cp /tsc-lib/cmake-build-debug/src/Block/libTSC.so /tsc-lib/built_bin/libTSC_debug.so;


