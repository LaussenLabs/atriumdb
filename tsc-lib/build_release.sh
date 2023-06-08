#!/bin/bash

#change into project directory
cd /tsc-lib;

#****windows binary****
#remove old build files
rm -rf cmake-build-release/**;

#load windows toolchain file amd setup build files
cmake -Bcmake-build-release -DCMAKE_TOOLCHAIN_FILE=windows-TC-mingw.cmake -DCMAKE_BUILD_TYPE='Release';

#build binary
cmake --build cmake-build-release --target Block;

#copy binary to output directory
cp /tsc-lib/cmake-build-release/src/Block/libTSC.dll /tsc-lib/built_bin/libTSC.dll;

#****linux binary****
rm -rf cmake-build-release/**;

#setup cmake flags and build files 
cmake -Bcmake-build-release -H. -DCMAKE_BUILD_TYPE='Release';
cmake --build cmake-build-release --target Block;

#copy binary to output directory
cp /tsc-lib/cmake-build-release/src/Block/libTSC.so /tsc-lib/built_bin/libTSC.so;


