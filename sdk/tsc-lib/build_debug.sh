#!/bin/bash

cd /atriumdb/sdk/tsc-lib

#****windows binary****
#remove old build files from build dir
rm -rf cmake-build-debug/**;

#load windows toolchain file and setup build files
cmake -Bcmake-build-debug -DCMAKE_TOOLCHAIN_FILE=windows-TC-mingw.cmake -DCMAKE_BUILD_TYPE='Debug';

#build binary
cmake --build cmake-build-debug --target Block;

#make the directory if it doesnt exist
mkdir -p ../atriumdb/bin

#copy windows binary to output directory
cp cmake-build-debug/src/Block/libTSC.dll ../atriumdb/bin/libTSC_debug.dll;

#****linux binary****
rm -rf cmake-build-debug/**;

#setup cmake flags and build files 
cmake -Bcmake-build-debug -H. -DCMAKE_BUILD_TYPE='Debug';
cmake --build cmake-build-debug --target Block;

#copy linux binary to output directory
cp cmake-build-debug/src/Block/libTSC.so ../atriumdb/bin/libTSC_debug.so;


