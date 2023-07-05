#!/bin/bash

#****windows binary****
#remove old build files
rm -rf cmake-build-release/**;

#load windows toolchain file amd setup build files
cmake -Bcmake-build-release -DCMAKE_TOOLCHAIN_FILE=windows-TC-mingw.cmake -DCMAKE_BUILD_TYPE='Release';

#build binary
cmake --build cmake-build-release --target Block;

#make the directory if it doesnt exist
mkdir -p ../sdk/bin

#copy windows binary to output directory
cp cmake-build-release/src/Block/libTSC.dll ../sdk/bin/libTSC.dll;

#****linux binary****
rm -rf cmake-build-release/**;

#setup cmake flags and build files 
cmake -Bcmake-build-release -H. -DCMAKE_BUILD_TYPE='Release';
cmake --build cmake-build-release --target Block;

#copy linux binary to output directory
cp cmake-build-release/src/Block/libTSC.so ../sdk/bin/libTSC.so;


