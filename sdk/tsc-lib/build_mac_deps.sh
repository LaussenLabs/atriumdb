#!/bin/bash
#
# Build the libraries libTSC links against on macOS: static lz4 and zstd, and the LLVM OpenMP runtime (libomp).
#
# Homebrew builds its libraries for the macOS version they were bottled on, which would raise the minimum
# macOS version of any wheel that bundles them. Building from source lets the libraries target the same
# macOS version as the wheel.
#
# Usage:
#   MACOSX_DEPLOYMENT_TARGET=11.0 ./build_mac_deps.sh <install-prefix>   (the prefix must not exist yet)
#
# Then build libTSC against them with CMAKE_PREFIX_PATH=<install-prefix>.
#
set -euo pipefail

: "${MACOSX_DEPLOYMENT_TARGET:?MACOSX_DEPLOYMENT_TARGET must be set}"

LZ4_VERSION=1.10.0
LZ4_SHA256=537512904744b35e232912055ccf8ec66d768639ff3abe5788d90d792ec5f48b
ZSTD_VERSION=1.5.7
ZSTD_SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
LLVM_VERSION=21.1.8
OPENMP_SHA256=856b023748b41ac7b2c83fd8e9f765ff48a4df2fe6777d2811ef7c7ed8f2f977
LLVM_CMAKE_SHA256=85735f20fd8c81ecb0a09abb0c267018475420e93b65050cc5b7634eab744de9

# Refuse an existing prefix: a stale or planted .dylib in <prefix>/lib would be linked instead of the libraries built here.
mkdir "$1"
PREFIX="$(cd "$1" && pwd)"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
cd "${WORK_DIR}"

# fetch <url> <sha256>: download an archive, verify its checksum and unpack it into the working directory.
fetch() {
    local archive
    archive="$(basename "$1")"
    curl -fsSL --proto '=https' --tlsv1.2 --retry 3 --max-time 600 -o "${archive}" "$1"
    echo "$2  ${archive}" | shasum -a 256 -c -
    tar -xf "${archive}"
}

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET}"
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

# build <name> <source-dir> [cmake-args...]: configure, build and install one library.
build() {
    local name="$1" source_dir="$2"
    shift 2
    cmake -S "${source_dir}" -B "build-${name}" "${CMAKE_ARGS[@]}" "$@"
    cmake --build "build-${name}" --parallel
    cmake --install "build-${name}"
}

fetch "https://github.com/lz4/lz4/releases/download/v${LZ4_VERSION}/lz4-${LZ4_VERSION}.tar.gz" "${LZ4_SHA256}"
build lz4 "lz4-${LZ4_VERSION}/build/cmake" \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DLZ4_BUILD_CLI=OFF -DLZ4_BUILD_LEGACY_LZ4C=OFF

fetch "https://github.com/facebook/zstd/releases/download/v${ZSTD_VERSION}/zstd-${ZSTD_VERSION}.tar.gz" "${ZSTD_SHA256}"
build zstd "zstd-${ZSTD_VERSION}/build/cmake" \
    -DZSTD_BUILD_SHARED=OFF -DZSTD_BUILD_STATIC=ON -DZSTD_BUILD_PROGRAMS=OFF -DZSTD_BUILD_TESTS=OFF

# The OpenMP runtime uses the LLVM CMake modules, which it expects in a directory next to its own source.
LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}"
fetch "${LLVM_URL}/openmp-${LLVM_VERSION}.src.tar.xz" "${OPENMP_SHA256}"
fetch "${LLVM_URL}/cmake-${LLVM_VERSION}.src.tar.xz" "${LLVM_CMAKE_SHA256}"
mv "openmp-${LLVM_VERSION}.src" openmp
mv "cmake-${LLVM_VERSION}.src" cmake
# An absolute install name lets delocate find the library from the libraries linked against it.
build omp openmp -DCMAKE_INSTALL_NAME_DIR="${PREFIX}/lib" -DLIBOMP_INSTALL_ALIASES=OFF

echo "Installed lz4, zstd and libomp into ${PREFIX} (macOS ${MACOSX_DEPLOYMENT_TARGET})"
