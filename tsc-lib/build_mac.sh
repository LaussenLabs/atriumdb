#!/bin/bash
#
# Build the native macOS shared library (libTSC.dylib) for the Python SDK and
# place it in sdk/bin where AtriumSDK loads it from.
#
# Requirements (install once):
#   - Xcode Command Line Tools:  xcode-select --install
#   - CMake:                     brew install cmake
#   - OpenMP runtime:            brew install libomp
#   - Compression libraries:     brew install lz4 zstd
#
# Usage:
#   ./build_mac.sh            # Release build (default)
#   ./build_mac.sh Debug      # Debug build
#
set -euo pipefail

# Always run from the directory this script lives in (tsc-lib).
cd "$(dirname "$0")"

BUILD_TYPE="${1:-Release}"
BUILD_DIR="cmake-build-macos"

# Remove old build files for a clean configure.
rm -rf "${BUILD_DIR}"

# Configure and build only the Block target (the shared library the SDK uses).
cmake -B"${BUILD_DIR}" -H. -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
cmake --build "${BUILD_DIR}" --target Block

# Copy the native dylib into the SDK's bin folder.
mkdir -p ../sdk/bin
cp "${BUILD_DIR}/src/Block/libTSC.dylib" ../sdk/bin/libTSC.dylib

echo "Built ../sdk/bin/libTSC.dylib (${BUILD_TYPE})"
