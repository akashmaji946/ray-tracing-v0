#!/bin/bash

# Build script for Shot project

echo "Building Shot project..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build the project
make -j$(nproc)

echo "Build complete! Executable is in build/Shot"

