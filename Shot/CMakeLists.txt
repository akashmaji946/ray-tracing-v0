cmake_minimum_required(VERSION 3.18)
project(Shot LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
)

# Gather source files
file(GLOB_RECURSE SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cu
    ${CMAKE_SOURCE_DIR}/common/*.cpp
    ${CMAKE_SOURCE_DIR}/common/*.cu
)

# Add main.cu
set(MAIN_CU ${CMAKE_SOURCE_DIR}/main.cu)

# Add executable
add_executable(Shot ${MAIN_CU} ${SRC_FILES})

# Set CUDA architectures if needed (optional)
# set_target_properties(Shot PROPERTIES CUDA_ARCHITECTURES 52 60 61 75 86)

# For header-only libraries, nothing else is needed
