# Commons library
add_library(commons STATIC
    commons.cpp
)

# Set include directories for commons
target_include_directories(commons PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Set C++ standard for commons
set_target_properties(commons PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
