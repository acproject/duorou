# Utils library
add_library(duorou_utils
    string_utils.cpp
)

target_include_directories(duorou_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set C++ standard
target_compile_features(duorou_utils PUBLIC cxx_std_17)