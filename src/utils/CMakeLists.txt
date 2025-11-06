# Utils library
add_library(duorou_utils STATIC
    string_utils.cpp
    ../core/logger.cpp
    object_store.cpp
)

target_include_directories(duorou_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
)

# Set C++ standard
target_compile_features(duorou_utils PUBLIC cxx_std_17)