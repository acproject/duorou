cmake_minimum_required(VERSION 3.16)
project(llama_cpp_extension)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp/src
)

# Define source files
set(EXTENSION_SOURCES
    arch_mapping.cpp
    model_config_manager.cpp
    vision_model_handler.cpp
    attention_handler.cpp
    compatibility_checker.cpp
    gguf_modifier.cpp
)

# Define header files
set(EXTENSION_HEADERS
    arch_mapping.h
    model_config_manager.h
    vision_model_handler.h
    attention_handler.h
    compatibility_checker.h
    gguf_modifier.h
)

# Create static library for the extension
add_library(llama_cpp_extension STATIC
    ${EXTENSION_SOURCES}
    ${EXTENSION_HEADERS}
)

# Link with llama.cpp library (using object files for now)
# Note: In a real project, you would link with the actual llama.cpp library
# target_link_libraries(llama_cpp_extension
#     llama
#     ggml
# )

# Compiler flags
target_compile_options(llama_cpp_extension PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
    -O3
)

# Add integration test executable
add_executable(integration_test
    integration_test.cpp
)

# Link integration test with the extension library
target_link_libraries(integration_test
    llama_cpp_extension
    # llama
    # ggml
)

# Add test target
enable_testing()
add_test(NAME integration_test COMMAND integration_test)

# Install targets
install(TARGETS llama_cpp_extension
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${EXTENSION_HEADERS}
    DESTINATION include/llama_cpp_extension
)

# Print configuration summary
message(STATUS "=== Llama.cpp Extension Configuration ===")
message(STATUS "Extension sources: ${EXTENSION_SOURCES}")
message(STATUS "Extension headers: ${EXTENSION_HEADERS}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "=========================================")