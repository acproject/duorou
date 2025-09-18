# Model module CMakeLists.txt

# Set minimum CMake version
cmake_minimum_required(VERSION 3.16)

# Model library sources
set(MODEL_SOURCES
    vocabulary.cpp
    byte_pair_encoding.cpp
    qwen_text_model.cpp
    qwen_vision_model.cpp
    qwen_image_processor.cpp
    qwen_multimodal_model.cpp
    sentence_piece.cpp
    model.cpp
    simple_integration_demo.cpp
    integrated_model_example.cpp
)

# Model library headers
set(MODEL_HEADERS
    base_model.h
    text_processor.h
    vocabulary.h
    byte_pair_encoding.h
    sentence_piece.h
    model.h
    qwen_text_model.h
    qwen_vision_model.h
    qwen_image_processor.h
    qwen_multimodal_model.h
    simple_integration_demo.h
    integrated_model_example.h
)

# Create model static library
add_library(duorou_model STATIC ${MODEL_SOURCES})

# Set target properties
set_target_properties(duorou_model PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Include directories
target_include_directories(duorou_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/ml
    ${CMAKE_SOURCE_DIR}/src/kvcache
    ${CMAKE_SOURCE_DIR}/src/extensions/ollama/gguf
)

# Compiler flags
target_compile_options(duorou_model PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

# Link libraries
target_link_libraries(duorou_model PRIVATE
    duorou_ml
    duorou_kvcache
    duorou_ollama_extension
)

# Test executable
add_executable(test_model test_model.cpp)

# Integration test executable
add_executable(test_integration test_integration.cpp)

# Link test with model library
target_link_libraries(test_model PRIVATE duorou_model)
target_link_libraries(test_integration PRIVATE duorou_model)

# Set test properties
set_target_properties(test_model PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

set_target_properties(test_integration PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Add test to CTest
enable_testing()
add_test(NAME ModelTests COMMAND test_model)
add_test(NAME IntegrationTests COMMAND test_integration)

# Install targets
install(TARGETS duorou_model
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${MODEL_HEADERS}
    DESTINATION include/duorou/model
)