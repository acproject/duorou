# Model module CMakeLists.txt

# Set minimum CMake version
cmake_minimum_required(VERSION 3.15)

# Model library
set(MODEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sentence_piece.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/byte_pair_encoding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qwen_text_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qwen_vision_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qwen_image_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qwen_multimodal_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_factory.cpp
)

set(MODEL_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/model.h
    ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary.h
    ${CMAKE_CURRENT_SOURCE_DIR}/sentence_piece.h
    ${CMAKE_CURRENT_SOURCE_DIR}/byte_pair_encoding.h
    ${CMAKE_CURRENT_SOURCE_DIR}/text_processor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/qwen_text_model.h
    ${CMAKE_CURRENT_SOURCE_DIR}/qwen_vision_model.h
    ${CMAKE_CURRENT_SOURCE_DIR}/qwen_image_processor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/qwen_multimodal_model.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_factory.h
)

add_library(duorou_model STATIC ${MODEL_SOURCES} ${MODEL_HEADERS})

# Include directories
# Add include directories for GGUF parser
include_directories(${CMAKE_SOURCE_DIR}/src/extensions/ollama)
include_directories(${CMAKE_SOURCE_DIR}/src/extensions/ollama/gguf)

# Set standards and compile options
set_target_properties(duorou_model PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Link dependencies
# Assuming duorou and llama targets exist elsewhere in the project

# Tests and executables
add_executable(test_model ${CMAKE_CURRENT_SOURCE_DIR}/test_model.cpp)
add_dependencies(test_model duorou_model)
# Link against model and ollama extension for GGUF symbols
target_link_libraries(test_model PRIVATE duorou_model duorou_ollama_extension)

add_executable(tokenizer_cli ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_cli.cpp)
add_dependencies(tokenizer_cli duorou_model)
# Link against model and ollama extension (factory may touch GGUF symbols)
target_link_libraries(tokenizer_cli PRIVATE duorou_model duorou_ollama_extension)

add_executable(tokenizer_golden_test ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_golden_test.cpp)
add_dependencies(tokenizer_golden_test duorou_model)
# Link against model and ollama extension (for consistency)
target_link_libraries(tokenizer_golden_test PRIVATE duorou_model duorou_ollama_extension)

# 将 tokenizer_golden_test 纳入 CTest（使用环境变量驱动数据路径）
if(BUILD_TESTING)
    add_test(NAME TokenizerGoldenTest COMMAND tokenizer_golden_test)
    set_tests_properties(TokenizerGoldenTest PROPERTIES
        ENVIRONMENT "DUOROU_TOKENIZER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/golden_tokenizer;DUOROU_TOKENIZER_TYPE=bpe;DUOROU_ADD_SPECIAL=0;DUOROU_GOLDEN_ENCODE=${CMAKE_CURRENT_SOURCE_DIR}/golden_tokenizer/encode.tsv;DUOROU_GOLDEN_DECODE=${CMAKE_CURRENT_SOURCE_DIR}/golden_tokenizer/decode.tsv"
    )
endif()

# Install targets
install(TARGETS duorou_model ARCHIVE DESTINATION lib)
install(FILES ${MODEL_HEADERS} DESTINATION include/duorou/model)