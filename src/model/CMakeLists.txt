# Model module CMakeLists.txt

# Set minimum CMake version
cmake_minimum_required(VERSION 3.15)

# Model library
set(MODEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sentence_piece.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/byte_pair_encoding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_factory.cpp
)

set(MODEL_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/model.h
    ${CMAKE_CURRENT_SOURCE_DIR}/vocabulary.h
    ${CMAKE_CURRENT_SOURCE_DIR}/sentence_piece.h
    ${CMAKE_CURRENT_SOURCE_DIR}/byte_pair_encoding.h
    ${CMAKE_CURRENT_SOURCE_DIR}/text_processor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_factory.h
)

add_library(duorou_model STATIC ${MODEL_SOURCES} ${MODEL_HEADERS})

# Include directories
# Add include directories for GGUF parser
include_directories(${CMAKE_SOURCE_DIR}/src/extensions/ollama)
include_directories(${CMAKE_SOURCE_DIR}/src/fs/gguf)

# Set standards and compile options
set_target_properties(duorou_model PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Link dependencies
target_link_libraries(duorou_model PUBLIC duorou_utils)

# Conditionally link Ollama extension when enabled
if(DUOROU_ENABLE_OLLAMA)
    target_link_libraries(duorou_model PUBLIC duorou_ollama_extension)
endif()
# Assuming duorou and llama targets exist elsewhere in the project

# Link against model and ollama extension for GGUF symbols
# target_link_libraries(test_model PRIVATE duorou_model duorou_ollama_extension)

# 创建 tokenizer_golden_test 可执行文件（用于词元器黄金测试），仅在启用 Ollama 扩展时构建
if(DUOROU_ENABLE_OLLAMA)
    add_executable(tokenizer_golden_test ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_golden_test.cpp)

    target_include_directories(tokenizer_golden_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/extensions/ollama
        ${CMAKE_SOURCE_DIR}/src/fs/gguf
    )

    target_link_libraries(tokenizer_golden_test
        duorou_model
        duorou_ollama_extension
    )

    set_target_properties(tokenizer_golden_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

    # 将 tokenizer_golden_test 纳入 CTest（使用环境变量驱动数据路径）
    if(BUILD_TESTING)
        add_test(NAME TokenizerGoldenTest COMMAND tokenizer_golden_test)
        set_tests_properties(TokenizerGoldenTest PROPERTIES
            ENVIRONMENT "DUOROU_TOKENIZER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/golden_tokenizer;DUOROU_TOKENIZER_TYPE=bpe;DUOROU_ADD_SPECIAL=0;DUOROU_GOLDEN_ENCODE=${CMAKE_CURRENT_SOURCE_DIR}/golden_tokenizer/encode.tsv;DUOROU_GOLDEN_DECODE=${CMAKE_CURRENT_SOURCE_DIR}/golden_tokenizer/decode.tsv"
        )
    endif()
endif()

# Install targets
install(TARGETS duorou_model ARCHIVE DESTINATION lib)
install(FILES ${MODEL_HEADERS} DESTINATION include/duorou/model)