# Ollama扩展模块

# 设置模块名称
set(OLLAMA_EXTENSION_NAME "duorou_ollama_extension")

# 收集源文件
set(OLLAMA_EXTENSION_SOURCES
    gguf_parser.cpp
    qwen25vl_modular_engine.cpp
    ollama_model_manager.cpp
    ollama_path_resolver.cpp
    text_processor.cpp
    sentencepiece_processor.cpp
    algorithms/bpe_processor.cpp
    algorithms/ggml_attention.cpp
    algorithms/ggml_feed_forward.cpp
)

# 收集头文件
set(OLLAMA_EXTENSION_HEADERS
    gguf_parser.h
    qwen25vl_modular_engine.h
    ollama_model_manager.h
    ollama_path_resolver.h
    text_processor.h
    sentencepiece_processor.h
    algorithms/bpe_processor.h
    algorithms/ggml_attention.h
    algorithms/ggml_feed_forward.h
)

# 创建静态库
add_library(${OLLAMA_EXTENSION_NAME} STATIC
    ${OLLAMA_EXTENSION_SOURCES}
    ${OLLAMA_EXTENSION_HEADERS}
)

# 设置目标属性
set_target_properties(${OLLAMA_EXTENSION_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# 测试程序已移除

# 包含目录
target_include_directories(${OLLAMA_EXTENSION_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# 编译定义
target_compile_definitions(${OLLAMA_EXTENSION_NAME}
    PRIVATE
        DUOROU_OLLAMA_EXTENSION
)

# 编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${OLLAMA_EXTENSION_NAME} PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
    )
elseif(MSVC)
    target_compile_options(${OLLAMA_EXTENSION_NAME} PRIVATE
        /W4
        /wd4100  # 未引用的形参
    )
endif()

# 链接库
target_link_libraries(${OLLAMA_EXTENSION_NAME} PRIVATE
    ggml-base
    ggml-cpu
)

# 平台特定链接库
if(APPLE)
    target_link_libraries(${OLLAMA_EXTENSION_NAME} PRIVATE
        "-framework Accelerate"
    )
endif()

# 安装规则
install(TARGETS ${OLLAMA_EXTENSION_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${OLLAMA_EXTENSION_HEADERS}
    DESTINATION include/duorou/extensions/ollama
)

# 测试（如果启用）
if(BUILD_TESTING)
    # 添加测试子目录
    # add_subdirectory(tests)
endif()

# 示例（如果启用）
if(BUILD_EXAMPLES)
    # 添加示例子目录
    # add_subdirectory(examples)
endif()

# 打印配置信息
message(STATUS "Ollama Extension Configuration:")
message(STATUS "  Target: ${OLLAMA_EXTENSION_NAME}")
message(STATUS "  Sources: ${OLLAMA_EXTENSION_SOURCES}")
message(STATUS "  Headers: ${OLLAMA_EXTENSION_HEADERS}")
message(STATUS "  Include dirs: ${CMAKE_CURRENT_SOURCE_DIR}")