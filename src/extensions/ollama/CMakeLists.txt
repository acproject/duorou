# Ollama扩展模块

# 设置模块名称
set(OLLAMA_EXTENSION_NAME "duorou_ollama_extension")

# 收集源文件
set(OLLAMA_EXTENSION_SOURCES
    gguf_parser.cpp
    ollama_model_manager.cpp
    ollama_path_resolver.cpp
    inference_engine.cpp
    ../../fs/ggml/ggml_wrapper.cpp
    ../../fs/gguf/gguf_wrapper.cpp
)

# llama.cpp源文件
set(LLAMA_CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp/src/llama-vocab.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp/src/unicode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp/src/unicode-data.cpp
)

# 收集头文件
set(OLLAMA_EXTENSION_HEADERS
    gguf_parser.h
    ollama_model_manager.h
    ollama_path_resolver.h
    inference_engine.h
    ../../fs/ggml/ggml_wrapper.h
    ../../fs/gguf/gguf_wrapper.h
)

# 创建静态库
add_library(${OLLAMA_EXTENSION_NAME} STATIC
    ${OLLAMA_EXTENSION_SOURCES}
    ${OLLAMA_EXTENSION_HEADERS}
    ${LLAMA_CPP_SOURCES}
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
        ${CMAKE_CURRENT_SOURCE_DIR}/../../fs/ggml
        ${CMAKE_CURRENT_SOURCE_DIR}/../../fs/gguf
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp/ggml/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# 链接库
target_link_libraries(${OLLAMA_EXTENSION_NAME}
    PUBLIC
        duorou_ml
        duorou_kvcache
        llama
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

# 安装规则
install(TARGETS ${OLLAMA_EXTENSION_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${OLLAMA_EXTENSION_HEADERS}
    DESTINATION include/duorou/extensions/ollama
)

# 测试程序已移除

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