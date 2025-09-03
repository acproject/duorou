# Ollama扩展模块

# 设置模块名称
set(OLLAMA_EXTENSION_NAME "duorou_ollama_extension")

# 收集源文件
set(OLLAMA_EXTENSION_SOURCES
    ollama_model_loader.cpp
    model_path_manager.cpp
    modelfile_parser.cpp
    compatibility_checker.cpp
    gguf_modifier.cpp
    config_manager.cpp
)

# 收集头文件
set(OLLAMA_EXTENSION_HEADERS
    ollama_model_loader.h
    model_path_manager.h
    modelfile_parser.h
    compatibility_checker.h
    gguf_modifier.h
    config_manager.h
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

# 创建测试程序
add_executable(test_ollama test_ollama.cpp)
target_link_libraries(test_ollama 
    ${OLLAMA_EXTENSION_NAME}
    llama
    ggml
    ggml-cpu
    ggml-blas
    ggml-metal
    ggml-base
)
set_target_properties(test_ollama PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 包含目录
target_include_directories(${OLLAMA_EXTENSION_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
)

# 链接依赖
target_link_libraries(${OLLAMA_EXTENSION_NAME}
    PRIVATE
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