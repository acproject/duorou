# KVCache 模块 CMakeLists.txt

# 设置最低 CMake 版本
cmake_minimum_required(VERSION 3.16)

# 收集源文件
set(KVCACHE_SOURCES
    cache.cpp
    causal.cpp
    encoder.cpp
    wrapper.cpp
)

# 收集头文件
set(KVCACHE_HEADERS
    cache.h
    causal.h
    encoder.h
    wrapper.h
)

# 创建静态库
add_library(duorou_kvcache STATIC ${KVCACHE_SOURCES})

# 设置目标属性
set_target_properties(duorou_kvcache PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# 设置编译选项
if(MSVC)
    target_compile_options(duorou_kvcache PRIVATE
        /W4
        /O2
    )
else()
    target_compile_options(duorou_kvcache PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -O2
    )
endif()

# 设置包含目录
target_include_directories(duorou_kvcache PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 如果是 Debug 模式，添加调试信息
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        target_compile_options(duorou_kvcache PRIVATE /Od /Zi)
    else()
        target_compile_options(duorou_kvcache PRIVATE -g -O0)
    endif()
    target_compile_definitions(duorou_kvcache PRIVATE DEBUG=1)
endif()

# 安装头文件（可选）
install(FILES ${KVCACHE_HEADERS}
    DESTINATION include/duorou/kvcache
)

# 安装库文件（可选）
install(TARGETS duorou_kvcache
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)