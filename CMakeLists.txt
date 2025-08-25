cmake_minimum_required(VERSION 3.14)
project(duorou VERSION 1.0.0 LANGUAGES C CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置编译选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译器特定选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
elseif(MSVC)
    add_compile_options(/W4)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi)
    else()
        add_compile_options(/O2 /DNDEBUG)
    endif()
endif()

# 平台检测
if(APPLE)
    set(DUOROU_PLATFORM "macos")
elseif(WIN32)
    set(DUOROU_PLATFORM "windows")
elseif(UNIX)
    set(DUOROU_PLATFORM "linux")
else()
    set(DUOROU_PLATFORM "unknown")
endif()

message(STATUS "Building for platform: ${DUOROU_PLATFORM}")

# 选项配置
option(DUOROU_ENABLE_CUDA "Enable CUDA support" OFF)
option(DUOROU_ENABLE_METAL "Enable Metal support (macOS only)" OFF)
option(DUOROU_ENABLE_VULKAN "Enable Vulkan support" OFF)
option(DUOROU_ENABLE_OPENCL "Enable OpenCL support" OFF)
option(DUOROU_BUILD_TESTS "Build tests" OFF)
option(DUOROU_BUILD_EXAMPLES "Build examples" OFF)

# 自动启用平台特定选项
if(APPLE AND NOT DUOROU_ENABLE_METAL)
    set(DUOROU_ENABLE_METAL ON)
    message(STATUS "Auto-enabled Metal support for macOS")
endif()

# 查找依赖包
find_package(PkgConfig REQUIRED)

# GTK4 依赖
pkg_check_modules(GTK4 REQUIRED gtk4)
if(GTK4_FOUND)
    message(STATUS "Found GTK4: ${GTK4_VERSION}")
else()
    message(FATAL_ERROR "GTK4 not found. Please install GTK4 development packages.")
endif()

# 线程支持
find_package(Threads REQUIRED)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp
    ${GTK4_INCLUDE_DIRS}
)

# 链接目录
link_directories(
    ${GTK4_LIBRARY_DIRS}
)

# 编译定义
add_definitions(
    ${GTK4_CFLAGS_OTHER}
    -DDUOROU_VERSION="${PROJECT_VERSION}"
    -DDUOROU_PLATFORM="${DUOROU_PLATFORM}"
)

# llama.cpp 子项目配置
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

# 根据选项配置 llama.cpp
if(DUOROU_ENABLE_CUDA)
    set(LLAMA_CUDA ON CACHE BOOL "" FORCE)
    message(STATUS "Enabled CUDA support for llama.cpp")
endif()

if(DUOROU_ENABLE_METAL)
    set(LLAMA_METAL ON CACHE BOOL "" FORCE)
    message(STATUS "Enabled Metal support for llama.cpp")
endif()

if(DUOROU_ENABLE_VULKAN)
    set(LLAMA_VULKAN ON CACHE BOOL "" FORCE)
    message(STATUS "Enabled Vulkan support for llama.cpp")
endif()

# stable-diffusion.cpp 子项目配置
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# 根据选项配置 stable-diffusion.cpp
if(DUOROU_ENABLE_CUDA)
    set(SD_CUDA ON CACHE BOOL "" FORCE)
    message(STATUS "Enabled CUDA support for stable-diffusion.cpp")
endif()

if(DUOROU_ENABLE_METAL)
    set(SD_METAL ON CACHE BOOL "" FORCE)
    message(STATUS "Enabled Metal support for stable-diffusion.cpp")
endif()

if(DUOROU_ENABLE_VULKAN)
    set(SD_VULKAN ON CACHE BOOL "" FORCE)
    message(STATUS "Enabled Vulkan support for stable-diffusion.cpp")
endif()

if(DUOROU_ENABLE_OPENCL)
    set(SD_OPENCL ON CACHE BOOL "" FORCE)
    message(STATUS "Enabled OpenCL support for stable-diffusion.cpp")
endif()

# 添加子项目
add_subdirectory(third_party/llama.cpp)
add_subdirectory(third_party/stable-diffusion.cpp)

# 源文件
set(DUOROU_SOURCES
    src/main.cpp
    src/core/application.cpp
    src/core/config_manager.cpp
    src/core/logger.cpp
    src/core/model_manager.cpp
    src/core/workflow_engine.cpp
    src/models/llama_wrapper.cpp
    src/models/stable_diffusion_wrapper.cpp
    src/gui/main_window.cpp
    src/gui/chat_view.cpp
    src/gui/image_view.cpp
    src/gui/settings_dialog.cpp
    src/gui/system_tray.cpp
    src/utils/file_utils.cpp
    src/utils/string_utils.cpp
    src/utils/image_utils.cpp
)

# 头文件
set(DUOROU_HEADERS
    src/core/application.h
    src/core/config_manager.h
    src/core/logger.h
    src/core/model_manager.h
    src/core/workflow_engine.h
    src/models/llama_wrapper.h
    src/models/stable_diffusion_wrapper.h
    src/gui/main_window.h
    src/gui/chat_view.h
    src/gui/image_view.h
    src/gui/settings_dialog.h
    src/gui/system_tray.h
    src/utils/file_utils.h
    src/utils/string_utils.h
    src/utils/image_utils.h
)

# 创建源文件目录（如果不存在）
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/models)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/gui)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

# 主可执行文件
add_executable(duorou ${DUOROU_SOURCES} ${DUOROU_HEADERS})

# 链接库
target_link_libraries(duorou
    llama
    stable-diffusion
    ${GTK4_LIBRARIES}
    Threads::Threads
)

# 平台特定链接
if(APPLE)
    target_link_libraries(duorou
        "-framework Foundation"
        "-framework AppKit"
    )
elseif(WIN32)
    target_link_libraries(duorou
        user32
        shell32
        ole32
    )
elseif(UNIX)
    target_link_libraries(duorou
        dl
        m
    )
endif()

# 编译器特定链接选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(duorou PRIVATE
        $<$<CONFIG:Debug>:-rdynamic>
    )
endif()

# 安装配置
install(TARGETS duorou
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 安装资源文件
install(DIRECTORY resources/
    DESTINATION share/duorou/resources
    PATTERN "*.in" EXCLUDE
)

# 安装桌面文件 (Linux)
if(UNIX AND NOT APPLE)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/resources/duorou.desktop.in
        ${CMAKE_CURRENT_BINARY_DIR}/duorou.desktop
        @ONLY
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/duorou.desktop
        DESTINATION share/applications
    )
endif()

# 测试配置
if(DUOROU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 示例配置
if(DUOROU_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 打包配置
set(CPACK_PACKAGE_NAME "duorou")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Multi-platform desktop AI application")
set(CPACK_PACKAGE_VENDOR "Duorou Project")
set(CPACK_PACKAGE_CONTACT "duorou@example.com")

if(WIN32)
    set(CPACK_GENERATOR "NSIS")
elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop")
else()
    set(CPACK_GENERATOR "DEB;RPM")
endif()

include(CPack)

# 显示配置摘要
message(STATUS "")
message(STATUS "Duorou Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Platform: ${DUOROU_PLATFORM}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  GTK4 Version: ${GTK4_VERSION}")
message(STATUS "  CUDA Support: ${DUOROU_ENABLE_CUDA}")
message(STATUS "  Metal Support: ${DUOROU_ENABLE_METAL}")
message(STATUS "  Vulkan Support: ${DUOROU_ENABLE_VULKAN}")
message(STATUS "  OpenCL Support: ${DUOROU_ENABLE_OPENCL}")
message(STATUS "  Build Tests: ${DUOROU_BUILD_TESTS}")
message(STATUS "  Build Examples: ${DUOROU_BUILD_EXAMPLES}")
message(STATUS "")