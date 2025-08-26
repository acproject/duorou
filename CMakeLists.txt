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
    
    # Windows vcpkg 支持
    if(DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
        message(STATUS "Using local vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
    else()
        message(WARNING "vcpkg not found. Please install vcpkg or set VCPKG_ROOT environment variable.")
        message(STATUS "To install vcpkg:")
        message(STATUS "  git clone https://github.com/Microsoft/vcpkg.git")
        message(STATUS "  cd vcpkg && .\\bootstrap-vcpkg.bat")
        message(STATUS "  Set VCPKG_ROOT environment variable to vcpkg directory")
    endif()
    
    # Windows 特定的 vcpkg 包配置
    set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet")
    
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
if(WIN32)
    # Windows 平台使用 vcpkg 查找 GTK4
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GTK4 gtk4)
    endif()
    
    if(NOT GTK4_FOUND)
        # 尝试使用 vcpkg 安装的 GTK4
        find_package(gtk QUIET)
        if(gtk_FOUND)
            set(GTK4_FOUND TRUE)
            set(GTK4_LIBRARIES gtk::gtk)
            set(GTK4_INCLUDE_DIRS ${gtk_INCLUDE_DIRS})
            message(STATUS "Found GTK4 via vcpkg")
        else()
            message(WARNING "GTK4 not found via vcpkg. Please install with: vcpkg install gtk:x64-windows")
            message(STATUS "Alternative: install MSYS2 and use: pacman -S mingw-w64-x86_64-gtk4")
        endif()
    endif()
else()
    # 非 Windows 平台使用 pkg-config
    pkg_check_modules(GTK4 REQUIRED gtk4)
endif()

if(GTK4_FOUND)
    message(STATUS "Found GTK4: ${GTK4_VERSION}")
else()
    message(FATAL_ERROR "GTK4 not found. Please install GTK4 development packages.")
endif()

# 线程支持
find_package(Threads REQUIRED)

# nlohmann_json 支持
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    # 如果系统没有安装，使用 FetchContent 下载
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    )
    FetchContent_MakeAvailable(nlohmann_json)
    message(STATUS "Downloaded nlohmann_json via FetchContent")
else()
    message(STATUS "Found nlohmann_json: ${nlohmann_json_VERSION}")
endif()

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
    -DHAVE_GTK
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

# 创建源文件目录（如果不存在）
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/models)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/gui)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/api)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/invokeai/core)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/invokeai/models)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/invokeai/pipeline)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/invokeai/utils)

# 源文件列表（当前为空，待后续开发时添加）
# 取消注释以下源文件并创建对应的.cpp文件来构建可执行文件
set(DUOROU_SOURCES_LIST
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
    src/gui/chat_session.cpp
    src/gui/chat_session_manager.cpp
    src/utils/file_utils.cpp
    src/utils/string_utils.cpp
    src/utils/image_utils.cpp
    src/api/api_server.cpp
    src/invokeai/core/invokeai_engine.cpp
    src/invokeai/models/model_manager.cpp
    src/invokeai/pipeline/image_pipeline.cpp
)

# 头文件列表（当前为空，待后续开发时添加）
set(DUOROU_HEADERS_LIST
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
    src/gui/chat_session.h
    src/gui/chat_session_manager.h
    src/utils/file_utils.h
    src/utils/string_utils.h
    src/utils/image_utils.h
    src/api/api_server.h
    src/invokeai/core/invokeai_engine.h
    src/invokeai/models/model_manager.h
    src/invokeai/pipeline/image_pipeline.h
)

# 检查实际存在的源文件
set(DUOROU_SOURCES)
set(DUOROU_HEADERS)
foreach(source_file ${DUOROU_SOURCES_LIST})
    set(full_path "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
    if(EXISTS "${full_path}")
        list(APPEND DUOROU_SOURCES "${source_file}")
        message(STATUS "Found source file: ${source_file}")
    else()
        message(WARNING "Source file not found: ${source_file} (checked: ${full_path})")
    endif()
endforeach()

foreach(header_file ${DUOROU_HEADERS_LIST})
    set(full_path "${CMAKE_CURRENT_SOURCE_DIR}/${header_file}")
    if(EXISTS "${full_path}")
        list(APPEND DUOROU_HEADERS "${header_file}")
        message(STATUS "Found header file: ${header_file}")
    else()
        message(WARNING "Header file not found: ${header_file} (checked: ${full_path})")
    endif()
endforeach()

# 如果没有找到源文件，直接使用源文件列表
if(NOT DUOROU_SOURCES)
    message(WARNING "No source files found through file existence check, using full source list")
    set(DUOROU_SOURCES ${DUOROU_SOURCES_LIST})
    set(DUOROU_HEADERS ${DUOROU_HEADERS_LIST})
endif()

# 只有当存在源文件时才构建可执行文件
if(DUOROU_SOURCES)
    list(LENGTH DUOROU_SOURCES DUOROU_SOURCES_COUNT)
    message(STATUS "Found ${DUOROU_SOURCES_COUNT} source files, building executable...")
    
    # 主可执行文件
    add_executable(duorou ${DUOROU_SOURCES} ${DUOROU_HEADERS})
    
    # 链接库
    target_link_libraries(duorou
        llama
        stable-diffusion
        ${GTK4_LIBRARIES}
        Threads::Threads
        nlohmann_json::nlohmann_json
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
else()
    message(STATUS "No source files found. Skipping executable build.")
    message(STATUS "To build the executable, create the source files listed in DUOROU_SOURCES_LIST.")
    message(STATUS "Start with creating src/main.cpp as the entry point.")
endif()

# 注意：链接库配置已移动到上面的条件块中

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