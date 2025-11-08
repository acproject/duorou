cmake_minimum_required(VERSION 3.14)
project(duorou VERSION 1.0.0 LANGUAGES C CXX)

# 启用 CTest/CTest 相关变量 (提供 BUILD_TESTING 选项)
include(CTest)
# 启用测试收集（生成CTestTestfile.cmake）
enable_testing()

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
    add_compile_options(/W4 /utf-8)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi /RTC1)
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

option(DUOROU_BUILD_EXAMPLES "Build examples" OFF)
# 新增：可选构建 GUI（GTK4）
option(DUOROU_BUILD_GUI "Build GUI components and executable (requires GTK4)" ON)

# 自动启用平台特定选项
if(APPLE AND NOT DUOROU_ENABLE_METAL)
    set(DUOROU_ENABLE_METAL ON)
    message(STATUS "Auto-enabled Metal support for macOS")
endif()

# 查找依赖包
find_package(PkgConfig REQUIRED)

# GTK4 依赖
if(DUOROU_BUILD_GUI)
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
        # 非 Windows 平台使用 pkg-config（非致命）
        pkg_check_modules(GTK4 QUIET gtk4)
    endif()
    
    if(GTK4_FOUND)
        message(STATUS "Found GTK4: ${GTK4_VERSION}")
    else()
        message(WARNING "GTK4 not found. Disabling GUI build. Set -DDUOROU_BUILD_GUI=OFF or install GTK4.")
        set(DUOROU_BUILD_GUI OFF CACHE BOOL "Build GUI components and executable (requires GTK4)" FORCE)
    endif()
else()
    message(STATUS "DUOROU_BUILD_GUI=OFF: Skipping GTK4 detection.")
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

# CURL 支持 (用于模型下载)
find_package(CURL REQUIRED)
if(CURL_FOUND)
    message(STATUS "Found CURL: ${CURL_VERSION_STRING}")
else()
    message(FATAL_ERROR "CURL not found. Please install libcurl development packages.")
endif()

# OpenSSL 支持 (用于SHA256计算)
find_package(OpenSSL REQUIRED)
if(OPENSSL_FOUND)
    message(STATUS "Found OpenSSL: ${OPENSSL_VERSION}")
else()
    message(FATAL_ERROR "OpenSSL not found. Please install OpenSSL development packages.")
endif()

# 视频和音频处理库
# FFmpeg 支持
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG QUIET
    libavcodec
    libavformat
    libavutil
    libavdevice
    libswscale
    libswresample
)

if(FFMPEG_FOUND)
    message(STATUS "Found FFmpeg libraries")
    set(HAVE_FFMPEG TRUE)
else()
    message(WARNING "FFmpeg not found. Video recording features will be limited.")
    message(STATUS "To install FFmpeg on macOS: brew install ffmpeg")
    message(STATUS "To install FFmpeg on Ubuntu: sudo apt install libavcodec-dev libavformat-dev libavutil-dev libavdevice-dev libswscale-dev libswresample-dev")
    set(HAVE_FFMPEG FALSE)
endif()

# GStreamer 支持 (备选视频处理库)
pkg_check_modules(GSTREAMER QUIET
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
    gstreamer-audio-1.0
)

if(GSTREAMER_FOUND)
    message(STATUS "Found GStreamer: ${GSTREAMER_VERSION}")
    set(HAVE_GSTREAMER TRUE)
else()
    message(WARNING "GStreamer not found. Alternative video processing will be unavailable.")
    message(STATUS "To install GStreamer on macOS: brew install gstreamer gst-plugins-base gst-plugins-good")
    message(STATUS "To install GStreamer on Ubuntu: sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev")
    set(HAVE_GSTREAMER FALSE)
endif()

# PortAudio 支持 (音频处理)
pkg_check_modules(PORTAUDIO QUIET portaudio-2.0)

if(PORTAUDIO_FOUND)
    message(STATUS "Found PortAudio: ${PORTAUDIO_VERSION}")
    set(HAVE_PORTAUDIO TRUE)
else()
    message(WARNING "PortAudio not found. Audio recording features will be limited.")
    message(STATUS "To install PortAudio on macOS: brew install portaudio")
    message(STATUS "To install PortAudio on Ubuntu: sudo apt install portaudio19-dev")
    set(HAVE_PORTAUDIO FALSE)
endif()

# OpenCV 支持 (图像和视频处理)
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio)

if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    set(HAVE_OPENCV TRUE)
else()
    message(WARNING "OpenCV not found. Advanced image/video processing will be unavailable.")
    message(STATUS "To install OpenCV on macOS: brew install opencv")
    message(STATUS "To install OpenCV on Ubuntu: sudo apt install libopencv-dev")
    set(HAVE_OPENCV FALSE)
endif()

# WebKitGTK 和 MD4C 支持（Markdown 渲染增强）
if(DUOROU_BUILD_GUI)
    # WebKitGTK: GTK4 通常对应 webkitgtk-6.0；部分环境仍为 webkit2gtk-4.x
    pkg_check_modules(WEBKIT2GTK QUIET webkitgtk-6.0)
    if(NOT WEBKIT2GTK_FOUND)
        pkg_check_modules(WEBKIT2GTK QUIET webkit2gtk-4.1)
    endif()
    if(NOT WEBKIT2GTK_FOUND)
        pkg_check_modules(WEBKIT2GTK QUIET webkit2gtk-4.0)
    endif()
    if(WEBKIT2GTK_FOUND)
        message(STATUS "Found WebKitGTK: ${WEBKIT2GTK_VERSION}")
        set(HAVE_WEBKIT2GTK TRUE)
    else()
        message(STATUS "WebKitGTK not found. MarkdownView will fall back to GtkTextView.")
        set(HAVE_WEBKIT2GTK FALSE)
    endif()

    # MD4C for Markdown->HTML conversion (md4c-html provides md_html, depends on md4c)
    pkg_check_modules(MD4C_HTML QUIET md4c-html)
    pkg_check_modules(MD4C QUIET md4c)
    if(MD4C_HTML_FOUND)
        message(STATUS "Found MD4C-HTML: ${MD4C_HTML_VERSION}")
        set(HAVE_MD4C TRUE)
    elseif(MD4C_FOUND)
        message(STATUS "Found MD4C (core) only: ${MD4C_VERSION}")
        message(STATUS "Note: md4c-html not found; HTML conversion will be disabled.")
        set(HAVE_MD4C FALSE)
    else()
        message(STATUS "MD4C not found. Using minimal fallback converter.")
        set(HAVE_MD4C FALSE)
    endif()
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/MiniMemory/src/server
)

# 添加视频音频库的包含目录
if(FFMPEG_FOUND)
    include_directories(${FFMPEG_INCLUDE_DIRS})
endif()

if(GSTREAMER_FOUND)
    include_directories(${GSTREAMER_INCLUDE_DIRS})
endif()

if(PORTAUDIO_FOUND)
    include_directories(${PORTAUDIO_INCLUDE_DIRS})
endif()

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# 添加 WebKitGTK/MD4C 的包含目录
if(WEBKIT2GTK_FOUND)
    include_directories(${WEBKIT2GTK_INCLUDE_DIRS})
endif()
if(MD4C_HTML_FOUND)
    include_directories(${MD4C_HTML_INCLUDE_DIRS})
endif()
if(MD4C_FOUND)
    include_directories(${MD4C_INCLUDE_DIRS})
endif()

# 链接目录（仅在启用 GUI 且找到 GTK4 时添加）
if(DUOROU_BUILD_GUI AND GTK4_FOUND)
    link_directories(${GTK4_LIBRARY_DIRS})
endif()

# 添加视频音频库的链接目录
if(FFMPEG_FOUND)
    link_directories(${FFMPEG_LIBRARY_DIRS})
endif()

if(GSTREAMER_FOUND)
    link_directories(${GSTREAMER_LIBRARY_DIRS})
endif()

if(PORTAUDIO_FOUND)
    link_directories(${PORTAUDIO_LIBRARY_DIRS})
endif()

# 添加 WebKitGTK/MD4C 的链接目录
if(WEBKIT2GTK_FOUND)
    link_directories(${WEBKIT2GTK_LIBRARY_DIRS})
endif()
if(MD4C_HTML_FOUND)
    link_directories(${MD4C_HTML_LIBRARY_DIRS})
endif()
if(MD4C_FOUND)
    link_directories(${MD4C_LIBRARY_DIRS})
endif()

# 编译定义（基础定义 + 条件 GTK 定义）
add_definitions(
    -DDUOROU_VERSION="${PROJECT_VERSION}"
    -DDUOROU_PLATFORM="${DUOROU_PLATFORM}"
)

if(DUOROU_BUILD_GUI AND GTK4_FOUND)
    add_definitions(${GTK4_CFLAGS_OTHER} -DHAVE_GTK)
endif()

# 添加视频音频库的编译定义
if(FFMPEG_FOUND)
    add_definitions(-DHAVE_FFMPEG ${FFMPEG_CFLAGS_OTHER})
endif()

if(GSTREAMER_FOUND)
    add_definitions(-DHAVE_GSTREAMER ${GSTREAMER_CFLAGS_OTHER})
endif()

if(PORTAUDIO_FOUND)
    add_definitions(-DHAVE_PORTAUDIO ${PORTAUDIO_CFLAGS_OTHER})
endif()

if(OpenCV_FOUND)
    add_definitions(-DHAVE_OPENCV)
endif()

# 编译定义：WebKitGTK / MD4C
if(WEBKIT2GTK_FOUND)
    add_definitions(-DHAVE_WEBKIT2GTK ${WEBKIT2GTK_CFLAGS_OTHER})
endif()
if(MD4C_HTML_FOUND)
    add_definitions(-DHAVE_MD4C ${MD4C_HTML_CFLAGS_OTHER})
elseif(MD4C_FOUND)
    # Core md4c present but html helper missing; do not define HAVE_MD4C
    add_definitions(${MD4C_CFLAGS_OTHER})
endif()

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
# Ensure mtmd library (multimodal tools) is available for linking
add_subdirectory(third_party/llama.cpp/tools/mtmd EXCLUDE_FROM_ALL)
add_subdirectory(third_party/stable-diffusion.cpp)
add_subdirectory(src/model)
if(TARGET duorou_model)
    target_include_directories(duorou_model PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    )
    target_link_libraries(duorou_model PRIVATE ggml)
endif()
# MiniMemory作为独立服务运行，不需要编译到主项目中

# 添加工具模块
add_subdirectory(src/utils)

# 添加扩展模块
add_subdirectory(src/extensions)

# 添加 kvcache 模块
add_subdirectory(src/kvcache)

# 移除自研 ml 模块，仅保留 llama.cpp 作为推理引擎
# add_subdirectory(src/ml)

# 添加 Model 模块
# (已在上方初始化 src/model；移除重复 add_subdirectory 以避免二次使用同一 binary 目录)

# 集成测试
if(BUILD_TESTING)
    include(${CMAKE_CURRENT_SOURCE_DIR}/src/CMakeLists_integration_test.txt)
endif()

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

# 源文件列表（只包含实际存在的文件）
set(DUOROU_SOURCES_LIST
    src/main.cpp
    src/core/application.cpp
    src/core/config_manager.cpp
    src/core/logger.cpp
    src/core/model_manager.cpp
    src/core/model_path_manager.cpp
    src/core/text_generator.cpp
    src/core/image_generator.cpp
    src/core/stb_impl.cpp
    src/core/model_downloader.cpp

    src/core/mtmd_demo.cpp

    src/core/ollama_model_loader.cpp
    src/core/modelfile_parser.cpp
    src/core/resource_manager.cpp
    src/core/workflow_engine.cpp
    src/core/model_switch_task.cpp

    src/gui/main_window.cpp
    src/gui/chat_view.cpp
    src/gui/image_view.cpp
    src/gui/settings_dialog.cpp
    src/gui/system_tray.cpp
    src/gui/markdown_view.cpp
    src/gui/chat_session.cpp
    src/gui/chat_session_manager.cpp
    src/gui/session_storage_adapter.cpp
    src/gui/video_display_window.cpp
    src/gui/enhanced_video_capture_window.cpp
    src/gui/video_source_dialog.cpp
    # src/api/api_server.cpp  # 注释：暂时禁用API服务器
    src/media/video_capture.cpp
    src/media/audio_capture.cpp
)

# macOS平台特定源文件
if(APPLE)
    list(APPEND DUOROU_SOURCES_LIST src/platform/macos_tray.mm)
    list(APPEND DUOROU_SOURCES_LIST src/media/macos_screen_capture.mm)
endif()

# 头文件列表（只包含实际存在的文件）
set(DUOROU_HEADERS_LIST
    src/core/application.h
    src/core/config_manager.h
    src/core/logger.h
    src/core/model_manager.h
    src/core/model_path_manager.h
    src/core/text_generator.h
    src/core/image_generator.h
    src/core/model_downloader.h

    src/core/ollama_model_loader.h
    src/core/resource_manager.h
    src/core/workflow_engine.h
    src/core/model_switch_task.h
    src/gui/main_window.h
    src/gui/chat_view.h
    src/gui/image_view.h
    src/gui/settings_dialog.h
    src/gui/system_tray.h
    src/gui/markdown_view.h
    src/gui/chat_session.h
    src/gui/chat_session_manager.h
    src/gui/session_storage_adapter.h
    src/gui/video_display_window.h
    src/gui/enhanced_video_capture_window.h
    src/gui/video_source_dialog.h
    src/api/api_server.h
    src/media/video_capture.h
    src/media/audio_capture.h
    src/media/video_frame.h
)

# macOS平台特定头文件
if(APPLE)
    list(APPEND DUOROU_HEADERS_LIST src/platform/macos_tray.h)
    list(APPEND DUOROU_HEADERS_LIST src/media/macos_screen_capture.h)
endif()

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

# 只有当存在源文件且启用 GUI 时才构建可执行文件
if(DUOROU_BUILD_GUI AND DUOROU_SOURCES)
    list(LENGTH DUOROU_SOURCES DUOROU_SOURCES_COUNT)
    message(STATUS "Found ${DUOROU_SOURCES_COUNT} source files, building executable...")
    
    # 主可执行文件
    add_executable(duorou ${DUOROU_SOURCES} ${DUOROU_HEADERS})
    
    # 设置包含目录（GTK 目录按条件添加）
    target_include_directories(duorou PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp
    )
    target_include_directories(duorou_model PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)
    if(GTK4_FOUND)
        target_include_directories(duorou PRIVATE ${GTK4_INCLUDE_DIRS})
    endif()

    target_link_libraries(duorou_model PRIVATE ggml)
    
    # 链接库（GTK4 按条件链接）
    target_link_libraries(duorou
        llama
        mtmd
        stable-diffusion
        duorou_ollama_extension
        duorou_model
        duorou_utils
        Threads::Threads
        nlohmann_json::nlohmann_json
        CURL::libcurl
        OpenSSL::SSL
        OpenSSL::Crypto
    )
    if(GTK4_FOUND)
        target_link_libraries(duorou ${GTK4_LIBRARIES})
    endif()
    if(WEBKIT2GTK_FOUND)
        target_link_libraries(duorou ${WEBKIT2GTK_LIBRARIES})
        message(STATUS "Linking WebKitGTK libraries: ${WEBKIT2GTK_LIBRARIES}")
    endif()
    if(MD4C_HTML_FOUND)
        target_link_libraries(duorou ${MD4C_HTML_LIBRARIES})
        message(STATUS "Linking MD4C-HTML libraries: ${MD4C_HTML_LIBRARIES}")
    elseif(MD4C_FOUND)
        target_link_libraries(duorou ${MD4C_LIBRARIES})
        message(STATUS "Linking MD4C libraries: ${MD4C_LIBRARIES}")
    endif()
    
    # 链接视频音频处理库
    if(FFMPEG_FOUND)
        target_link_libraries(duorou ${FFMPEG_LIBRARIES})
        message(STATUS "Linking FFmpeg libraries: ${FFMPEG_LIBRARIES}")
    endif()
    
    if(GSTREAMER_FOUND)
        target_link_libraries(duorou ${GSTREAMER_LIBRARIES})
        message(STATUS "Linking GStreamer libraries: ${GSTREAMER_LIBRARIES}")
    endif()
    
    if(PORTAUDIO_FOUND)
        target_link_libraries(duorou ${PORTAUDIO_LIBRARIES})
        message(STATUS "Linking PortAudio libraries: ${PORTAUDIO_LIBRARIES}")
    endif()
    
    if(OpenCV_FOUND)
        target_link_libraries(duorou ${OpenCV_LIBS})
        message(STATUS "Linking OpenCV libraries: ${OpenCV_LIBS}")
    endif()
    
    # 平台特定链接
    if(APPLE)
        target_link_libraries(duorou
            "-framework Foundation"
            "-framework AppKit"
            "-framework Cocoa"
            "-framework ScreenCaptureKit"
            "-framework CoreGraphics"
            "-framework CoreMedia"
            "-framework CoreVideo"
            "-framework AVFoundation"
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
elseif(NOT DUOROU_BUILD_GUI)
    message(STATUS "DUOROU_BUILD_GUI=OFF: Skipping GUI executable build.")
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

# 复制GUI资源文件到build目录（仅在启用 GUI 时）
if(DUOROU_BUILD_GUI)
    set(GUI_BUILD_DIR "${CMAKE_BINARY_DIR}/src/gui")
    file(MAKE_DIRECTORY "${GUI_BUILD_DIR}")

    # 复制CSS和图片文件
    set(GUI_RESOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/styles.css"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/duorou01.png"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/video-off.png"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/video-on.png"
    )

    foreach(RESOURCE_FILE ${GUI_RESOURCE_FILES})
        get_filename_component(FILENAME ${RESOURCE_FILE} NAME)
        configure_file(${RESOURCE_FILE} "${GUI_BUILD_DIR}/${FILENAME}" COPYONLY)
    endforeach()

    message(STATUS "GUI resources copied to: ${GUI_BUILD_DIR}")
endif()

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

message(STATUS "  Build Examples: ${DUOROU_BUILD_EXAMPLES}")
message(STATUS "")
