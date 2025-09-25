# ML module CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# Set ML module source files
set(ML_SOURCES
    # Backend source files
    backend/backend.cpp
    backend/cpu_backend.cpp
    backend/ggml_backend.cpp
    
    # Core source files
    tensor.cpp
    context.cpp
    
    # NN layer source files
    nn/linear.cpp
    nn/attention.cpp
    nn/activation.cpp
)

# Set ML module header files
set(ML_HEADERS
    # Backend header files
    backend/backend.h
    backend/cpu_backend.h
    backend/ggml_backend.h
    
    # Core header files
    tensor.h
    context.h
    
    # NN layer header files
    nn/linear.h
    nn/attention.h
    nn/activation.h
)

# Create ML static library
add_library(duorou_ml STATIC ${ML_SOURCES})

# Set include directories
target_include_directories(duorou_ml PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link dependency libraries
target_link_libraries(duorou_ml
    duorou_kvcache  # Link KV cache module
)

# Set compilation options
target_compile_features(duorou_ml PUBLIC cxx_std_17)

# 设置编译器特定选项
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(duorou_ml PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -O3
        -march=native
    )
endif()

# 添加预处理器定义
target_compile_definitions(duorou_ml PRIVATE
    DUOROU_ML_VERSION_MAJOR=1
    DUOROU_ML_VERSION_MINOR=0
    DUOROU_ML_VERSION_PATCH=0
)

# 创建测试程序（可选）
# option(BUILD_ML_TESTS "Build ML module tests" ON)
# if(BUILD_ML_TESTS)
#     add_executable(test_ml test_ml.cpp)
#     target_link_libraries(test_ml duorou_ml)
#     target_include_directories(test_ml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# endif()

# 安装规则
install(TARGETS duorou_ml
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/duorou/ml
    FILES_MATCHING PATTERN "*.h"
)

# 添加子目录（如果有GPU后端等）
# add_subdirectory(backend/cuda)
# add_subdirectory(backend/metal)