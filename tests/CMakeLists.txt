# 测试配置
cmake_minimum_required(VERSION 3.14)

# 查找测试框架（可选）
find_package(GTest QUIET)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/stable-diffusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/MiniMemory/src/server
)

# 基础测试程序（不依赖外部测试框架）
set(BASIC_TESTS
    test_minimemory
    test_workflow_engine_enhanced
    test_workflow_engine_core
    test_model_downloader
    test_ollama_model_loader
)

# 创建基础测试
foreach(test_name ${BASIC_TESTS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.cpp")
        add_executable(${test_name} ${test_name}.cpp)
        
        # 链接必要的库
        target_link_libraries(${test_name}
            llama
            stable-diffusion
            Threads::Threads
        )
        
        # 如果是工作流引擎测试，需要链接更多源文件
        if(test_name STREQUAL "test_workflow_engine_enhanced")
            target_sources(${test_name} PRIVATE
                ../src/core/workflow_engine.cpp
                ../src/core/resource_manager.cpp
                ../src/core/model_switch_task.cpp
                ../src/core/logger.cpp
            )
        endif()
        
        # 如果是工作流引擎核心测试
        if(test_name STREQUAL "test_workflow_engine_core")
            target_sources(${test_name} PRIVATE
                ../src/core/workflow_engine.cpp
                ../src/core/resource_manager.cpp
            )
        endif()
        
        # 如果是MiniMemory测试
        if(test_name STREQUAL "test_minimemory")
            target_sources(${test_name} PRIVATE
                ../third_party/MiniMemory/src/server/DataStore.cpp
                ../third_party/MiniMemory/src/server/CommandHandler.cpp
                ../third_party/MiniMemory/src/server/ConfigParser.cpp
                ../third_party/MiniMemory/src/server/RespParser.cpp
                ../third_party/MiniMemory/src/server/TcpServer.cpp
            )
        endif()
        
        # 如果是模型下载器测试
        if(test_name STREQUAL "test_model_downloader")
            target_sources(${test_name} PRIVATE
                ../src/core/model_downloader.cpp
                ../src/core/model_path_manager.cpp
                ../src/core/logger.cpp
            )
            # 链接curl和nlohmann_json
            find_package(CURL REQUIRED)
            find_package(nlohmann_json REQUIRED)
            find_package(OpenSSL REQUIRED)
            target_link_libraries(${test_name}
                CURL::libcurl
                nlohmann_json::nlohmann_json
                OpenSSL::SSL
                OpenSSL::Crypto
            )
        endif()
        
        # 如果是ollama模型加载器测试
        if(test_name STREQUAL "test_ollama_model_loader")
            target_sources(${test_name} PRIVATE
                ../src/core/ollama_model_loader.cpp
                ../src/core/model_path_manager.cpp
                ../src/core/logger.cpp
            )
            # 链接nlohmann_json
            find_package(nlohmann_json REQUIRED)
            target_link_libraries(${test_name}
                nlohmann_json::nlohmann_json
            )
        endif()
        
        # 添加到测试
        add_test(NAME ${test_name} COMMAND ${test_name})
        
        message(STATUS "Added test: ${test_name}")
    else()
        message(WARNING "Test source file not found: ${test_name}.cpp")
    endif()
endforeach()

# 如果找到了GTest，创建GTest测试
if(GTest_FOUND)
    set(GTEST_TESTS
        # 在这里添加使用GTest的测试
    )
    
    foreach(test_name ${GTEST_TESTS})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.cpp")
            add_executable(${test_name} ${test_name}.cpp)
            target_link_libraries(${test_name}
                GTest::gtest
                GTest::gtest_main
                llama
                stable-diffusion
                Threads::Threads
            )
            add_test(NAME ${test_name} COMMAND ${test_name})
            message(STATUS "Added GTest: ${test_name}")
        endif()
    endforeach()
else()
    message(STATUS "GTest not found, skipping GTest-based tests")
endif()

# 设置测试属性
set_tests_properties(${BASIC_TESTS} PROPERTIES
    TIMEOUT 300  # 5分钟超时
)

message(STATUS "Test configuration completed")