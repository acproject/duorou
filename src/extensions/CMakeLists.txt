# Extensions模块

message(STATUS "Configuring Extensions...")

# 添加llama_cpp扩展 (已集成到third_party)
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp")
#     add_subdirectory(llama_cpp)
#     message(STATUS "  Added llama_cpp extension")
# endif()

# 添加stable_diffusion_cpp扩展 (已集成到third_party)
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/stable_diffusion_cpp")
#     add_subdirectory(stable_diffusion_cpp)
#     message(STATUS "  Added stable_diffusion_cpp extension")
# endif()

# 添加ollama扩展
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ollama")
    add_subdirectory(ollama)
    message(STATUS "  Added ollama extension")
endif()

message(STATUS "Extensions configuration complete")