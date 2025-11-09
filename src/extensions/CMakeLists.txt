# Extensions模块

message(STATUS "Configuring Extensions...")

# 是否启用 Ollama 扩展（可通过 -DDUOROU_ENABLE_OLLAMA=OFF 关闭以规避编译问题）
option(DUOROU_ENABLE_OLLAMA "Build Ollama extension" ON)

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
if(DUOROU_ENABLE_OLLAMA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ollama")
    add_subdirectory(ollama)
    message(STATUS "  Added ollama extension")
else()
    message(STATUS "  Ollama extension disabled or not found")
endif()

message(STATUS "Extensions configuration complete")