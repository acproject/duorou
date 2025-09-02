#include "gguf_wrapper.h"
#include "gguf.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <vector>

namespace duorou {
namespace extensions {
namespace llama_cpp {

std::string GGUFWrapper::createTempGGUFWithDimensionSections(
    const std::string& original_path,
    const std::string& temp_path) {
    
    // 简单方法：直接复制原文件，然后尝试用Python脚本修改
    // 或者返回原文件路径，让llama.cpp处理缺失的键
    
    // 首先尝试复制文件
    std::ifstream src(original_path, std::ios::binary);
    std::ofstream dst(temp_path, std::ios::binary);
    
    if (!src.is_open() || !dst.is_open()) {
        throw std::runtime_error("Failed to copy original file");
    }
    
    dst << src.rdbuf();
    src.close();
    dst.close();
    
    // 暂时返回复制的文件，不修改元数据
    // TODO: 实现安全的GGUF元数据修改
    return temp_path;
}

bool GGUFWrapper::isMissingDimensionSections(const std::string& file_path) {
    struct gguf_init_params params = {
        /*.no_alloc   =*/ true,
        /*.ctx        =*/ nullptr,
    };
    
    struct gguf_context* ctx = gguf_init_from_file(file_path.c_str(), params);
    if (!ctx) {
        return false;
    }
    
    const char* key_name = "qwen2vl.rope.dimension_sections";
    int key_index = gguf_find_key(ctx, key_name);
    
    gguf_free(ctx);
    return key_index < 0;
}

bool GGUFWrapper::readMropeSections(
    const std::string& file_path,
    int32_t mrope_sections[4]) {
    
    struct gguf_init_params params = {
        /*.no_alloc   =*/ true,
        /*.ctx        =*/ nullptr,
    };
    
    struct gguf_context* ctx = gguf_init_from_file(file_path.c_str(), params);
    if (!ctx) {
        return false;
    }
    
    const char* key_name = "qwen2.rope.mrope_section";
    int key_index = gguf_find_key(ctx, key_name);
    
    if (key_index < 0) {
        gguf_free(ctx);
        return false;
    }
    
    enum gguf_type key_type = gguf_get_kv_type(ctx, key_index);
    if (key_type != GGUF_TYPE_ARRAY) {
        gguf_free(ctx);
        return false;
    }
    
    enum gguf_type arr_type = gguf_get_arr_type(ctx, key_index);
    size_t arr_n = gguf_get_arr_n(ctx, key_index);
    
    if (arr_type != GGUF_TYPE_INT32 || arr_n != 4) {
        gguf_free(ctx);
        return false;
    }
    
    const int32_t* arr_data = (const int32_t*)gguf_get_arr_data(ctx, key_index);
    for (int i = 0; i < 4; i++) {
        mrope_sections[i] = arr_data[i];
    }
    
    gguf_free(ctx);
    return true;
}



} // namespace llama_cpp
} // namespace extensions
} // namespace duorou