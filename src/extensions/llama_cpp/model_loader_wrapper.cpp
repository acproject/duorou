#include "model_loader_wrapper.h"
#include "gguf_wrapper.h"
#include "ggml.h"
#include "gguf.h"
#include <iostream>
#include <cstring>
#include <filesystem>

namespace duorou {
namespace extensions {
namespace llama_cpp {

struct llama_model* ModelLoaderWrapper::loadModelWithArchMapping(
    const std::string& model_path,
    llama_model_params params) {
    
    std::cout << "[ModelLoaderWrapper] Loading model: " << model_path << std::endl;
    
    std::string original_arch, mapped_arch;
    bool needs_mapping = checkArchitectureMapping(model_path, original_arch, mapped_arch);
    
    std::vector<llama_model_kv_override> overrides;
    std::string actual_model_path = model_path;
    std::string temp_file_path;
    
    if (needs_mapping) {
        std::cout << "Architecture mapping detected: " << original_arch << " -> " << mapped_arch << std::endl;
        
        // 参考Ollama的做法，qwen25vl直接映射到qwen2vl架构
        // 不需要修改模型文件，在代码层面处理架构差异
        std::cout << "Using direct architecture mapping without file modification" << std::endl;
        
        // 创建kv_overrides来覆盖架构字段
        overrides = createArchOverrides(mapped_arch, model_path);
        
        std::cout << "Loading model with architecture override: " << mapped_arch << std::endl;
    } else {
        std::cout << "No architecture mapping needed for: " << original_arch << std::endl;
    }
    
    // 设置kv_overrides到model_params中
    if (!overrides.empty()) {
        params.kv_overrides = overrides.data();
    }
    
    // 使用llama.cpp加载模型
    
    struct llama_model* model = llama_model_load_from_file(actual_model_path.c_str(), params);
    
    // 清理临时文件
    if (!temp_file_path.empty() && std::filesystem::exists(temp_file_path)) {
        std::filesystem::remove(temp_file_path);
        std::cout << "Cleaned up temporary file: " << temp_file_path << std::endl;
    }
    
    // 记录加载结果
    if (model) {
        std::cout << "[ModelLoaderWrapper] Model loaded successfully" << std::endl;
    } else {
        std::cout << "[ModelLoaderWrapper] Failed to load model" << std::endl;
    }
    
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return nullptr;
    }
    
    std::cout << "Successfully loaded model: " << model_path << std::endl;
    return model;
}

bool ModelLoaderWrapper::checkArchitectureMapping(
    const std::string& model_path,
    std::string& original_arch,
    std::string& mapped_arch) {
    
    // 使用gguf直接读取架构信息
    struct ggml_context* ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };
    
    struct gguf_context* gguf_ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!gguf_ctx) {
        std::cerr << "Failed to read GGUF file: " << model_path << std::endl;
        return false;
    }
    
    // 查找架构字段
    const char* arch_key = "general.architecture";
    int arch_index = gguf_find_key(gguf_ctx, arch_key);
    if (arch_index < 0) {
        std::cerr << "Architecture key not found in model" << std::endl;
        gguf_free(gguf_ctx);
        if (ctx) {
            ggml_free(ctx);
        }
        return false;
    }
    
    // 获取架构字符串
    const char* arch_str = gguf_get_val_str(gguf_ctx, arch_index);
    if (!arch_str) {
        std::cerr << "Failed to read architecture string" << std::endl;
        gguf_free(gguf_ctx);
        if (ctx) {
            ggml_free(ctx);
        }
        return false;
    }
    
    original_arch = std::string(arch_str);
    
    // 清理资源
    gguf_free(gguf_ctx);
    if (ctx) {
        ggml_free(ctx);
    }
    
    // 检查是否需要映射
    if (ArchMapping::needsMapping(original_arch)) {
        mapped_arch = ArchMapping::getMappedArchitecture(original_arch);
        return true;
    }
    
    mapped_arch = original_arch;
    return false;
}

std::vector<llama_model_kv_override> ModelLoaderWrapper::createArchOverrides(
    const std::string& mapped_arch,
    const std::string& model_path) {
    
    std::vector<llama_model_kv_override> overrides;
    
    // 创建架构覆盖
    llama_model_kv_override arch_override = {};
    strcpy(arch_override.key, "general.architecture");
    arch_override.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
    
    // 复制字符串到固定大小数组
    strncpy(arch_override.val_str, mapped_arch.c_str(), sizeof(arch_override.val_str) - 1);
    arch_override.val_str[sizeof(arch_override.val_str) - 1] = '\0';
    
    overrides.push_back(arch_override);
    
    // 参考Ollama的做法，为qwen25vl->qwen2vl映射添加必要的基础参数
    if (mapped_arch == "qwen2vl") {
        // 添加context_length - 这是qwen2vl架构必需的参数
        llama_model_kv_override context_override = {};
        strcpy(context_override.key, "qwen2vl.context_length");
        context_override.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        context_override.val_i64 = 128000;  // qwen2.5vl的默认上下文长度
        overrides.push_back(context_override);
        
        // 添加embedding_length - qwen2vl架构必需的参数
        llama_model_kv_override embedding_override = {};
        strcpy(embedding_override.key, "qwen2vl.embedding_length");
        embedding_override.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        embedding_override.val_i64 = 3584;  // qwen2.5vl的embedding维度
        overrides.push_back(embedding_override);
        
        // 添加block_count - qwen2vl架构必需的参数
        llama_model_kv_override block_count_override = {};
        strcpy(block_count_override.key, "qwen2vl.block_count");
        block_count_override.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        block_count_override.val_i64 = 28;  // qwen2.5vl的层数
        overrides.push_back(block_count_override);
        
        // 添加attention相关参数
        llama_model_kv_override head_count_override = {};
        strcpy(head_count_override.key, "qwen2vl.attention.head_count");
        head_count_override.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        head_count_override.val_i64 = 28;
        overrides.push_back(head_count_override);
        
        llama_model_kv_override head_count_kv_override = {};
        strcpy(head_count_kv_override.key, "qwen2vl.attention.head_count_kv");
        head_count_kv_override.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        head_count_kv_override.val_i64 = 4;
        overrides.push_back(head_count_kv_override);
        
        // 添加feed_forward_length参数
        llama_model_kv_override feed_forward_override = {};
        strcpy(feed_forward_override.key, "qwen2vl.feed_forward_length");
        feed_forward_override.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        feed_forward_override.val_i64 = 18944;
        overrides.push_back(feed_forward_override);
        
        // 添加rope相关参数
        llama_model_kv_override rope_freq_override = {};
        strcpy(rope_freq_override.key, "qwen2vl.rope.freq_base");
        rope_freq_override.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        rope_freq_override.val_f64 = 1000000.0;
        overrides.push_back(rope_freq_override);
        
        // 添加rope.dimension_sections - qwen2vl的关键参数
        // 这是一个数组参数，需要特殊处理
        // 注意：llama_model_kv_override不直接支持数组，这里先尝试单个值
        llama_model_kv_override rope_dim_override = {};
        strcpy(rope_dim_override.key, "qwen2vl.rope.dimension_sections");
        rope_dim_override.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        strcpy(rope_dim_override.val_str, "[128,128,128]");  // qwen2.5vl的rope维度分段
        overrides.push_back(rope_dim_override);
        
        // 添加attention.layer_norm_rms_epsilon参数
        llama_model_kv_override rms_epsilon_override = {};
        strcpy(rms_epsilon_override.key, "qwen2vl.attention.layer_norm_rms_epsilon");
        rms_epsilon_override.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        rms_epsilon_override.val_f64 = 1e-6;
        overrides.push_back(rms_epsilon_override);
    }
    
    // 添加终止符 - llama.cpp需要NULL终止的数组
    llama_model_kv_override terminator = {};
    terminator.key[0] = '\0';  // 空键表示终止
    overrides.push_back(terminator);
    
    return overrides;
}



} // namespace llama_cpp
} // namespace extensions
} // namespace duorou