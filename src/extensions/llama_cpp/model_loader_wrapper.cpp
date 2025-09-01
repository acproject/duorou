#include "model_loader_wrapper.h"
#include "ggml.h"
#include "gguf.h"
#include <iostream>
#include <cstring>

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
    
    if (needs_mapping) {
        std::cout << "Architecture mapping detected: " << original_arch << " -> " << mapped_arch << std::endl;
        
        // 创建kv_overrides来覆盖架构字段
        overrides = createArchOverrides(mapped_arch);
        
        std::cout << "Loading model with architecture override: " << mapped_arch << std::endl;
    } else {
        std::cout << "No architecture mapping needed for: " << original_arch << std::endl;
    }
    
    // 设置kv_overrides到model_params中
    if (!overrides.empty()) {
        params.kv_overrides = overrides.data();
    }
    
    // 使用llama.cpp加载模型
    
    struct llama_model* model = llama_model_load_from_file(model_path.c_str(), params);
    
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
    const std::string& mapped_arch) {
    
    std::vector<llama_model_kv_override> overrides;
    
    // 创建架构覆盖
    llama_model_kv_override arch_override = {};
    strcpy(arch_override.key, "general.architecture");
    arch_override.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
    
    // 复制字符串到固定大小数组
    strncpy(arch_override.val_str, mapped_arch.c_str(), sizeof(arch_override.val_str) - 1);
    arch_override.val_str[sizeof(arch_override.val_str) - 1] = '\0';
    
    overrides.push_back(arch_override);
    
    // 添加终止符 - llama.cpp需要NULL终止的数组
    llama_model_kv_override terminator = {};
    terminator.key[0] = '\0';  // 空键表示终止
    overrides.push_back(terminator);
    
    return overrides;
}

} // namespace llama_cpp
} // namespace extensions
} // namespace duorou