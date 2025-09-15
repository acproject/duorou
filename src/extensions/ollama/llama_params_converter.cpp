#include "llama_params_converter.h"
#include "../../../third_party/llama.cpp/include/llama.h"
#include <iostream>
#include <cstring>

namespace duorou {
namespace extensions {
namespace ollama {

llama_model_params LlamaParamsConverter::createFromGGUF(
    const GGUFParser& parser,
    const llama_model_params* custom_params
) {
    // 获取默认参数
    llama_model_params params = getDefaultParams();
    
    // 从GGUF解析器提取配置
    extractGpuConfig(parser, params);
    extractMemoryConfig(parser, params);
    extractKvOverrides(parser, params);
    
    // 应用自定义覆盖
    if (custom_params) {
        applyCustomOverrides(custom_params, params);
    }
    
    return params;
}

llama_model_params LlamaParamsConverter::createFromFile(
    const std::string& gguf_file_path,
    const llama_model_params* custom_params
) {
    GGUFParser parser(false); // 不启用详细输出
    
    if (!parser.parseFile(gguf_file_path)) {
        std::cerr << "Failed to parse GGUF file: " << gguf_file_path << std::endl;
        return getDefaultParams();
    }
    
    return createFromGGUF(parser, custom_params);
}

llama_model_params LlamaParamsConverter::getDefaultParams() {
    llama_model_params params = llama_model_default_params();
    
    // 设置一些合理的默认值
    params.use_mmap = true;        // 启用内存映射以提高性能
    params.use_mlock = false;      // 默认不锁定内存
    params.vocab_only = false;     // 加载完整模型
    params.check_tensors = true;   // 验证张量
    
    return params;
}

bool LlamaParamsConverter::validateParams(const llama_model_params& params) {
    // 基本验证
    if (params.n_gpu_layers < 0) {
        std::cerr << "Invalid n_gpu_layers: " << params.n_gpu_layers << std::endl;
        return false;
    }
    
    if (params.main_gpu < 0) {
        std::cerr << "Invalid main_gpu: " << params.main_gpu << std::endl;
        return false;
    }
    
    return true;
}

void LlamaParamsConverter::printParams(const llama_model_params& params) {
    std::cout << "=== llama_model_params Configuration ===" << std::endl;
    std::cout << "n_gpu_layers: " << params.n_gpu_layers << std::endl;
    std::cout << "main_gpu: " << params.main_gpu << std::endl;
    std::cout << "use_mmap: " << (params.use_mmap ? "true" : "false") << std::endl;
    std::cout << "use_mlock: " << (params.use_mlock ? "true" : "false") << std::endl;
    std::cout << "vocab_only: " << (params.vocab_only ? "true" : "false") << std::endl;
    std::cout << "check_tensors: " << (params.check_tensors ? "true" : "false") << std::endl;
    std::cout << "========================================" << std::endl;
}

void LlamaParamsConverter::extractGpuConfig(
    const GGUFParser& parser,
    llama_model_params& params
) {
    // 尝试从GGUF元数据中获取GPU配置
    // 注意：GGUF文件通常不包含GPU配置信息，这些通常是运行时配置
    
    // 检查是否有自定义的GPU层数配置
    const auto* gpu_layers_kv = parser.getMetadata("duorou.gpu_layers");
    if (gpu_layers_kv && gpu_layers_kv->type == GGUFType::INT32) {
        params.n_gpu_layers = gpu_layers_kv->asInt32();
    }
    
    // 检查主GPU配置
    const auto* main_gpu_kv = parser.getMetadata("duorou.main_gpu");
    if (main_gpu_kv && main_gpu_kv->type == GGUFType::INT32) {
        params.main_gpu = main_gpu_kv->asInt32();
    }
}

void LlamaParamsConverter::extractMemoryConfig(
    const GGUFParser& parser,
    llama_model_params& params
) {
    // 检查内存映射配置
    const auto* use_mmap_kv = parser.getMetadata("duorou.use_mmap");
    if (use_mmap_kv && use_mmap_kv->type == GGUFType::BOOL) {
        params.use_mmap = use_mmap_kv->asBool();
    }
    
    // 检查内存锁定配置
    const auto* use_mlock_kv = parser.getMetadata("duorou.use_mlock");
    if (use_mlock_kv && use_mlock_kv->type == GGUFType::BOOL) {
        params.use_mlock = use_mlock_kv->asBool();
    }
    
    // 检查是否只加载词汇表
    const auto* vocab_only_kv = parser.getMetadata("duorou.vocab_only");
    if (vocab_only_kv && vocab_only_kv->type == GGUFType::BOOL) {
        params.vocab_only = vocab_only_kv->asBool();
    }
}

void LlamaParamsConverter::extractKvOverrides(
    const GGUFParser& parser,
    llama_model_params& params
) {
    // 这里可以根据GGUF中的特定元数据创建键值覆盖
    // 例如，如果需要覆盖某些模型参数
    
    // 目前设置为nullptr，表示不使用键值覆盖
    params.kv_overrides = nullptr;
    
    // 如果将来需要支持键值覆盖，可以在这里实现
    // 例如：
    // static std::vector<llama_model_kv_override> overrides;
    // overrides.clear();
    // 
    // const auto* context_length_kv = parser.getMetadata("llama.context_length");
    // if (context_length_kv && context_length_kv->type == GGUFType::UINT64) {
    //     llama_model_kv_override override_item;
    //     strncpy(override_item.key, "context_length", sizeof(override_item.key) - 1);
    //     override_item.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    //     override_item.val_i64 = static_cast<int64_t>(context_length_kv->asUInt64());
    //     overrides.push_back(override_item);
    // }
    // 
    // if (!overrides.empty()) {
    //     // 添加结束标记
    //     llama_model_kv_override end_marker = {0};
    //     overrides.push_back(end_marker);
    //     params.kv_overrides = overrides.data();
    // }
}

void LlamaParamsConverter::applyCustomOverrides(
    const llama_model_params* custom_params,
    llama_model_params& params
) {
    if (!custom_params) return;
    
    // 应用自定义参数覆盖
    if (custom_params->n_gpu_layers >= 0) {
        params.n_gpu_layers = custom_params->n_gpu_layers;
    }
    
    if (custom_params->main_gpu >= 0) {
        params.main_gpu = custom_params->main_gpu;
    }
    
    // 复制其他重要参数
    params.use_mmap = custom_params->use_mmap;
    params.use_mlock = custom_params->use_mlock;
    params.vocab_only = custom_params->vocab_only;
    params.check_tensors = custom_params->check_tensors;
    
    // 如果自定义参数有键值覆盖，使用它们
    if (custom_params->kv_overrides) {
        params.kv_overrides = custom_params->kv_overrides;
    }
    
    // 复制回调函数
    if (custom_params->progress_callback) {
        params.progress_callback = custom_params->progress_callback;
        params.progress_callback_user_data = custom_params->progress_callback_user_data;
    }
}

} // namespace ollama
} // namespace extensions
} // namespace duorou