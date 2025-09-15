#include "llama_model_loader.h"
#include "../../../third_party/llama.cpp/include/llama.h"
#include <iostream>
#include <cstring>

namespace duorou {
namespace extensions {
namespace ollama {

// 静态变量用于跟踪后端初始化状态
static bool backend_initialized = false;

llama_model* LlamaModelLoader::loadModelWithParams(
    const std::string& gguf_file_path,
    const llama_model_params* custom_params
) {
    // 确保后端已初始化
    ensureBackendInitialized();
    
    // 解析GGUF文件
    GGUFParser parser(false); // 不启用详细输出
    if (!parser.parseFile(gguf_file_path)) {
        std::cerr << "Failed to parse GGUF file: " << gguf_file_path << std::endl;
        return nullptr;
    }
    
    return loadModelFromGGUF(parser, gguf_file_path, custom_params);
}

llama_model* LlamaModelLoader::loadModelFromGGUF(
    const GGUFParser& parser,
    const std::string& gguf_file_path,
    const llama_model_params* custom_params
) {
    // 确保后端已初始化
    ensureBackendInitialized();
    
    // 验证兼容性
    if (!validateCompatibility(parser)) {
        std::cerr << "GGUF file is not compatible with llama.cpp" << std::endl;
        return nullptr;
    }
    
    // 创建llama_model_params
    llama_model_params params = LlamaParamsConverter::createFromGGUF(parser, custom_params);
    
    // 验证参数
    if (!LlamaParamsConverter::validateParams(params)) {
        std::cerr << "Invalid model parameters" << std::endl;
        return nullptr;
    }
    
    // 使用llama.cpp加载模型
    llama_model* model = llama_model_load_from_file(gguf_file_path.c_str(), params);
    
    if (!model) {
        std::cerr << "Failed to load model with llama.cpp: " << gguf_file_path << std::endl;
        return nullptr;
    }
    
    // 验证加载的模型
    if (!validateModel(model)) {
        std::cerr << "Loaded model validation failed" << std::endl;
        llama_free_model(model);
        return nullptr;
    }
    
    std::cout << "Successfully loaded model: " << gguf_file_path << std::endl;
    return model;
}

llama_context* LlamaModelLoader::createContext(
    llama_model* model,
    const GGUFParser& parser,
    const llama_context_params* custom_ctx_params
) {
    if (!model) {
        std::cerr << "Cannot create context: model is null" << std::endl;
        return nullptr;
    }
    
    // 获取上下文参数
    llama_context_params ctx_params = getDefaultContextParams(parser);
    
    // 应用自定义参数覆盖
    if (custom_ctx_params) {
        // 复制重要的自定义参数
        ctx_params.n_ctx = custom_ctx_params->n_ctx;
        ctx_params.n_batch = custom_ctx_params->n_batch;
        ctx_params.n_ubatch = custom_ctx_params->n_ubatch;
        ctx_params.n_seq_max = custom_ctx_params->n_seq_max;
        ctx_params.n_threads = custom_ctx_params->n_threads;
        ctx_params.n_threads_batch = custom_ctx_params->n_threads_batch;
        ctx_params.rope_scaling_type = custom_ctx_params->rope_scaling_type;
        ctx_params.pooling_type = custom_ctx_params->pooling_type;
        ctx_params.rope_freq_base = custom_ctx_params->rope_freq_base;
        ctx_params.rope_freq_scale = custom_ctx_params->rope_freq_scale;
        ctx_params.yarn_ext_factor = custom_ctx_params->yarn_ext_factor;
        ctx_params.yarn_attn_factor = custom_ctx_params->yarn_attn_factor;
        ctx_params.yarn_beta_fast = custom_ctx_params->yarn_beta_fast;
        ctx_params.yarn_beta_slow = custom_ctx_params->yarn_beta_slow;
        ctx_params.yarn_orig_ctx = custom_ctx_params->yarn_orig_ctx;
        ctx_params.defrag_thold = custom_ctx_params->defrag_thold;
        ctx_params.type_k = custom_ctx_params->type_k;
        ctx_params.type_v = custom_ctx_params->type_v;
        ctx_params.logits_all = custom_ctx_params->logits_all;
        ctx_params.embeddings = custom_ctx_params->embeddings;
        ctx_params.offload_kqv = custom_ctx_params->offload_kqv;
        ctx_params.flash_attn = custom_ctx_params->flash_attn;
        ctx_params.no_perf = custom_ctx_params->no_perf;
    }
    
    // 创建上下文
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    
    if (!ctx) {
        std::cerr << "Failed to create llama context" << std::endl;
        return nullptr;
    }
    
    std::cout << "Successfully created llama context" << std::endl;
    return ctx;
}

llama_context_params LlamaModelLoader::getDefaultContextParams(
    const GGUFParser& parser
) {
    llama_context_params ctx_params = llama_context_default_params();
    
    // 从GGUF解析器提取上下文参数
    extractContextParams(parser, ctx_params);
    
    return ctx_params;
}

bool LlamaModelLoader::validateModel(llama_model* model) {
    if (!model) {
        return false;
    }
    
    // 检查模型的基本属性
    int32_t vocab_size = llama_n_vocab(model);
    if (vocab_size <= 0) {
        std::cerr << "Invalid vocab size: " << vocab_size << std::endl;
        return false;
    }
    
    int32_t n_ctx_train = llama_n_ctx_train(model);
    if (n_ctx_train <= 0) {
        std::cerr << "Invalid training context size: " << n_ctx_train << std::endl;
        return false;
    }
    
    int32_t n_embd = llama_n_embd(model);
    if (n_embd <= 0) {
        std::cerr << "Invalid embedding size: " << n_embd << std::endl;
        return false;
    }
    
    return true;
}

void LlamaModelLoader::freeModel(llama_model* model) {
    if (model) {
        llama_free_model(model);
    }
}

void LlamaModelLoader::freeContext(llama_context* ctx) {
    if (ctx) {
        llama_free(ctx);
    }
}

void LlamaModelLoader::printModelInfo(
    llama_model* model,
    const GGUFParser& parser
) {
    if (!model) {
        std::cout << "Model is null" << std::endl;
        return;
    }
    
    const auto& arch = parser.getArchitecture();
    
    std::cout << "=== Model Information ===" << std::endl;
    std::cout << "Architecture: " << arch.name << std::endl;
    std::cout << "Vocab Size: " << llama_n_vocab(model) << std::endl;
    std::cout << "Context Size (Training): " << llama_n_ctx_train(model) << std::endl;
    std::cout << "Embedding Size: " << llama_n_embd(model) << std::endl;
    std::cout << "Layer Count: " << llama_n_layer(model) << std::endl;
    
    // GGUF架构信息
    std::cout << "\n=== GGUF Architecture Info ===" << std::endl;
    std::cout << "Block Count: " << arch.block_count << std::endl;
    std::cout << "Attention Heads: " << arch.attention_head_count << std::endl;
    std::cout << "KV Attention Heads: " << arch.attention_head_count_kv << std::endl;
    std::cout << "Feed Forward Length: " << arch.feed_forward_length << std::endl;
    std::cout << "RoPE Freq Base: " << arch.rope_freq_base << std::endl;
    std::cout << "Has Vision: " << (arch.has_vision ? "YES" : "NO") << std::endl;
    std::cout << "========================" << std::endl;
}

void LlamaModelLoader::ensureBackendInitialized() {
    if (!backend_initialized) {
        llama_backend_init();
        backend_initialized = true;
        std::cout << "Llama backend initialized" << std::endl;
    }
}

void LlamaModelLoader::extractContextParams(
    const GGUFParser& parser,
    llama_context_params& ctx_params
) {
    const auto& arch = parser.getArchitecture();
    
    // 设置上下文长度
    if (arch.context_length > 0) {
        ctx_params.n_ctx = arch.context_length;
    }
    
    // 从GGUF元数据中提取其他参数
    const auto* rope_freq_base_kv = parser.getMetadata("llama.rope.freq_base");
    if (rope_freq_base_kv && rope_freq_base_kv->type == GGUFType::FLOAT32) {
        ctx_params.rope_freq_base = rope_freq_base_kv->asFloat32();
    } else if (arch.rope_freq_base > 0) {
        ctx_params.rope_freq_base = arch.rope_freq_base;
    }
    
    // 设置合理的默认值
    if (ctx_params.n_batch == 0) {
        ctx_params.n_batch = 512;
    }
    
    if (ctx_params.n_threads == 0) {
        ctx_params.n_threads = 4; // 默认4个线程
    }
}

bool LlamaModelLoader::validateCompatibility(
    const GGUFParser& parser
) {
    const auto& arch = parser.getArchitecture();
    
    // 检查架构是否受支持
    if (!GGUFParser::isSupportedArchitecture(arch.name)) {
        std::cerr << "Unsupported architecture: " << arch.name << std::endl;
        return false;
    }
    
    // 检查基本参数
    if (arch.vocab_size == 0) {
        std::cerr << "Invalid vocab size: " << arch.vocab_size << std::endl;
        return false;
    }
    
    if (arch.context_length == 0) {
        std::cerr << "Invalid context length: " << arch.context_length << std::endl;
        return false;
    }
    
    if (arch.embedding_length == 0) {
        std::cerr << "Invalid embedding length: " << arch.embedding_length << std::endl;
        return false;
    }
    
    if (arch.block_count == 0) {
        std::cerr << "Invalid block count: " << arch.block_count << std::endl;
        return false;
    }
    
    return true;
}

} // namespace ollama
} // namespace extensions
} // namespace duorou