#include "llama_model_loader.h"
#include "llama_params_converter.h"
#include "../third_party/llama.cpp/include/llama.h"
#include <iostream>
#include <stdexcept>

namespace duorou {
namespace extensions {
namespace ollama {

struct llama_model* LlamaModelLoader::loadModel(const std::string& model_path, 
                                               const llama_model_params& params) {
    ensureBackendInitialized();
    
    struct llama_model* model = llama_model_load_from_file(model_path.c_str(), params);
    if (!model) {
        throw std::runtime_error("Failed to load model from: " + model_path);
    }
    
    return model;
}

struct llama_model* LlamaModelLoader::loadModelWithGGUF(const GGUFParser& parser) {
    // 从GGUF解析器创建参数
    llama_model_params params = LlamaParamsConverter::createFromGGUFParser(parser);
    
    // 获取模型文件路径
    std::string model_path = parser.getFilePath();
    
    return loadModel(model_path, params);
}

struct llama_context* LlamaModelLoader::createContext(struct llama_model* model, 
                                                     const ModelInfo& model_info) {
    if (!model) {
        throw std::invalid_argument("Model cannot be null");
    }
    
    llama_context_params ctx_params = llama_context_default_params();
    
    // 设置上下文参数
    ctx_params.n_ctx = model_info.context_length;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = -1;
    ctx_params.n_threads_batch = -1;
    ctx_params.embeddings = false;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    
    struct llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        throw std::runtime_error("Failed to create llama context");
    }
    
    return ctx;
}

llama_model_params LlamaModelLoader::getDefaultModelParams() {
    return llama_model_default_params();
}

bool LlamaModelLoader::validateModel(struct llama_model* model, const ModelInfo& model_info) {
    if (!model) {
        std::cerr << "Error: Model is null" << std::endl;
        return false;
    }
    
    // 验证词汇表大小 - 使用llama_model_n_vocab
    int32_t vocab_size = llama_model_n_vocab(model);
    if (vocab_size != static_cast<int32_t>(model_info.vocab_size)) {
        std::cerr << "Warning: Vocabulary size mismatch. Expected: " 
                  << model_info.vocab_size << ", Got: " << vocab_size << std::endl;
    }
    
    return true;
}

void LlamaModelLoader::freeModel(struct llama_model* model) {
    if (model) {
        llama_model_free(model);
    }
}

void LlamaModelLoader::freeContext(struct llama_context* ctx) {
    if (ctx) {
        llama_context_free(ctx);
    }
}

void LlamaModelLoader::printModelInfo(struct llama_model* model) {
    if (!model) {
        std::cout << "Model: null" << std::endl;
        return;
    }
    
    std::cout << "Model Information:" << std::endl;
    std::cout << "  Vocabulary size: " << llama_model_n_vocab(model) << std::endl;
    std::cout << "  Context length: " << llama_model_n_ctx_train(model) << std::endl;
    std::cout << "  Embedding size: " << llama_model_n_embd(model) << std::endl;
}

void LlamaModelLoader::ensureBackendInitialized() {
    static bool initialized = false;
    if (!initialized) {
        llama_backend_init();
        initialized = true;
    }
}

llama_context_params LlamaModelLoader::extractContextParams(const ModelInfo& model_info) {
    llama_context_params params = llama_context_default_params();
    
    params.n_ctx = model_info.context_length;
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_seq_max = 1;
    params.embeddings = false;
    
    return params;
}

bool LlamaModelLoader::validateCompatibility(const GGUFParser& parser, 
                                           const ModelInfo& model_info) {
    // 检查架构兼容性
    if (parser.getArchitecture() != model_info.architecture) {
        std::cerr << "Architecture mismatch: GGUF=" << parser.getArchitecture() 
                  << ", ModelInfo=" << model_info.architecture << std::endl;
        return false;
    }
    
    // 检查上下文长度
    uint32_t gguf_ctx_length = parser.getContextLength();
    if (gguf_ctx_length != model_info.context_length) {
        std::cerr << "Context length mismatch: GGUF=" << gguf_ctx_length 
                  << ", ModelInfo=" << model_info.context_length << std::endl;
        return false;
    }
    
    return true;
}

} // namespace ollama
} // namespace extensions
} // namespace duorou