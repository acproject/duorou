#ifndef DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H
#define DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

// 临时类型定义，直到llama.h可用
typedef int32_t llama_token;

// 前向声明
#include "gguf_parser.h"
#include "../../kvcache/cache.h"

// 前向声明
namespace duorou {
namespace ml {
class Tensor;
class Context;
namespace nn {
class MultiHeadAttention;
}
}
}

namespace duorou {
namespace extensions {
namespace ollama {

// 推理引擎接口
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    
    // 文本生成
    virtual std::string generateText(
        const std::string& prompt,
        uint32_t max_tokens = 512,
        float temperature = 0.7f,
        float top_p = 0.9f
    ) = 0;
    
    // 检查引擎是否就绪
    virtual bool isReady() const = 0;
};

// 基于ML模块的推理引擎实现
class MLInferenceEngine : public InferenceEngine {
public:
    explicit MLInferenceEngine(const std::string& model_id);
    ~MLInferenceEngine();
    
    // 初始化引擎
    bool initialize();
    
    // 实现推理接口
    std::string generateText(
        const std::string& prompt,
        uint32_t max_tokens = 512,
        float temperature = 0.7f,
        float top_p = 0.9f
    ) override;
    
    bool isReady() const override;

private:
    std::string model_id_;
    bool initialized_;
    
    // ML组件指针（使用前向声明）
    ml::Context* ml_context_;
    ml::nn::MultiHeadAttention* attention_;
    
    // GGUF相关
    std::unique_ptr<GGUFParser> gguf_parser_;
    std::string model_path_;
    
    // KV缓存
    std::unique_ptr<kvcache::Cache> kv_cache_;
    kvcache::CacheConfig cache_config_;
    
    // 模型权重和配置
    std::vector<ml::Tensor*> model_weights_;
    uint32_t vocab_size_;
    uint32_t n_layers_;
    uint32_t n_heads_;
    uint32_t n_embd_;
    uint32_t n_ctx_;
    
    // RoPE频率
    std::vector<float> rope_freqs_;
    bool rope_initialized_;
    
    // 辅助方法
    std::string processText(const std::string& text);
    bool loadModel(const std::string& model_path);
    std::vector<llama_token> tokenize(const std::string& text);
    std::string detokenize(const std::vector<llama_token>& tokens);
    
    // 模型加载步骤
    bool parseModelConfig();
    bool loadModelWeights();
    bool initializeKVCache();
    bool precomputeRoPEFreqs();
    void cleanupResources();
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H