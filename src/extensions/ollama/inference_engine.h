#ifndef DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H
#define DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// 前向声明 llama.cpp 结构体和类型
struct llama_model;
struct llama_context;
struct llama_sampler;
typedef int32_t llama_token;

// 前向声明
#include "../../kvcache/cache.h"
#include "../../model/qwen_multimodal_model.h"
#include "../../model/text_processor.h"
#include "../../model/tokenizer_factory.h"
#include "../../model/vocabulary.h"
#include "gguf_parser.h"

// 前向声明
namespace duorou {
namespace ml {
class Tensor;
class Context;
namespace nn {
class MultiHeadAttention;
}
} // namespace ml
} // namespace duorou

namespace duorou {
namespace extensions {
namespace ollama {

// 推理引擎接口
class InferenceEngine {
public:
  virtual ~InferenceEngine() = default;

  // 文本生成
  virtual std::string generateText(const std::string &prompt,
                                   uint32_t max_tokens = 512,
                                   float temperature = 0.7f,
                                   float top_p = 0.9f) = 0;

  // 检查引擎是否就绪
  virtual bool isReady() const = 0;
};

// 基于ML模块的推理引擎实现
class MLInferenceEngine : public InferenceEngine {
public:
  explicit MLInferenceEngine(const std::string &model_id);
  ~MLInferenceEngine();

  // 初始化引擎
  bool initialize();

  // 实现推理接口
  std::string generateText(const std::string &prompt, uint32_t max_tokens = 512,
                           float temperature = 0.7f,
                           float top_p = 0.9f) override;

  bool isReady() const override;

private:
  std::string model_id_;
  bool initialized_;

  // ML组件指针（使用前向声明）
  duorou::ml::Context *ml_context_;
  duorou::ml::nn::MultiHeadAttention *attention_;

  // GGUF相关
  std::unique_ptr<GGUFParser> gguf_parser_;
  std::string model_path_;

  // KV缓存
  std::unique_ptr<kvcache::Cache> kv_cache_;
  kvcache::CacheConfig cache_config_;

  // 模型权重和配置
  std::vector<duorou::ml::Tensor *> model_weights_;
  std::unordered_map<std::string, duorou::ml::Tensor *>
      weight_map_; // 权重名称到张量的映射
  uint32_t vocab_size_;
  uint32_t n_layers_;
  uint32_t n_heads_;
  uint32_t n_kv_heads_; // 来自 GGUF 的 KV 头数（attention.head_count_kv）
  uint32_t n_embd_;
  uint32_t n_ctx_;

  // 注意力每头维度（来自GGUF）：Q的head_dim与K/V的head_dim_k（GQA场景）
  uint32_t head_dim_q_;
  uint32_t head_dim_k_;

  // RoPE参数
  std::vector<float> rope_freqs_;
  bool rope_initialized_;
  uint32_t rope_dim_;    // rope.dimension_count（若缺失则使用 head_dim）
  float rope_freq_base_; // rope.freq_base（默认 10000.0）

  // llama.cpp 相关成员
  llama_model *llama_model_;
  llama_context *llama_context_;
  llama_sampler *llama_sampler_;

  // 是否使用 llama.cpp 后端
  bool use_llama_backend_;

  // Tokenizer and Vocabulary
  std::shared_ptr<duorou::model::Vocabulary> vocab_;
  std::unique_ptr<duorou::model::TextProcessor> tokenizer_;
  duorou::model::TokenizerFactoryOptions tok_opts_;

  // Qwen Multimodal Model for internal forward mode
  std::unique_ptr<duorou::model::QwenMultimodalModel> qwen_model_;

  // 辅助方法
  std::string processText(const std::string &text);
  bool loadModel(const std::string &model_path);
  std::vector<llama_token> tokenize(const std::string &text);
  std::string detokenize(const std::vector<llama_token> &tokens);
  std::string generateIntelligentResponse(const std::string &prompt,
                                          uint32_t max_tokens,
                                          float temperature);

  // 模型加载步骤
  bool parseModelConfig();
  bool loadModelWeights();
  bool initializeKVCache();
  bool precomputeRoPEFreqs(); // 辅助方法
  bool loadLlamaModel(const std::string &model_path);
  bool initializeSampler();
  std::string generateWithLlama(const std::string &prompt, uint32_t max_tokens,
                                float temperature, float top_p);

  std::string generateWithGGLM(const std::string &prompt, uint32_t max_tokens,
                               float temperature, float top_p);
  // 内部 Forward 模式的文本生成
  std::string generateWithInternalForward(const std::string &prompt,
                                          uint32_t max_tokens,
                                          float temperature, float top_p);
  void cleanupResources();

  // 权重加载映射辅助方法
  bool mapTensorWeights();
  bool checkInternalForwardSupport();
  bool tryAutoFallback(const std::string &reason);
  duorou::ml::DataType convertGGMLDataType(GGMLTensorType ggmlType);
  bool loadTensorData(const std::string &tensorName,
                      duorou::ml::Tensor *tensor);
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H