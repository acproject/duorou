#ifndef QWEN25VL_INFERENCE_ENGINE_H
#define QWEN25VL_INFERENCE_ENGINE_H

#include "gguf_parser.h"
#include "text_processor.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Forward declarations
struct ModelArchitecture;

// SIMD support detection
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define SIMD_ENABLED
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define SIMD_ENABLED
#endif

// OpenBLAS support
#ifdef USE_OPENBLAS
#include <cblas.h>
#define BLAS_ENABLED
#endif

namespace duorou {
namespace extensions {
namespace ollama {

// 张量结构
struct Tensor {
  std::vector<float> data;
  std::vector<uint32_t> shape;
  uint32_t size;

  Tensor() : size(0) {}

  Tensor(const std::vector<uint32_t> &s) : shape(s), size(1) {
    for (uint32_t dim : shape) {
      size *= dim;
    }
    data.resize(size);
  }

  void reshape(const std::vector<uint32_t> &new_shape) {
    shape = new_shape;
    size = 1;
    for (uint32_t dim : shape) {
      size *= dim;
    }
    data.resize(size);
  }

  float &operator[](size_t index) { return data[index]; }

  const float &operator[](size_t index) const { return data[index]; }
};

// 注意力头结构
struct AttentionHead {
  Tensor query_weights;
  Tensor key_weights;
  Tensor value_weights;
  Tensor output_weights;

  Tensor query_bias;
  Tensor key_bias;
  Tensor value_bias;
  Tensor output_bias;
};

// Transformer层结构
struct TransformerLayer {
  // 多头注意力
  std::vector<AttentionHead> attention_heads;
  Tensor attention_norm_weights;
  Tensor attention_norm_bias;

  // 前馈网络
  Tensor ffn_gate_weights;
  Tensor ffn_up_weights;
  Tensor ffn_down_weights;
  Tensor ffn_gate_bias;
  Tensor ffn_up_bias;
  Tensor ffn_down_bias;

  // 层归一化
  Tensor ffn_norm_weights;
  Tensor ffn_norm_bias;
};

// 视觉编码器结构
struct VisionEncoder {
  // 卷积层
  std::vector<Tensor> conv_weights;
  std::vector<Tensor> conv_biases;

  // 位置编码
  Tensor position_embeddings;

  // Transformer层
  std::vector<TransformerLayer> layers;

  // 输出投影
  Tensor output_projection;
  Tensor output_bias;
};

// 模型配置结构
struct ModelConfig {
  uint32_t vocab_size;
  uint32_t hidden_size;
  uint32_t num_layers;
  uint32_t num_attention_heads;
  uint32_t intermediate_size;
  uint32_t max_position_embeddings;
  uint32_t rope_theta;
  float layer_norm_eps;

  // 视觉相关配置
  uint32_t vision_hidden_size;
  uint32_t vision_num_layers;
  uint32_t vision_num_attention_heads;
  uint32_t vision_intermediate_size;
  uint32_t image_size;
  uint32_t patch_size;

  ModelConfig() {
    vocab_size = 151936;
    hidden_size = 3584;
    num_layers = 28;
    num_attention_heads = 28;
    intermediate_size = 18944;
    max_position_embeddings = 32768;
    rope_theta = 1000000;
    layer_norm_eps = 1e-6;

    vision_hidden_size = 1280;
    vision_num_layers = 32;
    vision_num_attention_heads = 16;
    vision_intermediate_size = 5120;
    image_size = 448;
    patch_size = 14;
  }
};

// KV缓存结构
struct KVCache {
  std::vector<Tensor> key_cache;
  std::vector<Tensor> value_cache;
  uint32_t current_length;
  uint32_t max_length;

  KVCache() : current_length(0), max_length(0) {}

  void resize(uint32_t num_layers, uint32_t max_seq_len, uint32_t hidden_size) {
    max_length = max_seq_len;
    current_length = 0;
    key_cache.resize(num_layers);
    value_cache.resize(num_layers);

    for (uint32_t i = 0; i < num_layers; ++i) {
      key_cache[i].reshape({max_seq_len, hidden_size});
      value_cache[i].reshape({max_seq_len, hidden_size});
    }
  }

  void clear() { current_length = 0; }
};

// Qwen2.5VL推理引擎类
class Qwen25VLInferenceEngine {
public:
  // 构造函数和析构函数
  Qwen25VLInferenceEngine();
  explicit Qwen25VLInferenceEngine(bool verbose);
  ~Qwen25VLInferenceEngine();

  // 模型加载和管理
  bool loadModel(const std::string &model_path);
  bool unloadModel();
  bool isModelLoaded() const;

  // 文本生成
  std::string generateText(const std::string &prompt, int max_tokens = 100);
  std::string generateTextWithImage(const std::string &prompt,
                                    const std::string &image_path,
                                    int max_tokens = 100);

  // 批量生成
  std::vector<std::string>
  generateBatch(const std::vector<std::string> &prompts);

  // 流式生成
  void generateStream(const std::string &prompt,
                      std::function<void(const std::string &)> callback,
                      int max_tokens = 100);

  // 多模态生成（图像+文本）
  std::string
  generateTextWithImages(const std::string &prompt,
                         const std::vector<std::vector<float>> &image_features,
                         int max_tokens = 100);

  // 分词方法
  std::vector<int32_t> tokenize(const std::string &text);

  // 模型状态保存和加载
  bool saveState(const std::string &state_path) const;
  bool loadState(const std::string &state_path);

  // 性能优化
  void enableKVCache(bool enable = true);
  void setMaxSequenceLength(uint32_t max_length);
  void optimizeMemoryLayout();
  void optimizeComputationGraph();
  void warmupModel();

  // 量化支持
  bool enableQuantization(const std::string &quant_type = "q4_0");
  bool disableQuantization();

  // 并行处理
  void setNumThreads(uint32_t num_threads);
  void enableParallelProcessing(bool enable = true);

  // 采样参数设置
  void setTemperature(float temperature);
  void setTopP(float top_p);
  void setTopK(int top_k);
  void setRepetitionPenalty(float penalty);

  // 模型信息获取
  ModelConfig getModelConfig() const;
  std::string getModelInfo() const;
  size_t getModelSize() const;
  size_t calculateModelSize() const;

  // 性能统计
  double getInferenceTime() const;
  uint64_t getTokensGenerated() const;
  double getTokensPerSecond() const;
  void resetStatistics();

  // 词汇表操作
  std::string detokenize(const std::vector<int32_t> &tokens);
  int32_t getVocabSize() const;
  std::string getTokenString(int32_t token_id) const;
  int32_t getTokenId(const std::string &token) const;

  // 日志函数
  void log(const std::string &level, const std::string &message) const;

private:
  // 模型组件
  ModelConfig config_;
  std::unique_ptr<GGUFParser> gguf_parser_;

  // 模型权重
  Tensor token_embeddings_;
  std::vector<TransformerLayer> transformer_layers_;
  Tensor output_norm_weights_;
  Tensor output_norm_bias_;
  Tensor output_projection_;

  // 视觉编码器
  std::unique_ptr<VisionEncoder> vision_encoder_;

  // 文本处理器
  std::unique_ptr<TextProcessor> text_processor_;

  // 词汇表
  std::unordered_map<std::string, int32_t> vocab_;
  std::unordered_map<int32_t, std::string> reverse_vocab_;

  // 特殊token
  int32_t bos_token_id_;
  int32_t eos_token_id_;
  int32_t pad_token_id_;
  int32_t unk_token_id_;
  int32_t im_start_token_id_;
  int32_t im_end_token_id_;

  // KV缓存
  std::unique_ptr<KVCache> kv_cache_;
  bool kv_cache_enabled_;

  // RoPE频率
  std::vector<float> rope_freqs_;

  // 采样参数
  float temperature_;
  float top_p_;
  int top_k_;
  float repetition_penalty_;

  // 性能和状态
  bool model_loaded_;
  bool verbose_;
  uint32_t max_sequence_length_;
  uint32_t num_threads_;
  bool parallel_processing_enabled_;
  bool quantization_enabled_;
  std::string quantization_type_;

  // 统计信息
  mutable double total_inference_time_;
  mutable uint64_t total_tokens_generated_;
  
  // 重复惩罚历史
  std::vector<int32_t> generated_tokens_history_;

  // 内部方法
  bool loadWeights(const std::string &model_path);
  bool loadVocabulary();
  bool loadFallbackVocabulary();
  void loadDynamicSpecialTokens();
  bool loadTokenEmbedding();
  bool loadLayers();
  bool loadOutputWeights();
  bool loadVisionWeights();
  void precomputeRoPEFreqs();
  bool loadTensorFromGGUF(const std::string &tensor_name, Tensor &tensor);

  // 前向传播
  Tensor forward(const std::vector<int32_t> &input_ids);
  Tensor embedTokens(const std::vector<int32_t> &token_ids);
  Tensor applyLayerNorm(const Tensor &input, const Tensor &weights,
                        const Tensor &bias);
  Tensor applyRoPE(const Tensor &input, uint32_t position);
  Tensor multiHeadAttention(const Tensor &input, const TransformerLayer &layer,
                            uint32_t layer_idx);
  Tensor feedForward(const Tensor &input, const TransformerLayer &layer);
  Tensor
  processVisionInput(const std::vector<std::vector<float>> &image_features);

  // 采样方法
  int32_t sampleToken(const Tensor &logits);
  int32_t sampleTopK(const Tensor &logits, int k);
  int32_t sampleTopP(const Tensor &logits, float p);
  int32_t sampleTemperature(const Tensor &logits, float temp);

  // 工具方法
  void softmax(Tensor &tensor);
  void applyTemperature(Tensor &logits, float temperature);
  std::vector<std::pair<float, int32_t>> getTopKTokens(const Tensor &logits,
                                                       int k);
  float calculatePerplexity(const std::vector<int32_t> &tokens);

  // SIMD优化方法
  void vectorAdd(const float *a, const float *b, float *result, size_t size);
  void vectorMul(const float *a, const float *b, float *result, size_t size);
  void matrixMultiply(const float *a, const float *b, float *c, size_t m,
                      size_t n, size_t k);
  
  // 高效GEMM优化方法
    void batchedGEMM(const std::vector<const float*>& inputs,
                     const std::vector<const float*>& weights,
                     const std::vector<float*>& outputs,
                     size_t batch_size, size_t m, size_t n, size_t k);
    void optimizedMatMul(const float *a, const float *b, float *c,
                        size_t m, size_t n, size_t k, bool use_simd = true);
  void fusedAttentionKernel(const float *q, const float *k, const float *v,
                           float *output, size_t seq_len, size_t num_heads,
                           size_t head_dim, float scale, bool causal_mask = true);
  
  // Flash Attention风格的优化
  void blockwiseAttention(const float *q, const float *k, const float *v,
                         float *output, size_t seq_len, size_t num_heads,
                         size_t head_dim, float scale, size_t block_size = 64);
  
  // 批量RoPE处理
  void batchedRoPE(float *tensor, size_t seq_len, size_t num_heads,
                   size_t head_dim, uint32_t position_offset);

  // 内存管理
  void optimizeMemoryUsage();
  void clearCache();
  size_t getMemoryUsage() const;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // QWEN25VL_INFERENCE_ENGINE_H