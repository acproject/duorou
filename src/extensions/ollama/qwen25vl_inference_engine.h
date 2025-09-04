#ifndef DUOROU_EXTENSIONS_OLLAMA_QWEN25VL_INFERENCE_ENGINE_H
#define DUOROU_EXTENSIONS_OLLAMA_QWEN25VL_INFERENCE_ENGINE_H

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "gguf_parser.h"

namespace duorou {
namespace extensions {
namespace ollama {

// 前向声明
struct Tensor;
struct AttentionWeights;
struct FeedForwardWeights;
struct LayerWeights;

// 张量数据结构
struct Tensor {
    std::vector<float> data;
    std::vector<uint64_t> shape;
    uint64_t size;
    
    Tensor() : size(0) {}
    Tensor(const std::vector<uint64_t>& dims);
    
    // 张量操作
    void reshape(const std::vector<uint64_t>& new_shape);
    uint64_t getElementCount() const;
    float* getData() { return data.data(); }
    const float* getData() const { return data.data(); }
};

// 注意力权重
struct AttentionWeights {
    Tensor q_proj;  // query projection
    Tensor k_proj;  // key projection  
    Tensor v_proj;  // value projection
    Tensor o_proj;  // output projection
    
    // RoPE相关
    Tensor rope_freqs;
    
    bool isValid() const {
        return q_proj.size > 0 && k_proj.size > 0 && 
               v_proj.size > 0 && o_proj.size > 0;
    }
};

// 前馈网络权重
struct FeedForwardWeights {
    Tensor gate_proj;  // gate projection
    Tensor up_proj;    // up projection
    Tensor down_proj;  // down projection
    
    bool isValid() const {
        return gate_proj.size > 0 && up_proj.size > 0 && down_proj.size > 0;
    }
};

// 层权重
struct LayerWeights {
    AttentionWeights attention;
    FeedForwardWeights feed_forward;
    
    Tensor input_layernorm;
    Tensor post_attention_layernorm;
    
    bool isValid() const {
        return attention.isValid() && feed_forward.isValid() &&
               input_layernorm.size > 0 && post_attention_layernorm.size > 0;
    }
};

// 视觉编码器权重（用于多模态）
struct VisionWeights {
    Tensor patch_embedding;
    Tensor position_embedding;
    std::vector<LayerWeights> layers;
    Tensor final_layernorm;
    
    bool isValid() const {
        return patch_embedding.size > 0 && !layers.empty();
    }
};

// 模型权重
struct ModelWeights {
    Tensor token_embedding;
    std::vector<LayerWeights> layers;
    Tensor output_norm;
    Tensor output;
    
    // 视觉相关权重（可选）
    std::unique_ptr<VisionWeights> vision;
    
    bool isValid() const {
        return token_embedding.size > 0 && !layers.empty() && 
               output_norm.size > 0 && output.size > 0;
    }
};

// 推理状态
struct InferenceState {
    std::vector<int32_t> tokens;
    std::vector<std::vector<float>> kv_cache_k;  // key cache for each layer
    std::vector<std::vector<float>> kv_cache_v;  // value cache for each layer
    uint32_t sequence_length;
    uint32_t max_sequence_length;
    
    InferenceState(uint32_t max_seq_len, uint32_t num_layers, uint32_t head_dim);
    void reset();
    void addToken(int32_t token);
};

// Qwen2.5VL推理引擎
class Qwen25VLInferenceEngine {
public:
    explicit Qwen25VLInferenceEngine(bool verbose = false);
    ~Qwen25VLInferenceEngine();
    
    // 模型加载
    bool loadModel(const std::string& gguf_file_path);
    bool isModelLoaded() const { return model_loaded_; }
    
    // 推理接口
    std::vector<float> forward(const std::vector<int32_t>& input_tokens);
    std::vector<float> forwardWithImages(const std::vector<int32_t>& input_tokens,
                                        const std::vector<std::vector<float>>& image_features);
    
    // 文本生成
    std::string generateText(const std::string& prompt, uint32_t max_tokens = 512);
    std::string generateTextWithImages(const std::string& prompt,
                                      const std::vector<std::vector<float>>& image_features,
                                      uint32_t max_tokens = 512);
    
    // 采样
    int32_t sampleToken(const std::vector<float>& logits, float temperature = 0.7f, float top_p = 0.9f);
    
    // 模型信息
    const ModelArchitecture& getArchitecture() const { return architecture_; }
    uint32_t getVocabSize() const { return vocab_size_; }
    uint32_t getContextLength() const { return architecture_.context_length; }
    
    // 词汇表操作
    std::vector<int32_t> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int32_t>& tokens);
    
private:
    // 模型加载辅助函数
    bool loadWeights(const GGUFParser& parser);
    bool loadTokenEmbedding(const GGUFParser& parser);
    bool loadLayers(const GGUFParser& parser);
    bool loadOutputWeights(const GGUFParser& parser);
    bool loadVisionWeights(const GGUFParser& parser);
    bool loadVocabulary(const GGUFParser& parser);
    
    // 张量加载
    bool loadTensorFromGGUF(const GGUFParser& parser, const std::string& tensor_name, Tensor& tensor);
    bool convertGGUFTensorToFloat(const GGUFTensorInfo& tensor_info, 
                                 const std::string& file_path, Tensor& output);
    
    // 推理核心函数
    std::vector<float> runInference(const std::vector<int32_t>& tokens, 
                                   const std::vector<std::vector<float>>* image_features = nullptr);
    
    // 层计算
    void computeLayer(uint32_t layer_idx, 
                     const std::vector<float>& input,
                     std::vector<float>& output,
                     InferenceState& state);
    
    // 注意力计算
    void computeAttention(const AttentionWeights& weights,
                         const std::vector<float>& input,
                         std::vector<float>& output,
                         std::vector<float>& k_cache,
                         std::vector<float>& v_cache,
                         uint32_t seq_pos);
    
    // 前馈网络计算
    void computeFeedForward(const FeedForwardWeights& weights,
                           const std::vector<float>& input,
                           std::vector<float>& output);
    
    // 数学运算
    void matmul(const float* a, const float* b, float* c, 
               uint32_t m, uint32_t n, uint32_t k);
    void addBias(float* data, const float* bias, uint32_t size);
    void layerNorm(float* data, const float* weight, const float* bias, 
                  uint32_t size, float eps = 1e-6f);
    void rmsNorm(float* data, const float* weight, uint32_t size, float eps = 1e-6f);
    void silu(float* data, uint32_t size);
    void softmax(float* data, uint32_t size);
    
    // RoPE (Rotary Position Embedding)
    void applyRoPE(float* q, float* k, uint32_t head_dim, uint32_t pos);
    void precomputeRoPEFreqs();
    
    // 视觉处理
    std::vector<float> processImageFeatures(const std::vector<float>& image_features);
    
    // 工具函数
    void log(const std::string& level, const std::string& message) const;
    
    // 内存使用监控
    void logMemoryUsage(const std::string& context = "");
    
private:
    bool verbose_;
    bool model_loaded_;
    
    // 模型架构和权重
    ModelArchitecture architecture_;
    std::unique_ptr<ModelWeights> weights_;
    
    // 模型参数
    uint32_t vocab_size_;
    uint32_t embedding_dim_;
    uint32_t num_layers_;
    uint32_t num_heads_;
    uint32_t num_kv_heads_;
    uint32_t head_dim_;
    uint32_t ffn_dim_;
    float rms_norm_eps_;
    
    // RoPE参数
    float rope_freq_base_;
    std::vector<float> rope_freqs_;
    
    // 词汇表
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    
    // 推理状态
    std::unique_ptr<InferenceState> inference_state_;
    
    // 文件路径
    std::string model_file_path_;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_QWEN25VL_INFERENCE_ENGINE_H