#ifndef QWEN25VL_MODULAR_ENGINE_H
#define QWEN25VL_MODULAR_ENGINE_H

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <cstdint>
#include "algorithms/base_algorithm.h"
#include "algorithms/algorithm_factory.h"
#include "algorithms/multi_head_attention.h"
#include "algorithms/feed_forward.h"
#include "algorithms/rope_processor.h"
#include "algorithms/matrix_operations.h"

namespace duorou {
namespace extensions {
namespace ollama {

// Qwen特定的token ID定义
struct QwenTokens {
    static constexpr uint32_t ENDOFTEXT = 151643;    // <|endoftext|>
    static constexpr uint32_t IM_END = 151645;       // <|im_end|>
    static constexpr uint32_t IM_START = 151644;     // <|im_start|>
};

// ModelConfig在base_algorithm.h中定义

// Qwen2.5-VL模型配置
struct Qwen25VLConfig {
    uint32_t vocab_size = 152064;           // 词汇表大小
    uint32_t hidden_size = 3584;            // 隐藏层维度
    uint32_t intermediate_size = 18944;     // 前馈网络中间层维度
    uint32_t num_hidden_layers = 28;        // Transformer层数
    uint32_t num_attention_heads = 28;      // 注意力头数
    uint32_t num_key_value_heads = 4;       // KV头数（GQA）
    uint32_t max_position_embeddings = 32768; // 最大位置编码
    uint32_t max_window_layers = 21;        // 滑动窗口层数
    uint32_t sliding_window = 131072;       // 滑动窗口大小
    float rope_theta = 1000000.0f;          // RoPE基础频率
    float rms_norm_eps = 1e-6f;             // RMSNorm epsilon
    std::string activation = "silu";         // 激活函数类型
    bool use_sliding_window = true;         // 是否使用滑动窗口
    
    // 视觉相关配置
    uint32_t vision_hidden_size = 1280;     // 视觉编码器隐藏层维度
    uint32_t vision_intermediate_size = 5120; // 视觉前馈网络维度
    uint32_t vision_num_hidden_layers = 32;  // 视觉编码器层数
    uint32_t vision_num_attention_heads = 16; // 视觉注意力头数
    uint32_t image_size = 448;              // 图像尺寸
    uint32_t patch_size = 14;               // 图像块大小
    uint32_t num_channels = 3;              // 图像通道数
};

// 推理状态
struct InferenceState {
    std::vector<algorithms::Tensor> key_cache;    // K缓存
    std::vector<algorithms::Tensor> value_cache;  // V缓存
    uint32_t current_length = 0;                  // 当前序列长度
    uint32_t cache_position = 0;                  // 缓存位置
    bool is_prefill = true;                       // 是否为预填充阶段
};

// 流式生成回调函数类型
// 参数：新生成的token ID，是否为最后一个token
using StreamingCallback = std::function<void(uint32_t token_id, bool is_final)>;

// 流式生成状态
struct StreamingState {
    bool is_streaming = false;                    // 是否正在流式生成
    bool should_stop = false;                     // 是否应该停止生成
    StreamingCallback callback = nullptr;         // 回调函数
    std::vector<uint32_t> generated_tokens;       // 已生成的tokens
};

// 模块化Qwen2.5-VL推理引擎
class Qwen25VLModularEngine {
public:
    Qwen25VLModularEngine();
    ~Qwen25VLModularEngine();

    // 初始化引擎
    bool initialize(const Qwen25VLConfig& config);
    
    // 加载模型权重
    bool loadWeights(const std::string& model_path);
    
    // 文本推理
    std::vector<uint32_t> generateText(const std::vector<uint32_t>& input_ids,
                                      uint32_t max_length = 512,
                                      float temperature = 1.0f,
                                      uint32_t top_k = 50,
                                      float top_p = 0.9f);
    
    // 多模态推理（文本+图像）
    std::vector<uint32_t> generateMultimodal(const std::vector<uint32_t>& input_ids,
                                            const algorithms::Tensor& image_features,
                                            uint32_t max_length = 512,
                                            float temperature = 1.0f,
                                            uint32_t top_k = 50,
                                            float top_p = 0.9f);
    
    // 流式文本生成
    void generateTextStreaming(const std::vector<uint32_t>& input_ids,
                              StreamingCallback callback,
                              uint32_t max_length = 512,
                              float temperature = 1.0f,
                              uint32_t top_k = 50,
                              float top_p = 0.9f);
    
    // 流式多模态生成
    void generateMultimodalStreaming(const std::vector<uint32_t>& input_ids,
                                    const algorithms::Tensor& image_features,
                                    StreamingCallback callback,
                                    uint32_t max_length = 512,
                                    float temperature = 1.0f,
                                    uint32_t top_k = 50,
                                    float top_p = 0.9f);
    
    // 停止流式生成
    void stopStreaming();
    
    // 编码图像特征
    algorithms::Tensor encodeImage(const algorithms::Tensor& image);
    
    // 获取引擎状态
    bool isInitialized() const { return initialized_; }
    const Qwen25VLConfig& getConfig() const { return config_; }
    
    // 性能统计
    struct PerformanceStats {
        double total_inference_time = 0.0;  // 总推理时间(ms)
        uint32_t total_tokens = 0;           // 总token数
        double tokens_per_second = 0.0;      // tokens/秒
        uint32_t inference_count = 0;        // 推理次数
    };
    
    PerformanceStats getPerformanceStats() const { return perf_stats_; }
    void resetPerformanceStats();

private:
    // 配置和状态
    Qwen25VLConfig config_;
    bool initialized_ = false;
    InferenceState state_;
    StreamingState streaming_state_;
    PerformanceStats perf_stats_;
    
    // 算法组件
    std::unique_ptr<algorithms::MultiHeadAttention> attention_;
    std::unique_ptr<algorithms::FeedForward> feed_forward_;
    std::unique_ptr<algorithms::RoPEProcessor> rope_processor_;
    std::unique_ptr<algorithms::MatrixOperations> matrix_ops_;
    
    // 模型权重
    struct ModelWeights {
        algorithms::Tensor token_embeddings;     // 词嵌入
        algorithms::Tensor position_embeddings;  // 位置嵌入
        algorithms::Tensor norm_weight;          // 最终层归一化权重
        algorithms::Tensor lm_head_weight;       // 语言模型头权重
        
        // Transformer层权重
        std::vector<algorithms::Tensor> attention_weights;  // 注意力权重
        std::vector<algorithms::Tensor> ffn_weights;        // 前馈网络权重
        std::vector<algorithms::Tensor> layer_norm_weights; // 层归一化权重
        
        // 视觉编码器权重
        algorithms::Tensor vision_embeddings;    // 视觉嵌入
        std::vector<algorithms::Tensor> vision_attention_weights;
        std::vector<algorithms::Tensor> vision_ffn_weights;
        std::vector<algorithms::Tensor> vision_norm_weights;
    } weights_;
    
    // 核心计算方法
    algorithms::Tensor forwardTransformerLayer(const algorithms::Tensor& input,
                                              uint32_t layer_idx,
                                              const algorithms::Tensor* attention_mask = nullptr);
    
    algorithms::Tensor forwardVisionEncoder(const algorithms::Tensor& image);
    
    algorithms::Tensor applyRMSNorm(const algorithms::Tensor& input,
                                   const algorithms::Tensor& weight,
                                   float eps = 1e-6f);
    
    algorithms::Tensor applyEmbedding(const std::vector<uint32_t>& input_ids);
    
    algorithms::Tensor generateLogits(const algorithms::Tensor& hidden_states);
    
    uint32_t sampleToken(const algorithms::Tensor& logits,
                        float temperature,
                        uint32_t top_k,
                        float top_p,
                        const std::vector<uint32_t>& history = {},
                        float repetition_penalty = 1.0f);
    
    // 缓存管理
    void initializeKVCache();
    void updateKVCache(uint32_t layer_idx,
                      const algorithms::Tensor& key,
                      const algorithms::Tensor& value);
    
    // 工具方法
    algorithms::Tensor createAttentionMask(uint32_t seq_length, bool is_causal = true);
    void logPerformance(const std::string& operation, double time_ms);
    bool validateConfig(const Qwen25VLConfig& config);
    
    // 权重加载辅助方法
    bool loadTransformerWeights(const std::string& model_path);
    bool loadVisionWeights(const std::string& model_path);
    algorithms::Tensor loadTensorFromFile(const std::string& file_path);
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // QWEN25VL_MODULAR_ENGINE_H