#ifndef ROPE_PROCESSOR_H
#define ROPE_PROCESSOR_H

#include "base_algorithm.h"
#include <cmath>
#include <chrono>

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// RoPE（旋转位置编码）处理器
class RoPEProcessor : public IPositionalEncodingAlgorithm {
public:
  RoPEProcessor() = default;
  ~RoPEProcessor() override = default;

  bool initialize(const ModelConfig& config, const AlgorithmContext& context) override {
    context_ = context;
    
    // 从配置中获取参数
    rope_dim_ = config.rope_dim;
    rope_base_ = config.rope_base;
    rope_scale_ = config.rope_scale;
    max_position_embeddings_ = config.max_position_embeddings;
    
    // 预计算频率
    precomputeFrequencies();
    
    log("INFO", "RoPEProcessor initialized with rope_dim=" + std::to_string(rope_dim_) + 
        ", rope_base=" + std::to_string(rope_base_) + 
        ", rope_scale=" + std::to_string(rope_scale_));
    
    return true;
  }

  std::string getName() const override {
    return "RoPEProcessor";
  }

  std::string getVersion() const override {
    return "1.0.0";
  }

  bool validateInput(const Tensor& input) const override {
    if (input.shape.size() < 2) {
      return false;
    }
    
    // 检查最后一个维度是否是rope_dim的倍数
    return (input.shape.back() % rope_dim_) == 0;
  }

  Tensor apply(const Tensor& input, uint32_t position_offset = 0) override {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 验证输入
    if (!validateInput(input)) {
      throw std::invalid_argument("Invalid input tensor for RoPEProcessor");
    }
    
    // 创建输出张量
    Tensor output = input; // 复制输入
    
    // 应用RoPE
    applyInPlace(output, position_offset);
    
    // 更新统计信息
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    context_.total_time += duration.count() / 1000.0;
    context_.call_count++;
    
    return output;
  }

  void applyInPlace(Tensor& tensor, uint32_t position_offset = 0) override {
    uint32_t seq_len = tensor.shape[tensor.shape.size() - 2];
    uint32_t hidden_dim = tensor.shape.back();
    uint32_t batch_size = (tensor.shape.size() > 2) ? tensor.shape[0] : 1;
    
    // 对每个位置应用RoPE
    for (uint32_t b = 0; b < batch_size; ++b) {
      for (uint32_t pos = 0; pos < seq_len; ++pos) {
        uint32_t actual_pos = pos + position_offset;
        applyRoPEToPosition(tensor, b, pos, actual_pos, hidden_dim);
      }
    }
  }

  // 批量应用RoPE（优化版本）
  void applyBatch(std::vector<Tensor>& tensors, uint32_t position_offset = 0) {
    for (auto& tensor : tensors) {
      applyInPlace(tensor, position_offset);
    }
  }

  // 获取预计算的cos和sin值
  std::pair<std::vector<float>, std::vector<float>> getCosSinCache(uint32_t max_seq_len) {
    if (max_seq_len > cos_cache_.size()) {
      extendCache(max_seq_len);
    }
    
    std::vector<float> cos_values(cos_cache_.begin(), cos_cache_.begin() + max_seq_len);
    std::vector<float> sin_values(sin_cache_.begin(), sin_cache_.begin() + max_seq_len);
    
    return {cos_values, sin_values};
  }

private:
  uint32_t rope_dim_ = 128;
  float rope_base_ = 10000.0f;
  float rope_scale_ = 1.0f;
  uint32_t max_position_embeddings_ = 32768;
  
  std::vector<float> inv_freq_;
  std::vector<float> cos_cache_;
  std::vector<float> sin_cache_;

  void precomputeFrequencies() {
    // 计算逆频率：1 / (base^(2i/dim)) for i in [0, dim/2)
    inv_freq_.clear();
    inv_freq_.reserve(rope_dim_ / 2);
    
    for (uint32_t i = 0; i < rope_dim_ / 2; ++i) {
      float freq = 1.0f / std::pow(rope_base_, 2.0f * i / rope_dim_);
      inv_freq_.push_back(freq);
    }
    
    // 预计算cos和sin缓存
    extendCache(max_position_embeddings_);
  }

  void extendCache(uint32_t new_max_len) {
    size_t old_size = cos_cache_.size();
    
    cos_cache_.resize(new_max_len * rope_dim_ / 2);
    sin_cache_.resize(new_max_len * rope_dim_ / 2);
    
    for (uint32_t pos = old_size / (rope_dim_ / 2); pos < new_max_len; ++pos) {
      for (uint32_t i = 0; i < rope_dim_ / 2; ++i) {
        float angle = pos * inv_freq_[i] * rope_scale_;
        
        size_t cache_idx = pos * (rope_dim_ / 2) + i;
        cos_cache_[cache_idx] = std::cos(angle);
        sin_cache_[cache_idx] = std::sin(angle);
      }
    }
  }

  void applyRoPEToPosition(Tensor& tensor, uint32_t batch_idx, uint32_t seq_pos, 
                          uint32_t actual_pos, uint32_t hidden_dim) {
    // 确保缓存足够大
    if (actual_pos >= cos_cache_.size() / (rope_dim_ / 2)) {
      extendCache(actual_pos + 1);
    }
    
    // 对每个RoPE维度对应用旋转
    for (uint32_t head = 0; head < hidden_dim / rope_dim_; ++head) {
      for (uint32_t i = 0; i < rope_dim_ / 2; ++i) {
        // 获取要旋转的两个元素的索引
        size_t base_idx;
        if (tensor.shape.size() > 2) {
          base_idx = batch_idx * tensor.shape[1] * hidden_dim + 
                    seq_pos * hidden_dim + 
                    head * rope_dim_;
        } else {
          base_idx = seq_pos * hidden_dim + head * rope_dim_;
        }
        
        size_t idx1 = base_idx + i;
        size_t idx2 = base_idx + i + rope_dim_ / 2;
        
        // 获取cos和sin值
        size_t cache_idx = actual_pos * (rope_dim_ / 2) + i;
        float cos_val = cos_cache_[cache_idx];
        float sin_val = sin_cache_[cache_idx];
        
        // 应用旋转矩阵
        float x1 = tensor.data[idx1];
        float x2 = tensor.data[idx2];
        
        tensor.data[idx1] = x1 * cos_val - x2 * sin_val;
        tensor.data[idx2] = x1 * sin_val + x2 * cos_val;
      }
    }
  }
};

// 优化的RoPE处理器（使用SIMD）
class OptimizedRoPEProcessor : public RoPEProcessor {
public:
  std::string getName() const override {
    return "OptimizedRoPEProcessor";
  }

  void applyInPlace(Tensor& tensor, uint32_t position_offset = 0) override {
    if (context_.use_simd) {
      applyInPlaceSIMD(tensor, position_offset);
    } else {
      RoPEProcessor::applyInPlace(tensor, position_offset);
    }
  }

private:
  void applyInPlaceSIMD(Tensor& tensor, uint32_t position_offset) {
    // 这里可以添加SIMD优化的RoPE实现
    // 目前回退到基础实现
    RoPEProcessor::applyInPlace(tensor, position_offset);
  }
};

// 支持不同RoPE变体的处理器
class ExtendedRoPEProcessor : public RoPEProcessor {
public:
  enum class RoPEType {
    STANDARD,    // 标准RoPE
    LINEAR,      // 线性插值RoPE
    DYNAMIC,     // 动态RoPE
    YARN         // YaRN RoPE
  };

  ExtendedRoPEProcessor(RoPEType type = RoPEType::STANDARD) : rope_type_(type) {}

  std::string getName() const override {
    switch (rope_type_) {
      case RoPEType::LINEAR: return "LinearRoPEProcessor";
      case RoPEType::DYNAMIC: return "DynamicRoPEProcessor";
      case RoPEType::YARN: return "YaRNRoPEProcessor";
      default: return "StandardRoPEProcessor";
    }
  }

  bool initialize(const ModelConfig& config, const AlgorithmContext& context) override {
    rope_type_specific_init(config);
    return RoPEProcessor::initialize(config, context);
  }

private:
  RoPEType rope_type_;
  float linear_scale_factor_ = 1.0f;
  float dynamic_alpha_ = 1.0f;

  void rope_type_specific_init(const ModelConfig& config) {
    switch (rope_type_) {
      case RoPEType::LINEAR:
        // 线性插值RoPE的特定初始化
        linear_scale_factor_ = static_cast<float>(config.max_position_embeddings) / 
                              config.original_context_length;
        break;
      case RoPEType::DYNAMIC:
        // 动态RoPE的特定初始化
        dynamic_alpha_ = 1.0f; // 可以根据需要调整
        break;
      case RoPEType::YARN:
        // YaRN RoPE的特定初始化
        break;
      default:
        break;
    }
  }
};

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // ROPE_PROCESSOR_H