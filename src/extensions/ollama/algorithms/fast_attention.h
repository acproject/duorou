#ifndef FAST_ATTENTION_H
#define FAST_ATTENTION_H

#include "base_algorithm.h"
#include <cmath>
#include <algorithm>
#include <chrono>

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// 快速注意力算法实现
class FastAttention : public IAttentionAlgorithm {
public:
  FastAttention() = default;
  ~FastAttention() override = default;

  bool initialize(const ModelConfig& config, const AlgorithmContext& context) override {
    context_ = context;
    
    // 从配置中获取参数
    head_dim_ = config.hidden_size / config.num_attention_heads;
    num_heads_ = config.num_attention_heads;
    scale_factor_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    
    // 设置块大小用于优化
    block_size_ = std::min(static_cast<uint32_t>(64), head_dim_);
    
    log("INFO", "FastAttention initialized with head_dim=" + std::to_string(head_dim_) + 
        ", num_heads=" + std::to_string(num_heads_));
    
    return true;
  }

  std::string getName() const override {
    return "FastAttention";
  }

  std::string getVersion() const override {
    return "1.0.0";
  }

  bool validateInput(const Tensor& input) const override {
    if (input.shape.size() < 2) {
      return false;
    }
    
    // 检查最后一个维度是否匹配head_dim
    return input.shape.back() == head_dim_;
  }

  Tensor compute(const Tensor& query, const Tensor& key, const Tensor& value,
                const Tensor* mask = nullptr, float scale = 1.0f) override {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 验证输入
    if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
      throw std::invalid_argument("Invalid input tensors for FastAttention");
    }
    
    // 获取序列长度
    uint32_t seq_len_q = query.shape[query.shape.size() - 2];
    uint32_t seq_len_k = key.shape[key.shape.size() - 2];
    
    // 创建输出张量
    std::vector<uint32_t> output_shape = query.shape;
    Tensor output(output_shape);
    
    // 使用传入的scale或默认的scale_factor_
    float effective_scale = (scale != 1.0f) ? scale : scale_factor_;
    
    // 执行快速注意力计算
    if (seq_len_q == 1 && seq_len_k > 1) {
      // 增量解码模式
      computeIncrementalAttention(query, key, value, output, effective_scale, mask);
    } else {
      // 标准注意力计算
      computeStandardAttention(query, key, value, output, effective_scale, mask);
    }
    
    // 更新统计信息
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    context_.total_time += duration.count() / 1000.0; // 转换为毫秒
    context_.call_count++;
    
    return output;
  }

  Tensor computeWithCache(const Tensor& query, const Tensor& key, const Tensor& value,
                         Tensor& key_cache, Tensor& value_cache,
                         uint32_t cache_position, const Tensor* mask = nullptr,
                         float scale = 1.0f) override {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 更新KV缓存
    updateKVCache(key, value, key_cache, value_cache, cache_position);
    
    // 使用缓存计算注意力
    Tensor output = compute(query, key_cache, value_cache, mask, scale);
    
    // 更新统计信息
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    context_.total_time += duration.count() / 1000.0;
    context_.call_count++;
    
    return output;
  }

private:
  uint32_t head_dim_ = 128;
  uint32_t num_heads_ = 32;
  float scale_factor_ = 1.0f;
  uint32_t block_size_ = 64;

  void computeStandardAttention(const Tensor& query, const Tensor& key, const Tensor& value,
                               Tensor& output, float scale, const Tensor* mask) {
    uint32_t seq_len_q = query.shape[query.shape.size() - 2];
    uint32_t seq_len_k = key.shape[key.shape.size() - 2];
    
    // 计算注意力分数 Q * K^T
    std::vector<float> scores(seq_len_q * seq_len_k);
    
    for (uint32_t i = 0; i < seq_len_q; ++i) {
      for (uint32_t j = 0; j < seq_len_k; ++j) {
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim_; ++d) {
          score += query.data[i * head_dim_ + d] * key.data[j * head_dim_ + d];
        }
        scores[i * seq_len_k + j] = score * scale;
      }
    }
    
    // 应用mask（如果提供）
    if (mask) {
      for (uint32_t i = 0; i < seq_len_q; ++i) {
        for (uint32_t j = 0; j < seq_len_k; ++j) {
          if (mask->data[i * seq_len_k + j] == 0.0f) {
            scores[i * seq_len_k + j] = -std::numeric_limits<float>::infinity();
          }
        }
      }
    }
    
    // Softmax
    applySoftmax(scores, seq_len_q, seq_len_k);
    
    // 计算输出 Attention * V
    for (uint32_t i = 0; i < seq_len_q; ++i) {
      for (uint32_t d = 0; d < head_dim_; ++d) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < seq_len_k; ++j) {
          sum += scores[i * seq_len_k + j] * value.data[j * head_dim_ + d];
        }
        output.data[i * head_dim_ + d] = sum;
      }
    }
  }

  void computeIncrementalAttention(const Tensor& query, const Tensor& key, const Tensor& value,
                                  Tensor& output, float scale, const Tensor* mask) {
    // 增量解码：query只有一个token，key/value有多个token
    uint32_t seq_len_k = key.shape[key.shape.size() - 2];
    
    std::vector<float> scores(seq_len_k);
    
    // 计算注意力分数
    for (uint32_t j = 0; j < seq_len_k; ++j) {
      float score = 0.0f;
      for (uint32_t d = 0; d < head_dim_; ++d) {
        score += query.data[d] * key.data[j * head_dim_ + d];
      }
      scores[j] = score * scale;
    }
    
    // 应用mask
    if (mask) {
      for (uint32_t j = 0; j < seq_len_k; ++j) {
        if (mask->data[j] == 0.0f) {
          scores[j] = -std::numeric_limits<float>::infinity();
        }
      }
    }
    
    // Softmax
    applySoftmax(scores, 1, seq_len_k);
    
    // 计算输出
    for (uint32_t d = 0; d < head_dim_; ++d) {
      float sum = 0.0f;
      for (uint32_t j = 0; j < seq_len_k; ++j) {
        sum += scores[j] * value.data[j * head_dim_ + d];
      }
      output.data[d] = sum;
    }
  }

  void applySoftmax(std::vector<float>& scores, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows; ++i) {
      float* row = &scores[i * cols];
      
      // 找到最大值以提高数值稳定性
      float max_val = *std::max_element(row, row + cols);
      
      // 计算exp和sum
      float sum = 0.0f;
      for (uint32_t j = 0; j < cols; ++j) {
        row[j] = std::exp(row[j] - max_val);
        sum += row[j];
      }
      
      // 归一化
      if (sum > 0.0f) {
        for (uint32_t j = 0; j < cols; ++j) {
          row[j] /= sum;
        }
      }
    }
  }

  void updateKVCache(const Tensor& key, const Tensor& value,
                    Tensor& key_cache, Tensor& value_cache,
                    uint32_t cache_position) {
    uint32_t seq_len = key.shape[key.shape.size() - 2];
    
    // 确保缓存有足够的空间
    if (cache_position + seq_len > key_cache.shape[key_cache.shape.size() - 2]) {
      log("WARNING", "KV cache overflow, position=" + std::to_string(cache_position) + 
          ", seq_len=" + std::to_string(seq_len));
      return;
    }
    
    // 复制key和value到缓存
    for (uint32_t i = 0; i < seq_len; ++i) {
      for (uint32_t d = 0; d < head_dim_; ++d) {
        key_cache.data[(cache_position + i) * head_dim_ + d] = key.data[i * head_dim_ + d];
        value_cache.data[(cache_position + i) * head_dim_ + d] = value.data[i * head_dim_ + d];
      }
    }
  }
};

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // FAST_ATTENTION_H