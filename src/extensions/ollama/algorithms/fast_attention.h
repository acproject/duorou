#ifndef FAST_ATTENTION_H
#define FAST_ATTENTION_H

#include "base_algorithm.h"
#include <vector>
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
    
    // 检查最后一个维度是否合理（允许不同的维度）
    uint32_t last_dim = input.shape.back();
    return last_dim > 0 && last_dim <= 4096;
  }

  Tensor compute(const Tensor& query, const Tensor& key, const Tensor& value,
                const Tensor* mask = nullptr, float scale = 1.0f) override {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 验证输入
    if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
      throw std::invalid_argument("Invalid input tensors for FastAttention");
    }
    
    // 检查张量维度兼容性
    if (query.shape.size() < 2 || key.shape.size() < 2 || value.shape.size() < 2) {
      throw std::invalid_argument("Tensors must have at least 2 dimensions");
    }
    
    // 动态获取实际的head_dim
    uint32_t actual_head_dim = query.shape.back();
    if (actual_head_dim == 0 || actual_head_dim > 4096) {
      throw std::invalid_argument("Invalid head dimension: " + std::to_string(actual_head_dim));
    }
    
    // 确保key和value的维度与query匹配
    if (key.shape.back() != actual_head_dim || value.shape.back() != actual_head_dim) {
      throw std::invalid_argument("Dimension mismatch: query=" + std::to_string(actual_head_dim) + 
                                ", key=" + std::to_string(key.shape.back()) + 
                                ", value=" + std::to_string(value.shape.back()));
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
      computeIncrementalAttention(query, key, value, output, effective_scale, mask, actual_head_dim);
    } else {
      // 标准注意力计算
      computeStandardAttention(query, key, value, output, effective_scale, mask, actual_head_dim);
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
    
    // 输入验证
    if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
      throw std::invalid_argument("Invalid input tensors for FastAttention computeWithCache");
    }
    
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

  // 公有方法：更新KV缓存
  void updateKVCache(const Tensor& key, const Tensor& value,
                    Tensor& key_cache, Tensor& value_cache,
                    uint32_t cache_position) {
    // 动态获取head_dim
    uint32_t head_dim = key.shape.back();
    
    // 检查数据大小
    if (key.data.size() < head_dim || value.data.size() < head_dim) {
      throw std::runtime_error("Insufficient data size for cache update");
    }
    
    // 确保缓存有足够的空间
    uint32_t required_cache_size = (cache_position + 1) * head_dim;
    if (key_cache.data.size() < required_cache_size) {
      key_cache.data.resize(required_cache_size);
    }
    if (value_cache.data.size() < required_cache_size) {
      value_cache.data.resize(required_cache_size);
    }
    
    // 复制新的key和value到缓存
    for (uint32_t i = 0; i < head_dim; ++i) {
      key_cache.data[cache_position * head_dim + i] = key.data[i];
      value_cache.data[cache_position * head_dim + i] = value.data[i];
    }
    
    // 更新缓存的形状
    if (key_cache.shape.size() >= 2) {
      key_cache.shape[key_cache.shape.size() - 2] = cache_position + 1;
    }
    if (value_cache.shape.size() >= 2) {
      value_cache.shape[value_cache.shape.size() - 2] = cache_position + 1;
    }
  }

private:
  uint32_t head_dim_ = 128;
  uint32_t num_heads_ = 32;
  float scale_factor_ = 1.0f;
  uint32_t block_size_ = 64;

  void computeStandardAttention(const Tensor& query, const Tensor& key, const Tensor& value,
                               Tensor& output, float scale, const Tensor* mask, uint32_t head_dim) {
    // 边界检查
    if (query.shape.empty() || key.shape.empty() || value.shape.empty() || output.shape.empty()) {
      throw std::invalid_argument("Invalid tensor shapes in computeStandardAttention");
    }
    
    uint32_t seq_len_q = query.shape[query.shape.size() - 2];
    uint32_t seq_len_k = key.shape[key.shape.size() - 2];
    
    // 检查数据大小
    if (query.data.size() < seq_len_q * head_dim || 
        key.data.size() < seq_len_k * head_dim ||
        value.data.size() < seq_len_k * head_dim ||
        output.data.size() < seq_len_q * head_dim) {
      throw std::runtime_error("Tensor data size insufficient for attention computation");
    }
    
    // 计算注意力分数 Q * K^T
    std::vector<float> scores(seq_len_q * seq_len_k);
    
    for (uint32_t i = 0; i < seq_len_q; ++i) {
      for (uint32_t j = 0; j < seq_len_k; ++j) {
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
          score += query.data[i * head_dim + d] * key.data[j * head_dim + d];
        }
        scores[i * seq_len_k + j] = score * scale;
      }
    }
    
    // 应用mask（如果提供）
    if (mask && mask->data.size() >= seq_len_q * seq_len_k) {
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
      for (uint32_t d = 0; d < head_dim; ++d) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < seq_len_k; ++j) {
          sum += scores[i * seq_len_k + j] * value.data[j * head_dim + d];
        }
        output.data[i * head_dim + d] = sum;
      }
    }
  }

  void computeIncrementalAttention(const Tensor& query, const Tensor& key, const Tensor& value,
                                  Tensor& output, float scale, const Tensor* mask, uint32_t head_dim) {
    // 边界检查
    if (query.shape.empty() || key.shape.empty() || value.shape.empty() || output.shape.empty()) {
      throw std::invalid_argument("Invalid tensor shapes in computeIncrementalAttention");
    }
    
    // 增量解码：query只有一个token，key/value有多个token
    uint32_t seq_len_k = key.shape[key.shape.size() - 2];
    
    // 检查数据大小
    if (query.data.size() < head_dim || 
        key.data.size() < seq_len_k * head_dim ||
        value.data.size() < seq_len_k * head_dim ||
        output.data.size() < head_dim) {
      throw std::runtime_error("Tensor data size insufficient for incremental attention computation");
    }
    
    std::vector<float> scores(seq_len_k);
    
    // 计算注意力分数
    for (uint32_t j = 0; j < seq_len_k; ++j) {
      float score = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        score += query.data[d] * key.data[j * head_dim + d];
      }
      scores[j] = score * scale;
    }
    
    // 应用mask
    if (mask && mask->data.size() >= seq_len_k) {
      for (uint32_t j = 0; j < seq_len_k; ++j) {
        if (mask->data[j] == 0.0f) {
          scores[j] = -std::numeric_limits<float>::infinity();
        }
      }
    }
    
    // Softmax
    applySoftmax(scores, 1, seq_len_k);
    
    // 计算输出
    for (uint32_t d = 0; d < head_dim; ++d) {
      float sum = 0.0f;
      for (uint32_t j = 0; j < seq_len_k; ++j) {
        sum += scores[j] * value.data[j * head_dim + d];
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


};

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // FAST_ATTENTION_H