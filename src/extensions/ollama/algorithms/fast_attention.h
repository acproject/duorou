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
    std::cerr << "[DEBUG] FastAttention::compute called" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 验证输入
    if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
      throw std::invalid_argument("Invalid input tensors for FastAttention");
    }
    std::cerr << "[DEBUG] FastAttention input validation passed" << std::endl;
    
    // 早期退出机制：对于小序列长度使用简化算法
    const uint32_t seq_len = key.shape[key.shape.size() - 2];
    if (seq_len <= 16) {
      return computeSimpleAttention(query, key, value);
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
    
    // 使用内存池创建输出张量
    std::vector<uint32_t> output_shape = query.shape;
    auto& output_buffer = getTempOutputBuffer(seq_len_q * actual_head_dim);
    Tensor output(output_shape);
    output.data.assign(output_buffer.begin(), output_buffer.begin() + seq_len_q * actual_head_dim);
    
    // 使用传入的scale或默认的scale_factor_
    float effective_scale = (scale != 1.0f) ? scale : scale_factor_;
    
    // 执行快速注意力计算
    if (seq_len_k == 1) {
      // 特殊优化：当key/value只有一个token时的快速路径
      computeFastSingleKeyAttention(query, key, value, output, effective_scale, mask, actual_head_dim);
    } else if (seq_len_q == 1 && seq_len_k > 1) {
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

  // 公有方法：更新KV缓存（带重复检测优化）
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
    
    // 优化：检查是否已存在相同的key/value（避免重复计算）
    if (cache_position > 0 && isCacheHit(key, value, key_cache, value_cache, cache_position, head_dim)) {
      std::cerr << "[DEBUG] KV Cache hit detected at position " << cache_position 
                << ", skipping redundant update" << std::endl;
      return;
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
    
    std::cerr << "[DEBUG] KV Cache updated at position " << cache_position 
              << ", cache size: " << (cache_position + 1) << std::endl;
  }

private:
  uint32_t head_dim_ = 128;
  uint32_t num_heads_ = 32;
  float scale_factor_ = 1.0f;
  uint32_t block_size_ = 64;
  AlgorithmContext context_;
  
  // 内存池优化：预分配的临时缓存
  mutable std::vector<float> temp_scores_buffer_;
  mutable std::vector<float> temp_output_buffer_;
  
  // 获取临时分数缓存
  std::vector<float>& getTempScoresBuffer(size_t required_size) const {
    if (temp_scores_buffer_.size() < required_size) {
      temp_scores_buffer_.resize(required_size);
    }
    return temp_scores_buffer_;
  }
  
  // 获取临时输出缓存
  std::vector<float>& getTempOutputBuffer(size_t required_size) const {
    if (temp_output_buffer_.size() < required_size) {
      temp_output_buffer_.resize(required_size);
    }
    return temp_output_buffer_;
  }
  
  // 检测KV缓存命中（避免重复计算相同的key/value）
  bool isCacheHit(const Tensor& key, const Tensor& value,
                  const Tensor& key_cache, const Tensor& value_cache,
                  uint32_t cache_position, uint32_t head_dim) {
    // 简单的相似性检测：检查最近几个位置是否有相同的key/value
    const float tolerance = 1e-6f;
    const uint32_t check_range = std::min(cache_position, 3u); // 检查最近3个位置
    
    for (uint32_t pos = cache_position - check_range; pos < cache_position; ++pos) {
      bool key_match = true;
      bool value_match = true;
      
      // 检查key是否匹配
      for (uint32_t i = 0; i < head_dim && key_match; ++i) {
        float diff = std::abs(key.data[i] - key_cache.data[pos * head_dim + i]);
        if (diff > tolerance) {
          key_match = false;
        }
      }
      
      // 检查value是否匹配
      for (uint32_t i = 0; i < head_dim && value_match; ++i) {
        float diff = std::abs(value.data[i] - value_cache.data[pos * head_dim + i]);
        if (diff > tolerance) {
          value_match = false;
        }
      }
      
      // 如果key和value都匹配，则为缓存命中
      if (key_match && value_match) {
        return true;
      }
    }
    
    return false;
  }// 简化的注意力计算，用于小序列长度
  Tensor computeSimpleAttention(const Tensor& query, const Tensor& key, const Tensor& value) {
    const uint32_t seq_len_q = query.shape[query.shape.size() - 2];
    const uint32_t seq_len_k = key.shape[key.shape.size() - 2];
    const uint32_t head_dim = query.shape[query.shape.size() - 1];
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    Tensor output;
    output.shape = query.shape;
    output.data.resize(seq_len_q * head_dim);
    
    // 简单的三重循环实现，适用于小序列
    for (uint32_t i = 0; i < seq_len_q; ++i) {
      auto& scores_buffer = getTempScoresBuffer(seq_len_k);
    std::vector<float>& scores = scores_buffer;
      
      // 计算注意力分数
      for (uint32_t j = 0; j < seq_len_k; ++j) {
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
          score += query.data[i * head_dim + d] * key.data[j * head_dim + d];
        }
        scores[j] = score * scale;
      }
      
      // 应用softmax
      applySoftmax(scores, 1, seq_len_k);
      
      // 计算输出
      for (uint32_t d = 0; d < head_dim; ++d) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < seq_len_k; ++j) {
          sum += scores[j] * value.data[j * head_dim + d];
        }
        output.data[i * head_dim + d] = sum;
      }
    }
    
    return output;
  }

  // 特殊优化：当key/value只有一个token时的快速路径
  void computeFastSingleKeyAttention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    Tensor& output, float scale, const Tensor* mask, uint32_t head_dim) {
    std::cerr << "[DEBUG] FastAttention::computeFastSingleKeyAttention called with seq_len_q=" 
              << query.shape[query.shape.size() - 2] << ", seq_len_k=1, head_dim=" << head_dim << std::endl;
    
    uint32_t seq_len_q = query.shape[query.shape.size() - 2];
    
    // 检查数据大小
    if (query.data.size() < seq_len_q * head_dim || 
        key.data.size() < head_dim ||
        value.data.size() < head_dim ||
        output.data.size() < seq_len_q * head_dim) {
      throw std::runtime_error("Tensor data size insufficient for fast single key attention computation");
    }
    
    // 当只有一个key/value时，注意力权重对所有query位置都是1.0
    // 直接将value复制到所有输出位置，无需计算softmax
    const float* value_data = value.data.data();
    
    for (uint32_t i = 0; i < seq_len_q; ++i) {
      float* output_row = &output.data[i * head_dim];
      
      // 直接复制value到输出（4路展开优化）
      uint32_t d = 0;
      for (; d + 3 < head_dim; d += 4) {
        output_row[d] = value_data[d];
        output_row[d+1] = value_data[d+1];
        output_row[d+2] = value_data[d+2];
        output_row[d+3] = value_data[d+3];
      }
      
      // 处理剩余元素
      for (; d < head_dim; ++d) {
        output_row[d] = value_data[d];
      }
    }
    
    std::cerr << "[DEBUG] Fast single key attention computation completed" << std::endl;
  }

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
    
    // 使用内存池优化的注意力分数计算 Q * K^T
    auto& scores_buffer = getTempScoresBuffer(seq_len_q * seq_len_k);
    std::vector<float>& scores = scores_buffer;
    
    // 使用更高效的矩阵乘法：批量计算所有点积
    for (uint32_t i = 0; i < seq_len_q; ++i) {
      const float* q_row = &query.data[i * head_dim];
      for (uint32_t j = 0; j < seq_len_k; ++j) {
        const float* k_row = &key.data[j * head_dim];
        
        // 向量化点积计算
        float score = 0.0f;
        uint32_t d = 0;
        
        // 4路展开优化
        for (; d + 3 < head_dim; d += 4) {
          score += q_row[d] * k_row[d] + 
                   q_row[d+1] * k_row[d+1] + 
                   q_row[d+2] * k_row[d+2] + 
                   q_row[d+3] * k_row[d+3];
        }
        
        // 处理剩余元素
        for (; d < head_dim; ++d) {
          score += q_row[d] * k_row[d];
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
    
    // 优化的输出计算 Attention * V
    for (uint32_t i = 0; i < seq_len_q; ++i) {
      const float* attention_row = &scores[i * seq_len_k];
      float* output_row = &output.data[i * head_dim];
      
      // 初始化输出行为零
      std::fill(output_row, output_row + head_dim, 0.0f);
      
      // 向量化的加权求和
      for (uint32_t j = 0; j < seq_len_k; ++j) {
        const float attention_weight = attention_row[j];
        const float* value_row = &value.data[j * head_dim];
        
        // 4路展开优化
        uint32_t d = 0;
        for (; d + 3 < head_dim; d += 4) {
          output_row[d] += attention_weight * value_row[d];
          output_row[d+1] += attention_weight * value_row[d+1];
          output_row[d+2] += attention_weight * value_row[d+2];
          output_row[d+3] += attention_weight * value_row[d+3];
        }
        
        // 处理剩余元素
        for (; d < head_dim; ++d) {
          output_row[d] += attention_weight * value_row[d];
        }
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
    
    auto& scores_buffer = getTempScoresBuffer(seq_len_k);
    std::vector<float>& scores = scores_buffer;
    
    // 优化的注意力分数计算
    const float* q_data = query.data.data();
    for (uint32_t j = 0; j < seq_len_k; ++j) {
      const float* k_row = &key.data[j * head_dim];
      
      float score = 0.0f;
      uint32_t d = 0;
      
      // 4路展开优化
      for (; d + 3 < head_dim; d += 4) {
        score += q_data[d] * k_row[d] + 
                 q_data[d+1] * k_row[d+1] + 
                 q_data[d+2] * k_row[d+2] + 
                 q_data[d+3] * k_row[d+3];
      }
      
      // 处理剩余元素
      for (; d < head_dim; ++d) {
        score += q_data[d] * k_row[d];
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
    
    // 优化的输出计算
    std::fill(output.data.begin(), output.data.begin() + head_dim, 0.0f);
    
    for (uint32_t j = 0; j < seq_len_k; ++j) {
      const float attention_weight = scores[j];
      const float* value_row = &value.data[j * head_dim];
      
      // 4路展开优化
      uint32_t d = 0;
      for (; d + 3 < head_dim; d += 4) {
        output.data[d] += attention_weight * value_row[d];
        output.data[d+1] += attention_weight * value_row[d+1];
        output.data[d+2] += attention_weight * value_row[d+2];
        output.data[d+3] += attention_weight * value_row[d+3];
      }
      
      // 处理剩余元素
      for (; d < head_dim; ++d) {
        output.data[d] += attention_weight * value_row[d];
      }
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