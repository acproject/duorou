#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include "base_algorithm.h"
#include "fast_attention.h"
#include <vector>
#include <memory>
#include <chrono>

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// 多头注意力算法实现
class MultiHeadAttention : public IAttentionAlgorithm {
public:
  MultiHeadAttention() = default;
  ~MultiHeadAttention() override = default;

  bool initialize(const ModelConfig& config, const AlgorithmContext& context) override {
    context_ = context;
    
    // 从配置中获取参数
    hidden_size_ = config.hidden_size;
    num_heads_ = config.num_attention_heads;
    num_kv_heads_ = config.num_key_value_heads;
    head_dim_ = hidden_size_ / num_heads_;
    kv_head_dim_ = hidden_size_ / num_kv_heads_;
    
    // 计算组大小（用于GQA）
    group_size_ = num_heads_ / num_kv_heads_;
    
    // 初始化每个头的FastAttention实例
    attention_heads_.clear();
    attention_heads_.reserve(num_heads_);
    
    for (uint32_t i = 0; i < num_heads_; ++i) {
      auto head = std::make_unique<FastAttention>();
      
      // 为每个头创建单独的配置
      // 对于query头使用head_dim_，对于key/value头使用kv_head_dim_
      ModelConfig head_config = config;
      head_config.hidden_size = head_dim_;
      head_config.num_attention_heads = 1;
      
      if (!head->initialize(head_config, context)) {
        log("ERROR", "Failed to initialize attention head " + std::to_string(i));
        return false;
      }
      
      attention_heads_.push_back(std::move(head));
    }
    
    // 为key/value头创建单独的FastAttention实例
    kv_attention_heads_.clear();
    kv_attention_heads_.reserve(num_kv_heads_);
    
    for (uint32_t i = 0; i < num_kv_heads_; ++i) {
      auto kv_head = std::make_unique<FastAttention>();
      
      // 为key/value头创建配置
      ModelConfig kv_head_config = config;
      kv_head_config.hidden_size = kv_head_dim_;
      kv_head_config.num_attention_heads = 1;
      
      if (!kv_head->initialize(kv_head_config, context)) {
        log("ERROR", "Failed to initialize KV attention head " + std::to_string(i));
        return false;
      }
      
      kv_attention_heads_.push_back(std::move(kv_head));
    }
    
    log("INFO", "MultiHeadAttention initialized with " + std::to_string(num_heads_) + 
        " heads, head_dim=" + std::to_string(head_dim_) + 
        ", kv_heads=" + std::to_string(num_kv_heads_));
    
    return true;
  }

  std::string getName() const override {
    return "MultiHeadAttention";
  }

  std::string getVersion() const override {
    return "1.0.0";
  }

  bool validateInput(const Tensor& input) const override {
    if (input.shape.size() < 2) {
      return false;
    }
    
    // 检查最后一个维度是否匹配hidden_size
    return input.shape.back() == hidden_size_;
  }

  Tensor compute(const Tensor& query, const Tensor& key, const Tensor& value,
                const Tensor* mask = nullptr, float scale = 1.0f) override {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 验证输入
    if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
      throw std::invalid_argument("Invalid input tensors for MultiHeadAttention");
    }
    
    // 获取批次大小和序列长度
    uint32_t batch_size = 1;
    uint32_t seq_len_q = query.shape[query.shape.size() - 2];
    uint32_t seq_len_k = key.shape[key.shape.size() - 2];
    
    if (query.shape.size() > 2) {
      batch_size = query.shape[0];
    }
    
    // 分割Q、K、V到多个头
    auto query_heads = splitToHeads(query, num_heads_, head_dim_);
    auto key_heads = splitToHeads(key, num_kv_heads_, kv_head_dim_);
    auto value_heads = splitToHeads(value, num_kv_heads_, kv_head_dim_);
    
    // 为每个头计算注意力
    std::vector<Tensor> head_outputs;
    head_outputs.reserve(num_heads_);
    
    for (uint32_t h = 0; h < num_heads_; ++h) {
      // 对于GQA，多个查询头可能共享同一个键值头
      uint32_t kv_head_idx = h / group_size_;
      
      // 创建头级别的mask
      Tensor* head_mask = nullptr;
      Tensor head_mask_tensor;
      if (mask) {
        head_mask_tensor = createHeadMask(*mask, h, seq_len_q, seq_len_k);
        head_mask = &head_mask_tensor;
      }
      
      // 计算单头注意力
      Tensor head_output = attention_heads_[h]->compute(
        query_heads[h], key_heads[kv_head_idx], value_heads[kv_head_idx], 
        head_mask, scale
      );
      
      head_outputs.push_back(std::move(head_output));
    }
    
    // 合并所有头的输出
    Tensor output = concatenateHeads(head_outputs, batch_size, seq_len_q);
    
    // 更新统计信息
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    context_.total_time += duration.count() / 1000.0;
    context_.call_count++;
    
    return output;
  }

  Tensor computeWithCache(const Tensor& query, const Tensor& key, const Tensor& value,
                         Tensor& key_cache, Tensor& value_cache,
                         uint32_t cache_position, const Tensor* mask = nullptr,
                         float scale = 1.0f) override {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 分割输入到多个头
    auto query_heads = splitToHeads(query, num_heads_, head_dim_);
    auto key_heads = splitToHeads(key, num_kv_heads_, kv_head_dim_);
    auto value_heads = splitToHeads(value, num_kv_heads_, kv_head_dim_);
    
    // 分割缓存到多个头
    auto key_cache_heads = splitCacheToHeads(key_cache, num_kv_heads_, kv_head_dim_);
    auto value_cache_heads = splitCacheToHeads(value_cache, num_kv_heads_, kv_head_dim_);
    
    // 为每个头计算注意力
    std::vector<Tensor> head_outputs;
    head_outputs.reserve(num_heads_);
    
    uint32_t seq_len_q = query.shape[query.shape.size() - 2];
    uint32_t seq_len_k = key.shape[key.shape.size() - 2];
    uint32_t batch_size = (query.shape.size() > 2) ? query.shape[0] : 1;
    
    for (uint32_t h = 0; h < num_heads_; ++h) {
      uint32_t kv_head_idx = h / group_size_;
      
      // 创建头级别的mask
      Tensor* head_mask = nullptr;
      Tensor head_mask_tensor;
      if (mask) {
        head_mask_tensor = createHeadMask(*mask, h, seq_len_q, seq_len_k);
        head_mask = &head_mask_tensor;
      }
      
      // 使用缓存计算单头注意力
      // 使用attention_heads_处理query，kv_attention_heads_处理key/value缓存
      // 首先用kv_attention_heads_更新缓存
      kv_attention_heads_[kv_head_idx]->updateKVCache(
        key_heads[kv_head_idx], value_heads[kv_head_idx],
        key_cache_heads[kv_head_idx], value_cache_heads[kv_head_idx],
        cache_position
      );
      
      // 将kv缓存调整为query维度
      Tensor adjusted_key_cache = adjustKVCacheForQuery(key_cache_heads[kv_head_idx], head_dim_);
      Tensor adjusted_value_cache = adjustKVCacheForQuery(value_cache_heads[kv_head_idx], head_dim_);
      
      // 然后用attention_heads_计算注意力（query维度匹配）
      Tensor head_output = attention_heads_[h]->compute(
        query_heads[h], adjusted_key_cache, adjusted_value_cache,
        head_mask, scale
      );
      
      head_outputs.push_back(std::move(head_output));
    }
    
    // 更新原始缓存
    updateOriginalCache(key_cache_heads, value_cache_heads, key_cache, value_cache);
    
    // 合并所有头的输出
    Tensor output = concatenateHeads(head_outputs, batch_size, seq_len_q);
    
    // 更新统计信息
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    context_.total_time += duration.count() / 1000.0;
    context_.call_count++;
    
    return output;
  }

private:
  uint32_t hidden_size_ = 3584;
  uint32_t num_heads_ = 28;
  uint32_t num_kv_heads_ = 4;
  uint32_t head_dim_ = 128;
  uint32_t kv_head_dim_ = 896;
  uint32_t group_size_ = 7;
  
  std::vector<std::unique_ptr<FastAttention>> attention_heads_;
  std::vector<std::unique_ptr<FastAttention>> kv_attention_heads_;

  // 将张量分割到多个头
  std::vector<Tensor> splitToHeads(const Tensor& input, uint32_t num_heads, uint32_t head_dim) {
    // 边界检查
    if (input.shape.empty() || num_heads == 0 || head_dim == 0) {
      throw std::invalid_argument("Invalid parameters for splitToHeads");
    }
    
    std::vector<Tensor> heads;
    heads.reserve(num_heads);
    
    uint32_t seq_len = input.shape[input.shape.size() - 2];
    uint32_t batch_size = (input.shape.size() > 2) ? input.shape[0] : 1;
    
    // 检查输入数据大小
    size_t expected_size = batch_size * seq_len * num_heads * head_dim;
    if (input.data.size() < expected_size) {
      throw std::runtime_error("Input tensor data size insufficient for splitToHeads");
    }
    
    for (uint32_t h = 0; h < num_heads; ++h) {
      std::vector<uint32_t> head_shape;
      if (batch_size > 1) {
        head_shape = {batch_size, seq_len, head_dim};
      } else {
        head_shape = {seq_len, head_dim};
      }
      
      Tensor head_tensor(head_shape);
      
      // 复制对应头的数据
      for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
          for (uint32_t d = 0; d < head_dim; ++d) {
            size_t src_idx = b * seq_len * (num_heads * head_dim) + 
                           s * (num_heads * head_dim) + 
                           h * head_dim + d;
            size_t dst_idx = b * seq_len * head_dim + s * head_dim + d;
            
            if (batch_size == 1) {
              dst_idx = s * head_dim + d;
            }
            
            // 边界检查
            if (src_idx >= input.data.size() || dst_idx >= head_tensor.data.size()) {
              throw std::runtime_error("Array index out of bounds in splitToHeads");
            }
            
            head_tensor.data[dst_idx] = input.data[src_idx];
          }
        }
      }
      
      heads.push_back(std::move(head_tensor));
    }
    
    return heads;
  }

  // 将缓存分割到多个头
  std::vector<Tensor> splitCacheToHeads(const Tensor& cache, uint32_t num_heads, uint32_t head_dim) {
    // 边界检查
    if (cache.shape.empty() || num_heads == 0 || head_dim == 0) {
      throw std::invalid_argument("Invalid parameters for splitCacheToHeads");
    }
    
    std::vector<Tensor> cache_heads;
    cache_heads.reserve(num_heads);
    
    uint32_t max_seq_len = cache.shape[cache.shape.size() - 2];
    uint32_t batch_size = (cache.shape.size() > 2) ? cache.shape[0] : 1;
    
    // 检查缓存数据大小
    size_t expected_size = batch_size * max_seq_len * num_heads * head_dim;
    if (cache.data.size() < expected_size) {
      throw std::runtime_error("Cache tensor data size insufficient for splitCacheToHeads");
    }
    
    for (uint32_t h = 0; h < num_heads; ++h) {
      std::vector<uint32_t> head_shape;
      if (batch_size > 1) {
        head_shape = {batch_size, max_seq_len, head_dim};
      } else {
        head_shape = {max_seq_len, head_dim};
      }
      
      Tensor head_cache(head_shape);
      
      // 复制对应头的缓存数据
      for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < max_seq_len; ++s) {
          for (uint32_t d = 0; d < head_dim; ++d) {
            size_t src_idx = b * max_seq_len * (num_heads * head_dim) + 
                           s * (num_heads * head_dim) + 
                           h * head_dim + d;
            size_t dst_idx = b * max_seq_len * head_dim + s * head_dim + d;
            
            if (batch_size == 1) {
              dst_idx = s * head_dim + d;
            }
            
            // 边界检查
            if (src_idx >= cache.data.size() || dst_idx >= head_cache.data.size()) {
              throw std::runtime_error("Array index out of bounds in splitCacheToHeads");
            }
            
            head_cache.data[dst_idx] = cache.data[src_idx];
          }
        }
      }
      
      cache_heads.push_back(std::move(head_cache));
    }
    
    return cache_heads;
  }

  // 合并所有头的输出
  Tensor concatenateHeads(const std::vector<Tensor>& head_outputs, 
                         uint32_t batch_size, uint32_t seq_len) {
    // 边界检查
    if (head_outputs.empty() || head_outputs.size() != num_heads_) {
      throw std::invalid_argument("Invalid head_outputs size for concatenateHeads");
    }
    
    std::vector<uint32_t> output_shape;
    if (batch_size > 1) {
      output_shape = {batch_size, seq_len, hidden_size_};
    } else {
      output_shape = {seq_len, hidden_size_};
    }
    
    Tensor output(output_shape);
    
    for (uint32_t h = 0; h < num_heads_; ++h) {
      const Tensor& head_output = head_outputs[h];
      
      // 检查头输出的形状
      if (head_output.shape.empty()) {
        throw std::runtime_error("Empty head output shape in concatenateHeads");
      }
      
      for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
          for (uint32_t d = 0; d < head_dim_; ++d) {
            size_t src_idx = b * seq_len * head_dim_ + s * head_dim_ + d;
            size_t dst_idx = b * seq_len * hidden_size_ + 
                           s * hidden_size_ + 
                           h * head_dim_ + d;
            
            if (batch_size == 1) {
              src_idx = s * head_dim_ + d;
              dst_idx = s * hidden_size_ + h * head_dim_ + d;
            }
            
            // 边界检查
            if (src_idx >= head_output.data.size() || dst_idx >= output.data.size()) {
              throw std::runtime_error("Array index out of bounds in concatenateHeads");
            }
            
            output.data[dst_idx] = head_output.data[src_idx];
          }
        }
      }
    }
    
    return output;
  }

  // 创建头级别的mask
  Tensor createHeadMask(const Tensor& mask, uint32_t head_idx, 
                       uint32_t seq_len_q, uint32_t seq_len_k) {
    // 简化实现：假设mask对所有头都相同
    return mask;
  }

  // 更新原始缓存
  // 将KV缓存调整为query维度
  Tensor adjustKVCacheForQuery(const Tensor& kv_cache, uint32_t target_head_dim) {
    if (kv_cache.shape.empty()) {
      throw std::invalid_argument("Empty KV cache tensor");
    }
    
    uint32_t seq_len = kv_cache.shape[kv_cache.shape.size() - 2];
    uint32_t current_head_dim = kv_cache.shape.back();
    
    // 如果维度已经匹配，直接返回
    if (current_head_dim == target_head_dim) {
      return kv_cache;
    }
    
    // 创建新的张量形状
    std::vector<uint32_t> new_shape = kv_cache.shape;
    new_shape.back() = target_head_dim;
    
    Tensor adjusted_cache(new_shape);
    
    // 调整数据：如果目标维度小于当前维度，截取；如果大于，填充零
    for (uint32_t s = 0; s < seq_len; ++s) {
      for (uint32_t d = 0; d < target_head_dim; ++d) {
        if (d < current_head_dim) {
          // 复制现有数据
          adjusted_cache.data[s * target_head_dim + d] = 
            kv_cache.data[s * current_head_dim + d];
        } else {
          // 填充零
          adjusted_cache.data[s * target_head_dim + d] = 0.0f;
        }
      }
    }
    
    return adjusted_cache;
  }

  void updateOriginalCache(const std::vector<Tensor>& key_cache_heads,
                           const std::vector<Tensor>& value_cache_heads,
                           Tensor& key_cache, Tensor& value_cache) {
    // 边界检查
    if (key_cache.shape.empty() || value_cache.shape.empty()) {
      throw std::invalid_argument("Empty cache shapes in updateOriginalCache");
    }
    
    if (key_cache_heads.size() != num_kv_heads_ || value_cache_heads.size() != num_kv_heads_) {
      throw std::invalid_argument("Invalid cache heads size in updateOriginalCache");
    }
    
    uint32_t max_seq_len = key_cache.shape[key_cache.shape.size() - 2];
    uint32_t batch_size = (key_cache.shape.size() > 2) ? key_cache.shape[0] : 1;
    
    // 将头级别的缓存合并回原始缓存
    for (uint32_t h = 0; h < num_kv_heads_; ++h) {
      const Tensor& key_head = key_cache_heads[h];
      const Tensor& value_head = value_cache_heads[h];
      
      // 检查头缓存的形状
      if (key_head.shape.empty() || value_head.shape.empty()) {
        throw std::runtime_error("Empty head cache shapes in updateOriginalCache");
      }
      
      for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < max_seq_len; ++s) {
          for (uint32_t d = 0; d < kv_head_dim_; ++d) {
            size_t src_idx = b * max_seq_len * kv_head_dim_ + s * kv_head_dim_ + d;
            size_t dst_idx = b * max_seq_len * (num_kv_heads_ * kv_head_dim_) + 
                           s * (num_kv_heads_ * kv_head_dim_) + 
                           h * kv_head_dim_ + d;
            
            if (batch_size == 1) {
              src_idx = s * kv_head_dim_ + d;
              dst_idx = s * (num_kv_heads_ * kv_head_dim_) + h * kv_head_dim_ + d;
            }
            
            // 边界检查
            if (src_idx >= key_head.data.size() || src_idx >= value_head.data.size() ||
                dst_idx >= key_cache.data.size() || dst_idx >= value_cache.data.size()) {
              throw std::runtime_error("Array index out of bounds in updateOriginalCache");
            }
            
            key_cache.data[dst_idx] = key_head.data[src_idx];
            value_cache.data[dst_idx] = value_head.data[src_idx];
          }
        }
      }
    }
  }
};

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // MULTI_HEAD_ATTENTION_H