#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include "base_algorithm.h"
#include "fast_attention.h"
#include <vector>
#include <memory>
#include <chrono>
#include <stdexcept>
#include <string>

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
    if (num_kv_heads_ > 0) {
      group_size_ = num_heads_ / num_kv_heads_;
    } else {
      log("ERROR", "num_kv_heads_ cannot be zero");
      return false;
    }
    
    // 初始化注意力头
    attention_heads_.clear();
    kv_attention_heads_.clear();
    
    try {
      // 为每个查询头创建FastAttention实例
      for (uint32_t i = 0; i < num_heads_; ++i) {
        auto attention = std::make_unique<FastAttention>();
        if (!attention->initialize(config, context)) {
          log("ERROR", "Failed to initialize attention head " + std::to_string(i));
          return false;
        }
        attention_heads_.push_back(std::move(attention));
      }
      
      // 为每个键值头创建FastAttention实例
      for (uint32_t i = 0; i < num_kv_heads_; ++i) {
        auto kv_attention = std::make_unique<FastAttention>();
        if (!kv_attention->initialize(config, context)) {
          log("ERROR", "Failed to initialize KV attention head " + std::to_string(i));
          return false;
        }
        kv_attention_heads_.push_back(std::move(kv_attention));
      }
      
    } catch (const std::exception& e) {
      log("ERROR", "Exception during initialization: " + std::string(e.what()));
      return false;
    }
    
    log("INFO", "MultiHeadAttention initialized with " + std::to_string(num_heads_) + 
        " query heads and " + std::to_string(num_kv_heads_) + " key-value heads");
    
    return true;
  }

  std::string getName() const override {
    return "MultiHeadAttention";
  }

  std::string getVersion() const override {
    return "1.0.0";
  }

  bool validateInput(const Tensor& input) const override {
    if (input.data.empty() || input.shape.empty()) {
      return false;
    }
    return true;
  }

  Tensor compute(const Tensor& query, const Tensor& key, const Tensor& value,
                const Tensor* mask = nullptr, float scale = 1.0f) override {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
      // 输入验证
      if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
        throw std::invalid_argument("Invalid input tensors");
      }
      
      // 获取批次大小和序列长度
      uint32_t batch_size = query.shape[0];
      uint32_t seq_len = query.shape[1];
      
      // 分割查询、键、值到多个头
      auto query_heads = splitToHeads(query, num_heads_, head_dim_);
      auto key_heads = splitToHeads(key, num_kv_heads_, kv_head_dim_);
      auto value_heads = splitToHeads(value, num_kv_heads_, kv_head_dim_);
      
      std::vector<Tensor> head_outputs;
      head_outputs.reserve(num_heads_);
      
      // 为每个查询头计算注意力（GQA：每个查询头对应一个键值头）
      for (uint32_t i = 0; i < num_heads_; ++i) {
        uint32_t kv_head_idx = i / group_size_; // 计算对应的键值头索引
        
        if (kv_head_idx >= num_kv_heads_) {
          throw std::runtime_error("KV head index out of range");
        }
        
        // 使用对应的FastAttention实例计算注意力
        Tensor head_output = attention_heads_[i]->compute(
          query_heads[i], key_heads[kv_head_idx], value_heads[kv_head_idx], mask, scale
        );
        
        head_outputs.push_back(std::move(head_output));
      }
      
      // 连接所有头的输出
      Tensor result = concatenateHeads(head_outputs, batch_size, seq_len);
      
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
      context_.total_time += duration.count() / 1000.0; // 转换为毫秒
      context_.call_count++;
      
      return result;
      
    } catch (const std::exception& e) {
      log("ERROR", "MultiHeadAttention compute failed: " + std::string(e.what()));
      throw;
    }
  }

  Tensor computeWithCache(const Tensor& query, const Tensor& key, const Tensor& value,
                         Tensor& key_cache, Tensor& value_cache,
                         uint32_t cache_position, const Tensor* mask = nullptr,
                         float scale = 1.0f) override {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
      // 输入验证
      if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
        throw std::invalid_argument("Invalid input tensors");
      }
      
      // 获取批次大小和序列长度
      uint32_t batch_size = query.shape[0];
      uint32_t seq_len = query.shape[1];
      
      // 分割查询、键、值到多个头
      auto query_heads = splitToHeads(query, num_heads_, head_dim_);
      auto key_heads = splitToHeads(key, num_kv_heads_, kv_head_dim_);
      auto value_heads = splitToHeads(value, num_kv_heads_, kv_head_dim_);
      
      std::vector<Tensor> head_outputs;
      head_outputs.reserve(num_heads_);
      
      // 为每个查询头计算带缓存的注意力
      for (uint32_t i = 0; i < num_heads_; ++i) {
        uint32_t kv_head_idx = i / group_size_;
        
        if (kv_head_idx >= num_kv_heads_) {
          throw std::runtime_error("KV head index out of range");
        }
        
        // 使用对应的FastAttention实例计算带缓存的注意力
        Tensor head_output = attention_heads_[i]->computeWithCache(
          query_heads[i], key_heads[kv_head_idx], value_heads[kv_head_idx],
          key_cache, value_cache, cache_position, mask, scale
        );
        
        head_outputs.push_back(std::move(head_output));
      }
      
      // 连接所有头的输出
      Tensor result = concatenateHeads(head_outputs, batch_size, seq_len);
      
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
      context_.total_time += duration.count() / 1000.0;
      context_.call_count++;
      
      return result;
      
    } catch (const std::exception& e) {
      log("ERROR", "MultiHeadAttention computeWithCache failed: " + std::string(e.what()));
      throw;
    }
  }

private:
  uint32_t hidden_size_ = 3584;
  uint32_t num_heads_ = 28;
  uint32_t num_kv_heads_ = 4;
  uint32_t head_dim_ = 128;
  uint32_t kv_head_dim_ = 128;
  uint32_t group_size_ = 7;
  
  std::vector<std::unique_ptr<FastAttention>> attention_heads_;
  std::vector<std::unique_ptr<FastAttention>> kv_attention_heads_;
  AlgorithmContext context_;

  std::vector<Tensor> splitToHeads(const Tensor& input, uint32_t num_heads, uint32_t head_dim) {
    std::vector<Tensor> heads;
    heads.reserve(num_heads);
    
    if (input.shape.size() < 3) {
      throw std::invalid_argument("Input tensor must have at least 3 dimensions");
    }
    
    uint32_t batch_size = input.shape[0];
    uint32_t seq_len = input.shape[1];
    uint32_t hidden_size = input.shape[2];
    
    if (hidden_size != num_heads * head_dim) {
      throw std::invalid_argument("Hidden size mismatch: expected " + 
                                std::to_string(num_heads * head_dim) + 
                                ", got " + std::to_string(hidden_size));
    }
    
    for (uint32_t h = 0; h < num_heads; ++h) {
      Tensor head({batch_size, seq_len, head_dim});
      
      for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
          for (uint32_t d = 0; d < head_dim; ++d) {
            uint32_t input_idx = b * seq_len * hidden_size + s * hidden_size + h * head_dim + d;
            uint32_t head_idx = b * seq_len * head_dim + s * head_dim + d;
            
            if (input_idx < input.data.size() && head_idx < head.data.size()) {
              head.data[head_idx] = input.data[input_idx];
            }
          }
        }
      }
      
      heads.push_back(std::move(head));
    }
    
    return heads;
  }

  Tensor concatenateHeads(const std::vector<Tensor>& head_outputs, 
                         uint32_t batch_size, uint32_t seq_len) {
    if (head_outputs.empty()) {
      throw std::invalid_argument("No head outputs to concatenate");
    }
    
    uint32_t total_hidden_size = num_heads_ * head_dim_;
    Tensor result({batch_size, seq_len, total_hidden_size});
    
    for (uint32_t h = 0; h < num_heads_; ++h) {
      if (h >= head_outputs.size()) {
        throw std::runtime_error("Head index out of range");
      }
      
      const Tensor& head_output = head_outputs[h];
      
      if (head_output.shape.size() < 3 || 
          head_output.shape[0] != batch_size || 
          head_output.shape[1] != seq_len || 
          head_output.shape[2] != head_dim_) {
        throw std::invalid_argument("Head output shape mismatch");
      }
      
      for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
          for (uint32_t d = 0; d < head_dim_; ++d) {
            uint32_t head_idx = b * seq_len * head_dim_ + s * head_dim_ + d;
            uint32_t result_idx = b * seq_len * total_hidden_size + s * total_hidden_size + h * head_dim_ + d;
            
            if (head_idx < head_output.data.size() && result_idx < result.data.size()) {
              result.data[result_idx] = head_output.data[head_idx];
            }
          }
        }
      }
    }
    
    return result;
  }

  void log(const std::string& level, const std::string& message) {
    if (context_.verbose) {
      std::cout << "[" << level << "] MultiHeadAttention: " << message << std::endl;
    }
  }
};

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // MULTI_HEAD_ATTENTION_H