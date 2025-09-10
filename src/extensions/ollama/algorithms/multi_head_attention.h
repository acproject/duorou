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
    kv_head_dim_ = head_dim_;  // KV头维度应该与普通头维度相同
    
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
    if (input.shape.empty() || input.data.empty()) {
      return false;
    }
    return true;
  }

  Tensor compute(const Tensor& query, const Tensor& key, const Tensor& value,
                const Tensor* mask = nullptr, float scale = 1.0f) override {
    try {
      // 验证输入
      if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
        throw std::invalid_argument("Invalid input tensors");
      }
      
      // 分割到多个头
      auto query_heads = splitToHeads(query, num_heads_, head_dim_);
      auto key_heads = splitToHeads(key, num_kv_heads_, kv_head_dim_);
      auto value_heads = splitToHeads(value, num_kv_heads_, kv_head_dim_);
      
      std::vector<Tensor> head_outputs;
      head_outputs.reserve(num_heads_);
      
      // 为每个头计算注意力
      for (uint32_t i = 0; i < num_heads_; ++i) {
        uint32_t kv_head_idx = i / group_size_;
        
        Tensor head_output = attention_heads_[i]->compute(
          query_heads[i], key_heads[kv_head_idx], value_heads[kv_head_idx], 
          mask, scale);
        
        head_outputs.push_back(std::move(head_output));
      }
      
      // 连接所有头的输出
      uint32_t batch_size = query.shape.size() > 2 ? query.shape[0] : 1;
      uint32_t seq_len = query.shape[query.shape.size() - 2];
      
      return concatenateHeads(head_outputs, batch_size, seq_len);
      
    } catch (const std::exception& e) {
      log("ERROR", "MultiHeadAttention compute failed: " + std::string(e.what()));
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
    
    uint32_t seq_len = input.shape[input.shape.size() - 2];
    uint32_t batch_size = (input.shape.size() > 2) ? input.shape[0] : 1;
    
    for (uint32_t h = 0; h < num_heads; ++h) {
      std::vector<uint32_t> head_shape;
      if (batch_size > 1) {
        head_shape = {batch_size, seq_len, head_dim};
      } else {
        head_shape = {seq_len, head_dim};
      }
      
      Tensor head_tensor(head_shape);
      
      for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
          for (uint32_t d = 0; d < head_dim; ++d) {
            size_t src_idx = b * seq_len * (num_heads * head_dim) + 
                           s * (num_heads * head_dim) + 
                           h * head_dim + d;
            size_t dst_idx = b * seq_len * head_dim + s * head_dim + d;
            
            if (batch_size == 1) {
              src_idx = s * (num_heads * head_dim) + h * head_dim + d;
              dst_idx = s * head_dim + d;
            }
            
            if (src_idx < input.data.size() && dst_idx < head_tensor.data.size()) {
              head_tensor.data[dst_idx] = input.data[src_idx];
            }
          }
        }
      }
      
      heads.push_back(std::move(head_tensor));
    }
    
    return heads;
  }

  Tensor concatenateHeads(const std::vector<Tensor>& head_outputs, 
                         uint32_t batch_size, uint32_t seq_len) {
    if (head_outputs.empty()) {
      throw std::invalid_argument("Empty head outputs");
    }
    
    std::vector<uint32_t> output_shape;
    if (batch_size > 1) {
      output_shape = {batch_size, seq_len, hidden_size_};
    } else {
      output_shape = {seq_len, hidden_size_};
    }
    
    Tensor output(output_shape);
    
    for (uint32_t h = 0; h < head_outputs.size(); ++h) {
      const Tensor& head_output = head_outputs[h];
      
      for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
          for (uint32_t d = 0; d < head_dim_; ++d) {
            size_t src_idx = b * seq_len * head_dim_ + s * head_dim_ + d;
            size_t dst_idx = b * seq_len * hidden_size_ + s * hidden_size_ + h * head_dim_ + d;
            
            if (batch_size == 1) {
              src_idx = s * head_dim_ + d;
              dst_idx = s * hidden_size_ + h * head_dim_ + d;
            }
            
            if (src_idx < head_output.data.size() && dst_idx < output.data.size()) {
              output.data[dst_idx] = head_output.data[src_idx];
            }
          }
        }
      }
    }
    
    return output;
  }

  void log(const std::string& level, const std::string& message) {
    if (context_.verbose) {
      std::cout << "[" << level << "] " << getName() << ": " << message << std::endl;
    }
  }
};

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // MULTI_HEAD_ATTENTION_H