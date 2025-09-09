#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include "base_algorithm.h"
#include <cmath>
#include <chrono>

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// 前馈网络算法实现
class FeedForward : public IFeedForwardAlgorithm {
public:
  FeedForward() = default;
  ~FeedForward() override = default;

  bool initialize(const ModelConfig& config, const AlgorithmContext& context) override {
    context_ = context;
    
    // 从配置中获取参数
    hidden_size_ = config.hidden_size;
    intermediate_size_ = config.intermediate_size;
    
    log("INFO", "FeedForward initialized with hidden_size=" + std::to_string(hidden_size_) + 
        ", intermediate_size=" + std::to_string(intermediate_size_));
    
    return true;
  }

  std::string getName() const override {
    return "FeedForward";
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

  Tensor compute(const Tensor& input, const Tensor& gate_weights,
                const Tensor& up_weights, const Tensor& down_weights) override {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 验证输入
    if (!validateInput(input)) {
      throw std::invalid_argument("Invalid input tensor for FeedForward");
    }
    
    // 验证权重张量
    if (!validateWeights(gate_weights, up_weights, down_weights)) {
      throw std::invalid_argument("Invalid weight tensors for FeedForward");
    }
    
    // 获取序列长度和批次大小
    uint32_t seq_len = input.shape[input.shape.size() - 2];
    uint32_t batch_size = (input.shape.size() > 2) ? input.shape[0] : 1;
    
    // 创建中间张量
    std::vector<uint32_t> intermediate_shape;
    if (batch_size > 1) {
      intermediate_shape = {batch_size, seq_len, intermediate_size_};
    } else {
      intermediate_shape = {seq_len, intermediate_size_};
    }
    
    Tensor gate_output(intermediate_shape);
    Tensor up_output(intermediate_shape);
    
    // 计算门控投影和上投影
    computeLinearProjection(input, gate_weights, gate_output);
    computeLinearProjection(input, up_weights, up_output);
    
    // 应用SwiGLU激活函数
    applySwiGLU(gate_output, up_output);
    
    // 创建输出张量
    std::vector<uint32_t> output_shape = input.shape;
    Tensor output(output_shape);
    
    // 计算下投影
    computeLinearProjection(gate_output, down_weights, output);
    
    // 更新统计信息
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    context_.total_time += duration.count() / 1000.0;
    context_.call_count++;
    
    return output;
  }

private:
  uint32_t hidden_size_ = 3584;
  uint32_t intermediate_size_ = 18944;

  bool validateWeights(const Tensor& gate_weights, const Tensor& up_weights, 
                      const Tensor& down_weights) const {
    // 验证门控权重：[hidden_size, intermediate_size]
    if (gate_weights.shape.size() != 2 || 
        gate_weights.shape[0] != hidden_size_ || 
        gate_weights.shape[1] != intermediate_size_) {
      return false;
    }
    
    // 验证上投影权重：[hidden_size, intermediate_size]
    if (up_weights.shape.size() != 2 || 
        up_weights.shape[0] != hidden_size_ || 
        up_weights.shape[1] != intermediate_size_) {
      return false;
    }
    
    // 验证下投影权重：[intermediate_size, hidden_size]
    if (down_weights.shape.size() != 2 || 
        down_weights.shape[0] != intermediate_size_ || 
        down_weights.shape[1] != hidden_size_) {
      return false;
    }
    
    return true;
  }

  void computeLinearProjection(const Tensor& input, const Tensor& weights, Tensor& output) {
    // 边界检查
    if (input.shape.empty() || weights.shape.empty() || output.shape.empty()) {
      throw std::invalid_argument("Empty tensor shapes in computeLinearProjection");
    }
    
    uint32_t seq_len = input.shape[input.shape.size() - 2];
    uint32_t batch_size = (input.shape.size() > 2) ? input.shape[0] : 1;
    uint32_t input_dim = input.shape.back();
    uint32_t output_dim = output.shape.back();
    
    // 检查数据大小
    size_t expected_input_size = batch_size * seq_len * input_dim;
    size_t expected_weight_size = input_dim * output_dim;
    size_t expected_output_size = batch_size * seq_len * output_dim;
    
    if (input.data.size() < expected_input_size ||
        weights.data.size() < expected_weight_size ||
        output.data.size() < expected_output_size) {
      throw std::runtime_error("Insufficient tensor data size in computeLinearProjection");
    }
    
    // 执行矩阵乘法：input @ weights
    for (uint32_t b = 0; b < batch_size; ++b) {
      for (uint32_t s = 0; s < seq_len; ++s) {
        for (uint32_t o = 0; o < output_dim; ++o) {
          float sum = 0.0f;
          
          for (uint32_t i = 0; i < input_dim; ++i) {
            size_t input_idx = b * seq_len * input_dim + s * input_dim + i;
            size_t weight_idx = i * output_dim + o;
            
            if (batch_size == 1) {
              input_idx = s * input_dim + i;
            }
            
            // 边界检查
            if (input_idx >= input.data.size() || weight_idx >= weights.data.size()) {
              throw std::runtime_error("Array index out of bounds in computeLinearProjection");
            }
            
            sum += input.data[input_idx] * weights.data[weight_idx];
          }
          
          size_t output_idx = b * seq_len * output_dim + s * output_dim + o;
          if (batch_size == 1) {
            output_idx = s * output_dim + o;
          }
          
          // 边界检查
          if (output_idx >= output.data.size()) {
            throw std::runtime_error("Array index out of bounds in computeLinearProjection output");
          }
          
          output.data[output_idx] = sum;
        }
      }
    }
  }

  void applySwiGLU(Tensor& gate_output, const Tensor& up_output) {
    // SwiGLU: gate_output = silu(gate_output) * up_output
    // 其中 silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    // 边界检查
    if (gate_output.data.size() != up_output.data.size()) {
      throw std::invalid_argument("Mismatched tensor sizes in applySwiGLU");
    }
    
    size_t tensor_size = std::min(gate_output.size, gate_output.data.size());
    tensor_size = std::min(tensor_size, up_output.data.size());
    
    for (size_t i = 0; i < tensor_size; ++i) {
      // 边界检查
      if (i >= gate_output.data.size() || i >= up_output.data.size()) {
        throw std::runtime_error("Array index out of bounds in applySwiGLU");
      }
      
      float x = gate_output.data[i];
      
      // 计算SiLU激活函数
      float silu_x;
      if (x > 0) {
        // 数值稳定的计算方式
        float exp_neg_x = std::exp(-x);
        silu_x = x / (1.0f + exp_neg_x);
      } else {
        // 当x <= 0时的数值稳定计算
        float exp_x = std::exp(x);
        silu_x = x * exp_x / (1.0f + exp_x);
      }
      
      // 应用门控机制
      gate_output.data[i] = silu_x * up_output.data[i];
    }
  }

  // SIMD优化版本的线性投影（可选）
  void computeLinearProjectionSIMD(const Tensor& input, const Tensor& weights, Tensor& output) {
    if (!context_.use_simd) {
      computeLinearProjection(input, weights, output);
      return;
    }
    
    // 这里可以添加SIMD优化的实现
    // 目前回退到标准实现
    computeLinearProjection(input, weights, output);
  }

  // BLAS优化版本的线性投影（可选）
  void computeLinearProjectionBLAS(const Tensor& input, const Tensor& weights, Tensor& output) {
    if (!context_.use_blas) {
      computeLinearProjection(input, weights, output);
      return;
    }
    
    // 这里可以添加BLAS优化的实现
    // 目前回退到标准实现
    computeLinearProjection(input, weights, output);
  }
};

// 专门的SwiGLU前馈网络实现
class SwiGLUFeedForward : public FeedForward {
public:
  std::string getName() const override {
    return "SwiGLUFeedForward";
  }
  
  // 可以在这里添加SwiGLU特定的优化
};

// 专门的GELU前馈网络实现
class GELUFeedForward : public IFeedForwardAlgorithm {
public:
  GELUFeedForward() = default;
  ~GELUFeedForward() override = default;

  bool initialize(const ModelConfig& config, const AlgorithmContext& context) override {
    context_ = context;
    hidden_size_ = config.hidden_size;
    intermediate_size_ = config.intermediate_size;
    
    log("INFO", "GELUFeedForward initialized");
    return true;
  }

  std::string getName() const override {
    return "GELUFeedForward";
  }

  std::string getVersion() const override {
    return "1.0.0";
  }

  bool validateInput(const Tensor& input) const override {
    return input.shape.size() >= 2 && input.shape.back() == hidden_size_;
  }

  Tensor compute(const Tensor& input, const Tensor& gate_weights,
                const Tensor& up_weights, const Tensor& down_weights) override {
    // GELU前馈网络的实现
    // 这里可以实现GELU激活函数的版本
    
    // 目前简化实现，实际项目中可以完整实现
    throw std::runtime_error("GELUFeedForward not fully implemented");
  }

private:
  uint32_t hidden_size_ = 3584;
  uint32_t intermediate_size_ = 18944;
  
  float gelu(float x) {
    // GELU激活函数：0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + std::tanh(inner));
  }
};

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // FEED_FORWARD_H