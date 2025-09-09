#ifndef BASE_ALGORITHM_H
#define BASE_ALGORITHM_H

#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// ModelConfig定义
struct ModelConfig {
    uint32_t vocab_size = 152064;
    uint32_t hidden_size = 3584;
    uint32_t num_layers = 28;
    uint32_t num_attention_heads = 28;
    uint32_t num_key_value_heads = 4;
    uint32_t intermediate_size = 18944;
    uint32_t max_position_embeddings = 32768;
    float rope_theta = 1000000.0f;
    float layer_norm_eps = 1e-6f;
    float rms_norm_eps = 1e-6f;
    
    // RoPE相关配置
    uint32_t rope_dim = 128;
    float rope_base = 10000.0f;
    float rope_scale = 1.0f;
    uint32_t original_context_length = 32768;
};

// 张量结构（与主引擎保持一致）
struct Tensor {
  std::vector<float> data;
  std::vector<uint32_t> shape;
  size_t size;

  Tensor() : size(0) {}

  Tensor(const std::vector<uint32_t> &s) : shape(s), size(1) {
    for (uint32_t dim : shape) {
      if (dim == 0) {
        throw std::invalid_argument("Tensor dimension cannot be zero");
      }
      if (size > SIZE_MAX / dim) {
        throw std::overflow_error("Tensor size overflow");
      }
      size *= dim;
    }
    data.resize(size);
  }

  void reshape(const std::vector<uint32_t> &new_shape) {
    size_t new_size = 1;
    for (uint32_t dim : new_shape) {
      if (dim == 0) {
        throw std::invalid_argument("Tensor dimension cannot be zero");
      }
      if (new_size > SIZE_MAX / dim) {
        throw std::overflow_error("Tensor size overflow");
      }
      new_size *= dim;
    }
    if (new_size != size) {
      data.resize(new_size);
      size = new_size;
    }
    shape = new_shape;
  }
};

// 算法执行上下文
struct AlgorithmContext {
  bool verbose = false;
  uint32_t num_threads = 1;
  bool use_simd = true;
  bool use_blas = false;
  std::string device = "cpu";
  
  // 性能统计
  mutable double total_time = 0.0;
  mutable uint64_t call_count = 0;
};

// 基础算法接口
class IAlgorithm {
public:
  virtual ~IAlgorithm() = default;
  
  // 初始化算法
  virtual bool initialize(const ModelConfig& config, const AlgorithmContext& context) = 0;
  
  // 获取算法名称
  virtual std::string getName() const = 0;
  
  // 获取算法版本
  virtual std::string getVersion() const = 0;
  
  // 验证输入张量
  virtual bool validateInput(const Tensor& input) const = 0;
  
  // 获取性能统计
  virtual double getAverageTime() const {
    return context_.call_count > 0 ? context_.total_time / context_.call_count : 0.0;
  }
  
  virtual uint64_t getCallCount() const {
    return context_.call_count;
  }
  
  virtual void resetStatistics() {
    context_.total_time = 0.0;
    context_.call_count = 0;
  }

protected:
  AlgorithmContext context_;
  
  // 日志记录
  virtual void log(const std::string& level, const std::string& message) const {
    if (context_.verbose) {
      // 简单的日志输出，实际项目中可以集成更复杂的日志系统
      std::cout << "[" << level << "] " << getName() << ": " << message << std::endl;
    }
  }
};

// 注意力算法基类
class IAttentionAlgorithm : public IAlgorithm {
public:
  virtual ~IAttentionAlgorithm() = default;
  
  // 计算注意力
  virtual Tensor compute(const Tensor& query, const Tensor& key, const Tensor& value,
                        const Tensor* mask = nullptr, float scale = 1.0f) = 0;
  
  // 支持KV缓存的注意力计算
  virtual Tensor computeWithCache(const Tensor& query, const Tensor& key, const Tensor& value,
                                 Tensor& key_cache, Tensor& value_cache,
                                 uint32_t cache_position, const Tensor* mask = nullptr,
                                 float scale = 1.0f) {
    // 默认实现：不使用缓存
    return compute(query, key, value, mask, scale);
  }
};

// 前馈网络算法基类
class IFeedForwardAlgorithm : public IAlgorithm {
public:
  virtual ~IFeedForwardAlgorithm() = default;
  
  // 前馈网络计算
  virtual Tensor compute(const Tensor& input, const Tensor& gate_weights,
                        const Tensor& up_weights, const Tensor& down_weights) = 0;
};

// 位置编码算法基类
class IPositionalEncodingAlgorithm : public IAlgorithm {
public:
  virtual ~IPositionalEncodingAlgorithm() = default;
  
  // 应用位置编码
  virtual Tensor apply(const Tensor& input, uint32_t position_offset = 0) = 0;
  
  // 批量应用位置编码
  virtual void applyInPlace(Tensor& tensor, uint32_t position_offset = 0) = 0;
};

// 矩阵运算算法基类
class IMatrixAlgorithm : public IAlgorithm {
public:
  virtual ~IMatrixAlgorithm() = default;
  
  // 矩阵乘法
  virtual void multiply(const float* a, const float* b, float* c,
                       size_t m, size_t n, size_t k) = 0;
  
  // 向量运算
  virtual void vectorAdd(const float* a, const float* b, float* result, size_t size) = 0;
  virtual void vectorMul(const float* a, const float* b, float* result, size_t size) = 0;
};

// 算法工厂基类
template<typename T>
class AlgorithmFactory {
public:
  virtual ~AlgorithmFactory() = default;
  virtual std::unique_ptr<T> create(const std::string& algorithm_type) = 0;
  virtual std::vector<std::string> getSupportedTypes() const = 0;
};

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // BASE_ALGORITHM_H