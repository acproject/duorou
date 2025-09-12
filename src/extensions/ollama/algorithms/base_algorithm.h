#ifndef BASE_ALGORITHM_H
#define BASE_ALGORITHM_H

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// 内存池类 - 减少动态内存分配开销
class MemoryPool {
public:
  static MemoryPool& getInstance() {
    static MemoryPool instance;
    return instance;
  }
  
  // 获取指定大小的内存块
  std::vector<float>* getBuffer(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 查找合适大小的缓存
    auto it = free_buffers_.find(size);
    if (it != free_buffers_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }
    
    // 创建新的缓存
    auto buffer = std::make_unique<std::vector<float>>();
    buffer->reserve(size);
    buffer->resize(size);
    
    auto* ptr = buffer.get();
    allocated_buffers_.push_back(std::move(buffer));
    return ptr;
  }
  
  // 归还内存块到池中
  void returnBuffer(std::vector<float>* buffer, size_t size) {
    if (!buffer) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 清理数据但保留容量
    buffer->clear();
    buffer->resize(size);
    
    // 限制每个大小的缓存数量，避免内存泄漏
    if (free_buffers_[size].size() < max_buffers_per_size_) {
      free_buffers_[size].push_back(buffer);
    }
  }
  
  // 清理内存池
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    free_buffers_.clear();
    allocated_buffers_.clear();
  }
  
  // 获取内存池统计信息
  size_t getTotalAllocatedBuffers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_buffers_.size();
  }
  
private:
  MemoryPool() = default;
  ~MemoryPool() = default;
  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;
  
  mutable std::mutex mutex_;
  std::unordered_map<size_t, std::vector<std::vector<float>*>> free_buffers_;
  std::vector<std::unique_ptr<std::vector<float>>> allocated_buffers_;
  static constexpr size_t max_buffers_per_size_ = 10; // 限制每个大小的缓存数量
};

// 模型配置结构体
struct ModelConfig {
  uint32_t vocab_size = 152064;
  uint32_t hidden_size = 3584;
  uint32_t num_layers = 28;
  uint32_t num_attention_heads = 28;
  uint32_t num_key_value_heads = 4;
  uint32_t intermediate_size = 18944;
  uint32_t max_position_embeddings = 131072; // 与Qwen2.5-VL配置保持一致
  float rope_theta = 1000000.0f;
  float layer_norm_eps = 1e-6f;
  float rms_norm_eps = 1e-6f;

  // RoPE相关配置
  uint32_t rope_dim = 128;
  float rope_base = 10000.0f;
  float rope_scale = 1.0f;
  uint32_t original_context_length = 32768;
};

// 张量结构（支持内存池优化）
struct Tensor {
  std::vector<float> data;
  std::vector<uint32_t> shape;
  size_t size;
  bool use_memory_pool = false;
  std::vector<float>* pooled_buffer = nullptr;

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
  
  // 使用内存池的构造函数
  Tensor(const std::vector<uint32_t> &s, bool use_pool) : shape(s), size(1), use_memory_pool(use_pool) {
    for (uint32_t dim : shape) {
      if (dim == 0) {
        throw std::invalid_argument("Tensor dimension cannot be zero");
      }
      if (size > SIZE_MAX / dim) {
        throw std::overflow_error("Tensor size overflow");
      }
      size *= dim;
    }
    
    if (use_memory_pool) {
      pooled_buffer = MemoryPool::getInstance().getBuffer(size);
      // 使用pooled_buffer的引用，避免数据复制
      data.clear();
      data.reserve(size);
      data.resize(size);
      // 将pooled_buffer的数据复制到data中
      if (pooled_buffer && pooled_buffer->size() >= size) {
        std::copy(pooled_buffer->begin(), pooled_buffer->begin() + size, data.begin());
      }
    } else {
      data.resize(size);
    }
  }
  
  // 析构函数 - 归还内存池缓存
  ~Tensor() {
    if (use_memory_pool && pooled_buffer) {
      // 将数据复制回pooled_buffer
      if (pooled_buffer->size() >= size) {
        std::copy(data.begin(), data.end(), pooled_buffer->begin());
      }
      MemoryPool::getInstance().returnBuffer(pooled_buffer, size);
    }
  }
  
  // 拷贝构造函数
  Tensor(const Tensor& other) : shape(other.shape), size(other.size), use_memory_pool(false) {
    data = other.data;
    // 不复制内存池相关信息，避免双重释放
  }
  
  // 赋值操作符
  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      // 先释放当前的内存池缓存
      if (use_memory_pool && pooled_buffer) {
        MemoryPool::getInstance().returnBuffer(pooled_buffer, size);
        pooled_buffer = nullptr;
      }
      
      shape = other.shape;
      size = other.size;
      data = other.data;
      use_memory_pool = false; // 赋值后不使用内存池
    }
    return *this;
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
  MemoryPool* memory_pool = nullptr; // 内存池指针

  // 性能统计
  mutable double total_time = 0.0;
  mutable uint64_t call_count = 0;
};

// 基础算法接口
class IAlgorithm {
public:
  virtual ~IAlgorithm() = default;

  // 初始化算法
  virtual bool initialize(const ModelConfig &config,
                          const AlgorithmContext &context) = 0;

  // 获取算法名称
  virtual std::string getName() const = 0;

  // 获取算法版本
  virtual std::string getVersion() const = 0;

  // 验证输入张量
  virtual bool validateInput(const Tensor &input) const = 0;

  // 获取性能统计
  virtual double getAverageTime() const {
    return context_.call_count > 0 ? context_.total_time / context_.call_count
                                   : 0.0;
  }

  virtual uint64_t getCallCount() const { return context_.call_count; }

  virtual void resetStatistics() {
    context_.total_time = 0.0;
    context_.call_count = 0;
  }

protected:
  AlgorithmContext context_;

  // 日志记录
  virtual void log(const std::string &level, const std::string &message) const {
    if (context_.verbose) {
      // 简单的日志输出，实际项目中可以集成更复杂的日志系统
      std::cout << "[" << level << "] " << getName() << ": " << message
                << std::endl;
    }
  }
};

// 注意力算法基类
class IAttentionAlgorithm : public IAlgorithm {
public:
  virtual ~IAttentionAlgorithm() = default;

  // 计算注意力
  virtual Tensor compute(const Tensor &query, const Tensor &key,
                         const Tensor &value, const Tensor *mask = nullptr,
                         float scale = 1.0f) = 0;

  // 支持KV缓存的注意力计算
  virtual Tensor computeWithCache(const Tensor &query, const Tensor &key,
                                  const Tensor &value, Tensor &key_cache,
                                  Tensor &value_cache, uint32_t cache_position,
                                  const Tensor *mask = nullptr,
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
  virtual Tensor compute(const Tensor &input, const Tensor &gate_weights,
                         const Tensor &up_weights,
                         const Tensor &down_weights) = 0;
};

// 位置编码算法基类
class IPositionalEncodingAlgorithm : public IAlgorithm {
public:
  virtual ~IPositionalEncodingAlgorithm() = default;

  // 应用位置编码
  virtual Tensor apply(const Tensor &input, uint32_t position_offset = 0) = 0;

  // 批量应用位置编码
  virtual void applyInPlace(Tensor &tensor, uint32_t position_offset = 0) = 0;
};

// 矩阵运算算法基类
class IMatrixAlgorithm : public IAlgorithm {
public:
  virtual ~IMatrixAlgorithm() = default;

  // 矩阵乘法
  virtual void multiply(const float *a, const float *b, float *c, size_t m,
                        size_t n, size_t k) = 0;

  // 向量运算
  virtual void vectorAdd(const float *a, const float *b, float *result,
                         size_t size) = 0;
  virtual void vectorMul(const float *a, const float *b, float *result,
                         size_t size) = 0;
};

// 算法工厂基类
template <typename T> class AlgorithmFactory {
public:
  virtual ~AlgorithmFactory() = default;
  virtual std::unique_ptr<T> create(const std::string &algorithm_type) = 0;
  virtual std::vector<std::string> getSupportedTypes() const = 0;
};

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // BASE_ALGORITHM_H