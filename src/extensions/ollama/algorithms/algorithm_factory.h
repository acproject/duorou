#ifndef ALGORITHM_FACTORY_H
#define ALGORITHM_FACTORY_H

#include "base_algorithm.h"
#include "fast_attention.h"
#include "multi_head_attention.h"
#include "feed_forward.h"
#include "rope_processor.h"
#include <unordered_map>
#include <functional>
#include <vector>
#include <memory>
#include <string>

namespace duorou::extensions::ollama::algorithms {

// 注意力算法工厂
class AttentionAlgorithmFactory : public AlgorithmFactory<IAttentionAlgorithm> {
public:
  AttentionAlgorithmFactory() {
    registerAlgorithms();
  }

  std::unique_ptr<IAttentionAlgorithm> create(const std::string& algorithm_type) override {
    auto it = creators_.find(algorithm_type);
    if (it != creators_.end()) {
      return it->second();
    }
    return nullptr;
  }

  std::vector<std::string> getSupportedTypes() const override {
    std::vector<std::string> types;
    types.reserve(creators_.size());
    for (const auto& pair : creators_) {
      types.push_back(pair.first);
    }
    return types;
  }

private:
  std::unordered_map<std::string, std::function<std::unique_ptr<IAttentionAlgorithm>()>> creators_;

  void registerAlgorithms() {
    creators_["fast_attention"] = []() {
      return std::make_unique<FastAttention>();
    };
    
    creators_["multi_head_attention"] = []() {
      return std::make_unique<MultiHeadAttention>();
    };
    
    // 可以添加更多注意力算法
    creators_["standard_attention"] = []() {
      return std::make_unique<FastAttention>(); // 使用FastAttention作为标准实现
    };
  }
};

// 前馈网络算法工厂
class FeedForwardAlgorithmFactory : public AlgorithmFactory<IFeedForwardAlgorithm> {
public:
  FeedForwardAlgorithmFactory() {
    registerAlgorithms();
  }

  std::unique_ptr<IFeedForwardAlgorithm> create(const std::string& algorithm_type) override {
    auto it = creators_.find(algorithm_type);
    if (it != creators_.end()) {
      return it->second();
    }
    return nullptr;
  }

  std::vector<std::string> getSupportedTypes() const override {
    std::vector<std::string> types;
    types.reserve(creators_.size());
    for (const auto& pair : creators_) {
      types.push_back(pair.first);
    }
    return types;
  }

private:
  std::unordered_map<std::string, std::function<std::unique_ptr<IFeedForwardAlgorithm>()>> creators_;

  void registerAlgorithms() {
    creators_["swiglu"] = []() {
      return std::make_unique<SwiGLUFeedForward>();
    };
    
    creators_["gelu"] = []() {
      return std::make_unique<GELUFeedForward>();
    };
    
    creators_["standard"] = []() {
      return std::make_unique<FeedForward>();
    };
  }
};

// 位置编码算法工厂
class PositionalEncodingAlgorithmFactory : public AlgorithmFactory<IPositionalEncodingAlgorithm> {
public:
  PositionalEncodingAlgorithmFactory() {
    registerAlgorithms();
  }

  std::unique_ptr<IPositionalEncodingAlgorithm> create(const std::string& algorithm_type) override {
    auto it = creators_.find(algorithm_type);
    if (it != creators_.end()) {
      return it->second();
    }
    return nullptr;
  }

  std::vector<std::string> getSupportedTypes() const override {
    std::vector<std::string> types;
    types.reserve(creators_.size());
    for (const auto& pair : creators_) {
      types.push_back(pair.first);
    }
    return types;
  }

private:
  std::unordered_map<std::string, std::function<std::unique_ptr<IPositionalEncodingAlgorithm>()>> creators_;

  void registerAlgorithms() {
    creators_["rope"] = []() {
      return std::make_unique<RoPEProcessor>();
    };
    
    creators_["optimized_rope"] = []() {
      return std::make_unique<OptimizedRoPEProcessor>();
    };
    
    creators_["linear_rope"] = []() {
      return std::make_unique<ExtendedRoPEProcessor>(ExtendedRoPEProcessor::RoPEType::LINEAR);
    };
    
    creators_["dynamic_rope"] = []() {
      return std::make_unique<ExtendedRoPEProcessor>(ExtendedRoPEProcessor::RoPEType::DYNAMIC);
    };
    
    creators_["yarn_rope"] = []() {
      return std::make_unique<ExtendedRoPEProcessor>(ExtendedRoPEProcessor::RoPEType::YARN);
    };
  }
};

// 统一算法管理器
class AlgorithmManager {
public:
  static AlgorithmManager& getInstance() {
    static AlgorithmManager instance;
    return instance;
  }

  // 获取注意力算法
  std::unique_ptr<IAttentionAlgorithm> createAttentionAlgorithm(const std::string& type) {
    return attention_factory_.create(type);
  }

  // 获取前馈网络算法
  std::unique_ptr<IFeedForwardAlgorithm> createFeedForwardAlgorithm(const std::string& type) {
    return feedforward_factory_.create(type);
  }

  // 获取位置编码算法
  std::unique_ptr<IPositionalEncodingAlgorithm> createPositionalEncodingAlgorithm(const std::string& type) {
    return positional_factory_.create(type);
  }

  // 获取支持的算法类型
  std::vector<std::string> getSupportedAttentionTypes() const {
    return attention_factory_.getSupportedTypes();
  }

  std::vector<std::string> getSupportedFeedForwardTypes() const {
    return feedforward_factory_.getSupportedTypes();
  }

  std::vector<std::string> getSupportedPositionalEncodingTypes() const {
    return positional_factory_.getSupportedTypes();
  }

  // 注册自定义算法
  template<typename T>
  void registerAttentionAlgorithm(const std::string& name) {
    static_assert(std::is_base_of_v<IAttentionAlgorithm, T>, "T must inherit from IAttentionAlgorithm");
    // 这里可以添加动态注册的实现
  }

  template<typename T>
  void registerFeedForwardAlgorithm(const std::string& name) {
    static_assert(std::is_base_of_v<IFeedForwardAlgorithm, T>, "T must inherit from IFeedForwardAlgorithm");
    // 这里可以添加动态注册的实现
  }

  template<typename T>
  void registerPositionalEncodingAlgorithm(const std::string& name) {
    static_assert(std::is_base_of_v<IPositionalEncodingAlgorithm, T>, "T must inherit from IPositionalEncodingAlgorithm");
    // 这里可以添加动态注册的实现
  }

private:
  AttentionAlgorithmFactory attention_factory_;
  FeedForwardAlgorithmFactory feedforward_factory_;
  PositionalEncodingAlgorithmFactory positional_factory_;

  AlgorithmManager() = default;
  ~AlgorithmManager() = default;
  AlgorithmManager(const AlgorithmManager&) = delete;
  AlgorithmManager& operator=(const AlgorithmManager&) = delete;
};

// 便利函数
inline std::unique_ptr<IAttentionAlgorithm> createAttentionAlgorithm(const std::string& type) {
  return AlgorithmManager::getInstance().createAttentionAlgorithm(type);
}

inline std::unique_ptr<IFeedForwardAlgorithm> createFeedForwardAlgorithm(const std::string& type) {
  return AlgorithmManager::getInstance().createFeedForwardAlgorithm(type);
}

inline std::unique_ptr<IPositionalEncodingAlgorithm> createPositionalEncodingAlgorithm(const std::string& type) {
  return AlgorithmManager::getInstance().createPositionalEncodingAlgorithm(type);
}

// 算法配置结构
struct AlgorithmConfig {
  std::string attention_type = "multi_head_attention";
  std::string feedforward_type = "swiglu";
  std::string positional_encoding_type = "rope";
  
  bool use_optimized_attention = true;
  bool use_kv_cache = true;
  bool enable_parallel_heads = true;
  
  AlgorithmContext context;
  
  AlgorithmConfig() {
    context.verbose = false;
    context.num_threads = 1;
    context.use_simd = true;
    context.use_blas = false;
    context.device = "cpu";
  }
};

// 算法套件：包含完整的算法组合
class AlgorithmSuite {
public:
  AlgorithmSuite(const AlgorithmConfig& config = AlgorithmConfig()) : config_(config) {}

  bool initialize(const ModelConfig& model_config) {
    // 创建算法实例
    attention_ = createAttentionAlgorithm(config_.attention_type);
    feedforward_ = createFeedForwardAlgorithm(config_.feedforward_type);
    positional_encoding_ = createPositionalEncodingAlgorithm(config_.positional_encoding_type);
    
    if (!attention_ || !feedforward_ || !positional_encoding_) {
      return false;
    }
    
    // 初始化所有算法
    bool success = true;
    success &= attention_->initialize(model_config, config_.context);
    success &= feedforward_->initialize(model_config, config_.context);
    success &= positional_encoding_->initialize(model_config, config_.context);
    
    return success;
  }

  IAttentionAlgorithm* getAttentionAlgorithm() const {
    return attention_.get();
  }

  IFeedForwardAlgorithm* getFeedForwardAlgorithm() const {
    return feedforward_.get();
  }

  IPositionalEncodingAlgorithm* getPositionalEncodingAlgorithm() const {
    return positional_encoding_.get();
  }

  const AlgorithmConfig& getConfig() const {
    return config_;
  }

private:
  AlgorithmConfig config_;
  std::unique_ptr<IAttentionAlgorithm> attention_;
  std::unique_ptr<IFeedForwardAlgorithm> feedforward_;
  std::unique_ptr<IPositionalEncodingAlgorithm> positional_encoding_;
};

} // namespace duorou::extensions::ollama::algorithms

#endif // ALGORITHM_FACTORY_H