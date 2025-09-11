#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include "base_algorithm.h"
#include "fast_attention.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// 多头注意力算法实现
class MultiHeadAttention : public IAttentionAlgorithm {
public:
  MultiHeadAttention() = default;
  ~MultiHeadAttention() override = default;

  bool initialize(const ModelConfig &config,
                  const AlgorithmContext &context) override {
    context_ = context;

    // 从配置中获取参数
    hidden_size_ = config.hidden_size;
    num_heads_ = config.num_attention_heads;
    num_kv_heads_ = config.num_key_value_heads;
    head_dim_ = hidden_size_ / num_heads_;
    kv_head_dim_ = head_dim_; // In GQA, KV heads have same dimension as query heads

    // 验证配置参数
    if (hidden_size_ == 0 || num_heads_ == 0 || num_kv_heads_ == 0) {
      log("ERROR", "Invalid configuration: zero dimensions");
      return false;
    }

    if (hidden_size_ % num_heads_ != 0) {
      log("ERROR", "Hidden size must be divisible by number of heads");
      return false;
    }

    if (num_heads_ % num_kv_heads_ != 0) {
      log("ERROR", "Number of heads must be divisible by number of KV heads");
      return false;
    }

    group_size_ = num_heads_ / num_kv_heads_;

    // 初始化注意力头
    attention_heads_.clear();
    kv_attention_heads_.clear();

    // 为每个query头创建注意力算法实例
    for (uint32_t i = 0; i < num_heads_; ++i) {
      auto attention = std::make_unique<FastAttention>();
      if (!attention->initialize(config, context)) {
        log("ERROR",
            "Failed to initialize attention head " + std::to_string(i));
        return false;
      }
      attention_heads_.push_back(std::move(attention));
    }

    // 为每个key-value头创建注意力算法实例
    for (uint32_t i = 0; i < num_kv_heads_; ++i) {
      auto kv_attention = std::make_unique<FastAttention>();
      if (!kv_attention->initialize(config, context)) {
        log("ERROR",
            "Failed to initialize KV attention head " + std::to_string(i));
        return false;
      }
      kv_attention_heads_.push_back(std::move(kv_attention));
    }

    log("INFO", "MultiHeadAttention initialized with " +
                    std::to_string(num_heads_) + " heads, " +
                    std::to_string(num_kv_heads_) +
                    " KV heads, head_dim=" + std::to_string(head_dim_));

    return true;
  }

  std::string getName() const override { return "MultiHeadAttention"; }

  std::string getVersion() const override { return "1.0.0"; }

  bool validateInput(const Tensor &input) const override {
    return !input.data.empty() && !input.shape.empty();
  }

  Tensor compute(const Tensor &query, const Tensor &key, const Tensor &value,
                 const Tensor *mask = nullptr, float scale = 1.0f) override {
    auto start_time = std::chrono::high_resolution_clock::now();

    // 验证输入
    if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
      throw std::invalid_argument(
          "Invalid input tensors for MultiHeadAttention");
    }

    // 检查张量维度兼容性
    if (query.shape.size() < 2 || key.shape.size() < 2 ||
        value.shape.size() < 2) {
      throw std::invalid_argument("Tensors must have at least 2 dimensions");
    }

    // 确保输入张量至少有3个维度用于头分割
    // 创建深拷贝以避免修改原始张量
    Tensor query_3d;
    query_3d.data = query.data;
    query_3d.shape = query.shape;
    query_3d.size = query.size;

    Tensor key_3d;
    key_3d.data = key.data;
    key_3d.shape = key.shape;
    key_3d.size = key.size;

    Tensor value_3d;
    value_3d.data = value.data;
    value_3d.shape = value.shape;
    value_3d.size = value.size;

    // 如果输入是2D张量，添加batch维度
    if (query.shape.size() == 2) {
      query_3d.shape.insert(query_3d.shape.begin(), 1);
      // 重新计算size以保持一致性
      query_3d.size = 1;
      for (auto dim : query_3d.shape) {
        query_3d.size *= dim;
      }
    }
    if (key.shape.size() == 2) {
      key_3d.shape.insert(key_3d.shape.begin(), 1);
      key_3d.size = 1;
      for (auto dim : key_3d.shape) {
        key_3d.size *= dim;
      }
    }
    if (value.shape.size() == 2) {
      value_3d.shape.insert(value_3d.shape.begin(), 1);
      value_3d.size = 1;
      for (auto dim : value_3d.shape) {
        value_3d.size *= dim;
      }
    }

    // 获取批次大小和序列长度
    uint32_t batch_size = query_3d.shape[0];
    uint32_t seq_len_q = query_3d.shape[1];
    uint32_t seq_len_k = key_3d.shape[1];

    // 调试：打印传入splitToHeads的张量形状
    std::cerr
        << "[DEBUG] About to call splitToHeads for query_3d with shape: [";
    for (size_t i = 0; i < query_3d.shape.size(); ++i) {
      std::cerr << query_3d.shape[i];
      if (i < query_3d.shape.size() - 1)
        std::cerr << ", ";
    }
    std::cerr << "], num_heads=" << num_heads_ << ", head_dim=" << head_dim_
              << std::endl;

    // 分割为多个头
    auto query_heads = splitToHeads(query_3d, num_heads_, head_dim_);
    // 对于K和V，使用实际的维度而不是预期的维度
    uint32_t actual_kv_dim = key_3d.shape[2] / num_kv_heads_;
    auto key_heads = splitToHeads(key_3d, num_kv_heads_, actual_kv_dim);
    auto value_heads = splitToHeads(value_3d, num_kv_heads_, actual_kv_dim);

    // 对每个头执行注意力计算
    std::vector<Tensor> head_outputs;
    head_outputs.reserve(num_heads_);

    std::cerr << "[DEBUG] MultiHeadAttention: Processing " << num_heads_ << " attention heads" << std::endl;
    
    // 使用OpenMP并行化多头注意力计算
#ifdef _OPENMP
    // 设置线程数，但不超过头数和可用线程数
    int num_threads = std::min(static_cast<int>(num_heads_), omp_get_max_threads());
    if (context_.num_threads > 0) {
      num_threads = std::min(num_threads, static_cast<int>(context_.num_threads));
    }
    omp_set_num_threads(num_threads);
    
    std::cerr << "[DEBUG] Using OpenMP with " << num_threads << " threads for " << num_heads_ << " heads" << std::endl;
    
    #pragma omp parallel for schedule(dynamic)
#endif
    for (uint32_t i = 0; i < num_heads_; ++i) {
#ifdef _OPENMP
      int thread_id = omp_get_thread_num();
      std::cerr << "[DEBUG] Thread " << thread_id << " processing attention head " << i << "/" << num_heads_ << std::endl;
#else
      std::cerr << "[DEBUG] Processing attention head " << i << "/" << num_heads_ << std::endl;
#endif
      // 使用分组查询注意力：多个query头共享同一个key-value头
      uint32_t kv_head_idx = i / group_size_;

      Tensor head_output =
          attention_heads_[i]->compute(query_heads[i], key_heads[kv_head_idx],
                                       value_heads[kv_head_idx], mask, scale);

#ifdef _OPENMP
      #pragma omp critical
#endif
      {
        head_outputs.push_back(std::move(head_output));
      }
#ifdef _OPENMP
      std::cerr << "[DEBUG] Thread " << thread_id << " completed attention head " << i << std::endl;
#else
      std::cerr << "[DEBUG] Completed attention head " << i << std::endl;
#endif
    }
    std::cerr << "[DEBUG] MultiHeadAttention: All heads processed" << std::endl;

    // 连接所有头的输出
    Tensor result = concatenateHeads(head_outputs, batch_size, seq_len_q);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    context_.total_time += duration.count() / 1000.0;
    context_.call_count++;

    return result;
  }

  Tensor computeWithCache(const Tensor &query, const Tensor &key,
                          const Tensor &value, Tensor &key_cache,
                          Tensor &value_cache, uint32_t cache_position,
                          const Tensor *mask = nullptr,
                          float scale = 1.0f) override {
    auto start_time = std::chrono::high_resolution_clock::now();

    // 验证输入
    if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
      throw std::invalid_argument(
          "Invalid input tensors for MultiHeadAttention");
    }

    // 确保输入张量至少有3个维度用于头分割
    // 创建深拷贝以避免修改原始张量
    Tensor query_3d;
    query_3d.data = query.data;
    query_3d.shape = query.shape;
    query_3d.size = query.size;

    Tensor key_3d;
    key_3d.data = key.data;
    key_3d.shape = key.shape;
    key_3d.size = key.size;

    Tensor value_3d;
    value_3d.data = value.data;
    value_3d.shape = value.shape;
    value_3d.size = value.size;

    // 如果输入是2D张量，添加batch维度
    if (query.shape.size() == 2) {
      query_3d.shape.insert(query_3d.shape.begin(), 1);
      query_3d.size = 1;
      for (auto dim : query_3d.shape) {
        query_3d.size *= dim;
      }
    }
    if (key.shape.size() == 2) {
      key_3d.shape.insert(key_3d.shape.begin(), 1);
      key_3d.size = 1;
      for (auto dim : key_3d.shape) {
        key_3d.size *= dim;
      }
    }
    if (value.shape.size() == 2) {
      value_3d.shape.insert(value_3d.shape.begin(), 1);
      value_3d.size = 1;
      for (auto dim : value_3d.shape) {
        value_3d.size *= dim;
      }
    }

    // 获取批次大小和序列长度
    uint32_t batch_size = query_3d.shape[0];
    uint32_t seq_len_q = query_3d.shape[1];

    // 分割为多个头
    auto query_heads = splitToHeads(query_3d, num_heads_, head_dim_);
    // 对于K和V，使用实际的维度而不是预期的维度
    uint32_t actual_kv_dim =
        key_3d.shape[key_3d.shape.size() - 1] / num_kv_heads_;
    auto key_heads = splitToHeads(key_3d, num_kv_heads_, actual_kv_dim);
    auto value_heads = splitToHeads(value_3d, num_kv_heads_, actual_kv_dim);

    // 分割缓存（现在KV缓存已经是3维格式）
    // KV缓存使用正确的kv_head_dim_而不是actual_kv_dim
    uint32_t cache_kv_dim = key_cache.shape[key_cache.shape.size() - 1] / num_kv_heads_;
    auto key_cache_heads =
        splitToHeads(key_cache, num_kv_heads_, cache_kv_dim);
    auto value_cache_heads =
        splitToHeads(value_cache, num_kv_heads_, cache_kv_dim);

    // 对每个头执行注意力计算
    std::vector<Tensor> head_outputs;
    head_outputs.reserve(num_heads_);

    // 使用OpenMP并行化多头注意力计算（带缓存）
#ifdef _OPENMP
    // 设置线程数，但不超过头数和可用线程数
    int num_threads = std::min(static_cast<int>(num_heads_), omp_get_max_threads());
    if (context_.num_threads > 0) {
      num_threads = std::min(num_threads, static_cast<int>(context_.num_threads));
    }
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for schedule(dynamic)
#endif
    for (uint32_t i = 0; i < num_heads_; ++i) {
      // 使用分组查询注意力
      uint32_t kv_head_idx = i / group_size_;

      Tensor head_output = attention_heads_[i]->computeWithCache(
          query_heads[i], key_heads[kv_head_idx], value_heads[kv_head_idx],
          key_cache_heads[kv_head_idx], value_cache_heads[kv_head_idx],
          cache_position, mask, scale);

#ifdef _OPENMP
      #pragma omp critical
#endif
      {
        head_outputs.push_back(std::move(head_output));
      }
    }

    // 连接所有头的输出
    Tensor result = concatenateHeads(head_outputs, batch_size, seq_len_q);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    context_.total_time += duration.count() / 1000.0;
    context_.call_count++;

    return result;
  }

private:
  uint32_t hidden_size_ = 3584;
  uint32_t num_heads_ = 28;
  uint32_t num_kv_heads_ = 4;
  uint32_t head_dim_ = 128;    // hidden_size / num_heads
  uint32_t kv_head_dim_ = 128; // Same as head_dim in GQA architecture
  uint32_t group_size_ = 7;

  std::vector<std::unique_ptr<FastAttention>> attention_heads_;
  std::vector<std::unique_ptr<FastAttention>> kv_attention_heads_;
  AlgorithmContext context_;

  std::vector<Tensor> splitToHeads(const Tensor &input, uint32_t num_heads,
                                   uint32_t head_dim) {
    if (input.data.empty() || input.shape.empty()) {
      std::cerr << "[ERROR] splitToHeads: Input tensor is empty" << std::endl;
      throw std::invalid_argument("Input tensor is empty");
    }

    // 调试：打印输入张量的详细信息
    std::cerr << "[DEBUG] splitToHeads input shape size: " << input.shape.size()
              << std::endl;
    std::cerr << "[DEBUG] splitToHeads input shape: [";
    for (size_t i = 0; i < input.shape.size(); ++i) {
      std::cerr << input.shape[i];
      if (i < input.shape.size() - 1)
        std::cerr << ", ";
    }
    std::cerr << "], num_heads=" << num_heads << ", head_dim=" << head_dim
              << std::endl;
    std::cerr << "[DEBUG] splitToHeads input data size: " << input.data.size()
              << ", tensor size: " << input.size << std::endl;

    // 检查张量维度，必须至少有3个维度用于头分割
    if (input.shape.size() < 3) {
      std::cerr << "[ERROR] splitToHeads: Input tensor shape size is "
                << input.shape.size() << ", expected at least 3" << std::endl;
      std::cerr << "[ERROR] splitToHeads: This usually indicates a dimension "
                   "mismatch in cache tensors or input processing"
                << std::endl;
      throw std::invalid_argument(
          "Input tensor must have at least 3 dimensions for head splitting");
    }

    uint32_t batch_size = input.shape[0];
    uint32_t seq_len = input.shape[1];
    uint32_t total_dim = input.shape[2];

    // 验证维度匹配
    if (total_dim != num_heads * head_dim) {
      throw std::runtime_error("Hidden size mismatch: expected " +
                               std::to_string(num_heads * head_dim) + ", got " +
                               std::to_string(total_dim) +
                               " (num_heads=" + std::to_string(num_heads) +
                               ", head_dim=" + std::to_string(head_dim) + ")");
    }

    std::vector<Tensor> heads;
    heads.reserve(num_heads);

    for (uint32_t i = 0; i < num_heads; ++i) {
      Tensor head_tensor({batch_size, seq_len, head_dim});

      // 复制对应头的数据
      for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
          for (uint32_t d = 0; d < head_dim; ++d) {
            uint32_t src_idx =
                b * seq_len * total_dim + s * total_dim + i * head_dim + d;
            uint32_t dst_idx = b * seq_len * head_dim + s * head_dim + d;
            head_tensor.data[dst_idx] = input.data[src_idx];
          }
        }
      }

      heads.push_back(std::move(head_tensor));
    }

    return heads;
  }

  Tensor concatenateHeads(const std::vector<Tensor> &head_outputs,
                          uint32_t batch_size, uint32_t seq_len) {
    if (head_outputs.empty()) {
      throw std::invalid_argument("No head outputs to concatenate");
    }

    uint32_t head_dim = head_outputs[0].shape[2];
    uint32_t total_dim = num_heads_ * head_dim;

    Tensor result({batch_size, seq_len, total_dim});

    for (uint32_t i = 0; i < num_heads_; ++i) {
      const Tensor &head_output = head_outputs[i];

      for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t s = 0; s < seq_len; ++s) {
          for (uint32_t d = 0; d < head_dim; ++d) {
            uint32_t src_idx = b * seq_len * head_dim + s * head_dim + d;
            uint32_t dst_idx =
                b * seq_len * total_dim + s * total_dim + i * head_dim + d;
            result.data[dst_idx] = head_output.data[src_idx];
          }
        }
      }
    }

    return result;
  }

  void log(const std::string &level, const std::string &message) {
    if (context_.verbose) {
      std::cout << "[" << level << "] MultiHeadAttention: " << message
                << std::endl;
    }
  }
};

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // MULTI_HEAD_ATTENTION_H