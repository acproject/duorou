#include "ggml_attention.h"
#include "../../../../third_party/llama.cpp/ggml/include/ggml-cpu.h"
#include "../../../../third_party/llama.cpp/ggml/src/ggml-impl.h"
#include <cmath>
#include <cstring>
#include <iostream>

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

GGMLAttention::GGMLAttention()
    : ctx_(nullptr), gf_(nullptr), backend_(nullptr), buffer_(nullptr),
      hidden_size_(0), num_heads_(0), head_dim_(0), max_seq_len_(0),
      num_threads_(1), use_simd_(true), verbose_(false) {}

GGMLAttention::~GGMLAttention() { cleanupGGMLContext(); }

bool GGMLAttention::initialize(const ModelConfig &config,
                               const AlgorithmContext &context) {
  // Store configuration
  hidden_size_ = config.hidden_size;
  num_heads_ = config.num_attention_heads;
  head_dim_ = hidden_size_ / num_heads_;
  max_seq_len_ = config.max_position_embeddings;

  // Store algorithm context
  num_threads_ = context.num_threads;
  use_simd_ = context.use_simd;
  verbose_ = context.verbose;
  context_ = context;

  if (verbose_) {
    log("INFO", "Initializing GGMLAttention with hidden_size=" +
                    std::to_string(hidden_size_) +
                    ", num_heads=" + std::to_string(num_heads_) +
                    ", head_dim=" + std::to_string(head_dim_));
  }

  // Validate configuration
  if (hidden_size_ == 0 || num_heads_ == 0 || hidden_size_ % num_heads_ != 0) {
    log("ERROR", "Invalid attention configuration");
    return false;
  }

  // Initialize GGML context
  if (!initializeGGMLContext()) {
    log("ERROR", "Failed to initialize GGML context");
    return false;
  }

  if (verbose_) {
    log("INFO", "GGMLAttention initialized successfully");
  }

  return true;
}

Tensor GGMLAttention::compute(const Tensor &query, const Tensor &key,
                              const Tensor &value, const Tensor *mask,
                              float scale) {
  if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
    throw std::runtime_error("Invalid input tensors for GGMLAttention");
  }

  if (verbose_) {
    log("INFO", "Starting GGML attention computation using standard flash "
                "attention with RoPE");
    log("DEBUG", "Input tensor shapes - Q: [" + std::to_string(query.shape[0]) +
                     "," + std::to_string(query.shape[1]) + "," +
                     std::to_string(query.shape[2]) + "]");
    log("DEBUG", "Input tensor shapes - K: [" + std::to_string(key.shape[0]) +
                     "," + std::to_string(key.shape[1]) + "," +
                     std::to_string(key.shape[2]) + "]");
    log("DEBUG", "Input tensor shapes - V: [" + std::to_string(value.shape[0]) +
                     "," + std::to_string(value.shape[1]) + "," +
                     std::to_string(value.shape[2]) + "]");
  }

  // Validate tensor dimensions for GGML flash attention compatibility
  if (query.shape.size() != 3 || key.shape.size() != 3 ||
      value.shape.size() != 3) {
    throw std::runtime_error("GGML flash attention requires 3D tensors [batch, "
                             "seq_len, hidden_dim]");
  }

  // Check dimension compatibility
  if (query.shape[2] != key.shape[2] || key.shape[2] != value.shape[2]) {
    throw std::runtime_error(
        "Hidden dimensions must match: Q=" + std::to_string(query.shape[2]) +
        ", K=" + std::to_string(key.shape[2]) +
        ", V=" + std::to_string(value.shape[2]));
  }

  // Convert tensors to GGML format (keep original dimensions for flash
  // attention)
  struct ggml_tensor *q_ggml = tensorToGGML(query, "query", false);
  struct ggml_tensor *k_ggml = tensorToGGML(key, "key", false);
  struct ggml_tensor *v_ggml = tensorToGGML(value, "value", false);

  // Apply RoPE to Q and K tensors using ggml_rope_ext
  // RoPE parameters: n_dims (head_dim), mode (0 for normal), n_ctx
  // (max_seq_len), freq_base (10000.0), freq_scale (1.0), ext_factor (0.0),
  // attn_factor (1.0), beta_fast (32.0), beta_slow (1.0)
  struct ggml_tensor *q_rope = ggml_rope_ext(ctx_, q_ggml, nullptr, nullptr,
                                             head_dim_, // n_dims
                                             0, // mode (0 = normal RoPE)
                                             max_seq_len_, // n_ctx
                                             10000.0f,     // freq_base
                                             1.0f,         // freq_scale
                                             0.0f,         // ext_factor
                                             1.0f,         // attn_factor
                                             32.0f,        // beta_fast
                                             1.0f          // beta_slow
  );
  ggml_set_name(q_rope, "query_rope");

  struct ggml_tensor *k_rope = ggml_rope_ext(ctx_, k_ggml, nullptr, nullptr,
                                             head_dim_, // n_dims
                                             0, // mode (0 = normal RoPE)
                                             max_seq_len_, // n_ctx
                                             10000.0f,     // freq_base
                                             1.0f,         // freq_scale
                                             0.0f,         // ext_factor
                                             1.0f,         // attn_factor
                                             32.0f,        // beta_fast
                                             1.0f          // beta_slow
  );
  ggml_set_name(k_rope, "key_rope");

  if (verbose_) {
    log("DEBUG", "GGML tensor shapes - Q: [" + std::to_string(q_ggml->ne[0]) +
                     "," + std::to_string(q_ggml->ne[1]) + "," +
                     std::to_string(q_ggml->ne[2]) + "]");
    log("DEBUG", "GGML tensor shapes - K: [" + std::to_string(k_ggml->ne[0]) +
                     "," + std::to_string(k_ggml->ne[1]) + "," +
                     std::to_string(k_ggml->ne[2]) + "]");
    log("DEBUG", "GGML tensor shapes - V: [" + std::to_string(v_ggml->ne[0]) +
                     "," + std::to_string(v_ggml->ne[1]) + "," +
                     std::to_string(v_ggml->ne[2]) + "]");
  }

  // Prepare mask tensor if provided
  struct ggml_tensor *mask_ggml = nullptr;
  if (mask != nullptr) {
    mask_ggml = tensorToGGML(*mask, "mask", false);
  }

  // Calculate attention scale
  float attention_scale =
      (scale != 1.0f) ? scale : (1.0f / sqrtf(static_cast<float>(head_dim_)));

  // Use standard GGML flash attention with RoPE-applied Q and K tensors
  struct ggml_tensor *output = ggml_flash_attn_ext(ctx_, q_rope, k_rope, v_ggml,
                                                   mask_ggml, attention_scale,
                                                   0.0f, // max_bias
                                                   0.0f  // logit_softcap
  );
  ggml_set_name(output, "flash_attention_output");

  // Build computation graph using ggml_build_forward_expand
  // This ensures all operations are executed in depth-first order
  gf_ = ggml_new_graph(ctx_);
  ggml_build_forward_expand(gf_, output);

  if (verbose_) {
    log("INFO", "Built computation graph with " + std::to_string(gf_->n_nodes) +
                    " nodes using flash attention");

    // Verify tensor data pointers before execution
    log("DEBUG", "Before execution - Q tensor data ptr: " +
                     std::to_string((uintptr_t)q_ggml->data));
    log("DEBUG", "Before execution - K tensor data ptr: " +
                     std::to_string((uintptr_t)k_ggml->data));
    log("DEBUG", "Before execution - V tensor data ptr: " +
                     std::to_string((uintptr_t)v_ggml->data));
  }

  // Execute the computation graph using ggml_graph_compute_with_ctx
  // This runs ggml_compute_forward() on the flash attention node
  // which will call the optimized flash attention implementation
  enum ggml_status status =
      ggml_graph_compute_with_ctx(ctx_, gf_, num_threads_);

  if (verbose_) {
    log("INFO", "Completed graph computation with " +
                    std::to_string(num_threads_) +
                    " threads, status: " + std::to_string(status));

    // Verify tensor data pointers after execution
    log("DEBUG", "After execution - Q tensor data ptr: " +
                     std::to_string((uintptr_t)q_ggml->data));
    log("DEBUG", "After execution - K tensor data ptr: " +
                     std::to_string((uintptr_t)k_ggml->data));
    log("DEBUG", "After execution - V tensor data ptr: " +
                     std::to_string((uintptr_t)v_ggml->data));
    log("DEBUG", "After execution - Output tensor data ptr: " +
                     std::to_string((uintptr_t)output->data));

    // Verify output tensor has valid data
    if (output->data && ggml_nelements(output) > 0) {
      const float *output_data = (const float *)output->data;
      std::string sample_values = "First few output values: ";
      for (int i = 0; i < std::min(5, (int)ggml_nelements(output)); ++i) {
        sample_values += std::to_string(output_data[i]) + " ";
      }
      log("DEBUG", sample_values);
    } else {
      log("WARNING",
          "Output tensor data pointer is null or empty after execution");
    }
  }

  // Convert result back to Tensor format
  Tensor result = ggmlToTensor(output);

  if (verbose_) {
    log("INFO", "GGMLAttention compute completed successfully using standard "
                "flash attention");
  }

  return result;
}

Tensor GGMLAttention::computeWithCache(const Tensor &query, const Tensor &key,
                                       const Tensor &value, Tensor &key_cache,
                                       Tensor &value_cache,
                                       uint32_t cache_position,
                                       uint32_t head_idx, const Tensor *mask,
                                       float scale) {
  // Enhanced KV cache implementation
  try {
    if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
      throw std::runtime_error("Invalid input tensors for cached attention");
    }

    // Initialize cache if empty
    if (key_cache.data.empty() || value_cache.data.empty()) {
      key_cache = key;
      value_cache = value;
    } else {
      // Update cache at specified position
      // For simplicity, we'll append new key/value to cache
      // In a full implementation, this would handle position-based updates
      updateCache(key_cache, key, cache_position);
      updateCache(value_cache, value, cache_position);
    }

    // Convert cached tensors to GGML format
    struct ggml_tensor *q_ggml = tensorToGGML(query, "cached_query", true);
    struct ggml_tensor *k_ggml = tensorToGGML(key_cache, "cached_key", true);
    struct ggml_tensor *v_ggml =
        tensorToGGML(value_cache, "cached_value", true);

    // Compute attention scores with cached K
    float attention_scale =
        (scale != 1.0f) ? scale : (1.0f / sqrtf(static_cast<float>(head_dim_)));
    struct ggml_tensor *scores =
        computeAttentionScores(q_ggml, k_ggml, attention_scale);

    // Apply mask if provided
    if (mask != nullptr) {
      struct ggml_tensor *mask_ggml = tensorToGGML(*mask, "mask");
      scores = applyAttentionMask(scores, mask_ggml);
    }

    // Apply softmax
    scores = ggml_soft_max(ctx_, scores);
    ggml_set_name(scores, "cached_attention_weights");

    // Compute final output with cached V
    struct ggml_tensor *output = computeAttentionOutput(scores, v_ggml);

    // Build and compute the graph
    gf_ = ggml_new_graph(ctx_);
    ggml_build_forward_expand(gf_, output);
    ggml_graph_compute_with_ctx(ctx_, gf_, num_threads_);

    // Convert result back to Tensor format
    Tensor result = ggmlToTensor(output);

    if (verbose_) {
      log("INFO", "GGMLAttention cached compute completed for head " +
                      std::to_string(head_idx));
    }

    return result;

  } catch (const std::exception &e) {
    log("ERROR",
        "Cached attention computation failed: " + std::string(e.what()));
    // Fallback to non-cached computation
    return compute(query, key, value, mask, scale);
  }
}

bool GGMLAttention::validateInput(const Tensor &input) const {
  if (input.data.empty() || input.shape.empty()) {
    return false;
  }

  // Check if tensor has at least 2 dimensions
  if (input.shape.size() < 2) {
    return false;
  }

  // Verify data size matches shape
  size_t expected_size = 1;
  for (uint32_t dim : input.shape) {
    expected_size *= dim;
  }

  return input.data.size() == expected_size;
}

bool GGMLAttention::initializeGGMLContext() {
  // Calculate memory requirements
  size_t mem_size =
      256 * 1024 * 1024; // 256MB should be sufficient for most cases

  // Initialize GGML context
  struct ggml_init_params params = {
      /*.mem_size   =*/mem_size,
      /*.mem_buffer =*/nullptr,
      /*.no_alloc   =*/false,
  };

  ctx_ = ggml_init(params);
  if (!ctx_) {
    return false;
  }

  // Initialize backend
  backend_ = ggml_backend_cpu_init();
  if (!backend_) {
    ggml_free(ctx_);
    ctx_ = nullptr;
    return false;
  }

  return true;
}

void GGMLAttention::cleanupGGMLContext() {
  if (buffer_) {
    ggml_backend_buffer_free(buffer_);
    buffer_ = nullptr;
  }

  if (backend_) {
    ggml_backend_free(backend_);
    backend_ = nullptr;
  }

  if (ctx_) {
    ggml_free(ctx_);
    ctx_ = nullptr;
  }
}

struct ggml_tensor *GGMLAttention::tensorToGGML(const Tensor &tensor,
                                                const std::string &name,
                                                bool reshape_for_matmul) {
  // Validate tensor data
  if (tensor.shape.empty()) {
    throw std::runtime_error("Empty tensor shape in tensorToGGML for tensor: " +
                             name);
  }

  // Check for valid data size
  size_t expected_size = 1;
  for (auto dim : tensor.shape) {
    if (dim <= 0) {
      throw std::runtime_error("Invalid dimension " + std::to_string(dim) +
                               " in tensor: " + name);
    }
    expected_size *= dim;
  }

  if (tensor.data.size() != expected_size) {
    throw std::runtime_error("Tensor data size mismatch for " + name +
                             ": expected " + std::to_string(expected_size) +
                             ", got " + std::to_string(tensor.data.size()));
  }

  // Create GGML tensor with the same shape
  std::vector<int64_t> ggml_shape(tensor.shape.begin(), tensor.shape.end());

  struct ggml_tensor *ggml_tensor;

  // For matrix multiplication, reshape 3D tensors with batch_size=1 to 2D
  // GGML expects dimensions in reverse order: [cols, rows] for 2D tensors
  if (reshape_for_matmul && tensor.shape.size() == 3 && tensor.shape[0] == 1) {
    // Reshape [1, seq_len, hidden_size] to [hidden_size, seq_len] for GGML
    ggml_tensor =
        ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, ggml_shape[2], ggml_shape[1]);
  } else if (tensor.shape.size() == 1) {
    ggml_tensor = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, ggml_shape[0]);
  } else if (tensor.shape.size() == 2) {
    // GGML expects [cols, rows] order for 2D tensors
    ggml_tensor =
        ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, ggml_shape[1], ggml_shape[0]);
  } else if (tensor.shape.size() == 3) {
    // For 3D tensors: [batch, seq_len, hidden_size] -> [hidden_size, seq_len,
    // batch]
    ggml_tensor = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, ggml_shape[2],
                                     ggml_shape[1], ggml_shape[0]);
  } else if (tensor.shape.size() == 4) {
    ggml_tensor =
        ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, ggml_shape[3], ggml_shape[2],
                           ggml_shape[1], ggml_shape[0]);
  } else {
    throw std::runtime_error("Unsupported tensor dimension: " +
                             std::to_string(tensor.shape.size()));
  }

  if (!ggml_tensor) {
    throw std::runtime_error("Failed to create GGML tensor for: " + name);
  }

  // Copy data - need to transpose for GGML's column-major layout
  if (reshape_for_matmul && tensor.shape.size() == 3 && tensor.shape[0] == 1) {
    // Transpose from [seq_len, hidden_size] to [hidden_size, seq_len]
    float *dst = (float *)ggml_tensor->data;
    const float *src = tensor.data.data();
    int seq_len = tensor.shape[1];
    int hidden_size = tensor.shape[2];

    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j < hidden_size; ++j) {
        dst[j * seq_len + i] = src[i * hidden_size + j];
      }
    }
  } else if (tensor.shape.size() == 2) {
    // Transpose from [rows, cols] to [cols, rows]
    float *dst = (float *)ggml_tensor->data;
    const float *src = tensor.data.data();
    int rows = tensor.shape[0];
    int cols = tensor.shape[1];

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        dst[j * rows + i] = src[i * cols + j];
      }
    }
  } else if (tensor.shape.size() == 3) {
    // For 3D tensors: transpose from [batch, seq_len, hidden_size] to
    // [hidden_size, seq_len, batch]
    float *dst = (float *)ggml_tensor->data;
    const float *src = tensor.data.data();
    int batch = tensor.shape[0];
    int seq_len = tensor.shape[1];
    int hidden_size = tensor.shape[2];

    for (int b = 0; b < batch; ++b) {
      for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < hidden_size; ++h) {
          // src: [b][s][h] -> dst: [h][s][b]
          dst[h * seq_len * batch + s * batch + b] =
              src[b * seq_len * hidden_size + s * hidden_size + h];
        }
      }
    }
  } else {
    // Direct copy for other cases
    memcpy(ggml_tensor->data, tensor.data.data(),
           tensor.data.size() * sizeof(float));
  }

  // Set name if provided
  if (!name.empty()) {
    ggml_set_name(ggml_tensor, name.c_str());
  }

  return ggml_tensor;
}

Tensor GGMLAttention::ggmlToTensor(const struct ggml_tensor *ggml_tensor) {
  // Extract shape
  std::vector<uint32_t> shape;
  for (int i = 0; i < ggml_n_dims(ggml_tensor); ++i) {
    shape.push_back(static_cast<uint32_t>(ggml_tensor->ne[i]));
  }

  // Create result tensor
  Tensor result(shape);

  // Copy data
  size_t data_size = ggml_nbytes(ggml_tensor) / sizeof(float);
  result.data.resize(data_size);
  memcpy(result.data.data(), ggml_tensor->data, ggml_nbytes(ggml_tensor));

  return result;
}

struct ggml_tensor *
GGMLAttention::computeAttentionScores(struct ggml_tensor *query,
                                      struct ggml_tensor *key, float scale) {
  // Transpose key for matrix multiplication
  struct ggml_tensor *key_t = ggml_transpose(ctx_, key);
  ggml_set_name(key_t, "key_transposed");

  // Compute Q * K^T
  struct ggml_tensor *scores = ggml_mul_mat(ctx_, key_t, query);
  ggml_set_name(scores, "attention_scores_raw");

  // Apply scaling
  if (scale != 1.0f) {
    scores = ggml_scale(ctx_, scores, scale);
    ggml_set_name(scores, "attention_scores_scaled");
  }

  if (verbose_) {
    std::cout << "[DEBUG] computeAttentionScores: scale=" << scale
              << ", scores shape=[" << scores->ne[0] << "," << scores->ne[1]
              << "]" << std::endl;
  }

  return scores;
}

struct ggml_tensor *
GGMLAttention::applyAttentionMask(struct ggml_tensor *scores,
                                  struct ggml_tensor *mask) {
  // Add mask to scores (masked positions should have large negative values)
  struct ggml_tensor *masked_scores = ggml_add(ctx_, scores, mask);
  ggml_set_name(masked_scores, "masked_attention_scores");

  return masked_scores;
}

struct ggml_tensor *
GGMLAttention::computeAttentionOutput(struct ggml_tensor *scores,
                                      struct ggml_tensor *value) {
  // Compute attention_weights * V
  struct ggml_tensor *output = ggml_mul_mat(ctx_, value, scores);
  ggml_set_name(output, "attention_output");

  return output;
}

// Free function: high-performance linear projection using GGML
void GGMLAttention::updateCache(Tensor &cache, const Tensor &new_data,
                                uint32_t position) {
  // 简化的缓存更新实现：直接追加新数据
  // 在实际应用中，这里应该根据position进行精确的位置更新
  if (cache.data.empty()) {
    cache = new_data;
    return;
  }

  // 如果缓存已满或需要扩展，简单地用新数据替换
  // 更复杂的实现会管理滑动窗口或环形缓冲区
  if (position >= cache.shape[0]) {
    // 扩展缓存或重置
    cache = new_data;
  } else {
    // 简单追加策略：将新数据连接到现有缓存
    size_t old_size = cache.data.size();
    size_t new_size = new_data.data.size();
    cache.data.resize(old_size + new_size);
    std::memcpy(cache.data.data() + old_size, new_data.data.data(),
                new_size * sizeof(float));

    // 更新形状（假设在序列维度上扩展）
    if (!cache.shape.empty()) {
      cache.shape[0] += new_data.shape[0];
    }
  }
}

Tensor computeLinear(const Tensor &a, const Tensor &w) {
  // Validate inputs
  if (a.shape.size() < 2 || w.shape.size() != 2) {
    throw std::invalid_argument("computeLinear: invalid input shapes");
  }

  // Derive M, K from a; K, N from w
  uint32_t M = 0, K_a = 0;
  if (a.shape.size() == 3) {
    if (a.shape[0] != 1) {
      throw std::invalid_argument(
          "computeLinear: only batch=1 supported for 3D input");
    }
    M = a.shape[1];
    K_a = a.shape[2];
  } else {
    M = a.shape[0];
    K_a = a.shape[1];
  }

  uint32_t K_w = w.shape[0];
  uint32_t N = w.shape[1];

  if (K_a != K_w) {
    std::cerr << "[ERROR] computeLinear: K mismatch a.K=" << K_a
              << " vs w.K=" << K_w << std::endl;
    throw std::invalid_argument("computeLinear: inner dim mismatch");
  }

  // Create ggml context (short-lived)
  size_t mem_size = 64 * 1024 * 1024; // 64MB
  struct ggml_init_params params = {
      /*.mem_size   =*/mem_size,
      /*.mem_buffer =*/nullptr,
      /*.no_alloc   =*/false,
  };
  struct ggml_context *ctx = ggml_init(params);
  if (!ctx) {
    throw std::runtime_error("computeLinear: ggml_init failed");
  }

  // Build ggml tensors (note: ggml uses unconventional matmul order and dims):
  // - For a conventional A[M,K], create ggml A_g with ne[0]=K (cols), ne[1]=M (rows)
  // - For a conventional W[K,N], create ggml W_g with ne[0]=K (cols), ne[1]=N (rows)
  // Then call C = ggml_mul_mat(ctx, W_g, A_g) which yields C with ne[0]=N, ne[1]=M

  // A_g: [K, M] in ggml dims (represents conventional [M, K])
  struct ggml_tensor *A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K_a, M);
  {
    float *dst = (float *)A->data;
    const float *src = a.data.data();
    // Copy data: src is row-major [M,K], dst is column-major with dims [K,M]
    for (uint32_t i = 0; i < M; ++i) {
      for (uint32_t j = 0; j < K_a; ++j) {
        dst[j * M + i] = src[i * K_a + j];
      }
    }
  }
  ggml_set_name(A, "A_KM");

  // W_g: [K, N] in ggml dims (represents conventional [K, N])
  struct ggml_tensor *W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K_w, N);
  {
    float *dst = (float *)W->data;
    const float *src = w.data.data();
    // If weight is provided as [N,K], transpose it; otherwise assume [K,N]
    if (w.shape[0] == N && w.shape[1] == K_w) {
      // src is row-major [N,K] -> dst column-major with dims [K,N]
      for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t k = 0; k < K_w; ++k) {
          dst[k + n * K_w] = src[n * K_w + k];
        }
      }
    } else {
      // src is row-major [K,N] -> dst column-major with dims [K,N]
      for (uint32_t k = 0; k < K_w; ++k) {
        for (uint32_t n = 0; n < N; ++n) {
          dst[k + n * K_w] = src[k * N + n];
        }
      }
    }
  }
  ggml_set_name(W, "W_KN");

  // Debug: Print tensor dimensions before matrix multiplication
  std::cout << "[DEBUG] computeLinear: (conventional) A[M,K]=" << M << "," << K_a
            << " ; W[K,N]=" << K_w << "," << N << std::endl;
  std::cout << "[DEBUG] computeLinear: (ggml dims) A.ne[0]=" << A->ne[0] << ", A.ne[1]=" << A->ne[1]
            << " ; W.ne[0]=" << W->ne[0] << ", W.ne[1]=" << W->ne[1] << std::endl;

  // Check ggml_can_mul_mat conditions for ggml_mul_mat(W, A)
  bool can_mul = (W->ne[0] == A->ne[0]) && (A->ne[2] % W->ne[2] == 0) && (A->ne[3] % W->ne[3] == 0);
  std::cout << "[DEBUG] ggml_can_mul_mat check: W.ne[0]=" << W->ne[0] << ", A.ne[0]=" << A->ne[0] << std::endl;
  std::cout << "[DEBUG] W.ne[2]=" << W->ne[2] << ", W.ne[3]=" << W->ne[3] << std::endl;
  std::cout << "[DEBUG] A.ne[2]=" << A->ne[2] << ", A.ne[3]=" << A->ne[3] << std::endl;
  std::cout << "[DEBUG] can_mul_mat result: " << can_mul << std::endl;

  // C = W_g x A_g ; ggml_mul_mat(W, A) -> C.ne[0]=W.ne[1](N), C.ne[1]=A.ne[1](M)
  struct ggml_tensor *C = ggml_mul_mat(ctx, W, A);
  ggml_set_name(C, "C_MN");

  // Graph and compute
  struct ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, C);
  ggml_graph_compute_with_ctx(ctx, gf, 4);

  // Copy back to Tensor (row-major [M,N], or [1,M,N] if 3D input)
  Tensor out(a.shape.size() == 3 ? std::vector<uint32_t>{1, M, N}
                                 : std::vector<uint32_t>{M, N});
  out.data.resize((size_t)M * N);

  // ggml C result has dims ne[0]=N, ne[1]=M (column-major)
  // Convert to row-major [M,N]
  const float *csrc = (const float *)C->data;
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      out.data[i * N + j] = csrc[j * M + i];
    }
  }

  ggml_free(ctx);
  return out;
}

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou