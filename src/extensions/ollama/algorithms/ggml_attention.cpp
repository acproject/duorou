#include "ggml_attention.h"
#include "../../../../third_party/llama.cpp/ggml/include/ggml-cpu.h"
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
      num_threads_(1), use_simd_(true), verbose_(false) {
}

GGMLAttention::~GGMLAttention() {
    cleanupGGMLContext();
}

bool GGMLAttention::initialize(const ModelConfig& config, const AlgorithmContext& context) {
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
        log("INFO", "Initializing GGMLAttention with hidden_size=" + std::to_string(hidden_size_) +
                   ", num_heads=" + std::to_string(num_heads_) + ", head_dim=" + std::to_string(head_dim_));
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

Tensor GGMLAttention::compute(const Tensor& query, const Tensor& key, const Tensor& value,
                             const Tensor* mask, float scale) {
    if (!validateInput(query) || !validateInput(key) || !validateInput(value)) {
        throw std::runtime_error("Invalid input tensors for GGMLAttention");
    }
    
    // Convert tensors to GGML format (reshape 3D to 2D for matrix multiplication)
    struct ggml_tensor* q_ggml = tensorToGGML(query, "query", true);
    struct ggml_tensor* k_ggml = tensorToGGML(key, "key", true);
    struct ggml_tensor* v_ggml = tensorToGGML(value, "value", true);
    
    // Compute attention scores: Q * K^T
    float attention_scale = (scale != 1.0f) ? scale : (1.0f / sqrtf(static_cast<float>(head_dim_)));
    struct ggml_tensor* scores = computeAttentionScores(q_ggml, k_ggml, attention_scale);
    
    // Apply mask if provided
    if (mask != nullptr) {
        struct ggml_tensor* mask_ggml = tensorToGGML(*mask, "mask");
        scores = applyAttentionMask(scores, mask_ggml);
    }
    
    // Apply softmax
    scores = ggml_soft_max(ctx_, scores);
    ggml_set_name(scores, "attention_weights");
    
    // Compute final output: attention_weights * V
    struct ggml_tensor* output = computeAttentionOutput(scores, v_ggml);
    
    // Build and compute the graph
    gf_ = ggml_new_graph(ctx_);
    ggml_build_forward_expand(gf_, output);
    
    // Set number of threads
    ggml_graph_compute_with_ctx(ctx_, gf_, num_threads_);
    
    // Convert result back to Tensor format
    Tensor result = ggmlToTensor(output);
    
    if (verbose_) {
        log("INFO", "GGMLAttention compute completed");
    }
    
    return result;
}

Tensor GGMLAttention::computeWithCache(const Tensor& query, const Tensor& key, const Tensor& value,
                                      Tensor& key_cache, Tensor& value_cache, uint32_t cache_position,
                                      uint32_t head_idx, const Tensor* mask, float scale) {
    // For now, implement a simple version without actual caching
    // This can be enhanced later with proper KV cache management
    return compute(query, key, value, mask, scale);
}

bool GGMLAttention::validateInput(const Tensor& input) const {
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
    size_t mem_size = 256 * 1024 * 1024; // 256MB should be sufficient for most cases
    
    // Initialize GGML context
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
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

struct ggml_tensor* GGMLAttention::tensorToGGML(const Tensor& tensor, const std::string& name, bool reshape_for_matmul) {
    // Create GGML tensor with the same shape
    std::vector<int64_t> ggml_shape(tensor.shape.begin(), tensor.shape.end());
    
    struct ggml_tensor* ggml_tensor;
    
    // For matrix multiplication, reshape 3D tensors with batch_size=1 to 2D
    // GGML expects dimensions in reverse order: [cols, rows] for 2D tensors
    if (reshape_for_matmul && tensor.shape.size() == 3 && tensor.shape[0] == 1) {
        // Reshape [1, seq_len, hidden_size] to [hidden_size, seq_len] for GGML
        ggml_tensor = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, ggml_shape[2], ggml_shape[1]);
    } else if (tensor.shape.size() == 1) {
        ggml_tensor = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, ggml_shape[0]);
    } else if (tensor.shape.size() == 2) {
        // GGML expects [cols, rows] order for 2D tensors
        ggml_tensor = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, ggml_shape[1], ggml_shape[0]);
    } else if (tensor.shape.size() == 3) {
        ggml_tensor = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, ggml_shape[2], ggml_shape[1], ggml_shape[0]);
    } else if (tensor.shape.size() == 4) {
        ggml_tensor = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, ggml_shape[3], ggml_shape[2], ggml_shape[1], ggml_shape[0]);
    } else {
        throw std::runtime_error("Unsupported tensor dimension: " + std::to_string(tensor.shape.size()));
    }
    
    // Copy data - need to transpose for GGML's column-major layout
    if (reshape_for_matmul && tensor.shape.size() == 3 && tensor.shape[0] == 1) {
        // Transpose from [seq_len, hidden_size] to [hidden_size, seq_len]
        float* dst = (float*)ggml_tensor->data;
        const float* src = tensor.data.data();
        int seq_len = tensor.shape[1];
        int hidden_size = tensor.shape[2];
        
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                dst[j * seq_len + i] = src[i * hidden_size + j];
            }
        }
    } else if (tensor.shape.size() == 2) {
        // Transpose from [rows, cols] to [cols, rows]
        float* dst = (float*)ggml_tensor->data;
        const float* src = tensor.data.data();
        int rows = tensor.shape[0];
        int cols = tensor.shape[1];
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    } else {
        // Direct copy for other cases
        memcpy(ggml_tensor->data, tensor.data.data(), tensor.data.size() * sizeof(float));
    }
    
    // Set name if provided
    if (!name.empty()) {
        ggml_set_name(ggml_tensor, name.c_str());
    }
    
    return ggml_tensor;
}

Tensor GGMLAttention::ggmlToTensor(const struct ggml_tensor* ggml_tensor) {
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

struct ggml_tensor* GGMLAttention::computeAttentionScores(struct ggml_tensor* query, 
                                                         struct ggml_tensor* key,
                                                         float scale) {
    // Transpose key for matrix multiplication
    struct ggml_tensor* key_t = ggml_transpose(ctx_, key);
    ggml_set_name(key_t, "key_transposed");
    
    // Compute Q * K^T
    struct ggml_tensor* scores = ggml_mul_mat(ctx_, key_t, query);
    ggml_set_name(scores, "attention_scores");
    
    // Apply scaling
    if (scale != 1.0f) {
        scores = ggml_scale(ctx_, scores, scale);
        ggml_set_name(scores, "scaled_attention_scores");
    }
    
    return scores;
}

struct ggml_tensor* GGMLAttention::applyAttentionMask(struct ggml_tensor* scores,
                                                     struct ggml_tensor* mask) {
    // Add mask to scores (masked positions should have large negative values)
    struct ggml_tensor* masked_scores = ggml_add(ctx_, scores, mask);
    ggml_set_name(masked_scores, "masked_attention_scores");
    
    return masked_scores;
}

struct ggml_tensor* GGMLAttention::computeAttentionOutput(struct ggml_tensor* scores,
                                                         struct ggml_tensor* value) {
    // Compute attention_weights * V
    struct ggml_tensor* output = ggml_mul_mat(ctx_, value, scores);
    ggml_set_name(output, "attention_output");
    
    return output;
}

// Free function: high-performance linear projection using GGML
Tensor computeLinear(const Tensor& a, const Tensor& w) {
    // Validate inputs
    if (a.shape.size() < 2 || w.shape.size() != 2) {
        throw std::invalid_argument("computeLinear: invalid input shapes");
    }

    // Derive M, K from a; K, N from w
    uint32_t M = 0, K_a = 0;
    if (a.shape.size() == 3) {
        if (a.shape[0] != 1) {
            throw std::invalid_argument("computeLinear: only batch=1 supported for 3D input");
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
        std::cerr << "[ERROR] computeLinear: K mismatch a.K=" << K_a << " vs w.K=" << K_w << std::endl;
        throw std::invalid_argument("computeLinear: inner dim mismatch");
    }

    // Create ggml context (short-lived)
    size_t mem_size = 64 * 1024 * 1024; // 64MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        throw std::runtime_error("computeLinear: ggml_init failed");
    }

    // Build ggml tensors:
    // A: [K, M]
    struct ggml_tensor* A = nullptr;
    if (a.shape.size() == 3) {
        A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K_a, M);
        // copy transpose from [M,K] row-major -> [K,M] column-major
        float* dst = (float*)A->data;
        const float* src = a.data.data();
        for (uint32_t i = 0; i < M; ++i) {
            for (uint32_t j = 0; j < K_a; ++j) {
                dst[j * M + i] = src[i * K_a + j];
            }
        }
    } else { // 2D
        A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K_a, M);
        float* dst = (float*)A->data;
        const float* src = a.data.data();
        for (uint32_t i = 0; i < M; ++i) {
            for (uint32_t j = 0; j < K_a; ++j) {
                dst[j * M + i] = src[i * K_a + j];
            }
        }
    }
    ggml_set_name(A, "A_KM");

    // W: [K, N] -> ggml [K, N]
    struct ggml_tensor* W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K_w, N);
    {
        float* dst = (float*)W->data;
        const float* src = w.data.data();
        for (uint32_t k = 0; k < K_w; ++k) {
            for (uint32_t n = 0; n < N; ++n) {
                dst[k * N + n] = src[k * N + n];
            }
        }
    }
    ggml_set_name(W, "W_KN");

    // C = A x W ; ggml_mul_mat expects A.ne[0] == W.ne[0] == K
    struct ggml_tensor* C = ggml_mul_mat(ctx, A, W);
    ggml_set_name(C, "C_MN");

    // Graph and compute
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    // Copy back to Tensor (row-major [M,N], or [1,M,N] if 3D input)
    Tensor out(a.shape.size() == 3 ? std::vector<uint32_t>{1, M, N}
                                   : std::vector<uint32_t>{M, N});
    out.data.resize((size_t)M * N);

    // ggml C is [M,N] with ne0=N, ne1=M; data layout column-major
    const float* csrc = (const float*)C->data;
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