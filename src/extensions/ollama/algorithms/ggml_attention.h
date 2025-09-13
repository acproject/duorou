#ifndef GGML_ATTENTION_H
#define GGML_ATTENTION_H

#include "base_algorithm.h"
#include "../../../../third_party/llama.cpp/ggml/include/ggml.h"
#include "../../../../third_party/llama.cpp/ggml/include/ggml-backend.h"
#include <stdint.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

class GGMLAttention : public IAttentionAlgorithm {
public:
    GGMLAttention();
    ~GGMLAttention() override;
    
    // IAlgorithm interface
    bool initialize(const ModelConfig& config, const AlgorithmContext& context) override;
    std::string getName() const override { return "GGMLAttention"; }
    std::string getVersion() const override { return "GGML-1.0"; }
    bool validateInput(const Tensor& input) const override;
    
    // IAttentionAlgorithm interface
    Tensor compute(const Tensor& query, const Tensor& key, const Tensor& value,
                   const Tensor* mask = nullptr, float scale = 1.0f) override;
    
    Tensor computeWithCache(const Tensor& query, const Tensor& key, const Tensor& value,
                           Tensor& key_cache, Tensor& value_cache, uint32_t cache_position,
                           uint32_t head_idx, const Tensor* mask = nullptr,
                           float scale = 1.0f) override;
    
    // Logging method
    void log(const std::string& level, const std::string& message) const {
        if (verbose_) {
            std::cerr << "[" << level << "] GGMLAttention: " << message << std::endl;
        }
    }
    
private:
    // GGML context and computation graph
    struct ggml_context* ctx_;
    struct ggml_cgraph* gf_;
    ggml_backend_t backend_;
    ggml_backend_buffer_t buffer_;
    
    // Model configuration
    uint32_t hidden_size_;
    uint32_t num_heads_;
    uint32_t head_dim_;
    uint32_t max_seq_len_;
    
    // Runtime configuration
    uint32_t num_threads_;
    bool use_simd_;
    bool verbose_;
    
    // GGML management methods
    bool initializeGGMLContext();
    void cleanupGGMLContext();
    
    // Tensor conversion methods
    struct ggml_tensor* tensorToGGML(const Tensor& tensor, const std::string& name = "", bool reshape_for_matmul = false);
    Tensor ggmlToTensor(const struct ggml_tensor* ggml_tensor);
    
    // Attention computation helpers
    struct ggml_tensor* computeAttentionScores(struct ggml_tensor* query, 
                                              struct ggml_tensor* key,
                                              float scale);
    struct ggml_tensor* applyAttentionMask(struct ggml_tensor* scores,
                                          struct ggml_tensor* mask);
    struct ggml_tensor* computeAttentionOutput(struct ggml_tensor* scores,
                                              struct ggml_tensor* value);
};

// New: High-performance linear projection via GGML
// Compute C = A x W where:
// - If A is 3D [1, M, K], it is treated as [M, K]; result returns as [1, M, N]
// - If A is 2D [M, K], result returns as [M, N]
// - W must be 2D [K, N]
Tensor computeLinear(const Tensor& a, const Tensor& w);

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // GGML_ATTENTION_H