#ifndef DUOROU_ML_NN_ATTENTION_H
#define DUOROU_ML_NN_ATTENTION_H

#ifdef __cplusplus
#include <vector>
#include <cstdint>

#include "../tensor.h"
#include "../context.h"
#include "../../kvcache/cache.h"

namespace duorou {
namespace ml {
namespace nn {

// Configuration for Rotary Position Embedding (RoPE)
struct RoPEConfig {
    int64_t dimension = 0;         // number of dims to rotate (<= headDim)
    float theta = 10000.0f;        // base frequency (rope.freq_base)
    std::vector<int64_t> sections; // optional mrope sections
};

// Multi-head attention mechanism
class MultiHeadAttention {
public:
    // Constructor
    MultiHeadAttention(int64_t embedDim, int64_t numHeads, 
                      int64_t kvHeads = -1, bool bias = true, 
                      float dropout = 0.0f);
    
    ~MultiHeadAttention() = default;
    
    // Forward pass
    Tensor forward(Context& ctx, const Tensor& query, 
                  const Tensor& key = {}, const Tensor& value = {},
                  kvcache::Cache* cache = nullptr, const Tensor& mask = {});
    
    // Forward pass with attention sinks
    Tensor forwardWithSinks(Context& ctx, const Tensor& query,
                           const Tensor& key, const Tensor& value,
                           const Tensor& sinks, float scale,
                           kvcache::Cache* cache = nullptr);
    
    // Copy and move semantics
    MultiHeadAttention(const MultiHeadAttention& other) = delete;
    MultiHeadAttention& operator=(const MultiHeadAttention& other) = delete;
    MultiHeadAttention(MultiHeadAttention&& other) = default;
    MultiHeadAttention& operator=(MultiHeadAttention&& other) = default;
    
    // Weight initialization
    void initializeWeights(Context& ctx, const std::string& method = "xavier_uniform");

    // Set weights from host vectors (allocates and attaches backend)
    bool setWeights(Context& ctx,
                    const std::vector<float>& qW,
                    const std::vector<float>& kW,
                    const std::vector<float>& vW,
                    const std::vector<float>& oW,
                    const std::vector<float>* qB = nullptr,
                    const std::vector<float>* kB = nullptr,
                    const std::vector<float>* vB = nullptr,
                    const std::vector<float>* oB = nullptr);

    // Configure RoPE (optional). If not set, defaults are used.
    void setRoPEConfig(const RoPEConfig& cfg) { ropeCfg_ = cfg; ropeCfgSet_ = true; }
    bool hasRoPEConfig() const { return ropeCfgSet_; }
    
    // Getters
    int64_t embedDim() const { return embedDim_; }
    int64_t numHeads() const { return numHeads_; }
    int64_t kvHeads() const { return kvHeads_; }
    int64_t headDim() const { return headDim_; }
    bool hasBias() const { return hasBias_; }
    float dropout() const { return dropout_; }
    
private:
    int64_t embedDim_;
    int64_t numHeads_;
    int64_t kvHeads_;
    int64_t headDim_;
    bool hasBias_;
    float dropout_;
    
    // Weight matrices
    // Shapes:
    //  - queryWeight_:  [embedDim, numHeads * headDim]
    //  - keyWeight_:    [embedDim, kvHeads * headDim]
    //  - valueWeight_:  [embedDim, kvHeads * headDim]
    //  - outputWeight_: [numHeads * headDim, embedDim]
    Tensor queryWeight_;
    Tensor keyWeight_;
    Tensor valueWeight_;
    Tensor outputWeight_;
    
    // Bias vectors (if enabled)
    Tensor queryBias_;   // [numHeads * headDim]
    Tensor keyBias_;     // [kvHeads * headDim]
    Tensor valueBias_;   // [kvHeads * headDim]
    Tensor outputBias_;  // [embedDim]
    
    // Helper methods
    Tensor scaledDotProductAttention(Context& ctx, const Tensor& q, 
                                   const Tensor& k, const Tensor& v,
                                   const Tensor& mask = {});
    Tensor applyRotaryPositionEmbedding(Context& ctx, const Tensor& tensor, 
                                      int64_t seqLen, int64_t offset = 0);

    // RoPE config state
    RoPEConfig ropeCfg_{};
    bool ropeCfgSet_ = false;
};

// Standalone attention functions
Tensor attention(Context& ctx, const Tensor& query, const Tensor& key, 
                const Tensor& value, float scale, kvcache::Cache* cache = nullptr);

Tensor attentionWithSinks(Context& ctx, const Tensor& query, const Tensor& key,
                         const Tensor& value, const Tensor& sinks, float scale,
                         kvcache::Cache* cache = nullptr);

} // namespace nn
} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_NN_ATTENTION_H

#endif // __cplusplus