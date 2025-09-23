#ifndef DUOROU_ML_NN_ATTENTION_H
#define DUOROU_ML_NN_ATTENTION_H

#include "../tensor.h"
#include "../context.h"
#include "../../kvcache/cache.h"

namespace duorou {
namespace ml {
namespace nn {

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
    Tensor queryWeight_;  // [embedDim, embedDim]
    Tensor keyWeight_;    // [embedDim, kvHeads * headDim]
    Tensor valueWeight_;  // [embedDim, kvHeads * headDim]
    Tensor outputWeight_; // [embedDim, embedDim]
    
    // Bias vectors
    Tensor queryBias_;
    Tensor keyBias_;
    Tensor valueBias_;
    Tensor outputBias_;
    
    // Helper methods
    Tensor scaledDotProductAttention(Context& ctx, const Tensor& q, 
                                   const Tensor& k, const Tensor& v,
                                   const Tensor& mask = {});
    Tensor applyRotaryPositionEmbedding(Context& ctx, const Tensor& tensor, 
                                      int64_t seqLen, int64_t offset = 0);
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