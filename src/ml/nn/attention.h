#ifndef DUOROU_ML_NN_ATTENTION_H
#define DUOROU_ML_NN_ATTENTION_H

#include "../tensor.h"
#include "../context.h"
#include "../../kvcache/cache.h"

namespace duorou {
namespace ml {
namespace nn {

// 多头注意力机制
class MultiHeadAttention {
public:
    // 构造函数
    MultiHeadAttention(int64_t embedDim, int64_t numHeads, 
                      int64_t kvHeads = -1, bool bias = true, 
                      float dropout = 0.0f);
    
    ~MultiHeadAttention() = default;
    
    // 前向传播
    Tensor forward(Context& ctx, const Tensor& query, 
                  const Tensor& key = Tensor(), const Tensor& value = Tensor(),
                  kvcache::Cache* cache = nullptr, const Tensor& mask = Tensor());
    
    // 带Sink Token的注意力
    Tensor forwardWithSinks(Context& ctx, const Tensor& query,
                           const Tensor& key, const Tensor& value,
                           const Tensor& sinks, float scale,
                           kvcache::Cache* cache = nullptr);
    
    // 参数访问
    const Tensor& getQueryWeight() const { return queryWeight_; }
    const Tensor& getKeyWeight() const { return keyWeight_; }
    const Tensor& getValueWeight() const { return valueWeight_; }
    const Tensor& getOutputWeight() const { return outputWeight_; }
    
    // 参数初始化
    void initializeWeights(Context& ctx, const std::string& method = "xavier_uniform");
    
    // 层信息
    int64_t getEmbedDim() const { return embedDim_; }
    int64_t getNumHeads() const { return numHeads_; }
    int64_t getKVHeads() const { return kvHeads_; }
    int64_t getHeadDim() const { return headDim_; }
    float getDropout() const { return dropout_; }
    
private:
    int64_t embedDim_;
    int64_t numHeads_;
    int64_t kvHeads_;
    int64_t headDim_;
    bool hasBias_;
    float dropout_;
    
    // 权重矩阵
    Tensor queryWeight_;  // [embedDim, embedDim]
    Tensor keyWeight_;    // [embedDim, kvHeads * headDim]
    Tensor valueWeight_;  // [embedDim, kvHeads * headDim]
    Tensor outputWeight_; // [embedDim, embedDim]
    
    // 偏置
    Tensor queryBias_;
    Tensor keyBias_;
    Tensor valueBias_;
    Tensor outputBias_;
    
    // 辅助方法
    Tensor scaledDotProductAttention(Context& ctx, const Tensor& q, 
                                   const Tensor& k, const Tensor& v,
                                   const Tensor& mask = Tensor());
    Tensor applyRotaryPositionEmbedding(Context& ctx, const Tensor& tensor, 
                                      int64_t seqLen, int64_t offset = 0);
};

// 标准注意力函数（类似Ollama的实现）
Tensor attention(Context& ctx, const Tensor& query, const Tensor& key, 
                const Tensor& value, float scale, kvcache::Cache* cache = nullptr);

Tensor attentionWithSinks(Context& ctx, const Tensor& query, const Tensor& key,
                         const Tensor& value, const Tensor& sinks, float scale,
                         kvcache::Cache* cache = nullptr);

} // namespace nn
} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_NN_ATTENTION_H