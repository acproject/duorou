#include "attention.h"
#include <cmath>

namespace duorou {
namespace ml {
namespace nn {

// MultiHeadAttention 实现
MultiHeadAttention::MultiHeadAttention(int64_t embedDim, int64_t numHeads, 
                                     int64_t kvHeads, bool bias, float dropout)
    : embedDim_(embedDim), numHeads_(numHeads), 
      kvHeads_(kvHeads == -1 ? numHeads : kvHeads), 
      headDim_(embedDim / numHeads), hasBias_(bias), dropout_(dropout),
      queryWeight_({embedDim, embedDim}),
      keyWeight_({embedDim, kvHeads_ * headDim_}),
      valueWeight_({embedDim, kvHeads_ * headDim_}),
      outputWeight_({embedDim, embedDim}) {
    
    if (hasBias_) {
        queryBias_ = Tensor({embedDim});
        keyBias_ = Tensor({kvHeads_ * headDim_});
        valueBias_ = Tensor({kvHeads_ * headDim_});
        outputBias_ = Tensor({embedDim});
    }
}

Tensor MultiHeadAttention::forward(Context& ctx, const Tensor& query, 
                                  const Tensor& key, const Tensor& value,
                                  kvcache::Cache* cache, const Tensor& mask) {
    // 处理3D张量：[batch_size, seq_len, embed_dim]
    // 为了简化实现，我们将3D张量重塑为2D进行矩阵乘法
    auto queryShape = query.shape();
    bool is3D = queryShape.size() == 3;
    
    Tensor q, k, v;
    if (is3D) {
        // 重塑为2D: [batch_size * seq_len, embed_dim]
        int64_t batchSeq = queryShape[0] * queryShape[1];
        int64_t embedDim = queryShape[2];
        
        Tensor query2D = query.reshape({batchSeq, embedDim});
        q = query2D.matmul(ctx, queryWeight_);
        
        if (key.data() && value.data()) {
            Tensor key2D = key.reshape({batchSeq, embedDim});
            Tensor value2D = value.reshape({batchSeq, embedDim});
            k = key2D.matmul(ctx, keyWeight_);
            v = value2D.matmul(ctx, valueWeight_);
        }
    } else {
        // 2D张量直接处理
        q = query.matmul(ctx, queryWeight_);
        if (key.data() && value.data()) {
            k = key.matmul(ctx, keyWeight_);
            v = value.matmul(ctx, valueWeight_);
        }
    }
    
    if (hasBias_) {
        q = q.add(ctx, queryBias_);
        if (k.data() && v.data()) {
            k = k.add(ctx, keyBias_);
            v = v.add(ctx, valueBias_);
        }
    }
    
    // 如果有缓存，存储 K, V
    if (cache && k.data() && v.data()) {
        // cache->put(ctx, k, v); // 需要实现缓存接口
    }
    
    // 执行注意力计算
    Tensor attnOutput = scaledDotProductAttention(ctx, q, k, v, mask);
    
    // 输出投影 - 需要在2D形式下进行
    Tensor output;
    if (is3D) {
        // 保持2D形式进行输出投影
        output = attnOutput.matmul(ctx, outputWeight_);
        if (hasBias_) {
            output = output.add(ctx, outputBias_);
        }
        // 然后重塑回3D
        output = output.reshape({queryShape[0], queryShape[1], output.shape()[1]});
    } else {
        // 2D张量直接处理
        output = attnOutput.matmul(ctx, outputWeight_);
        if (hasBias_) {
            output = output.add(ctx, outputBias_);
        }
    }
    
    return output;
}

Tensor MultiHeadAttention::forwardWithSinks(Context& ctx, const Tensor& query,
                                           const Tensor& key, const Tensor& value,
                                           const Tensor& sinks, float scale,
                                           kvcache::Cache* cache) {
    // 这是一个简化的实现，实际需要处理 sink tokens
    return forward(ctx, query, key, value, cache);
}

void MultiHeadAttention::initializeWeights(Context& ctx, const std::string& method) {
    // 初始化所有权重矩阵
    queryWeight_.allocate();
    keyWeight_.allocate();
    valueWeight_.allocate();
    outputWeight_.allocate();
    
    if (hasBias_) {
        queryBias_.allocate();
        keyBias_.allocate();
        valueBias_.allocate();
        outputBias_.allocate();
    }
    
    // 使用 Xavier 初始化（简化实现）
    // 实际应该根据 method 参数选择不同的初始化方法
}

Tensor MultiHeadAttention::scaledDotProductAttention(Context& ctx, const Tensor& q, 
                                                   const Tensor& k, const Tensor& v,
                                                   const Tensor& mask) {
    // 计算注意力分数: scores = q @ k.T / sqrt(d_k)
    float scale = 1.0f / std::sqrt(static_cast<float>(headDim_));
    
    // 这里需要实现实际的注意力计算
    // 简化实现，返回 v（实际需要完整的注意力机制）
    return v;
}

Tensor MultiHeadAttention::applyRotaryPositionEmbedding(Context& ctx, const Tensor& tensor, 
                                                       int64_t seqLen, int64_t offset) {
    // RoPE 实现（简化）
    return tensor;
}

// 全局注意力函数
Tensor attention(Context& ctx, const Tensor& query, const Tensor& key, 
                const Tensor& value, float scale, kvcache::Cache* cache) {
    // 简化的注意力实现
    // 实际需要完整的 scaled dot-product attention
    return value;
}

Tensor attentionWithSinks(Context& ctx, const Tensor& query, const Tensor& key,
                         const Tensor& value, const Tensor& sinks, float scale,
                         kvcache::Cache* cache) {
    // 带 sink tokens 的注意力实现
    return attention(ctx, query, key, value, scale, cache);
}

} // namespace nn
} // namespace ml
} // namespace duorou