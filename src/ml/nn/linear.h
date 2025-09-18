#ifndef DUOROU_ML_NN_LINEAR_H
#define DUOROU_ML_NN_LINEAR_H

#include "../tensor.h"
#include "../context.h"

namespace duorou {
namespace ml {
namespace nn {

// 线性层（全连接层）
class Linear {
public:
    // 构造函数
    Linear(int64_t inFeatures, int64_t outFeatures, bool bias = true);
    
    // 拷贝和移动
    Linear(const Linear& other) = delete;
    Linear& operator=(const Linear& other) = delete;
    Linear(Linear&& other) noexcept;
    Linear& operator=(Linear&& other) noexcept;
    
    ~Linear() = default;
    
    // 前向传播
    Tensor forward(Context& ctx, const Tensor& input);
    
    // 参数访问
    const Tensor& getWeight() const { return weight_; }
    const Tensor& getBias() const { return bias_; }
    Tensor& getWeight() { return weight_; }
    Tensor& getBias() { return bias_; }
    
    // 参数初始化
    void initializeWeights(Context& ctx, const std::string& method = "xavier_uniform");
    void initializeBias(Context& ctx, float value = 0.0f);
    
    // 层信息
    int64_t getInFeatures() const { return inFeatures_; }
    int64_t getOutFeatures() const { return outFeatures_; }
    bool hasBias() const { return hasBias_; }
    
    // 参数统计
    int64_t getParameterCount() const;
    
private:
    int64_t inFeatures_;
    int64_t outFeatures_;
    bool hasBias_;
    
    Tensor weight_;  // [outFeatures, inFeatures]
    Tensor bias_;    // [outFeatures]
};

// 批量线性层（用于专家混合等场景）
class LinearBatch {
public:
    LinearBatch(int64_t inFeatures, int64_t outFeatures, int64_t batchSize, bool bias = true);
    
    // 前向传播，indices指定使用哪个专家
    Tensor forward(Context& ctx, const Tensor& input, const Tensor& indices);
    
    // 参数访问
    const Tensor& getWeight() const { return weight_; }
    const Tensor& getBias() const { return bias_; }
    
    // 参数初始化
    void initializeWeights(Context& ctx, const std::string& method = "xavier_uniform");
    void initializeBias(Context& ctx, float value = 0.0f);
    
    // 层信息
    int64_t getInFeatures() const { return inFeatures_; }
    int64_t getOutFeatures() const { return outFeatures_; }
    int64_t getBatchSize() const { return batchSize_; }
    bool hasBias() const { return hasBias_; }
    
private:
    int64_t inFeatures_;
    int64_t outFeatures_;
    int64_t batchSize_;
    bool hasBias_;
    
    Tensor weight_;  // [batchSize, outFeatures, inFeatures]
    Tensor bias_;    // [batchSize, outFeatures]
};

} // namespace nn
} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_NN_LINEAR_H