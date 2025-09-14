#ifndef GGML_FEED_FORWARD_H
#define GGML_FEED_FORWARD_H

#ifdef __cplusplus

#include "base_algorithm.h"
#include <vector>
#include <string>

// Forward declarations for GGML types to avoid include conflicts
struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// 基于ggml的前馈网络实现
class GGMLFeedForward : public IFeedForwardAlgorithm {
public:
    GGMLFeedForward() = default;
    ~GGMLFeedForward() override;

    // 初始化算法
    bool initialize(const ModelConfig &config, const AlgorithmContext &context) override;

    // 获取算法名称
    std::string getName() const override { return "GGMLFeedForward"; }

    // 获取算法版本
    std::string getVersion() const override { return "1.0.0"; }

    // 验证输入张量
    bool validateInput(const Tensor &input) const override;

    // 前馈网络计算 - 使用ggml_compute_forward
    Tensor compute(const Tensor &input, const Tensor &gate_weights,
                   const Tensor &up_weights, const Tensor &down_weights) override;

protected:
    // GGML上下文
    struct ggml_context *ctx_;
    struct ggml_cgraph *gf_;
    
    // 配置参数
    ModelConfig config_;
    AlgorithmContext context_;
    
    // 内部缓存和状态
    std::vector<uint8_t> work_buffer_;
    size_t work_buffer_size_;
    
    // 内部方法
    void cleanup();
    bool allocateWorkBuffer(size_t size);
    
    // GGML辅助方法
    struct ggml_tensor *tensorToGGML(struct ggml_context *ctx, const Tensor &tensor);
    Tensor ggmlToTensor(struct ggml_tensor *ggml_tensor);
    
    // 激活函数
    struct ggml_tensor *applyActivation(struct ggml_context *ctx, struct ggml_tensor *input, const std::string &activation_type = "swiglu");
    
    // 日志方法
    void log(const std::string& level, const std::string& message) const;
};

// SwiGLU前馈网络
class SwiGLUFeedForward : public GGMLFeedForward {
public:
    std::string getName() const override { return "SwiGLUFeedForward"; }
    
    Tensor compute(const Tensor &input, const Tensor &gate_weights,
                   const Tensor &up_weights, const Tensor &down_weights) override;
};

// GELU前馈网络
class GELUFeedForward : public GGMLFeedForward {
public:
    std::string getName() const override { return "GELUFeedForward"; }
    
    Tensor compute(const Tensor &input, const Tensor &gate_weights,
                   const Tensor &up_weights, const Tensor &down_weights) override;
};

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // __cplusplus

#endif // GGML_FEED_FORWARD_H