#ifndef DUOROU_ML_NN_LAYER_NORM_H
#define DUOROU_ML_NN_LAYER_NORM_H

#include "../context.h"
#include "../tensor.h"

namespace duorou {
namespace ml {
namespace nn {

// Layer Normalization层
class LayerNorm {
public:
  // 构造函数
  LayerNorm(int64_t normalizedShape, float eps = 1e-5f,
            bool elementwiseAffine = true);
  LayerNorm(const std::vector<int64_t> &normalizedShape, float eps = 1e-5f,
            bool elementwiseAffine = true);

  // 拷贝和移动
  LayerNorm(const LayerNorm &other) = delete;
  LayerNorm &operator=(const LayerNorm &other) = delete;
  LayerNorm(LayerNorm &&other) noexcept;
  LayerNorm &operator=(LayerNorm &&other) noexcept;

  ~LayerNorm() = default;

  // 前向传播
  Tensor forward(Context &ctx, const Tensor &input);

  // 参数访问
  const Tensor &getWeight() const { return weight_; }
  const Tensor &getBias() const { return bias_; }
  Tensor &getWeight() { return weight_; }
  Tensor &getBias() { return bias_; }

  // 参数初始化
  void initializeWeights(Context &ctx);
  void initializeBias(Context &ctx);

  // 层信息
  const std::vector<int64_t> &getNormalizedShape() const {
    return normalizedShape_;
  }
  float getEps() const { return eps_; }
  bool hasElementwiseAffine() const { return elementwiseAffine_; }

  // 参数统计
  int64_t getParameterCount() const;

private:
  std::vector<int64_t> normalizedShape_;
  float eps_;
  bool elementwiseAffine_;

  Tensor weight_; // [normalizedShape...]
  Tensor bias_;   // [normalizedShape...]

  // 计算归一化维度的总大小
  int64_t getNormalizedSize() const;
};

// RMS Normalization层（用于某些Transformer变体）
class RMSNorm {
public:
  // 构造函数
  RMSNorm(int64_t normalizedShape, float eps = 1e-6f);
  RMSNorm(const std::vector<int64_t> &normalizedShape, float eps = 1e-6f);

  // 拷贝和移动
  RMSNorm(const RMSNorm &other) = delete;
  RMSNorm &operator=(const RMSNorm &other) = delete;
  RMSNorm(RMSNorm &&other) noexcept;
  RMSNorm &operator=(RMSNorm &&other) noexcept;

  ~RMSNorm() = default;

  // 前向传播
  Tensor forward(Context &ctx, const Tensor &input);

  // 参数访问
  const Tensor &getWeight() const { return weight_; }
  Tensor &getWeight() { return weight_; }

  // 参数初始化
  void initializeWeights(Context &ctx);

  // 层信息
  const std::vector<int64_t> &getNormalizedShape() const {
    return normalizedShape_;
  }
  float getEps() const { return eps_; }

  // 参数统计
  int64_t getParameterCount() const;

private:
  std::vector<int64_t> normalizedShape_;
  float eps_;

  Tensor weight_; // [normalizedShape...]

  // 计算归一化维度的总大小
  int64_t getNormalizedSize() const;
};

} // namespace nn
} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_NN_LAYER_NORM_H