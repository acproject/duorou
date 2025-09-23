#include "layer_norm.h"
#include <cmath>
#include <numeric>

namespace duorou {
namespace ml {
namespace nn {

// LayerNorm implementation
LayerNorm::LayerNorm(int64_t normalizedShape, float eps, bool elementwiseAffine)
    : normalizedShape_({normalizedShape}), eps_(eps),
      elementwiseAffine_(elementwiseAffine) {

  if (elementwiseAffine_) {
    weight_ = Tensor(normalizedShape_);
    bias_ = Tensor(normalizedShape_);
  }
}

LayerNorm::LayerNorm(const std::vector<int64_t> &normalizedShape, float eps,
                     bool elementwiseAffine)
    : normalizedShape_(normalizedShape), eps_(eps),
      elementwiseAffine_(elementwiseAffine) {

  if (elementwiseAffine_) {
    weight_ = Tensor(normalizedShape_);
    bias_ = Tensor(normalizedShape_);
  }
}

LayerNorm::LayerNorm(LayerNorm &&other) noexcept
    : normalizedShape_(std::move(other.normalizedShape_)), eps_(other.eps_),
      elementwiseAffine_(other.elementwiseAffine_),
      weight_(std::move(other.weight_)), bias_(std::move(other.bias_)) {}

LayerNorm &LayerNorm::operator=(LayerNorm &&other) noexcept {
  if (this != &other) {
    normalizedShape_ = std::move(other.normalizedShape_);
    eps_ = other.eps_;
    elementwiseAffine_ = other.elementwiseAffine_;
    weight_ = std::move(other.weight_);
    bias_ = std::move(other.bias_);
  }
  return *this;
}

Tensor LayerNorm::forward(Context &ctx, const Tensor &input) {
  // Calculate mean
  Tensor mean = input.mean(ctx, -1, true);

  // Calculate variance
  Tensor centered = input.sub(ctx, mean);
  Tensor squared = centered.mul(ctx, centered);
  Tensor variance = squared.mean(ctx, -1, true);

  // Add epsilon to avoid division by zero
  Tensor eps_tensor = Tensor::ones(variance.shape(), variance.dtype());
  eps_tensor.allocate();
  // Manually set epsilon value
  float *eps_data = static_cast<float *>(eps_tensor.data());
  for (int64_t i = 0; i < eps_tensor.numel(); ++i) {
    eps_data[i] = eps_;
  }
  Tensor var_with_eps = variance.add(ctx, eps_tensor);

  // Calculate standard deviation (manual sqrt implementation)
  Tensor std_dev = Tensor::zeros(var_with_eps.shape(), var_with_eps.dtype());
  std_dev.allocate();
  float *var_data = static_cast<float *>(var_with_eps.data());
  float *std_data = static_cast<float *>(std_dev.data());
  for (int64_t i = 0; i < var_with_eps.numel(); ++i) {
    std_data[i] = std::sqrt(var_data[i]);
  }

  // Normalization
  Tensor normalized = centered.div(ctx, std_dev);

  // Apply weight and bias
  Tensor result = normalized.mul(ctx, weight_);
  if (bias_.data()) {
    result = result.add(ctx, bias_);
  }

  return result;
}

void LayerNorm::initializeWeights(Context &ctx) {
  if (elementwiseAffine_) {
    weight_.allocate();

    // Initialize to 1
    float *weightData = weight_.data<float>();
    int64_t numel = weight_.numel();
    for (int64_t i = 0; i < numel; ++i) {
      weightData[i] = 1.0f;
    }
  }
}

void LayerNorm::initializeBias(Context &ctx) {
  if (elementwiseAffine_) {
    bias_.allocate();

    // Initialize to 0
    float *biasData = bias_.data<float>();
    int64_t numel = bias_.numel();
    for (int64_t i = 0; i < numel; ++i) {
      biasData[i] = 0.0f;
    }
  }
}

int64_t LayerNorm::getParameterCount() const {
  if (elementwiseAffine_) {
    return weight_.numel() + bias_.numel();
  }
  return 0;
}

int64_t LayerNorm::getNormalizedSize() const {
  return std::accumulate(normalizedShape_.begin(), normalizedShape_.end(), 1LL,
                         std::multiplies<int64_t>());
}

// RMSNorm implementation
RMSNorm::RMSNorm(int64_t normalizedShape, float eps)
    : normalizedShape_({normalizedShape}), eps_(eps),
      weight_(normalizedShape_) {}

RMSNorm::RMSNorm(const std::vector<int64_t> &normalizedShape, float eps)
    : normalizedShape_(normalizedShape), eps_(eps), weight_(normalizedShape_) {}

RMSNorm::RMSNorm(RMSNorm &&other) noexcept
    : normalizedShape_(std::move(other.normalizedShape_)), eps_(other.eps_),
      weight_(std::move(other.weight_)) {}

RMSNorm &RMSNorm::operator=(RMSNorm &&other) noexcept {
  if (this != &other) {
    normalizedShape_ = std::move(other.normalizedShape_);
    eps_ = other.eps_;
    weight_ = std::move(other.weight_);
  }
  return *this;
}

Tensor RMSNorm::forward(Context &ctx, const Tensor &input) {
  // RMS normalization: x / sqrt(mean(x^2) + eps) * weight

  // Calculate square
  Tensor squared = input.mul(ctx, input);

  // Calculate mean
  Tensor mean_squared = squared.mean(ctx, -1, true);

  // Add epsilon
  Tensor eps_tensor = Tensor::ones(mean_squared.shape(), mean_squared.dtype());
  eps_tensor.allocate();
  float *eps_data = static_cast<float *>(eps_tensor.data());
  for (int64_t i = 0; i < eps_tensor.numel(); ++i) {
    eps_data[i] = eps_;
  }
  Tensor mean_with_eps = mean_squared.add(ctx, eps_tensor);

  // Calculate RMS (manual sqrt implementation)
  Tensor rms = Tensor::zeros(mean_with_eps.shape(), mean_with_eps.dtype());
  rms.allocate();
  float *mean_data = static_cast<float *>(mean_with_eps.data());
  float *rms_data = static_cast<float *>(rms.data());
  for (int64_t i = 0; i < mean_with_eps.numel(); ++i) {
    rms_data[i] = std::sqrt(mean_data[i]);
  }

  // Normalization
  Tensor normalized = input.div(ctx, rms);

  // Apply weight
  return normalized.mul(ctx, weight_);
}

void RMSNorm::initializeWeights(Context &ctx) {
  weight_.allocate();

  // Initialize to 1
  float *weightData = weight_.data<float>();
  int64_t numel = weight_.numel();
  for (int64_t i = 0; i < numel; ++i) {
    weightData[i] = 1.0f;
  }
}

int64_t RMSNorm::getParameterCount() const { return weight_.numel(); }

int64_t RMSNorm::getNormalizedSize() const {
  return std::accumulate(normalizedShape_.begin(), normalizedShape_.end(), 1LL,
                         std::multiplies<int64_t>());
}

} // namespace nn
} // namespace ml
} // namespace duorou