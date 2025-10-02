#include "linear.h"
#include <ggml.h>
#include <cmath>
#include <random>

namespace duorou {
namespace ml {
namespace nn {

// Linear implementation
Linear::Linear(int64_t inFeatures, int64_t outFeatures, bool bias)
    : inFeatures_(inFeatures), outFeatures_(outFeatures), hasBias_(bias),
      weight_({outFeatures, inFeatures}),
      bias_(hasBias_ ? std::vector<int64_t>{outFeatures}
                     : std::vector<int64_t>{}) {}

Linear::Linear(Linear &&other) noexcept
    : inFeatures_(other.inFeatures_), outFeatures_(other.outFeatures_),
      hasBias_(other.hasBias_), weight_(std::move(other.weight_)),
      bias_(std::move(other.bias_)) {}

Linear &Linear::operator=(Linear &&other) noexcept {
  if (this != &other) {
    inFeatures_ = other.inFeatures_;
    outFeatures_ = other.outFeatures_;
    hasBias_ = other.hasBias_;
    weight_ = std::move(other.weight_);
    bias_ = std::move(other.bias_);
  }
  return *this;
}

Tensor Linear::forward(Context &ctx, const Tensor &input) {
  // Perform matrix multiplication: output = input @ weight.T
  // Tensor output = input.matmul(ctx, weight_.transpose(0, 1));

  // // Add bias if available
  // if (hasBias_) {
  //     output = output.add(ctx, bias_);
  // }

  // return output;

  ggml_context *gctx = ctx.ggml_ctx();
  if (!gctx) {
    Tensor output = input.matmul(ctx, weight_.transpose(0, 1));
    return hasBias_ ? output.add(ctx, bias_) : output;
  }

  // use ggml path
  ggml_tensor *gg_in = input.to_ggml(gctx);
  ggml_tensor *gg_wt = weight_.to_ggml(gctx);
  ggml_tensor *gg_out = ggml_mul_mat(gctx, gg_wt, gg_in);

  if (hasBias_) {
    ggml_tensor *gg_b = bias_.to_ggml(gctx);
    // Broadcast bias across the second dimension to match gg_out shape
    ggml_tensor *gg_b_rep = ggml_repeat(gctx, gg_b, gg_out);
    gg_out = ggml_add(gctx, gg_out, gg_b_rep);
  }

  // create graph and compute it
  struct ggml_cgraph *gf = ggml_new_graph(gctx);
  ggml_build_forward_expand(gf, gg_out);
  ctx.compute(gf);

  Tensor host_out;
  host_out.from_ggml(gg_out);
  return host_out;
}

void Linear::initializeWeights(Context & /*ctx*/, const std::string &method) {
  weight_.allocate();

  if (method == "xavier_uniform") {
    // Xavier uniform initialization
    float limit = std::sqrt(6.0f / (inFeatures_ + outFeatures_));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);

    // Fill weight data
    float *weightData = weight_.data<float>();
    int64_t numel = weight_.numel();
    for (int64_t i = 0; i < numel; ++i) {
      weightData[i] = dis(gen);
    }
  } else if (method == "kaiming_uniform") {
    // Kaiming uniform initialization
    float limit = std::sqrt(6.0f / inFeatures_);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);

    float *weightData = weight_.data<float>();
    int64_t numel = weight_.numel();
    for (int64_t i = 0; i < numel; ++i) {
      weightData[i] = dis(gen);
    }
  }
}

void Linear::initializeBias(Context & /*ctx*/, float value) {
  if (hasBias_) {
    bias_.allocate();

    float *biasData = bias_.data<float>();
    int64_t numel = bias_.numel();
    for (int64_t i = 0; i < numel; ++i) {
      biasData[i] = value;
    }
  }
}

int64_t Linear::getParameterCount() const {
  int64_t count = weight_.numel();
  if (hasBias_) {
    count += bias_.numel();
  }
  return count;
}

// LinearBatch implementation
LinearBatch::LinearBatch(int64_t inFeatures, int64_t outFeatures,
                         int64_t batchSize, bool bias)
    : inFeatures_(inFeatures), outFeatures_(outFeatures), batchSize_(batchSize),
      hasBias_(bias), weight_({batchSize, outFeatures, inFeatures}),
      bias_(hasBias_ ? std::vector<int64_t>{batchSize, outFeatures}
                     : std::vector<int64_t>{}) {}

Tensor LinearBatch::forward(Context &ctx, const Tensor &input,
                            const Tensor & /*indices*/) {
  // Simplified implementation: select corresponding batch weights and biases
  // Actual implementation needs to select appropriate parameters based on
  // indices
  Tensor output = input.matmul(ctx, weight_.transpose(1, 2));

  if (hasBias_) {
    output = output.add(ctx, bias_);
  }

  return output;
}

void LinearBatch::initializeWeights(Context & /*ctx*/,
                                    const std::string &method) {
  weight_.allocate();

  if (method == "xavier_uniform") {
    float limit = std::sqrt(6.0f / (inFeatures_ + outFeatures_));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);

    float *data = weight_.data<float>();
    int64_t numel = weight_.numel();
    for (int64_t i = 0; i < numel; ++i) {
      data[i] = dis(gen);
    }
  }
}

void LinearBatch::initializeBias(Context & /*ctx*/, float value) {
  if (hasBias_) {
    bias_.allocate();

    float *data = bias_.data<float>();
    int64_t numel = bias_.numel();
    for (int64_t i = 0; i < numel; ++i) {
      data[i] = value;
    }
  }
}

} // namespace nn
} // namespace ml
} // namespace duorou