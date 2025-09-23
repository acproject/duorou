#include "linear.h"
#include <random>
#include <cmath>

namespace duorou {
namespace ml {
namespace nn {

// Linear implementation
Linear::Linear(int64_t inFeatures, int64_t outFeatures, bool bias)
    : inFeatures_(inFeatures), outFeatures_(outFeatures), hasBias_(bias),
      weight_({outFeatures, inFeatures}), bias_(hasBias_ ? std::vector<int64_t>{outFeatures} : std::vector<int64_t>{}) {
}

Linear::Linear(Linear&& other) noexcept
    : inFeatures_(other.inFeatures_), outFeatures_(other.outFeatures_), 
      hasBias_(other.hasBias_), weight_(std::move(other.weight_)), bias_(std::move(other.bias_)) {
}

Linear& Linear::operator=(Linear&& other) noexcept {
    if (this != &other) {
        inFeatures_ = other.inFeatures_;
        outFeatures_ = other.outFeatures_;
        hasBias_ = other.hasBias_;
        weight_ = std::move(other.weight_);
        bias_ = std::move(other.bias_);
    }
    return *this;
}

Tensor Linear::forward(Context& ctx, const Tensor& input) {
    // Perform matrix multiplication: output = input @ weight.T
    Tensor output = input.matmul(ctx, weight_.transpose(0, 1));
    
    // Add bias if available
    if (hasBias_) {
        output = output.add(ctx, bias_);
    }
    
    return output;
}

void Linear::initializeWeights(Context& /*ctx*/, const std::string& method) {
    weight_.allocate();
    
    if (method == "xavier_uniform") {
        // Xavier uniform initialization
        float limit = std::sqrt(6.0f / (inFeatures_ + outFeatures_));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-limit, limit);
        
        // Fill weight data
        float* weightData = weight_.data<float>();
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
        
        float* weightData = weight_.data<float>();
        int64_t numel = weight_.numel();
        for (int64_t i = 0; i < numel; ++i) {
            weightData[i] = dis(gen);
        }
    }
}

void Linear::initializeBias(Context& /*ctx*/, float value) {
    if (hasBias_) {
        bias_.allocate();
        
        float* biasData = bias_.data<float>();
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
LinearBatch::LinearBatch(int64_t inFeatures, int64_t outFeatures, int64_t batchSize, bool bias)
    : inFeatures_(inFeatures), outFeatures_(outFeatures), batchSize_(batchSize), hasBias_(bias),
      weight_({batchSize, outFeatures, inFeatures}), 
      bias_(hasBias_ ? std::vector<int64_t>{batchSize, outFeatures} : std::vector<int64_t>{}) {
}

Tensor LinearBatch::forward(Context& ctx, const Tensor& input, const Tensor& /*indices*/) {
    // Simplified implementation: select corresponding batch weights and biases
    // Actual implementation needs to select appropriate parameters based on indices
    Tensor output = input.matmul(ctx, weight_.transpose(1, 2));
    
    if (hasBias_) {
        output = output.add(ctx, bias_);
    }
    
    return output;
}

void LinearBatch::initializeWeights(Context& /*ctx*/, const std::string& method) {
    weight_.allocate();
    
    if (method == "xavier_uniform") {
        float limit = std::sqrt(6.0f / (inFeatures_ + outFeatures_));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-limit, limit);
        
        float* data = weight_.data<float>();
        int64_t numel = weight_.numel();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = dis(gen);
        }
    }
}

void LinearBatch::initializeBias(Context& /*ctx*/, float value) {
    if (hasBias_) {
        bias_.allocate();
        
        float* data = bias_.data<float>();
        int64_t numel = bias_.numel();
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = value;
        }
    }
}

} // namespace nn
} // namespace ml
} // namespace duorou