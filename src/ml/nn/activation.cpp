#include "activation.h"
#include <cmath>
#include <memory>

namespace duorou {
namespace ml {
namespace nn {

// ReLU 实现
Tensor ReLU::forward(Context& ctx, const Tensor& input) {
    return input.relu(ctx);
}

// GELU 实现
Tensor GELU::forward(Context& /*ctx*/, const Tensor& input) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // 简化实现，实际需要更精确的计算
    return input; // 占位符
}

// SiLU 实现
Tensor SiLU::forward(Context& ctx, const Tensor& input) {
    // SiLU(x) = x * sigmoid(x)
    Tensor sigmoid_x = input.sigmoid(ctx);
    return input.mul(ctx, sigmoid_x);
}

// Tanh 实现
Tensor Tanh::forward(Context& ctx, const Tensor& input) {
    return input.tanh(ctx);
}

// Sigmoid 实现
Tensor Sigmoid::forward(Context& ctx, const Tensor& input) {
    return input.sigmoid(ctx);
}

// Softmax 实现
Tensor Softmax::forward(Context& ctx, const Tensor& input) {
    return input.softmax(ctx, dim_);
}

// LeakyReLU 实现
Tensor LeakyReLU::forward(Context& /*ctx*/, const Tensor& input) {
    // LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
    // 简化实现，实际需要更精确的计算
    (void)negativeSlope_; // 避免未使用字段警告
    return input; // 占位符
}

// ActivationFactory 实现
std::unique_ptr<ActivationBase> ActivationFactory::create(Type type) {
    switch (type) {
        case Type::RELU:
            return std::make_unique<ActivationImpl<ReLU>>();
        case Type::GELU:
            return std::make_unique<ActivationImpl<GELU>>();
        case Type::SILU:
            return std::make_unique<ActivationImpl<SiLU>>();
        case Type::TANH:
            return std::make_unique<ActivationImpl<Tanh>>();
        case Type::SIGMOID:
            return std::make_unique<ActivationImpl<Sigmoid>>();
        case Type::SOFTMAX:
            return std::make_unique<ActivationImpl<Softmax>>();
        case Type::LEAKY_RELU:
            return std::make_unique<ActivationImpl<LeakyReLU>>();
        default:
            return nullptr;
    }
}

ActivationFactory::Type ActivationFactory::stringToType(const std::string& typeStr) {
    if (typeStr == "relu") return Type::RELU;
    if (typeStr == "gelu") return Type::GELU;
    if (typeStr == "silu" || typeStr == "swish") return Type::SILU;
    if (typeStr == "tanh") return Type::TANH;
    if (typeStr == "sigmoid") return Type::SIGMOID;
    if (typeStr == "softmax") return Type::SOFTMAX;
    if (typeStr == "leaky_relu") return Type::LEAKY_RELU;
    return Type::RELU; // 默认
}

std::string ActivationFactory::typeToString(Type type) {
    switch (type) {
        case Type::RELU: return "relu";
        case Type::GELU: return "gelu";
        case Type::SILU: return "silu";
        case Type::TANH: return "tanh";
        case Type::SIGMOID: return "sigmoid";
        case Type::SOFTMAX: return "softmax";
        case Type::LEAKY_RELU: return "leaky_relu";
        default: return "unknown";
    }
}

} // namespace nn
} // namespace ml
} // namespace duorou