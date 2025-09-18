#ifndef DUOROU_ML_NN_ACTIVATION_H
#define DUOROU_ML_NN_ACTIVATION_H

#include "../tensor.h"
#include "../context.h"

namespace duorou {
namespace ml {
namespace nn {

// ReLU激活函数
class ReLU {
public:
    ReLU() = default;
    Tensor forward(Context& ctx, const Tensor& input);
};

// GELU激活函数
class GELU {
public:
    GELU() = default;
    Tensor forward(Context& ctx, const Tensor& input);
};

// SiLU (Swish)激活函数
class SiLU {
public:
    SiLU() = default;
    Tensor forward(Context& ctx, const Tensor& input);
};

// Tanh激活函数
class Tanh {
public:
    Tanh() = default;
    Tensor forward(Context& ctx, const Tensor& input);
};

// Sigmoid激活函数
class Sigmoid {
public:
    Sigmoid() = default;
    Tensor forward(Context& ctx, const Tensor& input);
};

// Softmax激活函数
class Softmax {
public:
    Softmax(int dim = -1) : dim_(dim) {}
    Tensor forward(Context& ctx, const Tensor& input);
    
private:
    int dim_;
};

// LeakyReLU激活函数
class LeakyReLU {
public:
    LeakyReLU(float negativeSlope = 0.01f) : negativeSlope_(negativeSlope) {}
    Tensor forward(Context& ctx, const Tensor& input);
    
private:
    float negativeSlope_;
};

// 激活函数工厂
class ActivationFactory {
public:
    enum class Type {
        RELU,
        GELU,
        SILU,
        TANH,
        SIGMOID,
        SOFTMAX,
        LEAKY_RELU
    };
    
    static std::unique_ptr<class ActivationBase> create(Type type);
    static Type stringToType(const std::string& typeStr);
    static std::string typeToString(Type type);
};

// 激活函数基类
class ActivationBase {
public:
    virtual ~ActivationBase() = default;
    virtual Tensor forward(Context& ctx, const Tensor& input) = 0;
};

// 具体激活函数实现类
template<typename ActivationType>
class ActivationImpl : public ActivationBase {
public:
    template<typename... Args>
    ActivationImpl(Args&&... args) : activation_(std::forward<Args>(args)...) {}
    
    Tensor forward(Context& ctx, const Tensor& input) override {
        return activation_.forward(ctx, input);
    }
    
private:
    ActivationType activation_;
};

} // namespace nn
} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_NN_ACTIVATION_H