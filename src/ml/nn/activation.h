#ifndef DUOROU_ML_NN_ACTIVATION_H
#define DUOROU_ML_NN_ACTIVATION_H

#include "../tensor.h"
#include "../context.h"

namespace duorou {
namespace ml {
namespace nn {

// ReLU activation function
class ReLU {
public:
    ReLU() = default;
    Tensor forward(Context& ctx, const Tensor& input);
};

// GELU activation function
class GELU {
public:
    GELU() = default;
    Tensor forward(Context& ctx, const Tensor& input);
};

// SiLU (Swish) activation function
class SiLU {
public:
    SiLU() = default;
    Tensor forward(Context& ctx, const Tensor& input);
};

// Tanh activation function
class Tanh {
public:
    Tanh() = default;
    Tensor forward(Context& ctx, const Tensor& input);
};

// Sigmoid activation function
class Sigmoid {
public:
    Sigmoid() = default;
    Tensor forward(Context& ctx, const Tensor& input);
};

// Softmax activation function
class Softmax {
public:
    Softmax(int dim = -1) : dim_(dim) {}
    Tensor forward(Context& ctx, const Tensor& input);
    
private:
    int dim_;
};

// LeakyReLU activation function
class LeakyReLU {
public:
    LeakyReLU(float negativeSlope = 0.01f) : negativeSlope_(negativeSlope) {}
    Tensor forward(Context& ctx, const Tensor& input);
    
private:
    float negativeSlope_;
};

// Activation factory for creating activation functions
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

// Base class for polymorphic activation functions
class ActivationBase {
public:
    virtual ~ActivationBase() = default;
    virtual Tensor forward(Context& ctx, const Tensor& input) = 0;
};

// Template implementation for activation functions
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