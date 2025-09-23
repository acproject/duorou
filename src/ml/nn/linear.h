#ifndef DUOROU_ML_NN_LINEAR_H
#define DUOROU_ML_NN_LINEAR_H

#include "../tensor.h"
#include "../context.h"

namespace duorou {
namespace ml {
namespace nn {

// Linear layer (fully connected layer)
class Linear {
public:
    // Constructor
    Linear(int64_t inFeatures, int64_t outFeatures, bool bias = true);
    
    // Copy and move
    Linear(const Linear& other) = delete;
    Linear& operator=(const Linear& other) = delete;
    Linear(Linear&& other) noexcept;
    Linear& operator=(Linear&& other) noexcept;
    
    ~Linear() = default;
    
    // Forward propagation
    Tensor forward(Context& ctx, const Tensor& input);
    
    // Parameter access
    const Tensor& getWeight() const { return weight_; }
    const Tensor& getBias() const { return bias_; }
    Tensor& getWeight() { return weight_; }
    Tensor& getBias() { return bias_; }
    
    // Parameter initialization
    void initializeWeights(Context& ctx, const std::string& method = "xavier_uniform");
    void initializeBias(Context& ctx, float value = 0.0f);
    
    // Layer information
    int64_t getInFeatures() const { return inFeatures_; }
    int64_t getOutFeatures() const { return outFeatures_; }
    bool hasBias() const { return hasBias_; }
    
    // Parameter statistics
    int64_t getParameterCount() const;
    
    // Backend management
    void setBackend(Backend* backend);
    
private:
    int64_t inFeatures_;
    int64_t outFeatures_;
    bool hasBias_;
    
    Tensor weight_;  // [outFeatures, inFeatures]
    Tensor bias_;    // [outFeatures]
};

// Batch linear layer for efficient batch processing
class LinearBatch {
public:
    LinearBatch(int64_t inFeatures, int64_t outFeatures, int64_t batchSize, bool bias = true);
    
    // Forward propagation with batch indices
    Tensor forward(Context& ctx, const Tensor& input, const Tensor& indices);
    
    // Parameter access
    const Tensor& getWeight() const { return weight_; }
    const Tensor& getBias() const { return bias_; }
    
    // Parameter initialization
    void initializeWeights(Context& ctx, const std::string& method = "xavier_uniform");
    void initializeBias(Context& ctx, float value = 0.0f);
    
    // Layer information
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