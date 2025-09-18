#pragma once

#include "base_model.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace duorou {
namespace model {

// Grid structure for image patches
struct Grid {
    size_t temporal = 1;  // For video, usually 1 for images
    size_t height = 0;    // Number of patches in height
    size_t width = 0;     // Number of patches in width
    
    Grid() = default;
    Grid(size_t h, size_t w, size_t t = 1) : temporal(t), height(h), width(w) {}
    
    size_t totalPatches() const { return temporal * height * width; }
};

// Vision model configuration
struct VisionModelOptions {
    size_t hiddenSize = 1024;
    size_t numHeads = 16;
    size_t numLayers = 24;
    size_t patchSize = 14;
    size_t imageSize = 224;
    size_t numChannels = 3;
    size_t temporalPatchSize = 2;
    size_t spatialMergeSize = 2;
    float layerNormEps = 1e-6f;
    
    // Computed properties
    size_t patchDim() const { 
        return numChannels * temporalPatchSize * patchSize * patchSize; 
    }
    size_t numPatches() const { 
        return (imageSize / patchSize) * (imageSize / patchSize); 
    }
};

// Vision attention layer
class VisionAttention {
public:
    VisionAttention(const VisionModelOptions& options);
    ~VisionAttention() = default;
    
    // Forward pass with optional attention mask
    std::vector<float> forward(
        const std::vector<float>& input,
        const std::vector<float>& attentionMask = {}
    );
    
    // Load attention weights
    bool loadWeights(const std::string& weightsPath, size_t layerIndex);
    
private:
    VisionModelOptions options_;
    bool weightsLoaded_ = false;
    
    // Attention weight matrices
    std::vector<float> queryWeights_;
    std::vector<float> keyWeights_;
    std::vector<float> valueWeights_;
    std::vector<float> outputWeights_;
    std::vector<float> queryBias_;
    std::vector<float> keyBias_;
    std::vector<float> valueBias_;
    std::vector<float> outputBias_;
    
    // Helper methods
    std::vector<float> multiHeadAttention(
        const std::vector<float>& query,
        const std::vector<float>& key,
        const std::vector<float>& value,
        const std::vector<float>& mask = {}
    );
};

// Vision MLP (Multi-Layer Perceptron)
class VisionMLP {
public:
    VisionMLP(const VisionModelOptions& options);
    ~VisionMLP() = default;
    
    // Forward pass
    std::vector<float> forward(const std::vector<float>& input);
    
    // Load MLP weights
    bool loadWeights(const std::string& weightsPath, size_t layerIndex);
    
private:
    VisionModelOptions options_;
    bool weightsLoaded_ = false;
    
    // MLP weight matrices
    std::vector<float> fc1Weights_;
    std::vector<float> fc2Weights_;
    std::vector<float> fc1Bias_;
    std::vector<float> fc2Bias_;
    
    // Activation function (GELU)
    std::vector<float> gelu(const std::vector<float>& input);
};

// Vision transformer layer
class VisionTransformerLayer {
public:
    VisionTransformerLayer(const VisionModelOptions& options);
    ~VisionTransformerLayer() = default;
    
    // Forward pass
    std::vector<float> forward(
        const std::vector<float>& input,
        const std::vector<float>& attentionMask = {}
    );
    
    // Load layer weights
    bool loadWeights(const std::string& weightsPath, size_t layerIndex);
    
private:
    VisionModelOptions options_;
    std::unique_ptr<VisionAttention> attention_;
    std::unique_ptr<VisionMLP> mlp_;
    
    // Layer normalization weights
    std::vector<float> layerNorm1Weights_;
    std::vector<float> layerNorm1Bias_;
    std::vector<float> layerNorm2Weights_;
    std::vector<float> layerNorm2Bias_;
    
    // Helper methods
    std::vector<float> layerNorm(
        const std::vector<float>& input,
        const std::vector<float>& weights,
        const std::vector<float>& bias,
        float eps = 1e-6f
    );
};

// Rotary Position Embedding for vision
class VisionRotaryEmbedding {
public:
    VisionRotaryEmbedding(size_t dim, size_t maxSeqLen = 10000);
    ~VisionRotaryEmbedding() = default;
    
    // Apply rotary embedding to input
    std::vector<float> apply(
        const std::vector<float>& input,
        const std::vector<size_t>& positions
    );
    
private:
    size_t dim_;
    size_t maxSeqLen_;
    std::vector<float> cosCache_;
    std::vector<float> sinCache_;
    
    void buildCache();
    std::vector<float> rotateHalf(const std::vector<float>& input);
};

// Main Qwen Vision Model
class QwenVisionModel : public VisionModel {
public:
    QwenVisionModel();
    explicit QwenVisionModel(const VisionModelOptions& options);
    ~QwenVisionModel() override = default;
    
    // VisionModel interface implementation
    std::vector<float> processImage(const std::vector<uint8_t>& imageData) override;
    std::pair<size_t, size_t> getImageFeatureDims() const override;
    bool isLoaded() const override { return initialized_; }
    
    // Qwen-specific methods
    bool initialize(const std::string& configPath);
    bool loadModel(const std::string& modelPath);
    void setOptions(const VisionModelOptions& options);
    const VisionModelOptions& getOptions() const { return options_; }
    
    // Forward pass with grid information
    std::vector<float> forward(
        const std::vector<float>& pixelValues,
        const Grid& grid
    );
    
    // Patch embedding
    std::vector<float> patchEmbedding(const std::vector<float>& pixelValues);
    
    // Position embedding
    std::vector<float> positionEmbedding(
        const std::vector<float>& embeddings,
        const Grid& grid
    );
    
    // Create attention mask for block diagonal attention
    std::vector<float> createBlockDiagonalMask(
        size_t seqLength,
        const std::vector<size_t>& bounds
    );
    
private:
    VisionModelOptions options_;
    bool initialized_ = false;
    std::vector<std::unique_ptr<VisionTransformerLayer>> layers_;
    
    // Embedding layers
    std::vector<float> patchEmbeddingWeights_;
    std::vector<float> patchEmbeddingBias_;
    std::vector<float> positionEmbeddingWeights_;
    
    // Final layer norm
    std::vector<float> finalLayerNormWeights_;
    std::vector<float> finalLayerNormBias_;
    
    // Rotary position embedding
    std::unique_ptr<VisionRotaryEmbedding> rotaryEmbedding_;
    
    // Helper methods
    bool loadConfig(const std::string& configPath);
    bool loadWeights(const std::string& weightsPath);
    std::vector<float> layerNorm(
        const std::vector<float>& input,
        const std::vector<float>& weights,
        const std::vector<float>& bias,
        float eps = 1e-6f
    );
    
    // Image preprocessing
    std::vector<float> preprocessImage(const std::vector<uint8_t>& imageData);
    Grid calculateGrid(size_t imageHeight, size_t imageWidth);
};

// Factory function for creating Qwen vision models
std::unique_ptr<VisionModel> createQwenVisionModel(const std::string& configPath);

} // namespace model
} // namespace duorou