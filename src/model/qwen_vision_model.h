#pragma once

// Strong include guard to assist indexers and mixed-language builds
#ifndef DUOROU_MODEL_QWEN_VISION_MODEL_H
#define DUOROU_MODEL_QWEN_VISION_MODEL_H

// Ensure this header is only parsed by a C++ compiler
#if defined(__cplusplus)

#include "base_model.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <cstddef>

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
    size_t hiddenSize = 1280;
    size_t numHeads = 20; // Qwen2.5VL常见视觉头数接近 hiddenSize/64，后续可从GGUF覆盖
    size_t numLayers = 32;
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

// Vision patch merger: merges 2x2 patches -> concat 4*hidden (5120),
// then RMSNorm + MLP 5120 -> 5120 -> GELU -> 3584
class VisionPatchMerger {
public:
    VisionPatchMerger() = default;
    ~VisionPatchMerger() = default;

    // Configure dimensions; textHidden should match text embed dim (e.g., 3584)
    void configure(size_t visionHidden, size_t textHidden) {
        visionHidden_ = visionHidden;
        textHidden_ = textHidden;
        mergedDim_ = visionHidden_ * 4; // 2x2 spatial merge
        // Init RMSNorm scale
        lnScale_.assign(mergedDim_, 1.0f);
        // Init MLP weights (placeholders; real weights should be loaded via GGUF)
        mlpW1_.assign(mergedDim_ * mergedDim_, 0.0f);
        mlpB1_.assign(mergedDim_, 0.0f);
        mlpW2_.assign(mergedDim_ * textHidden_, 0.0f);
        mlpB2_.assign(textHidden_, 0.0f);
    }

    // Merge per 2x2 blocks in sequence order
    std::vector<float> forward(const std::vector<float>& visionSeq) const {
        if (visionHidden_ == 0 || textHidden_ == 0 || mergedDim_ == 0) return {};
        if (visionSeq.empty() || (visionSeq.size() % visionHidden_) != 0) return {};
        const size_t seq = visionSeq.size() / visionHidden_;
        const size_t group = 4; // 2x2 merge
        const size_t outSeq = seq / group;
        if (outSeq == 0) return {};
        std::vector<float> out(outSeq * textHidden_, 0.0f);
        // 1) concat 4 tokens -> 5120
        std::vector<float> merged(mergedDim_, 0.0f);
        for (size_t t = 0; t < outSeq; ++t) {
            // concat tokens [t*4 + i], i=0..3
            for (size_t i = 0; i < group; ++i) {
                const float* src = &visionSeq[(t*group + i) * visionHidden_];
                float* dst = &merged[i * visionHidden_];
                std::copy(src, src + visionHidden_, dst);
            }
            // 2) RMSNorm
            rmsNormInPlace(merged, lnScale_, 1e-6f);
            // 3) MLP: 5120->5120->GELU->3584
            std::vector<float> h1 = matmulVec(merged, mlpW1_, mergedDim_, mergedDim_);
            addBiasInPlace(h1, mlpB1_);
            geluInPlace(h1);
            std::vector<float> h2 = matmulVec(h1, mlpW2_, mergedDim_, textHidden_);
            addBiasInPlace(h2, mlpB2_);
            std::copy(h2.begin(), h2.end(), &out[t * textHidden_]);
        }
        return out;
    }

private:
    size_t visionHidden_ = 0;
    size_t textHidden_ = 0;
    size_t mergedDim_ = 0; // 4 * visionHidden
    // RMSNorm scale
    std::vector<float> lnScale_;
    // MLP 5120->5120->3584
    std::vector<float> mlpW1_;
    std::vector<float> mlpB1_;
    std::vector<float> mlpW2_;
    std::vector<float> mlpB2_;

    static void rmsNormInPlace(std::vector<float>& x, const std::vector<float>& scale, float eps) {
        const size_t n = x.size();
        double msq = 0.0;
        for (size_t i = 0; i < n; ++i) { msq += double(x[i]) * double(x[i]); }
        msq /= double(n);
        float inv = 1.0f / std::sqrt(float(msq) + eps);
        for (size_t i = 0; i < n; ++i) { x[i] = x[i] * inv * (i < scale.size() ? scale[i] : 1.0f); }
    }
    static void addBiasInPlace(std::vector<float>& x, const std::vector<float>& b) {
        const size_t n = std::min(x.size(), b.size());
        for (size_t i = 0; i < n; ++i) x[i] += b[i];
    }
    static void geluInPlace(std::vector<float>& x) {
        for (auto& v : x) {
            float t = v;
            v = 0.5f * t * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (t + 0.044715f * t * t * t)));
        }
    }
    static std::vector<float> matmulVec(const std::vector<float>& a, const std::vector<float>& w,
                                        size_t inDim, size_t outDim) {
        std::vector<float> y(outDim, 0.0f);
        if (w.size() != inDim * outDim) return y;
        for (size_t o = 0; o < outDim; ++o) {
            double acc = 0.0;
            const float* wp = &w[o * inDim];
            for (size_t i = 0; i < inDim; ++i) acc += double(a[i]) * double(wp[i]);
            y[o] = float(acc);
        }
        return y;
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

#else

// Fallback for C compilation units: provide a harmless typedef so the file
// remains syntactically valid without exposing any C++ constructs.
typedef int duorou_qwen_vision_model_requires_cplusplus;

#endif // defined(__cplusplus)

#endif // DUOROU_MODEL_QWEN_VISION_MODEL_H