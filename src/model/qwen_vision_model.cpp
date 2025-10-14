#define _USE_MATH_DEFINES
#include "qwen_vision_model.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>

namespace duorou {
namespace model {

// VisionAttention implementation
VisionAttention::VisionAttention(const VisionModelOptions& options) 
    : options_(options) {
    // Initialize weight matrices with proper sizes
    size_t hiddenSize = options_.hiddenSize;
    
    queryWeights_.resize(hiddenSize * hiddenSize);
    keyWeights_.resize(hiddenSize * hiddenSize);
    valueWeights_.resize(hiddenSize * hiddenSize);
    outputWeights_.resize(hiddenSize * hiddenSize);
    
    queryBias_.resize(hiddenSize);
    keyBias_.resize(hiddenSize);
    valueBias_.resize(hiddenSize);
    outputBias_.resize(hiddenSize);
}

std::vector<float> VisionAttention::forward(
    const std::vector<float>& input,
    const std::vector<float>& attentionMask) {
    
    if (!weightsLoaded_) {
        std::cerr << "Warning: VisionAttention weights not loaded" << std::endl;
        return input; // Return input unchanged if weights not loaded
    }
    
    // Suppress unused parameter warning
    (void)attentionMask;
    
    // TODO: Implement multi-head attention
    // For now, return input unchanged
    return input;
}

bool VisionAttention::loadWeights(const std::string& weightsPath, size_t layerIndex) {
    // TODO: Implement weight loading from file
    weightsLoaded_ = true;
    return true;
}

// VisionMLP implementation
VisionMLP::VisionMLP(const VisionModelOptions& options) 
    : options_(options) {
    size_t hiddenSize = options_.hiddenSize;
    size_t intermediateSize = hiddenSize * 4; // Common ratio for MLP
    
    fc1Weights_.resize(hiddenSize * intermediateSize);
    fc2Weights_.resize(intermediateSize * hiddenSize);
    fc1Bias_.resize(intermediateSize);
    fc2Bias_.resize(hiddenSize);
}

std::vector<float> VisionMLP::forward(const std::vector<float>& input) {
    if (!weightsLoaded_) {
        std::cerr << "Warning: VisionMLP weights not loaded" << std::endl;
        return input;
    }
    
    // TODO: Implement MLP forward pass
    return input;
}

std::vector<float> VisionMLP::gelu(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
    return output;
}

bool VisionMLP::loadWeights(const std::string& weightsPath, size_t layerIndex) {
    // TODO: Implement weight loading
    weightsLoaded_ = true;
    return true;
}

// VisionTransformerLayer implementation
VisionTransformerLayer::VisionTransformerLayer(const VisionModelOptions& options) 
    : options_(options) {
    attention_ = std::make_unique<VisionAttention>(options);
    mlp_ = std::make_unique<VisionMLP>(options);
    
    size_t hiddenSize = options_.hiddenSize;
    layerNorm1Weights_.resize(hiddenSize, 1.0f);
    layerNorm1Bias_.resize(hiddenSize, 0.0f);
    layerNorm2Weights_.resize(hiddenSize, 1.0f);
    layerNorm2Bias_.resize(hiddenSize, 0.0f);
}

std::vector<float> VisionTransformerLayer::forward(
    const std::vector<float>& input,
    const std::vector<float>& attentionMask) {
    
    // Pre-norm architecture
    auto normed1 = layerNorm(input, layerNorm1Weights_, layerNorm1Bias_, options_.layerNormEps);
    auto attnOutput = attention_->forward(normed1, attentionMask);
    
    // Residual connection
    std::vector<float> residual1(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        residual1[i] = input[i] + attnOutput[i];
    }
    
    auto normed2 = layerNorm(residual1, layerNorm2Weights_, layerNorm2Bias_, options_.layerNormEps);
    auto mlpOutput = mlp_->forward(normed2);
    
    // Final residual connection
    std::vector<float> output(residual1.size());
    for (size_t i = 0; i < residual1.size(); ++i) {
        output[i] = residual1[i] + mlpOutput[i];
    }
    
    return output;
}

std::vector<float> VisionTransformerLayer::layerNorm(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    float eps) {
    const size_t hidden = options_.hiddenSize;
    if (hidden == 0) {
        return input;
    }
    if (input.empty()) {
        return {};
    }
    if (input.size() % hidden != 0) {
        // Shape mismatch; avoid out-of-bounds and return input unchanged
        std::cerr << "[WARN] VisionTransformerLayer::layerNorm input size " << input.size()
                  << " not divisible by hidden " << hidden << std::endl;
        return input;
    }

    const bool has_scale = (weights.size() == hidden);
    const bool has_bias = (bias.size() == hidden);
    std::vector<float> output(input.size());
    const size_t seq_len = input.size() / hidden;

    for (size_t t = 0; t < seq_len; ++t) {
        const size_t base = t * hidden;
        // mean
        double mean = 0.0;
        for (size_t i = 0; i < hidden; ++i) {
            mean += static_cast<double>(input[base + i]);
        }
        mean /= static_cast<double>(hidden);
        // variance
        double var = 0.0;
        for (size_t i = 0; i < hidden; ++i) {
            double diff = static_cast<double>(input[base + i]) - mean;
            var += diff * diff;
        }
        var /= static_cast<double>(hidden);
        float invStd = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        for (size_t i = 0; i < hidden; ++i) {
            float scaled = (input[base + i] - static_cast<float>(mean)) * invStd;
            if (has_scale) scaled *= weights[i];
            if (has_bias) scaled += bias[i];
            output[base + i] = scaled;
        }
    }

    return output;
}

bool VisionTransformerLayer::loadWeights(const std::string& weightsPath, size_t layerIndex) {
    // Load weights for attention and MLP
    bool success = true;
    success &= attention_->loadWeights(weightsPath, layerIndex);
    success &= mlp_->loadWeights(weightsPath, layerIndex);
    
    // TODO: Load layer norm weights
    
    return success;
}

// VisionRotaryEmbedding implementation
VisionRotaryEmbedding::VisionRotaryEmbedding(size_t dim, size_t maxSeqLen) 
    : dim_(dim), maxSeqLen_(maxSeqLen) {
    buildCache();
}

void VisionRotaryEmbedding::buildCache() {
    cosCache_.resize(maxSeqLen_ * dim_);
    sinCache_.resize(maxSeqLen_ * dim_);
    
    for (size_t pos = 0; pos < maxSeqLen_; ++pos) {
        for (size_t i = 0; i < dim_; i += 2) {
            float theta = pos / std::pow(10000.0f, static_cast<float>(i) / dim_);
            cosCache_[pos * dim_ + i] = std::cos(theta);
            sinCache_[pos * dim_ + i] = std::sin(theta);
            if (i + 1 < dim_) {
                cosCache_[pos * dim_ + i + 1] = std::cos(theta);
                sinCache_[pos * dim_ + i + 1] = std::sin(theta);
            }
        }
    }
}

std::vector<float> VisionRotaryEmbedding::apply(
    const std::vector<float>& input,
    const std::vector<size_t>& positions) {
    
    std::vector<float> output = input;
    
    // TODO: Implement rotary embedding application
    
    return output;
}

std::vector<float> VisionRotaryEmbedding::rotateHalf(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    size_t half = input.size() / 2;
    
    for (size_t i = 0; i < half; ++i) {
        output[i] = -input[i + half];
        output[i + half] = input[i];
    }
    
    return output;
}

// QwenVisionModel implementation
QwenVisionModel::QwenVisionModel() {
    // Use default options
    options_ = VisionModelOptions{};
}

QwenVisionModel::QwenVisionModel(const VisionModelOptions& options) 
    : options_(options) {
}

bool QwenVisionModel::initialize(const std::string& configPath) {
    if (!loadConfig(configPath)) {
        std::cerr << "Failed to load config from: " << configPath << std::endl;
        return false;
    }
    
    // Initialize layers
    layers_.clear();
    for (size_t i = 0; i < options_.numLayers; ++i) {
        layers_.push_back(std::make_unique<VisionTransformerLayer>(options_));
    }
    
    // Initialize embeddings
    size_t patchDim = options_.patchDim();
    size_t hiddenSize = options_.hiddenSize;
    
    patchEmbeddingWeights_.resize(patchDim * hiddenSize);
    patchEmbeddingBias_.resize(hiddenSize);
    
    size_t maxPatches = (options_.imageSize / options_.patchSize) * 
                       (options_.imageSize / options_.patchSize);
    positionEmbeddingWeights_.resize(maxPatches * hiddenSize);
    
    // Initialize final layer norm
    finalLayerNormWeights_.resize(hiddenSize, 1.0f);
    finalLayerNormBias_.resize(hiddenSize, 0.0f);
    
    // Initialize rotary embedding
    rotaryEmbedding_ = std::make_unique<VisionRotaryEmbedding>(hiddenSize);
    
    initialized_ = true;
    return true;
}

bool QwenVisionModel::loadModel(const std::string& modelPath) {
    if (!initialized_) {
        std::cerr << "Model not initialized. Call initialize() first." << std::endl;
        return false;
    }
    
    return loadWeights(modelPath);
}

std::vector<float> QwenVisionModel::processImage(const std::vector<uint8_t>& imageData) {
    if (!initialized_) {
        std::cerr << "Model not initialized" << std::endl;
        return {};
    }
    if (imageData.empty()) {
        std::cerr << "[WARN] Empty image data" << std::endl;
        return {};
    }
    // Preprocess image data
    auto pixelValues = preprocessImage(imageData);
    if (pixelValues.empty()) {
        std::cerr << "[WARN] Preprocess produced empty pixel values" << std::endl;
        return {};
    }
    if (options_.numChannels == 0) {
        std::cerr << "[ERROR] numChannels is 0 in VisionModelOptions" << std::endl;
        return {};
    }
    if (pixelValues.size() % options_.numChannels != 0) {
        std::cerr << "[WARN] Pixel values size " << pixelValues.size()
                  << " not divisible by numChannels " << options_.numChannels << std::endl;
        // Fallback to configured image size to proceed
        size_t imageSize = options_.imageSize;
        if (imageSize == 0) {
            std::cerr << "[ERROR] Configured imageSize is 0" << std::endl;
            return {};
        }
        auto grid = calculateGrid(imageSize, imageSize);
        return forward(pixelValues, grid);
    }
    // Calculate grid based on image dimensions
    // For now, assume square image
    const size_t pixels = pixelValues.size() / options_.numChannels;
    size_t imageSize = static_cast<size_t>(std::sqrt(static_cast<double>(pixels)));
    if (imageSize * imageSize != pixels) {
        std::cerr << "[WARN] Inferred image is not square: pixels=" << pixels
                  << ", sqrt=" << imageSize << "; falling back to configured imageSize="
                  << options_.imageSize << std::endl;
        imageSize = options_.imageSize;
    }
    auto grid = calculateGrid(imageSize, imageSize);
    
    return forward(pixelValues, grid);
}

std::vector<float> QwenVisionModel::forward(
    const std::vector<float>& pixelValues,
    const Grid& grid) {
    
    // Patch embedding
    auto embeddings = patchEmbedding(pixelValues);
    
    // Position embedding
    embeddings = positionEmbedding(embeddings, grid);
    
    // Pass through transformer layers
    auto hidden = embeddings;
    for (auto& layer : layers_) {
        hidden = layer->forward(hidden);
    }
    
    // Final layer norm
    hidden = layerNorm(hidden, finalLayerNormWeights_, finalLayerNormBias_, options_.layerNormEps);
    
    return hidden;
}

std::vector<float> QwenVisionModel::patchEmbedding(const std::vector<float>& pixelValues) {
    // TODO: Implement patch embedding
    // For now, return a placeholder
    std::vector<float> embeddings(options_.hiddenSize * options_.numPatches(), 0.0f);
    return embeddings;
}

std::vector<float> QwenVisionModel::positionEmbedding(
    const std::vector<float>& embeddings,
    const Grid& grid) {
    
    // TODO: Implement position embedding
    return embeddings;
}

std::pair<size_t, size_t> QwenVisionModel::getImageFeatureDims() const {
    return {options_.numPatches(), options_.hiddenSize};
}

bool QwenVisionModel::loadConfig(const std::string& configPath) {
    // TODO: Load configuration from JSON file
    // For now, use default values
    return true;
}

bool QwenVisionModel::loadWeights(const std::string& weightsPath) {
    // TODO: Load model weights from file
    // Load weights for all layers
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (!layers_[i]->loadWeights(weightsPath, i)) {
            return false;
        }
    }
    
    return true;
}

std::vector<float> QwenVisionModel::layerNorm(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    float eps) {
    const size_t hidden = options_.hiddenSize;
    if (hidden == 0) {
        return input;
    }
    if (input.empty()) {
        return {};
    }
    if (input.size() % hidden != 0) {
        std::cerr << "[WARN] QwenVisionModel::layerNorm input size " << input.size()
                  << " not divisible by hidden " << hidden << std::endl;
        return input;
    }

    const bool has_scale = (weights.size() == hidden);
    const bool has_bias = (bias.size() == hidden);
    std::vector<float> output(input.size());
    const size_t seq_len = input.size() / hidden;

    for (size_t t = 0; t < seq_len; ++t) {
        const size_t base = t * hidden;
        // mean
        double mean = 0.0;
        for (size_t i = 0; i < hidden; ++i) {
            mean += static_cast<double>(input[base + i]);
        }
        mean /= static_cast<double>(hidden);
        // variance
        double var = 0.0;
        for (size_t i = 0; i < hidden; ++i) {
            double diff = static_cast<double>(input[base + i]) - mean;
            var += diff * diff;
        }
        var /= static_cast<double>(hidden);
        float invStd = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        for (size_t i = 0; i < hidden; ++i) {
            float scaled = (input[base + i] - static_cast<float>(mean)) * invStd;
            if (has_scale) scaled *= weights[i];
            if (has_bias) scaled += bias[i];
            output[base + i] = scaled;
        }
    }

    return output;
}

std::vector<float> QwenVisionModel::preprocessImage(const std::vector<uint8_t>& imageData) {
    // TODO: Implement proper image preprocessing
    // Convert uint8 to float and normalize
    std::vector<float> pixelValues(imageData.size());
    for (size_t i = 0; i < imageData.size(); ++i) {
        pixelValues[i] = static_cast<float>(imageData[i]) / 255.0f;
    }
    return pixelValues;
}

Grid QwenVisionModel::calculateGrid(size_t imageHeight, size_t imageWidth) {
    size_t patchHeight = imageHeight / options_.patchSize;
    size_t patchWidth = imageWidth / options_.patchSize;
    return Grid(patchHeight, patchWidth);
}

std::vector<float> QwenVisionModel::createBlockDiagonalMask(
    size_t seqLength,
    const std::vector<size_t>& bounds) {
    
    std::vector<float> mask(seqLength * seqLength, -std::numeric_limits<float>::infinity());
    
    // TODO: Implement block diagonal mask creation
    
    return mask;
}

void QwenVisionModel::setOptions(const VisionModelOptions& options) {
    options_ = options;
}

// Factory function
std::unique_ptr<VisionModel> createQwenVisionModel(const std::string& configPath) {
    auto model = std::make_unique<QwenVisionModel>();
    if (!model->initialize(configPath)) {
        return nullptr;
    }
    return std::move(model);
}

} // namespace model
} // namespace duorou