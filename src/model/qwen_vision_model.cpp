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
    
    const size_t hidden = options_.hiddenSize;
    if (input.empty() || hidden == 0 || (input.size() % hidden) != 0) return input;
    const size_t seq = input.size() / hidden;
    const size_t heads = options_.numHeads;
    const size_t headDim = hidden / heads;
    if (headDim == 0 || (hidden % heads) != 0) return input;

    // 1) Linear projections Q, K, V (simple matmul, no bias for brevity)
    auto matmul_seq = [&](const std::vector<float>& x, const std::vector<float>& W) {
        std::vector<float> y(seq * hidden, 0.0f);
        for (size_t t = 0; t < seq; ++t) {
            const float* xi = &x[t * hidden];
            float* yo = &y[t * hidden];
            for (size_t o = 0; o < hidden; ++o) {
                const float* wrow = &W[o * hidden];
                double acc = 0.0;
                for (size_t i = 0; i < hidden; ++i) acc += double(xi[i]) * double(wrow[i]);
                yo[o] = float(acc);
            }
        }
        return y;
    };
    std::vector<float> Q = matmul_seq(input, queryWeights_);
    std::vector<float> K = matmul_seq(input, keyWeights_);
    std::vector<float> V = matmul_seq(input, valueWeights_);

    // 2) 应用视觉RoPE到Q/K（占位：未接入全局rotaryEmbedding缓存，此处略）
    // TODO: 接入VisionRotaryEmbedding::apply并传递位置索引

    // 3) 缩放点积注意力 per head
    const float scale = 1.0f / std::sqrt(float(headDim));
    std::vector<float> out(seq * hidden, 0.0f);
    for (size_t h = 0; h < heads; ++h) {
        // head slices
        auto headSlice = [&](std::vector<float>& T) {
            std::vector<float> r(seq * headDim);
            for (size_t t = 0; t < seq; ++t) {
                std::copy(&T[t * hidden + h * headDim], &T[t * hidden + (h+1) * headDim], &r[t * headDim]);
            }
            return r;
        };
        std::vector<float> Qh = headSlice(Q);
        std::vector<float> Kh = headSlice(K);
        std::vector<float> Vh = headSlice(V);
        // attention weights: [seq, seq]
        std::vector<float> att(seq * seq, 0.0f);
        for (size_t t = 0; t < seq; ++t) {
            for (size_t s = 0; s < seq; ++s) {
                const float* qt = &Qh[t * headDim];
                const float* ks = &Kh[s * headDim];
                double dot = 0.0;
                for (size_t i = 0; i < headDim; ++i) dot += double(qt[i]) * double(ks[i]);
                att[t * seq + s] = float(dot) * scale;
            }
            // mask: 简化忽略; TODO: 使用attentionMask
            // softmax over s
            float maxv = -std::numeric_limits<float>::infinity();
            for (size_t s = 0; s < seq; ++s) maxv = std::max(maxv, att[t * seq + s]);
            double sum = 0.0;
            for (size_t s = 0; s < seq; ++s) {
                att[t * seq + s] = std::exp(att[t * seq + s] - maxv);
                sum += att[t * seq + s];
            }
            for (size_t s = 0; s < seq; ++s) att[t * seq + s] = float(att[t * seq + s] / sum);
        }
        // output per head: [seq, headDim]
        for (size_t t = 0; t < seq; ++t) {
            float* dst = &out[t * hidden + h * headDim];
            std::fill(dst, dst + headDim, 0.0f);
            for (size_t s = 0; s < seq; ++s) {
                const float a = att[t * seq + s];
                const float* vs = &Vh[s * headDim];
                for (size_t i = 0; i < headDim; ++i) dst[i] += a * vs[i];
            }
        }
    }

    // 4) 输出线性
    auto outProj = matmul_seq(out, outputWeights_);
    return outProj;
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
    const size_t hidden = options_.hiddenSize;
    if (hidden == 0 || input.empty() || (input.size() % hidden) != 0) return input;
    const size_t seq = input.size() / hidden;
    const size_t inter = hidden * 4; // 按图比例，常见为4倍
    // 简化：权重存储为fc1Weights_[inter*hidden]行主、fc2Weights_[hidden*inter]行主
    auto matmul_seq = [&](const std::vector<float>& x, const std::vector<float>& W, size_t outDim, size_t inDim) {
        std::vector<float> y(seq * outDim, 0.0f);
        for (size_t t = 0; t < seq; ++t) {
            const float* xi = &x[t * inDim];
            float* yo = &y[t * outDim];
            for (size_t o = 0; o < outDim; ++o) {
                const float* wrow = &W[o * inDim];
                double acc = 0.0;
                for (size_t i = 0; i < inDim; ++i) acc += double(xi[i]) * double(wrow[i]);
                yo[o] = float(acc);
            }
        }
        return y;
    };
    auto h1 = matmul_seq(input, fc1Weights_, inter, hidden);
    // add bias + SiLU
    for (size_t t = 0; t < seq; ++t) {
        for (size_t i = 0; i < inter; ++i) {
            float x = h1[t * inter + i] + (i < fc1Bias_.size() ? fc1Bias_[i] : 0.0f);
            h1[t * inter + i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        }
    }
    auto h2 = matmul_seq(h1, fc2Weights_, hidden, inter);
    for (size_t t = 0; t < seq; ++t) {
        for (size_t i = 0; i < hidden; ++i) {
            h2[t * hidden + i] += (i < fc2Bias_.size() ? fc2Bias_[i] : 0.0f);
        }
    }
    return h2;
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
    if (input.empty() || dim_ == 0) return output;
    const size_t seq = input.size() / dim_;
    if (seq == 0) return output;
    // positions.size() 可小于等于seq
    for (size_t t = 0; t < seq && t < positions.size(); ++t) {
        size_t pos = positions[t];
        const float* c = &cosCache_[pos * dim_];
        const float* s = &sinCache_[pos * dim_];
        for (size_t i = 0; i < dim_; i += 2) {
            float x0 = output[t * dim_ + i];
            float x1 = (i + 1 < dim_) ? output[t * dim_ + i + 1] : 0.0f;
            float ro0 = x0 * c[i] - x1 * s[i];
            float ro1 = x0 * s[i] + x1 * c[i];
            output[t * dim_ + i] = ro0;
            if (i + 1 < dim_) output[t * dim_ + i + 1] = ro1;
        }
    }
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
    
    // Patch merger: 将2x2串联并映射到文本维度
    VisionPatchMerger merger;
    merger.configure(options_.hiddenSize, /*textHidden=*/3584);
    auto merged = merger.forward(hidden);
    return merged;
}

std::vector<float> QwenVisionModel::patchEmbedding(const std::vector<float>& pixelValues) {
    // 近似Conv3d(3->hidden, kernel=(2,14,14), stride=(2,14,14))：
    // 这里采用每patch线性映射近似，忽略时间维度，按temporalPatchSize=2展开后取平均。
    const size_t patchDim = options_.patchDim();
    const size_t numPatches = options_.numPatches();
    std::vector<float> embeddings(options_.hiddenSize * numPatches, 0.0f);
    if (pixelValues.size() < patchDim * numPatches) return embeddings;
    // 权重初始化占位：使用patchEmbeddingWeights_作为[hidden, patchDim]
    if (patchEmbeddingWeights_.size() != options_.hiddenSize * patchDim) {
        patchEmbeddingWeights_.assign(options_.hiddenSize * patchDim, 0.0f);
    }
    for (size_t p = 0; p < numPatches; ++p) {
        const float* x = &pixelValues[p * patchDim];
        float* y = &embeddings[p * options_.hiddenSize];
        for (size_t o = 0; o < options_.hiddenSize; ++o) {
            const float* w = &patchEmbeddingWeights_[o * patchDim];
            double acc = 0.0;
            for (size_t i = 0; i < patchDim; ++i) acc += double(x[i]) * double(w[i]);
            y[o] = float(acc) + (o < patchEmbeddingBias_.size() ? patchEmbeddingBias_[o] : 0.0f);
        }
    }
    return embeddings;
}

std::vector<float> QwenVisionModel::positionEmbedding(
    const std::vector<float>& embeddings,
    const Grid& grid) {
    
    // 简单位置加性嵌入
    std::vector<float> out = embeddings;
    const size_t numPatches = options_.numPatches();
    if (positionEmbeddingWeights_.size() == numPatches * options_.hiddenSize) {
        for (size_t p = 0; p < numPatches; ++p) {
            for (size_t i = 0; i < options_.hiddenSize; ++i) {
                out[p * options_.hiddenSize + i] += positionEmbeddingWeights_[p * options_.hiddenSize + i];
            }
        }
    }
    return out;
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