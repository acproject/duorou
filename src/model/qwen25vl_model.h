#pragma once

#include "model.h"
#include "../ml/tensor.h"
#include "../ml/context.h"
#include "../ml/nn/linear.h"
#include "../ml/nn/attention.h"
#include "../ml/nn/layer_norm.h"
#include <memory>
#include <vector>
#include <string>
#include <map>

namespace duorou {
namespace model {

// Forward declarations
class QwenTextModel;
class QwenVisionModel;
class QwenImageProcessor;

// Qwen25VL specific configuration
struct Qwen25VLConfig {
    // Text model configuration
    size_t hiddenSize = 3584;
    size_t numHeads = 28;
    size_t numKVHeads = 4;
    size_t numLayers = 28;
    size_t vocabSize = 151936;
    size_t maxPositionEmbeddings = 32768;
    double ropeTheta = 1000000.0;
    double rmsNormEps = 1e-6;
    
    // Vision model configuration
    size_t visionHiddenSize = 1280;
    size_t visionNumHeads = 16;
    size_t visionNumLayers = 32;
    size_t patchSize = 14;
    size_t spatialMergeSize = 2;
    size_t maxPixels = 1003520; // 28*28*1280
    
    // Image processor configuration
    size_t numChannels = 3;
    size_t minPixels = 3136; // 56*56
    double rescaleFactor = 1.0 / 255.0;
    std::vector<double> imageMean = {0.485, 0.456, 0.406};
    std::vector<double> imageStd = {0.229, 0.224, 0.225};
    
    // Special tokens
    int32_t visionStartToken = 151652;
    int32_t visionEndToken = 151653;
    int32_t imageToken = 151655;
    int32_t videoToken = 151656;
    
    Qwen25VLConfig() = default;
    
    // Load configuration from file
    bool loadFromFile(const std::string& configPath);
    
    // Validate configuration
    bool validate() const;
};

// Grid structure for image processing
struct Grid {
    size_t height;
    size_t width;
    size_t temporal;
    
    Grid(size_t h = 0, size_t w = 0, size_t t = 1) 
        : height(h), width(w), temporal(t) {}
};

// Pixel values container
struct PixelValues {
    ml::Tensor data;
    Grid grid;
    
    PixelValues() = default;
    PixelValues(const ml::Tensor& tensor, const Grid& g) 
        : data(tensor), grid(g) {}
    
    // Create from raw image data
    static PixelValues fromImageData(const std::vector<float>& imageData, 
                                   size_t height, size_t width, size_t channels);
};

// Image processor for Qwen25VL
class QwenImageProcessor {
public:
    explicit QwenImageProcessor(const Qwen25VLConfig& config);
    ~QwenImageProcessor() = default;
    
    // Process image and return pixel values
    PixelValues processImage(const std::vector<uint8_t>& imageData, 
                           size_t width, size_t height, size_t channels);
    
    // Smart resize algorithm
    std::pair<size_t, size_t> smartResize(size_t height, size_t width) const;
    
private:
    Qwen25VLConfig config_;
    
    // Helper methods
    std::vector<float> normalize(const std::vector<float>& pixels) const;
    std::vector<float> resize(const std::vector<uint8_t>& imageData,
                            size_t origWidth, size_t origHeight, size_t channels,
                            size_t newWidth, size_t newHeight) const;
};

// Vision self-attention layer
class VisionSelfAttention {
public:
    VisionSelfAttention(size_t hiddenSize, size_t numHeads, size_t headDim);
    ~VisionSelfAttention() = default;
    
    ml::Tensor forward(ml::Context& ctx, const ml::Tensor& hiddenStates,
                      const ml::Tensor& cos, const ml::Tensor& sin,
                      const ml::Tensor& mask = ml::Tensor()) const;
    
    // Load weights from GGUF
    bool loadWeights(const std::map<std::string, ml::Tensor>& weights, 
                    const std::string& prefix);
    
private:
    std::unique_ptr<ml::nn::Linear> query_;
    std::unique_ptr<ml::nn::Linear> key_;
    std::unique_ptr<ml::nn::Linear> value_;
    std::unique_ptr<ml::nn::Linear> output_;
    
    size_t hiddenSize_;
    size_t numHeads_;
    size_t headDim_;
    
    // Helper methods
    ml::Tensor rotateHalf(ml::Context& ctx, const ml::Tensor& tensor) const;
    ml::Tensor applyRotaryEmbedding(ml::Context& ctx, const ml::Tensor& tensor,
                                   const ml::Tensor& cos, const ml::Tensor& sin) const;
};

// Vision MLP layer
class VisionMLP {
public:
    explicit VisionMLP(size_t hiddenSize, size_t intermediateSize);
    ~VisionMLP() = default;
    
    ml::Tensor forward(ml::Context& ctx, const ml::Tensor& hiddenStates) const;
    
    // Load weights from GGUF
    bool loadWeights(const std::map<std::string, ml::Tensor>& weights,
                    const std::string& prefix);
    
private:
    std::unique_ptr<ml::nn::Linear> gate_;
    std::unique_ptr<ml::nn::Linear> up_;
    std::unique_ptr<ml::nn::Linear> down_;
};

// Vision transformer layer
class VisionLayer {
public:
    VisionLayer(size_t hiddenSize, size_t numHeads, size_t intermediateSize);
    ~VisionLayer() = default;
    
    ml::Tensor forward(ml::Context& ctx, const ml::Tensor& hiddenStates,
                      const ml::Tensor& cos, const ml::Tensor& sin,
                      const ml::Tensor& mask = ml::Tensor()) const;
    
    // Load weights from GGUF
    bool loadWeights(const std::map<std::string, ml::Tensor>& weights,
                    const std::string& prefix);
    
private:
    std::unique_ptr<VisionSelfAttention> attention_;
    std::unique_ptr<VisionMLP> mlp_;
    std::unique_ptr<ml::nn::LayerNorm> attentionNorm_;
    std::unique_ptr<ml::nn::LayerNorm> mlpNorm_;
};

// Vision model
class QwenVisionModel {
public:
    explicit QwenVisionModel(const Qwen25VLConfig& config);
    ~QwenVisionModel() = default;
    
    // Forward pass
    ml::Tensor forward(ml::Context& ctx, const PixelValues& pixelValues) const;
    
    // Load weights from GGUF
    bool loadWeights(const std::map<std::string, ml::Tensor>& weights);
    
private:
    Qwen25VLConfig config_;
    std::vector<std::unique_ptr<VisionLayer>> layers_;
    std::unique_ptr<ml::nn::Linear> patchEmbedding_;
    std::unique_ptr<ml::nn::LayerNorm> layerNorm_;
    
    // Helper methods
    ml::Tensor createRotaryEmbedding(ml::Context& ctx, size_t seqLen) const;
    ml::Tensor createAttentionMask(ml::Context& ctx, size_t seqLen,
                                  const std::vector<size_t>& bounds) const;
};

// Text self-attention layer (similar to vision but with different parameters)
class TextSelfAttention {
public:
    TextSelfAttention(size_t hiddenSize, size_t numHeads, size_t numKVHeads, size_t headDim);
    ~TextSelfAttention() = default;
    
    ml::Tensor forward(ml::Context& ctx, const ml::Tensor& hiddenStates,
                      const ml::Tensor& cos, const ml::Tensor& sin,
                      const ml::Tensor& mask = ml::Tensor()) const;
    
    // Load weights from GGUF
    bool loadWeights(const std::map<std::string, ml::Tensor>& weights,
                    const std::string& prefix);
    
private:
    std::unique_ptr<ml::nn::Linear> query_;
    std::unique_ptr<ml::nn::Linear> key_;
    std::unique_ptr<ml::nn::Linear> value_;
    std::unique_ptr<ml::nn::Linear> output_;
    
    size_t hiddenSize_;
    size_t numHeads_;
    size_t numKVHeads_;
    size_t headDim_;
};

// Text MLP layer
class TextMLP {
public:
    explicit TextMLP(size_t hiddenSize, size_t intermediateSize);
    ~TextMLP() = default;
    
    ml::Tensor forward(ml::Context& ctx, const ml::Tensor& hiddenStates) const;
    
    // Load weights from GGUF
    bool loadWeights(const std::map<std::string, ml::Tensor>& weights,
                    const std::string& prefix);
    
private:
    std::unique_ptr<ml::nn::Linear> gate_;
    std::unique_ptr<ml::nn::Linear> up_;
    std::unique_ptr<ml::nn::Linear> down_;
};

// Text transformer layer
class TextLayer {
public:
    TextLayer(size_t hiddenSize, size_t numHeads, size_t numKVHeads, size_t intermediateSize);
    ~TextLayer() = default;
    
    ml::Tensor forward(ml::Context& ctx, const ml::Tensor& hiddenStates,
                      const ml::Tensor& cos, const ml::Tensor& sin,
                      const ml::Tensor& mask = ml::Tensor()) const;
    
    // Load weights from GGUF
    bool loadWeights(const std::map<std::string, ml::Tensor>& weights,
                    const std::string& prefix);
    
private:
    std::unique_ptr<TextSelfAttention> attention_;
    std::unique_ptr<TextMLP> mlp_;
    std::unique_ptr<ml::nn::LayerNorm> attentionNorm_;
    std::unique_ptr<ml::nn::LayerNorm> mlpNorm_;
};

// Text model
class QwenTextModel {
public:
    explicit QwenTextModel(const Qwen25VLConfig& config);
    ~QwenTextModel() = default;
    
    // Forward pass
    ml::Tensor forward(ml::Context& ctx, const ml::Tensor& inputIds,
                      const ml::Tensor& cos, const ml::Tensor& sin,
                      const ml::Tensor& mask = ml::Tensor()) const;
    
    // Load weights from GGUF
    bool loadWeights(const std::map<std::string, ml::Tensor>& weights);
    
private:
    Qwen25VLConfig config_;
    std::vector<std::unique_ptr<TextLayer>> layers_;
    std::unique_ptr<ml::nn::Linear> tokenEmbedding_;
    std::unique_ptr<ml::nn::LayerNorm> layerNorm_;
    std::unique_ptr<ml::nn::Linear> lmHead_;
};

// Main Qwen25VL model
class Qwen25VLModel : public Model {
public:
    explicit Qwen25VLModel(const Qwen25VLConfig& config = Qwen25VLConfig());
    ~Qwen25VLModel() override = default;
    
    // Model interface implementation
    bool load(const std::string& modelPath) override;
    bool isLoaded() const override;
    void unload() override;
    
    // Text processing
    std::vector<int32_t> encode(const std::string& text, bool addSpecial = true) override;
    std::string decode(const std::vector<int32_t>& tokens) override;
    
    // Generation
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt,
                                 size_t maxTokens = 100) override;
    std::string generateText(const std::string& prompt,
                           size_t maxTokens = 100) override;
    
    // Multimodal generation
    std::vector<int32_t> generateMultimodal(const std::vector<int32_t>& textTokens,
                                           const std::vector<PixelValues>& images,
                                           size_t maxTokens = 100);
    
    std::string generateMultimodalText(const std::string& text,
                                     const std::vector<PixelValues>& images,
                                     size_t maxTokens = 100);
    
    // Model information
    const ModelConfig& getConfig() const override;
    const TextProcessor* getTokenizer() const override;
    size_t getVocabSize() const override;
    size_t getContextLength() const override;
    
    std::string getModelName() const override;
    std::string getModelVersion() const override;
    std::map<std::string, std::string> getMetadata() const override;
    
    // Qwen25VL specific methods
    PixelValues processImage(const std::vector<uint8_t>& imageData,
                           size_t width, size_t height, size_t channels);
    
    ml::Tensor encodeMultimodal(ml::Context& ctx,
                               const std::vector<int32_t>& textTokens,
                               const std::vector<PixelValues>& images);
    
private:
    Qwen25VLConfig qwenConfig_;
    ModelConfig baseConfig_;
    
    std::unique_ptr<QwenTextModel> textModel_;
    std::unique_ptr<QwenVisionModel> visionModel_;
    std::unique_ptr<QwenImageProcessor> imageProcessor_;
    std::unique_ptr<TextProcessor> tokenizer_;
    
    bool loaded_;
    std::string modelPath_;
    std::map<std::string, std::string> metadata_;
    
    // Helper methods
    bool loadGGUFModel(const std::string& modelPath);
    std::map<std::string, ml::Tensor> loadTensorsFromGGUF(const std::string& modelPath);
    
    std::vector<int32_t> postTokenize(const std::vector<int32_t>& textTokens,
                                     const std::vector<PixelValues>& images);
    
    ml::Tensor createRotaryEmbedding(ml::Context& ctx, size_t seqLen) const;
    ml::Tensor createAttentionMask(ml::Context& ctx, size_t seqLen) const;
    
    // Token processing
    std::vector<int32_t> insertImageTokens(const std::vector<int32_t>& tokens,
                                          const std::vector<PixelValues>& images);
    
    size_t calculateImageTokenCount(const PixelValues& pixelValues) const;
};

// Factory function for creating Qwen25VL model
std::unique_ptr<Qwen25VLModel> createQwen25VLModel(const std::string& configPath = "");

} // namespace model
} // namespace duorou