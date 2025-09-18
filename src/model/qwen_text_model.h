#pragma once

#include "base_model.h"
#include "vocabulary.h"
#include "byte_pair_encoding.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace duorou {
namespace model {

// Text model options/configuration
struct TextModelOptions {
    size_t hiddenSize = 4096;
    size_t numHeads = 32;
    size_t numKVHeads = 32;
    size_t ropeDim = 128;
    size_t originalContextLength = 128000;
    float eps = 1e-6f;
    float ropeBase = 10000.0f;
    float ropeScale = 1.0f;
    size_t blockCount = 32;
    size_t embeddingLength = 4096;
};

// Self-attention layer implementation
class SelfAttention {
public:
    SelfAttention(const TextModelOptions& options);
    ~SelfAttention() = default;
    
    // Forward pass for self-attention
    std::vector<float> forward(
        const std::vector<float>& input,
        const std::vector<float>& attentionMask = {}
    );
    
    // Initialize weights (placeholder for actual weight loading)
    bool loadWeights(const std::string& weightsPath);
    
private:
    TextModelOptions options_;
    bool weightsLoaded_ = false;
    
    // Weight matrices (simplified representation)
    std::vector<float> queryWeights_;
    std::vector<float> keyWeights_;
    std::vector<float> valueWeights_;
    std::vector<float> outputWeights_;
};

// Feed-forward network layer
class FeedForward {
public:
    FeedForward(const TextModelOptions& options);
    ~FeedForward() = default;
    
    // Forward pass for FFN
    std::vector<float> forward(const std::vector<float>& input);
    
    // Initialize weights
    bool loadWeights(const std::string& weightsPath);
    
private:
    TextModelOptions options_;
    bool weightsLoaded_ = false;
    
    // Weight matrices
    std::vector<float> gateWeights_;
    std::vector<float> upWeights_;
    std::vector<float> downWeights_;
};

// Transformer layer combining attention and FFN
class TransformerLayer {
public:
    TransformerLayer(const TextModelOptions& options);
    ~TransformerLayer() = default;
    
    // Forward pass for transformer layer
    std::vector<float> forward(
        const std::vector<float>& input,
        const std::vector<float>& attentionMask = {}
    );
    
    // Load layer weights
    bool loadWeights(const std::string& weightsPath, size_t layerIndex);
    
private:
    TextModelOptions options_;
    std::unique_ptr<SelfAttention> attention_;
    std::unique_ptr<FeedForward> feedForward_;
    
    // Layer normalization weights
    std::vector<float> inputNormWeights_;
    std::vector<float> postAttentionNormWeights_;
};

// Qwen text model implementation
class QwenTextModel : public TextModel {
public:
    QwenTextModel();
    explicit QwenTextModel(const TextModelOptions& options);
    ~QwenTextModel() override = default;
    
    // BaseModel interface implementation
    std::vector<int32_t> encode(const std::string& text, bool addSpecial = true) override;
    std::string decode(const std::vector<int32_t>& ids) override;
    std::string getModelType() const override { return "qwen-text"; }
    size_t getVocabSize() const override;
    const Vocabulary* getVocabulary() const override;
    bool initialize(const std::string& configPath) override;
    bool isInitialized() const override { return initialized_; }
    
    // TextModel interface implementation
    std::vector<int32_t> generate(
        const std::vector<int32_t>& inputIds,
        size_t maxLength = 512,
        float temperature = 1.0f,
        float topP = 0.9f
    ) override;
    
    std::vector<float> forward(const std::vector<int32_t>& inputIds) override;
    
    // Qwen-specific methods
    bool loadModel(const std::string& modelPath);
    void setOptions(const TextModelOptions& options);
    const TextModelOptions& getOptions() const { return options_; }
    
    // Utility methods
    std::vector<float> embedTokens(const std::vector<int32_t>& tokenIds);
    std::vector<float> applyPositionalEncoding(
        const std::vector<float>& embeddings,
        size_t sequenceLength
    );
    
private:
    TextModelOptions options_;
    std::vector<std::unique_ptr<TransformerLayer>> layers_;
    
    // Embedding and output layers
    std::vector<float> tokenEmbeddings_;  // Token embedding weights
    std::vector<float> outputWeights_;    // Output projection weights
    std::vector<float> outputNormWeights_; // Final layer norm weights
    
    // Helper methods
    bool loadConfig(const std::string& configPath);
    bool loadWeights(const std::string& weightsPath);
    std::vector<float> layerNorm(
        const std::vector<float>& input,
        const std::vector<float>& weights,
        float eps = 1e-6f
    );
    std::vector<float> softmax(const std::vector<float>& logits);
    int32_t sampleToken(const std::vector<float>& probabilities, float temperature, float topP);
};

// Factory function for creating Qwen text models
std::unique_ptr<BaseModel> createQwenTextModel(const std::string& configPath);

} // namespace model
} // namespace duorou