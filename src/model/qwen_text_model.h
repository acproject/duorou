#pragma once

#include "base_model.h"
#include "vocabulary.h"
#include "byte_pair_encoding.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include "../ml/nn/attention.h"
#include "../ml/context.h"
#include "../ml/tensor.h"
#include "../kvcache/cache.h"

// Forward declare GGUFParser to avoid heavy include in header
namespace duorou { namespace extensions { namespace ollama { class GGUFParser; } } }

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
        duorou::ml::Context& ctx,
        const std::vector<float>& input,
        const std::vector<float>& attentionMask = {},
        duorou::kvcache::Cache* cache = nullptr
    );
    
    // Initialize weights (placeholder for actual weight loading)
    bool loadWeights(const std::string& weightsPath);
    // New overload: load weights for a specific layer from a parsed GGUF
    bool loadWeights(duorou::extensions::ollama::GGUFParser &parser, size_t layerIndex);

    // Set precomputed RoPE frequencies (size = ropeDim/2)
    void setRoPEFreqs(const std::vector<float>& freqs) { ropeFreqs_ = freqs; }
    // Control where RoPE is applied: in attention (default) or at embedding stage
    void setApplyRopeInAttention(bool v) { applyRopeInAttention_ = v; }

private:
    TextModelOptions options_;
    bool weightsLoaded_ = false;
    friend class TransformerLayer;
    // Weight matrices (simplified representation)
    std::vector<float> queryWeights_;
    std::vector<float> keyWeights_;
    std::vector<float> valueWeights_;
    std::vector<float> outputWeights_;

    // Multi-head attention implementation (Tensor-based)
    std::unique_ptr<duorou::ml::nn::MultiHeadAttention> mha_;

    // Precomputed RoPE frequencies
    std::vector<float> ropeFreqs_;
    // Whether to apply RoPE inside attention (true) or assume it has been applied on inputs (false)
    bool applyRopeInAttention_ = true;
    // Lazy-init flag: whether MHA has been bound with actual weights (allocated/attached)
    bool mhaWeightsReady_ = false;
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
    // New overload: load weights for a specific layer from a parsed GGUF
    bool loadWeights(duorou::extensions::ollama::GGUFParser &parser, size_t layerIndex);
    
private:
    TextModelOptions options_;
    bool weightsLoaded_ = false;
    friend class TransformerLayer;
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
        duorou::ml::Context& ctx,
        const std::vector<float>& input,
        const std::vector<float>& attentionMask = {},
        duorou::kvcache::Cache* cache = nullptr
    );
    
    // Load layer weights
    bool loadWeights(const std::string& weightsPath, size_t layerIndex);

    // Propagate precomputed RoPE frequencies to attention
    void setRoPEFreqs(const std::vector<float>& freqs);
    // Control where RoPE is applied for this layer (attention vs. embeddings)
    void setApplyRopeInAttention(bool v);
    
private:
    TextModelOptions options_;
    std::unique_ptr<SelfAttention> attention_;
    std::unique_ptr<FeedForward> feedForward_;
    
    // Layer normalization weights
    std::vector<float> inputNormWeights_;
    std::vector<float> postAttentionNormWeights_;

    // Helper: local LayerNorm operating on std::vector<float>
    std::vector<float> layerNormVec(
        const std::vector<float>& input,
        const std::vector<float>& weights,
        float eps
    );
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
    size_t getVocabSize() const override;
    const Vocabulary* getVocabulary() const override;
    bool initialize(const std::string& configPath) override;
    bool initialize(const std::string& configPath, bool skipVocabInit);
    std::string getModelType() const override; // implement BaseModel pure virtual
    bool isInitialized() const override;       // implement BaseModel pure virtual
    
    // TextModel interface implementation
    std::vector<int32_t> generate(
        const std::vector<int32_t>& inputIds,
        size_t maxLength = 512,
        float temperature = 1.0f,
        float topP = 0.9f
    ) override;
    
    std::vector<float> forward(const std::vector<int32_t>& inputIds) override;

    // New: Forward with KV cache and ML Context/Tensor for multimodal path
    duorou::ml::Tensor forward(
        duorou::ml::Context& ctx,
        const duorou::ml::Tensor& inputIds,
        duorou::kvcache::Cache* cache = nullptr
    );

    // Helper exposure
    size_t getHiddenSize() const;
    std::vector<float> computeLogitsFromHidden(const std::vector<float>& hidden);
    
    // Step-by-step decode: compute logits for the last token, optionally using KV Cache
    duorou::ml::Tensor stepDecode(
        duorou::ml::Context& ctx,
        const duorou::ml::Tensor& lastTokenId,
        duorou::kvcache::Cache* cache = nullptr
    );

    // New: nextToken helper using stepDecode with temperature and top-p sampling
    int32_t nextToken(
        duorou::ml::Context& ctx,
        const duorou::ml::Tensor& lastTokenId,
        duorou::kvcache::Cache* cache,
        float temperature,
        float topP
    );
    
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

    // Set precomputed RoPE frequencies (propagates to layers)
    void setRoPEFreqs(const std::vector<float>& freqs);
    // Control where RoPE is applied (attention vs. embeddings) for all layers
    void setApplyRopeInAttention(bool v);

    // Bind an external vocabulary/tokenizer from upper-level (e.g., multimodal model)
    // Ensures getVocabSize() reflects external vocab and logits size matches expected.
    void setExternalVocabulary(std::shared_ptr<Vocabulary> vocab);
    
private:
    TextModelOptions options_;
    std::vector<std::unique_ptr<TransformerLayer>> layers_;
    
    // Weights and buffers
    // Use BaseModel::vocabulary_ and BaseModel::tokenizer_
    std::vector<float> tokenEmbeddings_;  // Token embedding weights
    std::vector<float> outputWeights_;    // Output projection weights
    std::vector<float> outputNormWeights_; // Final layer norm weights

    // Precomputed RoPE frequencies
    std::vector<float> ropeFreqs_;
    
    // Internal helpers
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

std::unique_ptr<BaseModel> createQwenTextModel(const std::string& configPath);

} // namespace model
} // namespace duorou