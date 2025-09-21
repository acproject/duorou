#include "qwen_text_model.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <fstream>
#include <iostream>
#include "tokenizer_factory.h"

namespace duorou {
namespace model {

// SelfAttention implementation
SelfAttention::SelfAttention(const TextModelOptions& options) 
    : options_(options) {
    // Initialize weight matrices with proper dimensions
    size_t hiddenSize = options_.hiddenSize;
    queryWeights_.resize(hiddenSize * hiddenSize);
    keyWeights_.resize(hiddenSize * hiddenSize);
    valueWeights_.resize(hiddenSize * hiddenSize);
    outputWeights_.resize(hiddenSize * hiddenSize);
}

std::vector<float> SelfAttention::forward(
    const std::vector<float>& input,
    const std::vector<float>& attentionMask) {
    
    // Simplified attention implementation
    // In a real implementation, this would involve matrix multiplications
    // with query, key, value projections and scaled dot-product attention
    
    std::vector<float> output = input; // Placeholder
    return output;
}

bool SelfAttention::loadWeights(const std::string& weightsPath) {
    // Placeholder for weight loading
    weightsLoaded_ = true;
    return true;
}

// FeedForward implementation
FeedForward::FeedForward(const TextModelOptions& options) 
    : options_(options) {
    size_t hiddenSize = options_.hiddenSize;
    size_t ffnDim = hiddenSize * 4; // Typical FFN expansion
    
    gateWeights_.resize(hiddenSize * ffnDim);
    upWeights_.resize(hiddenSize * ffnDim);
    downWeights_.resize(ffnDim * hiddenSize);
}

std::vector<float> FeedForward::forward(const std::vector<float>& input) {
    // Simplified FFN implementation
    // Real implementation would involve SwiGLU activation
    std::vector<float> output = input; // Placeholder
    return output;
}

bool FeedForward::loadWeights(const std::string& weightsPath) {
    weightsLoaded_ = true;
    return true;
}

// TransformerLayer implementation
TransformerLayer::TransformerLayer(const TextModelOptions& options) 
    : options_(options) {
    attention_ = std::make_unique<SelfAttention>(options);
    feedForward_ = std::make_unique<FeedForward>(options);
    
    // Initialize layer norm weights
    inputNormWeights_.resize(options_.hiddenSize, 1.0f);
    postAttentionNormWeights_.resize(options_.hiddenSize, 1.0f);
}

std::vector<float> TransformerLayer::forward(
    const std::vector<float>& input,
    const std::vector<float>& attentionMask) {
    
    // Layer norm + attention + residual
    auto normed = input; // Placeholder for layer norm
    auto attentionOut = attention_->forward(normed, attentionMask);
    
    // Add residual connection
    std::vector<float> afterAttention(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        afterAttention[i] = input[i] + attentionOut[i];
    }
    
    // Layer norm + FFN + residual
    auto normed2 = afterAttention; // Placeholder for layer norm
    auto ffnOut = feedForward_->forward(normed2);
    
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = afterAttention[i] + ffnOut[i];
    }
    
    return output;
}

bool TransformerLayer::loadWeights(const std::string& weightsPath, size_t layerIndex) {
    // Load weights for this specific layer
    return attention_->loadWeights(weightsPath) && feedForward_->loadWeights(weightsPath);
}

// QwenTextModel implementation
QwenTextModel::QwenTextModel() {
    modelType_ = "qwen-text";
}

QwenTextModel::QwenTextModel(const TextModelOptions& options) 
    : options_(options) {
    modelType_ = "qwen-text";
    
    // Initialize transformer layers
    layers_.reserve(options_.blockCount);
    for (size_t i = 0; i < options_.blockCount; ++i) {
        layers_.push_back(std::make_unique<TransformerLayer>(options_));
    }
    
    // Initialize embedding and output weights
    size_t vocabSize = 151936; // Qwen default vocab size
    tokenEmbeddings_.resize(vocabSize * options_.hiddenSize);
    outputWeights_.resize(options_.hiddenSize * vocabSize);
    outputNormWeights_.resize(options_.hiddenSize, 1.0f);
}

std::vector<int32_t> QwenTextModel::encode(const std::string& text, bool addSpecial) {
    if (!tokenizer_) {
        std::cerr << "Error: Tokenizer not initialized. Cannot encode text." << std::endl;
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    auto tokens = tokenizer_->encode(text, addSpecial);
    if (tokens.empty() && !text.empty()) {
        std::cerr << "Warning: Tokenizer returned empty tokens for non-empty text: " << text << std::endl;
        // Return a fallback token (UNK token) to prevent empty vectors
        return {0}; // Assuming 0 is UNK token ID
    }
    
    return tokens;
}

std::string QwenTextModel::decode(const std::vector<int32_t>& ids) {
    if (!tokenizer_) {
        std::cerr << "Error: Tokenizer not initialized. Cannot decode tokens." << std::endl;
        throw std::runtime_error("Tokenizer not initialized");
    }
    return tokenizer_->decode(ids);
}

size_t QwenTextModel::getVocabSize() const {
    if (vocabulary_) {
        return vocabulary_->size();
    }
    return 151936; // Default Qwen vocab size
}

const Vocabulary* QwenTextModel::getVocabulary() const {
    return vocabulary_.get();
}

bool QwenTextModel::initialize(const std::string& configPath) {
    try {
        // Load configuration
        if (!loadConfig(configPath)) {
            std::cerr << "Failed to load config from: " << configPath << std::endl;
            return false;
        }
        
        // Initialize vocabulary and tokenizer
        vocabulary_ = std::make_unique<Vocabulary>();
        
        // Initialize vocabulary with default Qwen vocabulary
        std::vector<std::string> defaultVocab;
        std::vector<int32_t> defaultTypes;
        std::vector<float> defaultScores;
        std::vector<std::string> defaultMerges;
        
        // Create a minimal vocabulary for testing with byte-level support
        size_t vocabSize = getVocabSize();
        defaultVocab.reserve(vocabSize);
        defaultTypes.reserve(vocabSize);
        defaultScores.reserve(vocabSize);
        
        // Set special tokens first
        defaultVocab.push_back("<unk>");  // 0: UNK token
        defaultTypes.push_back(1); // Special token
        defaultScores.push_back(0.0f);
        
        defaultVocab.push_back("<bos>");  // 1: BOS token
        defaultTypes.push_back(1); // Special token
        defaultScores.push_back(0.0f);
        
        defaultVocab.push_back("<eos>");  // 2: EOS token
        defaultTypes.push_back(1); // Special token
        defaultScores.push_back(0.0f);
        
        // Add byte-level tokens (0-255) for proper BPE encoding
        for (int i = 0; i < 256; ++i) {
            std::string byteToken;
            if (i < 32 || i >= 127) {
                // Non-printable characters - use byte representation
                byteToken = "<0x" + std::to_string(i) + ">";
            } else {
                // Printable ASCII characters
                byteToken = std::string(1, static_cast<char>(i));
            }
            defaultVocab.push_back(byteToken);
            defaultTypes.push_back(0); // Normal token
            defaultScores.push_back(0.0f);
        }
        
        // Add some common Chinese characters for better support
        std::vector<std::string> commonChinese = {
            "你", "好", "我", "是", "的", "在", "有", "不", "了", "人",
            "他", "这", "中", "大", "为", "上", "个", "国", "一", "以",
            "要", "就", "出", "会", "可", "也", "都", "后", "自", "时"
        };
        
        for (const auto& ch : commonChinese) {
            if (defaultVocab.size() < vocabSize) {
                defaultVocab.push_back(ch);
                defaultTypes.push_back(0); // Normal token
                defaultScores.push_back(0.0f);
            }
        }
        
        // Fill remaining slots with generic tokens if needed
        while (defaultVocab.size() < vocabSize) {
            defaultVocab.push_back("<token_" + std::to_string(defaultVocab.size()) + ">");
            defaultTypes.push_back(0); // Normal token
            defaultScores.push_back(0.0f);
        }
        
        vocabulary_->initialize(defaultVocab, defaultTypes, defaultScores, defaultMerges);
        vocabulary_->setUNK({0});
        vocabulary_->setBOS({1}, true);
        vocabulary_->setEOS({2}, true);
        
        std::cout << "[DEBUG] Vocabulary initialized with " << vocabulary_->size() << " tokens" << std::endl;
        
        auto vocabPtr = std::shared_ptr<Vocabulary>(vocabulary_.get(), [](Vocabulary*){});
        
        // Create Qwen tokenizer via factory
        std::cout << "[DEBUG] Creating Qwen tokenizer via factory..." << std::endl;
        TokenizerFactoryOptions opts;
        opts.override_type = "bpe"; // Qwen uses BPE by default
        std::cout << "[DEBUG] Calling createTextProcessorForArchitecture with architecture='qwen', override_type='bpe'" << std::endl;
        tokenizer_ = createTextProcessorForArchitecture("qwen", vocabPtr, opts);
        
        // Verify tokenizer was created successfully
        if (!tokenizer_) {
            std::cerr << "[ERROR] Failed to create tokenizer for Qwen architecture" << std::endl;
            std::cerr << "[ERROR] createTextProcessorForArchitecture returned nullptr" << std::endl;
            return false;
        } else {
            std::cout << "[DEBUG] Tokenizer created successfully!" << std::endl;
        }
        
        // Initialize layers if not already done
        if (layers_.empty()) {
            layers_.reserve(options_.blockCount);
            for (size_t i = 0; i < options_.blockCount; ++i) {
                layers_.push_back(std::make_unique<TransformerLayer>(options_));
            }
        }
        
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing QwenTextModel: " << e.what() << std::endl;
        return false;
    }
}

std::vector<int32_t> QwenTextModel::generate(
    const std::vector<int32_t>& inputIds,
    size_t maxLength,
    float temperature,
    float topP) {
    
    if (!initialized_) {
        return {};
    }
    
    std::vector<int32_t> result = inputIds;
    
    for (size_t i = inputIds.size(); i < maxLength; ++i) {
        // Forward pass to get logits
        auto logits = forward(result);
        
        // Sample next token
        auto probabilities = softmax(logits);
        int32_t nextToken = sampleToken(probabilities, temperature, topP);
        
        result.push_back(nextToken);
        
        // Check for EOS token (simplified)
        if (nextToken == 151643) { // Qwen EOS token
            break;
        }
    }
    
    return result;
}

std::vector<float> QwenTextModel::forward(const std::vector<int32_t>& inputIds) {
    if (!initialized_ || inputIds.empty()) {
        return {};
    }
    
    // Embed tokens
    auto embeddings = embedTokens(inputIds);
    
    // Apply positional encoding
    embeddings = applyPositionalEncoding(embeddings, inputIds.size());
    
    // Pass through transformer layers
    auto hidden = embeddings;
    for (auto& layer : layers_) {
        hidden = layer->forward(hidden);
    }
    
    // Apply final layer norm
    hidden = layerNorm(hidden, outputNormWeights_);
    
    // Project to vocabulary size (simplified - just return last token's logits)
    size_t hiddenSize = options_.hiddenSize;
    size_t lastTokenStart = (inputIds.size() - 1) * hiddenSize;
    
    std::vector<float> logits(getVocabSize());
    // Simplified projection - in reality this would be a matrix multiplication
    for (size_t i = 0; i < logits.size() && i < hiddenSize; ++i) {
        logits[i] = hidden[lastTokenStart + i];
    }
    
    return logits;
}

bool QwenTextModel::loadModel(const std::string& modelPath) {
    // Load model weights from file
    return loadWeights(modelPath);
}

void QwenTextModel::setOptions(const TextModelOptions& options) {
    options_ = options;
}

std::vector<float> QwenTextModel::embedTokens(const std::vector<int32_t>& tokenIds) {
    size_t hiddenSize = options_.hiddenSize;
    std::vector<float> embeddings(tokenIds.size() * hiddenSize);
    
    // Simplified embedding lookup
    for (size_t i = 0; i < tokenIds.size(); ++i) {
        int32_t tokenId = tokenIds[i];
        size_t embeddingStart = i * hiddenSize;
        
        // Copy embedding (simplified - normally would index into embedding matrix)
        for (size_t j = 0; j < hiddenSize; ++j) {
            embeddings[embeddingStart + j] = static_cast<float>(tokenId) / 1000.0f; // Placeholder
        }
    }
    
    return embeddings;
}

std::vector<float> QwenTextModel::applyPositionalEncoding(
    const std::vector<float>& embeddings,
    size_t sequenceLength) {
    
    // Simplified positional encoding
    std::vector<float> result = embeddings;
    size_t hiddenSize = options_.hiddenSize;
    
    for (size_t pos = 0; pos < sequenceLength; ++pos) {
        for (size_t i = 0; i < hiddenSize; ++i) {
            size_t idx = pos * hiddenSize + i;
            // Add simple positional encoding
            result[idx] += std::sin(pos / 10000.0f * (i + 1));
        }
    }
    
    return result;
}

bool QwenTextModel::loadConfig(const std::string& configPath) {
    // Simplified config loading - in reality would parse JSON/YAML
    std::ifstream file(configPath);
    if (!file.is_open()) {
        // Use default configuration
        return true;
    }
    
    // Parse configuration file and update options_
    // For now, just use defaults
    return true;
}

bool QwenTextModel::loadWeights(const std::string& weightsPath) {
    // Load model weights from file
    // This would typically load from a binary format like GGUF
    return true;
}

std::vector<float> QwenTextModel::layerNorm(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    float eps) {
    
    std::vector<float> output = input;
    
    // Simplified layer normalization
    size_t hiddenSize = weights.size();
    size_t sequenceLength = input.size() / hiddenSize;
    
    for (size_t seq = 0; seq < sequenceLength; ++seq) {
        size_t start = seq * hiddenSize;
        
        // Calculate mean
        float mean = 0.0f;
        for (size_t i = 0; i < hiddenSize; ++i) {
            mean += input[start + i];
        }
        mean /= hiddenSize;
        
        // Calculate variance
        float variance = 0.0f;
        for (size_t i = 0; i < hiddenSize; ++i) {
            float diff = input[start + i] - mean;
            variance += diff * diff;
        }
        variance /= hiddenSize;
        
        // Normalize
        float stddev = std::sqrt(variance + eps);
        for (size_t i = 0; i < hiddenSize; ++i) {
            output[start + i] = (input[start + i] - mean) / stddev * weights[i];
        }
    }
    
    return output;
}

std::vector<float> QwenTextModel::softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    
    // Find max for numerical stability
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - maxLogit);
        sum += probabilities[i];
    }
    
    // Normalize
    for (size_t i = 0; i < probabilities.size(); ++i) {
        probabilities[i] /= sum;
    }
    
    return probabilities;
}

int32_t QwenTextModel::sampleToken(
    const std::vector<float>& probabilities,
    float temperature,
    float topP) {
    
    // Simple sampling implementation
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    // Apply temperature
    std::vector<float> scaledProbs(probabilities.size());
    for (size_t i = 0; i < probabilities.size(); ++i) {
        scaledProbs[i] = std::pow(probabilities[i], 1.0f / temperature);
    }
    
    // Renormalize
    float sum = 0.0f;
    for (float p : scaledProbs) {
        sum += p;
    }
    for (float& p : scaledProbs) {
        p /= sum;
    }
    
    // Simple random sampling (top-p sampling would be more complex)
    float random = dis(gen);
    float cumulative = 0.0f;
    
    for (size_t i = 0; i < scaledProbs.size(); ++i) {
        cumulative += scaledProbs[i];
        if (random <= cumulative) {
            return static_cast<int32_t>(i);
        }
    }
    
    return 0; // Fallback
}

// Factory function
std::unique_ptr<BaseModel> createQwenTextModel(const std::string& configPath) {
    auto model = std::make_unique<QwenTextModel>();
    if (model->initialize(configPath)) {
        return std::move(model);
    }
    return nullptr;
}

} // namespace model
} // namespace duorou