#include "model.h"
#include "byte_pair_encoding.h"
#include "sentence_piece.h"
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>

namespace duorou {
namespace model {

// BaseModel implementation
BaseModel::BaseModel() 
    : loaded_(false), modelName_("BaseModel"), modelVersion_("1.0") {
}

BaseModel::~BaseModel() {
    unload();
}

bool BaseModel::load(const std::string& modelPath) {
    if (loaded_) {
        unload();
    }
    
    modelPath_ = modelPath;
    
    // Load configuration
    if (!loadConfig(modelPath + "/config.json")) {
        return false;
    }
    
    // Load tokenizer
    if (!loadTokenizer(modelPath + "/tokenizer")) {
        return false;
    }
    
    // Load model weights
    if (!loadModel(modelPath + "/model.bin")) {
        return false;
    }
    
    loaded_ = true;
    return true;
}

bool BaseModel::isLoaded() const {
    return loaded_;
}

void BaseModel::unload() {
    tokenizer_.reset();
    vocabulary_.reset();
    loaded_ = false;
    modelPath_.clear();
    metadata_.clear();
}

std::vector<int32_t> BaseModel::encode(const std::string& text, bool addSpecial) {
    if (!tokenizer_) {
        return {};
    }
    return tokenizer_->encode(text, addSpecial);
}

std::string BaseModel::decode(const std::vector<int32_t>& tokens) {
    if (!tokenizer_) {
        return "";
    }
    return tokenizer_->decode(tokens);
}

std::vector<int32_t> BaseModel::generate(const std::vector<int32_t>& prompt, size_t maxTokens) {
    if (!loaded_ || !tokenizer_) {
        return {};
    }
    
    std::vector<int32_t> result = prompt;
    
    for (size_t i = 0; i < maxTokens; ++i) {
        // Limit context to model's context length
        std::vector<int32_t> context = result;
        if (context.size() > config_.context_length) {
            context = std::vector<int32_t>(
                result.end() - config_.context_length, 
                result.end()
            );
        }
        
        int32_t nextToken = sampleNext(context);
        if (nextToken < 0) {
            break;
        }
        
        result.push_back(nextToken);
        
        // Check for end-of-sequence token
        if (tokenizer_->isSpecial(nextToken, Special::EOS)) {
            break;
        }
    }
    
    return result;
}

std::string BaseModel::generateText(const std::string& prompt, size_t maxTokens) {
    auto promptTokens = encode(prompt, true);
    auto generatedTokens = generate(promptTokens, maxTokens);
    return decode(generatedTokens);
}

const ModelConfig& BaseModel::getConfig() const {
    return config_;
}

const TextProcessor* BaseModel::getTokenizer() const {
    return tokenizer_.get();
}

size_t BaseModel::getVocabSize() const {
    return config_.vocab_size;
}

size_t BaseModel::getContextLength() const {
    return config_.context_length;
}

std::string BaseModel::getModelName() const {
    return modelName_;
}

std::string BaseModel::getModelVersion() const {
    return modelVersion_;
}

std::map<std::string, std::string> BaseModel::getMetadata() const {
    return metadata_;
}

bool BaseModel::loadModel(const std::string& modelPath) {
    // Basic implementation - just check if file exists
    std::ifstream file(modelPath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // In a real implementation, this would load model weights
    // For now, just verify the file exists
    file.close();
    return true;
}

bool BaseModel::loadTokenizer(const std::string& tokenizerPath) {
    // Try to load vocabulary
    vocabulary_ = std::make_shared<Vocabulary>();
    
    // Load vocabulary from files (simplified implementation)
    std::vector<std::string> values;
    std::vector<int32_t> types;
    std::vector<float> scores;
    std::vector<std::string> merges;
    
    // Load vocabulary file
    std::ifstream vocabFile(tokenizerPath + "/vocab.txt");
    if (vocabFile.is_open()) {
        std::string line;
        while (std::getline(vocabFile, line)) {
            if (!line.empty()) {
                values.push_back(line);
                types.push_back(0); // Normal token
                scores.push_back(0.0f);
            }
        }
        vocabFile.close();
    }
    
    // Load merges file
    std::ifstream mergesFile(tokenizerPath + "/merges.txt");
    if (mergesFile.is_open()) {
        std::string line;
        while (std::getline(mergesFile, line)) {
            if (!line.empty()) {
                merges.push_back(line);
            }
        }
        mergesFile.close();
    }
    
    // Initialize vocabulary
    vocabulary_->initialize(values, types, scores, merges);
    
    if (values.empty()) {
        return false;
    }
    
    // Determine tokenizer type and create appropriate tokenizer
    if (config_.tokenizer_type == "bpe" || config_.tokenizer_type == "BytePairEncoding") {
        // Create BPE tokenizer
        std::string pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
        tokenizer_ = std::make_unique<BytePairEncoding>(pattern, vocabulary_);
    } else if (config_.tokenizer_type == "spm" || config_.tokenizer_type == "SentencePiece") {
        // Create SentencePiece tokenizer
        tokenizer_ = std::make_unique<SentencePiece>(vocabulary_);
    } else {
        // Default to BPE
        std::string pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
        tokenizer_ = std::make_unique<BytePairEncoding>(pattern, vocabulary_);
    }
    
    config_.vocab_size = vocabulary_->size();
    return true;
}

bool BaseModel::loadConfig(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        // Use default configuration
        config_ = ModelConfig();
        return true;
    }
    
    // Simple JSON-like parsing (basic implementation)
    std::string line;
    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        
        // Parse key-value pairs
        size_t colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            std::string key = line.substr(0, colonPos);
            std::string value = line.substr(colonPos + 1);
            
            // Remove quotes
            if (!key.empty() && key.front() == '"' && key.back() == '"') {
                key = key.substr(1, key.length() - 2);
            }
            if (!value.empty() && value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.length() - 2);
            }
            
            // Remove trailing comma
            if (!value.empty() && value.back() == ',') {
                value.pop_back();
            }
            
            // Set configuration values
            if (key == "architecture") {
                config_.architecture = value;
            } else if (key == "tokenizer_type") {
                config_.tokenizer_type = value;
            } else if (key == "vocab_size") {
                config_.vocab_size = std::stoul(value);
            } else if (key == "context_length") {
                config_.context_length = std::stoul(value);
            } else if (key == "embedding_dim") {
                config_.embedding_dim = std::stoul(value);
            } else if (key == "num_layers") {
                config_.num_layers = std::stoul(value);
            } else if (key == "num_heads") {
                config_.num_heads = std::stoul(value);
            } else if (key == "temperature") {
                config_.temperature = std::stod(value);
            } else if (key == "top_p") {
                config_.top_p = std::stod(value);
            } else if (key == "top_k") {
                config_.top_k = std::stoi(value);
            }
            
            // Store in metadata
            metadata_[key] = value;
        }
    }
    
    file.close();
    return true;
}

int32_t BaseModel::sampleNext(const std::vector<int32_t>& context) {
    // Compute logits for next token
    auto logits = computeLogits(context);
    if (logits.empty()) {
        return -1;
    }
    
    // Sample from logits
    return sampleFromLogits(logits);
}

std::vector<double> BaseModel::computeLogits(const std::vector<int32_t>& context) {
    // Basic implementation - return uniform distribution
    // In a real model, this would run inference
    std::vector<double> logits(config_.vocab_size, 0.0);
    
    // Simple heuristic: prefer common tokens
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = -std::log(static_cast<double>(i + 1));
    }
    
    return logits;
}

int32_t BaseModel::sampleFromLogits(const std::vector<double>& logits) {
    if (logits.empty()) {
        return -1;
    }
    
    // Apply temperature
    std::vector<double> scaledLogits(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scaledLogits[i] = logits[i] / config_.temperature;
    }
    
    // Convert to probabilities (softmax)
    double maxLogit = *std::max_element(scaledLogits.begin(), scaledLogits.end());
    std::vector<double> probs(scaledLogits.size());
    double sum = 0.0;
    
    for (size_t i = 0; i < scaledLogits.size(); ++i) {
        probs[i] = std::exp(scaledLogits[i] - maxLogit);
        sum += probs[i];
    }
    
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }
    
    // Apply top-k filtering
    if (config_.top_k > 0 && config_.top_k < static_cast<int32_t>(probs.size())) {
        std::vector<std::pair<double, int32_t>> probPairs;
        for (size_t i = 0; i < probs.size(); ++i) {
            probPairs.emplace_back(probs[i], static_cast<int32_t>(i));
        }
        
        std::partial_sort(probPairs.begin(), 
                         probPairs.begin() + config_.top_k, 
                         probPairs.end(),
                         std::greater<std::pair<double, int32_t>>());
        
        // Zero out probabilities outside top-k
        std::fill(probs.begin(), probs.end(), 0.0);
        double newSum = 0.0;
        for (int32_t i = 0; i < config_.top_k; ++i) {
            int32_t idx = probPairs[i].second;
            probs[idx] = probPairs[i].first;
            newSum += probs[idx];
        }
        
        // Renormalize
        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i] /= newSum;
        }
    }
    
    // Apply top-p (nucleus) sampling
    if (config_.top_p < 1.0) {
        std::vector<std::pair<double, int32_t>> probPairs;
        for (size_t i = 0; i < probs.size(); ++i) {
            if (probs[i] > 0.0) {
                probPairs.emplace_back(probs[i], static_cast<int32_t>(i));
            }
        }
        
        std::sort(probPairs.begin(), probPairs.end(), 
                 std::greater<std::pair<double, int32_t>>());
        
        double cumProb = 0.0;
        size_t cutoff = probPairs.size();
        for (size_t i = 0; i < probPairs.size(); ++i) {
            cumProb += probPairs[i].first;
            if (cumProb >= config_.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Zero out probabilities outside top-p
        std::fill(probs.begin(), probs.end(), 0.0);
        double newSum = 0.0;
        for (size_t i = 0; i < cutoff; ++i) {
            int32_t idx = probPairs[i].second;
            probs[idx] = probPairs[i].first;
            newSum += probs[idx];
        }
        
        // Renormalize
        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i] /= newSum;
        }
    }
    
    // Sample from the distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    
    return dist(gen);
}

// ModelFactory implementation
std::map<std::string, ModelFactory::ModelCreator> ModelFactory::creators_;

std::unique_ptr<Model> ModelFactory::createModel(const std::string& modelType) {
    auto it = creators_.find(modelType);
    if (it != creators_.end()) {
        return it->second();
    }
    
    // Default to BaseModel
    return std::make_unique<BaseModel>();
}

std::unique_ptr<Model> ModelFactory::loadModel(const std::string& modelPath) {
    auto model = std::make_unique<BaseModel>();
    if (model->load(modelPath)) {
        return std::move(model);
    }
    return nullptr;
}

std::vector<std::string> ModelFactory::getSupportedModels() {
    std::vector<std::string> models;
    models.push_back("BaseModel");
    
    for (const auto& pair : creators_) {
        models.push_back(pair.first);
    }
    
    return models;
}

void ModelFactory::registerModel(const std::string& modelType, ModelCreator creator) {
    creators_[modelType] = creator;
}

} // namespace model
} // namespace duorou