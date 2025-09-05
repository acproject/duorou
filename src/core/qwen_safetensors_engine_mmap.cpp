#include "qwen_safetensors_engine_mmap.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace duorou {

QwenSafeTensorsEngineMmap::QwenSafeTensorsEngineMmap(const QwenSafeTensorsConfig& config)
    : config_(config), state_(EngineState::UNINITIALIZED) {
    log("INFO", "Initializing QwenSafeTensorsEngineMmap with mmap support");
    log("INFO", "Model path: " + config_.model_path);
    log("INFO", "Max context length: " + std::to_string(config_.max_context_length));
    log("INFO", "Using memory mapping: " + std::string(config_.use_mmap ? "true" : "false"));
}

QwenSafeTensorsEngineMmap::~QwenSafeTensorsEngineMmap() {
    cleanup();
}

bool QwenSafeTensorsEngineMmap::loadModel() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (state_ == EngineState::READY) {
        log("INFO", "Model already loaded");
        return true;
    }
    
    setState(EngineState::LOADING);
    log("INFO", "Starting model loading with memory mapping...");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Initialize model loader with mmap
        model_loader_ = std::make_unique<SafeTensorsModelLoaderMmap>(config_.verbose);
        
        // Load the model
        if (!model_loader_->loadModel(config_.model_path)) {
            log("ERROR", "Failed to load SafeTensors model from: " + config_.model_path);
            setState(EngineState::ERROR);
            return false;
        }
        
        log("INFO", "SafeTensors files loaded successfully");
        
        // Load model components
        if (!loadArchitecture()) {
            log("ERROR", "Failed to load model architecture");
            setState(EngineState::ERROR);
            return false;
        }
        
        if (!loadVocabulary()) {
            log("ERROR", "Failed to load vocabulary");
            setState(EngineState::ERROR);
            return false;
        }
        
        if (!loadWeights()) {
            log("ERROR", "Failed to load model weights");
            setState(EngineState::ERROR);
            return false;
        }
        
        if (!validateModel()) {
            log("ERROR", "Model validation failed");
            setState(EngineState::ERROR);
            return false;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        setState(EngineState::READY);
        log("INFO", "Model loaded successfully in " + std::to_string(duration.count()) + " ms");
        log("INFO", "Memory usage: " + std::to_string(getMemoryUsage() / 1024 / 1024) + " MB");
        
        return true;
        
    } catch (const std::exception& e) {
        log("ERROR", "Exception during model loading: " + std::string(e.what()));
        setState(EngineState::ERROR);
        return false;
    }
}

bool QwenSafeTensorsEngineMmap::unloadModel() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (state_ == EngineState::UNINITIALIZED) {
        log("INFO", "Model already unloaded");
        return true;
    }
    
    log("INFO", "Unloading model...");
    cleanup();
    setState(EngineState::UNINITIALIZED);
    log("INFO", "Model unloaded successfully");
    
    return true;
}

bool QwenSafeTensorsEngineMmap::isModelLoaded() const {
    return state_ == EngineState::READY;
}

EngineState QwenSafeTensorsEngineMmap::getState() const {
    return state_.load();
}

std::string QwenSafeTensorsEngineMmap::getModelInfo() const {
    std::ostringstream oss;
    oss << "QwenSafeTensorsEngineMmap Model Information:\n";
    oss << "  Model Type: " << architecture_.model_type << "\n";
    oss << "  Vocabulary Size: " << architecture_.vocab_size << "\n";
    oss << "  Hidden Size: " << architecture_.hidden_size << "\n";
    oss << "  Number of Layers: " << architecture_.num_layers << "\n";
    oss << "  Attention Heads: " << architecture_.num_attention_heads << "\n";
    oss << "  Key-Value Heads: " << architecture_.num_key_value_heads << "\n";
    oss << "  Intermediate Size: " << architecture_.intermediate_size << "\n";
    oss << "  Max Position Embeddings: " << architecture_.max_position_embeddings << "\n";
    oss << "  RMS Norm Epsilon: " << architecture_.rms_norm_eps << "\n";
    oss << "  Memory Usage: " << (getMemoryUsage() / 1024 / 1024) << " MB\n";
    oss << "  Model Size: " << (getModelSize() / 1024 / 1024) << " MB\n";
    oss << "  State: ";
    
    switch (getState()) {
        case EngineState::UNINITIALIZED: oss << "Uninitialized"; break;
        case EngineState::LOADING: oss << "Loading"; break;
        case EngineState::READY: oss << "Ready"; break;
        case EngineState::GENERATING: oss << "Generating"; break;
        case EngineState::ERROR: oss << "Error"; break;
    }
    
    return oss.str();
}

std::vector<int> QwenSafeTensorsEngineMmap::tokenize(const std::string& text) const {
    if (!isModelLoaded()) {
        log("ERROR", "Model not loaded for tokenization");
        return {};
    }
    
    // Simple whitespace tokenization (placeholder)
    // In a real implementation, this would use a proper tokenizer
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        auto it = vocab_.find(word);
        if (it != vocab_.end()) {
            tokens.push_back(it->second);
        } else {
            // Use unknown token ID (typically 0 or a special value)
            tokens.push_back(0);
        }
    }
    
    log("DEBUG", "Tokenized '" + text + "' to " + std::to_string(tokens.size()) + " tokens");
    return tokens;
}

std::string QwenSafeTensorsEngineMmap::detokenize(const std::vector<int>& tokens) const {
    if (!isModelLoaded()) {
        log("ERROR", "Model not loaded for detokenization");
        return "";
    }
    
    std::ostringstream oss;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) oss << " ";
        
        auto it = reverse_vocab_.find(tokens[i]);
        if (it != reverse_vocab_.end()) {
            oss << it->second;
        } else {
            oss << "<unk>";
        }
    }
    
    return oss.str();
}

std::string QwenSafeTensorsEngineMmap::generate(const std::string& prompt, int max_tokens) {
    if (!isModelLoaded()) {
        log("ERROR", "Model not loaded for generation");
        return "";
    }
    
    setState(EngineState::GENERATING);
    log("INFO", "Generating text for prompt: '" + prompt + "'");
    
    auto input_tokens = tokenize(prompt);
    auto output_tokens = generateTokens(input_tokens, max_tokens);
    auto result = detokenize(output_tokens);
    
    setState(EngineState::READY);
    log("INFO", "Generated " + std::to_string(output_tokens.size()) + " tokens");
    
    return result;
}

std::vector<int> QwenSafeTensorsEngineMmap::generateTokens(const std::vector<int>& input_tokens, int max_tokens) {
    if (!isModelLoaded()) {
        log("ERROR", "Model not loaded for token generation");
        return {};
    }
    
    // Placeholder implementation - in a real scenario, this would perform actual inference
    // using the memory-mapped weights
    std::vector<int> result = input_tokens;
    
    // Simple mock generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, std::min(1000, static_cast<int>(architecture_.vocab_size)));
    
    for (int i = 0; i < max_tokens; ++i) {
        // In a real implementation, this would:
        // 1. Run forward pass through the model using mmap'd weights
        // 2. Apply temperature and sampling
        // 3. Select next token
        
        int next_token = dis(gen);
        result.push_back(next_token);
        
        // Simple stopping condition
        if (next_token == 2) { // Assume 2 is EOS token
            break;
        }
    }
    
    return result;
}

std::vector<std::string> QwenSafeTensorsEngineMmap::getSupportedQuantizations() const {
    return {"F32", "F16", "BF16", "Q8_0", "Q4_0"};
}

size_t QwenSafeTensorsEngineMmap::getMemoryUsage() const {
    if (!model_loader_) {
        return 0;
    }
    
    // Memory mapping doesn't load data into RAM until accessed
    // Return a conservative estimate
    return getModelSize() / 10; // Assume 10% of model size is actually in memory
}

size_t QwenSafeTensorsEngineMmap::getModelSize() const {
    if (!model_loader_) {
        return 0;
    }
    
    // Calculate total size of all tensors
    size_t total_size = 0;
    auto tensor_names = model_loader_->getAllTensorNames();
    
    for (const auto& name : tensor_names) {
        const TensorInfo* info = model_loader_->getTensorInfo(name);
        if (info) {
            total_size += info->data_size;
        }
    }
    
    return total_size;
}

std::vector<std::string> QwenSafeTensorsEngineMmap::getTensorNames() const {
    if (!model_loader_) {
        return {};
    }
    return model_loader_->getAllTensorNames();
}

const TensorInfo* QwenSafeTensorsEngineMmap::getTensorInfo(const std::string& name) const {
    if (!model_loader_) {
        return nullptr;
    }
    return model_loader_->getTensorInfo(name);
}

const void* QwenSafeTensorsEngineMmap::getTensorDataPtr(const std::string& name) const {
    if (!model_loader_) {
        return nullptr;
    }
    return model_loader_->getTensorDataPtr(name);
}

bool QwenSafeTensorsEngineMmap::loadArchitecture() {
    log("INFO", "Loading model architecture...");
    
    // Try to infer architecture from tensor shapes and names
    auto tensor_names = model_loader_->getAllTensorNames();
    
    // Look for common architecture tensors
    for (const auto& name : tensor_names) {
        const TensorInfo* info = model_loader_->getTensorInfo(name);
        if (!info) continue;
        
        if (name.find("embed_tokens") != std::string::npos || 
            name.find("token_embedding") != std::string::npos) {
            if (info->shape.size() >= 2) {
                architecture_.vocab_size = info->shape[0];
                architecture_.hidden_size = info->shape[1];
            }
        }
        
        if (name.find("layers.") != std::string::npos) {
            // Extract layer number to determine total layers
            size_t layer_pos = name.find("layers.");
            if (layer_pos != std::string::npos) {
                size_t dot_pos = name.find(".", layer_pos + 7);
                if (dot_pos != std::string::npos) {
                    std::string layer_str = name.substr(layer_pos + 7, dot_pos - layer_pos - 7);
                    try {
                        int layer_num = std::stoi(layer_str);
                        architecture_.num_layers = std::max(architecture_.num_layers, static_cast<uint32_t>(layer_num + 1));
                    } catch (...) {
                        // Ignore parsing errors
                    }
                }
            }
        }
    }
    
    // Set reasonable defaults if not found
    if (architecture_.vocab_size == 0) architecture_.vocab_size = 32000;
    if (architecture_.hidden_size == 0) architecture_.hidden_size = 4096;
    if (architecture_.num_layers == 0) architecture_.num_layers = 32;
    if (architecture_.num_attention_heads == 0) architecture_.num_attention_heads = 32;
    if (architecture_.num_key_value_heads == 0) architecture_.num_key_value_heads = 32;
    if (architecture_.intermediate_size == 0) architecture_.intermediate_size = 11008;
    if (architecture_.max_position_embeddings == 0) architecture_.max_position_embeddings = 2048;
    
    architecture_.model_type = "qwen";
    
    log("INFO", "Architecture loaded: " + std::to_string(architecture_.num_layers) + " layers, " +
                std::to_string(architecture_.hidden_size) + " hidden size, " +
                std::to_string(architecture_.vocab_size) + " vocab size");
    
    return true;
}

bool QwenSafeTensorsEngineMmap::loadVocabulary() {
    log("INFO", "Loading vocabulary...");
    
    // Create a simple vocabulary for testing
    // In a real implementation, this would load from tokenizer files
    vocab_.clear();
    reverse_vocab_.clear();
    
    // Add some common tokens
    std::vector<std::string> common_tokens = {
        "<unk>", "<s>", "</s>", "<pad>",
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "among", "throughout",
        "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "this", "that", "these", "those", "what", "which", "who", "when", "where", "why", "how"
    };
    
    for (size_t i = 0; i < common_tokens.size(); ++i) {
        vocab_[common_tokens[i]] = static_cast<int>(i);
        reverse_vocab_[static_cast<int>(i)] = common_tokens[i];
    }
    
    // Fill remaining vocabulary with placeholder tokens
    for (int i = common_tokens.size(); i < static_cast<int>(architecture_.vocab_size); ++i) {
        std::string token = "token_" + std::to_string(i);
        vocab_[token] = i;
        reverse_vocab_[i] = token;
    }
    
    log("INFO", "Vocabulary loaded: " + std::to_string(vocab_.size()) + " tokens");
    return true;
}

bool QwenSafeTensorsEngineMmap::loadWeights() {
    log("INFO", "Loading model weights using memory mapping...");
    
    auto tensor_names = model_loader_->getAllTensorNames();
    int loaded_tensors = 0;
    
    // Load token embeddings
    for (const auto& name : tensor_names) {
        if (name.find("embed_tokens") != std::string::npos || 
            name.find("token_embedding") != std::string::npos) {
            weights_.token_embeddings = getTensorDataPtr(name);
            if (weights_.token_embeddings) {
                log("DEBUG", "Loaded token embeddings: " + name);
                loaded_tensors++;
            }
            break;
        }
    }
    
    // Load layer weights
    weights_.layer_attention_weights.resize(architecture_.num_layers, nullptr);
    weights_.layer_ffn_weights.resize(architecture_.num_layers, nullptr);
    weights_.layer_norm_weights.resize(architecture_.num_layers, nullptr);
    
    for (uint32_t layer = 0; layer < architecture_.num_layers; ++layer) {
        std::string layer_prefix = "layers." + std::to_string(layer) + ".";
        
        for (const auto& name : tensor_names) {
            if (name.find(layer_prefix) == 0) {
                const void* ptr = getTensorDataPtr(name);
                if (ptr) {
                    if (name.find("attention") != std::string::npos) {
                        if (!weights_.layer_attention_weights[layer]) {
                            weights_.layer_attention_weights[layer] = ptr;
                            loaded_tensors++;
                        }
                    } else if (name.find("mlp") != std::string::npos || name.find("ffn") != std::string::npos) {
                        if (!weights_.layer_ffn_weights[layer]) {
                            weights_.layer_ffn_weights[layer] = ptr;
                            loaded_tensors++;
                        }
                    } else if (name.find("norm") != std::string::npos) {
                        if (!weights_.layer_norm_weights[layer]) {
                            weights_.layer_norm_weights[layer] = ptr;
                            loaded_tensors++;
                        }
                    }
                }
            }
        }
    }
    
    // Load output weights
    for (const auto& name : tensor_names) {
        if (name.find("lm_head") != std::string::npos || 
            name.find("output") != std::string::npos) {
            weights_.output_weights = getTensorDataPtr(name);
            if (weights_.output_weights) {
                log("DEBUG", "Loaded output weights: " + name);
                loaded_tensors++;
            }
            break;
        }
    }
    
    log("INFO", "Loaded " + std::to_string(loaded_tensors) + " weight tensors using memory mapping");
    return loaded_tensors > 0;
}

bool QwenSafeTensorsEngineMmap::validateModel() {
    log("INFO", "Validating model...");
    
    // Check that essential components are loaded
    if (!weights_.token_embeddings) {
        log("WARNING", "Token embeddings not found");
    }
    
    if (!weights_.output_weights) {
        log("WARNING", "Output weights not found");
    }
    
    // Check layer weights
    int valid_layers = 0;
    for (uint32_t i = 0; i < architecture_.num_layers; ++i) {
        if (weights_.layer_attention_weights[i] || 
            weights_.layer_ffn_weights[i] || 
            weights_.layer_norm_weights[i]) {
            valid_layers++;
        }
    }
    
    log("INFO", "Validation complete: " + std::to_string(valid_layers) + "/" + 
                std::to_string(architecture_.num_layers) + " layers have weights");
    
    return true; // Always return true for now, as this is a demonstration
}

std::vector<float> QwenSafeTensorsEngineMmap::forward(const std::vector<int>& tokens) {
    // Placeholder for actual forward pass
    // In a real implementation, this would use the memory-mapped weights
    // to perform the actual neural network computation
    
    std::vector<float> logits(architecture_.vocab_size, 0.0f);
    
    // Simple mock logits
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    for (auto& logit : logits) {
        logit = dis(gen);
    }
    
    return logits;
}

int QwenSafeTensorsEngineMmap::sampleToken(const std::vector<float>& logits) {
    // Apply temperature
    std::vector<float> scaled_logits = logits;
    for (auto& logit : scaled_logits) {
        logit /= config_.temperature;
    }
    
    // Simple sampling (in a real implementation, would use top-k/top-p)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(scaled_logits.begin(), scaled_logits.end());
    
    return dis(gen);
}

void QwenSafeTensorsEngineMmap::log(const std::string& level, const std::string& message) const {
    if (config_.verbose || level == "ERROR") {
        std::cout << "[QwenSafeTensorsEngineMmap " << level << "] " << message << std::endl;
    }
}

void QwenSafeTensorsEngineMmap::setState(EngineState state) {
    state_.store(state);
}

void QwenSafeTensorsEngineMmap::cleanup() {
    model_loader_.reset();
    vocab_.clear();
    reverse_vocab_.clear();
    
    // Clear weight pointers (memory is managed by model_loader_)
    weights_.token_embeddings = nullptr;
    weights_.layer_attention_weights.clear();
    weights_.layer_ffn_weights.clear();
    weights_.layer_norm_weights.clear();
    weights_.output_weights = nullptr;
    
    // Reset architecture
    architecture_ = ModelArchitecture{};
}

} // namespace duorou