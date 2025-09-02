#include "attention_handler.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Static member initialization
std::unordered_map<std::string, std::shared_ptr<AttentionHandler::AttentionConfig>> AttentionHandler::attentionConfigs;
bool AttentionHandler::initialized = false;

std::shared_ptr<AttentionHandler::AttentionConfig> AttentionHandler::getAttentionConfig(const std::string& architecture) {
    if (!initialized) {
        initialize();
    }
    
    auto it = attentionConfigs.find(architecture);
    if (it != attentionConfigs.end()) {
        return it->second;
    }
    
    return nullptr;
}

bool AttentionHandler::hasAdvancedAttention(const std::string& architecture) {
    auto config = getAttentionConfig(architecture);
    if (!config) return false;
    
    return config->type != AttentionType::STANDARD ||
           config->hasSlidingWindow ||
           config->hasAttentionSinks ||
           config->hasGlobalLocal ||
           config->hasSoftcapping;
}

bool AttentionHandler::usesSlidingWindow(const std::string& architecture) {
    auto config = getAttentionConfig(architecture);
    return config && config->hasSlidingWindow;
}

bool AttentionHandler::usesAttentionSinks(const std::string& architecture) {
    auto config = getAttentionConfig(architecture);
    return config && config->hasAttentionSinks;
}

int AttentionHandler::getEffectiveContextLength(
    const std::string& architecture,
    int layerIndex,
    int baseContextLength
) {
    auto config = getAttentionConfig(architecture);
    if (!config) return baseContextLength;
    
    // Global-local attention pattern
    if (config->hasGlobalLocal) {
        if (isGlobalLayer(layerIndex, config->globalLayerInterval)) {
            return baseContextLength; // Global attention
        } else {
            // Local attention with sliding window
            return std::min(baseContextLength, config->slidingWindowSize);
        }
    }
    
    // Standard sliding window
    if (config->hasSlidingWindow) {
        return std::min(baseContextLength, config->slidingWindowSize);
    }
    
    return baseContextLength;
}

std::vector<int> AttentionHandler::calculateAttentionMask(
    const AttentionHandler::AttentionConfig& config,
    int sequenceLength,
    int layerIndex
) {
    std::vector<int> mask;
    
    if (config.hasGlobalLocal) {
        if (isGlobalLayer(layerIndex, config.globalLayerInterval)) {
            // Global attention - attend to all positions
            mask.resize(sequenceLength, 1);
        } else {
            // Local attention with sliding window
            mask.resize(sequenceLength, 0);
            int windowSize = config.slidingWindowSize;
            for (int i = 0; i < sequenceLength; ++i) {
                int start = std::max(0, i - windowSize + 1);
                int end = std::min(sequenceLength, i + 1);
                for (int j = start; j < end; ++j) {
                    mask[j] = 1;
                }
            }
        }
    } else if (config.hasSlidingWindow) {
        // Standard sliding window
        mask.resize(sequenceLength, 0);
        int windowSize = config.slidingWindowSize;
        for (int i = 0; i < sequenceLength; ++i) {
            int start = std::max(0, i - windowSize + 1);
            int end = std::min(sequenceLength, i + 1);
            for (int j = start; j < end; ++j) {
                mask[j] = 1;
            }
        }
    } else {
        // Standard full attention
        mask.resize(sequenceLength, 1);
    }
    
    // Apply attention sinks
    if (config.hasAttentionSinks) {
        for (int sinkPos : config.sinkPositions) {
            if (sinkPos < sequenceLength) {
                mask[sinkPos] = 1;
            }
        }
    }
    
    return mask;
}

std::vector<float> AttentionHandler::applySoftcapping(
    const AttentionHandler::AttentionConfig& config,
    const std::vector<float>& logits,
    bool isFinal
) {
    if (!config.hasSoftcapping) {
        return logits;
    }
    
    std::vector<float> result = logits;
    float cap = isFinal ? config.finalLogitSoftcap : config.attentionLogitSoftcap;
    
    if (cap > 0.0f) {
        for (float& logit : result) {
            logit = softcap(logit, cap);
        }
    }
    
    return result;
}

std::unordered_map<std::string, float> AttentionHandler::getRoPEParams(const std::string& architecture) {
    auto config = getAttentionConfig(architecture);
    std::unordered_map<std::string, float> params;
    
    if (config) {
        params["base"] = config->ropeBase;
        params["scale"] = config->ropeScale;
        params["original_context_length"] = static_cast<float>(config->originalContextLength);
        params["use_neox"] = config->useNeoXRoPE ? 1.0f : 0.0f;
    } else {
        // Default RoPE parameters
        params["base"] = 10000.0f;
        params["scale"] = 1.0f;
        params["original_context_length"] = 2048.0f;
        params["use_neox"] = 0.0f;
    }
    
    return params;
}

void AttentionHandler::initialize() {
    if (initialized) return;
    
    createGemma2AttentionConfig();
    createGemma3AttentionConfig();
    createGemma3nAttentionConfig();
    createGptossAttentionConfig();
    createMistral3AttentionConfig();
    createQwen25vlAttentionConfig();
    createLlamaAttentionConfig();
    createQwen2AttentionConfig();
    createQwen3AttentionConfig();
    
    initialized = true;
}

void AttentionHandler::createGemma2AttentionConfig() {
    auto config = std::make_shared<AttentionConfig>();
    config->architecture = "gemma2";
    config->type = AttentionType::SLIDING_WINDOW;
    config->hasSlidingWindow = true;
    config->slidingWindowSize = 4096;
    config->hasSoftcapping = true;
    config->attentionLogitSoftcap = 50.0f;
    config->finalLogitSoftcap = 30.0f;
    config->ropeBase = 10000.0f;
    config->ropeScale = 1.0f;
    config->originalContextLength = 8192;
    
    attentionConfigs["gemma2"] = config;
}

void AttentionHandler::createGemma3AttentionConfig() {
    auto config = std::make_shared<AttentionConfig>();
    config->architecture = "gemma3";
    config->type = AttentionType::GLOBAL_LOCAL;
    config->hasGlobalLocal = true;
    config->globalLayerInterval = 6; // Every 6th layer is global
    config->hasSlidingWindow = true;
    config->slidingWindowSize = 4096;
    config->hasSoftcapping = true;
    config->attentionLogitSoftcap = 50.0f;
    config->finalLogitSoftcap = 30.0f;
    config->ropeBase = 10000.0f;
    config->ropeScale = 1.0f;
    config->originalContextLength = 8192;
    
    attentionConfigs["gemma3"] = config;
}

void AttentionHandler::createGemma3nAttentionConfig() {
    auto config = std::make_shared<AttentionConfig>();
    config->architecture = "gemma3n";
    config->type = AttentionType::GLOBAL_LOCAL;
    config->hasGlobalLocal = true;
    config->globalLayerInterval = 6;
    config->hasSlidingWindow = true;
    config->slidingWindowSize = 4096;
    config->hasSoftcapping = true;
    config->attentionLogitSoftcap = 50.0f;
    config->finalLogitSoftcap = 30.0f;
    config->ropeBase = 10000.0f;
    config->ropeScale = 1.0f;
    config->originalContextLength = 8192;
    
    attentionConfigs["gemma3n"] = config;
}

void AttentionHandler::createGptossAttentionConfig() {
    auto config = std::make_shared<AttentionConfig>();
    config->architecture = "gptoss";
    config->type = AttentionType::STANDARD;
    config->ropeBase = 10000.0f;
    config->ropeScale = 1.0f;
    config->originalContextLength = 2048;
    config->useNeoXRoPE = true;
    
    attentionConfigs["gptoss"] = config;
    attentionConfigs["gpt-oss"] = config; // Alternative name
}

void AttentionHandler::createMistral3AttentionConfig() {
    auto config = std::make_shared<AttentionConfig>();
    config->architecture = "mistral3";
    config->type = AttentionType::SLIDING_WINDOW;
    config->hasSlidingWindow = true;
    config->slidingWindowSize = 131072; // 128k sliding window
    config->ropeBase = 1000000.0f;
    config->ropeScale = 1.0f;
    config->originalContextLength = 131072;
    
    attentionConfigs["mistral3"] = config;
}

void AttentionHandler::createQwen25vlAttentionConfig() {
    auto config = std::make_shared<AttentionConfig>();
    config->architecture = "qwen25vl";
    config->type = AttentionType::STANDARD;
    config->ropeBase = 1000000.0f;
    config->ropeScale = 1.0f;
    config->originalContextLength = 32768;
    config->useNeoXRoPE = false;
    
    attentionConfigs["qwen25vl"] = config;
    attentionConfigs["qwen2.5vl"] = config; // Alternative name
}

void AttentionHandler::createLlamaAttentionConfig() {
    auto config = std::make_shared<AttentionConfig>();
    config->architecture = "llama";
    config->type = AttentionType::STANDARD;
    config->ropeBase = 10000.0f;
    config->ropeScale = 1.0f;
    config->originalContextLength = 2048;
    config->useNeoXRoPE = false;
    
    attentionConfigs["llama"] = config;
}

void AttentionHandler::createQwen2AttentionConfig() {
    auto config = std::make_shared<AttentionConfig>();
    config->architecture = "qwen2";
    config->type = AttentionType::STANDARD;
    config->ropeBase = 1000000.0f;
    config->ropeScale = 1.0f;
    config->originalContextLength = 32768;
    config->useNeoXRoPE = false;
    
    attentionConfigs["qwen2"] = config;
}

void AttentionHandler::createQwen3AttentionConfig() {
    auto config = std::make_shared<AttentionConfig>();
    config->architecture = "qwen3";
    config->type = AttentionType::STANDARD;
    config->ropeBase = 1000000.0f;
    config->ropeScale = 1.0f;
    config->originalContextLength = 32768;
    config->useNeoXRoPE = false;
    
    attentionConfigs["qwen3"] = config;
}

float AttentionHandler::softcap(float x, float cap) {
    if (cap <= 0.0f) return x;
    return cap * std::tanh(x / cap);
}

bool AttentionHandler::isGlobalLayer(int layerIndex, int interval) {
    return (layerIndex % interval) == 0;
}