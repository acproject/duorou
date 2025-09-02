#include "model_config_manager.h"
#include <algorithm>

// Static member definitions
std::unordered_map<std::string, std::shared_ptr<ModelConfigManager::ModelConfig>> ModelConfigManager::configs;
bool ModelConfigManager::initialized = false;

std::shared_ptr<ModelConfigManager::ModelConfig> ModelConfigManager::getConfig(const std::string& architecture) {
    if (!initialized) {
        initialize();
    }
    
    auto it = configs.find(architecture);
    if (it != configs.end()) {
        return it->second;
    }
    return nullptr;
}

bool ModelConfigManager::requiresSpecialHandling(const std::string& architecture) {
    auto config = getConfig(architecture);
    return config && (config->hasVision || config->hasSlidingWindow || 
                     config->hasAttentionSinks || config->requiresSpecialTokenHandling);
}

bool ModelConfigManager::hasVisionSupport(const std::string& architecture) {
    auto config = getConfig(architecture);
    return config && config->hasVision;
}

std::vector<std::string> ModelConfigManager::getOllamaEngineRequired() {
    return {
        "gemma3", "gemma3n", "mistral3", "llama4", "mllama", 
        "qwen25vl", "gptoss", "gpt-oss"
    };
}

void ModelConfigManager::initialize() {
    if (initialized) return;
    
    createGemma3Config();
    createGemma3nConfig();
    createMistral3Config();
    createQwen25vlConfig();
    createQwen3Config();
    createGptossConfig();
    createLlamaConfig();
    createQwen2Config();
    createGemma2Config();
    
    initialized = true;
}

void ModelConfigManager::createGemma3Config() {
    auto config = std::make_shared<ModelConfig>();
    config->architecture = "gemma3";
    config->hasVision = true;
    config->hasSlidingWindow = true;
    config->requiresSpecialTokenHandling = true;
    
    // Vision parameters
    config->imageSize = 224;
    config->patchSize = 14;
    config->tokensPerImage = 256;
    
    // Attention parameters
    config->slidingWindowSize = 4096;
    config->attentionLogitSoftcap = 50.0f;
    config->finalLogitSoftcap = 30.0f;
    
    // Special tokens
    config->specialTokenIds = {106}; // EOT token
    
    configs["gemma3"] = config;
}

void ModelConfigManager::createGemma3nConfig() {
    auto config = std::make_shared<ModelConfig>();
    config->architecture = "gemma3n";
    config->hasVision = false;
    config->hasSlidingWindow = true;
    config->requiresSpecialTokenHandling = true;
    
    // Attention parameters
    config->slidingWindowSize = 4096;
    config->attentionLogitSoftcap = 50.0f;
    config->finalLogitSoftcap = 30.0f;
    
    configs["gemma3n"] = config;
}

void ModelConfigManager::createMistral3Config() {
    auto config = std::make_shared<ModelConfig>();
    config->architecture = "mistral3";
    config->hasVision = true;
    config->requiresSpecialTokenHandling = true;
    
    // Vision parameters
    config->imageSize = 224;
    config->patchSize = 14;
    config->customIntParams["spatial_merge_size"] = 2;
    
    // Custom parameters
    config->customFloatParams["rms_norm_eps"] = 1e-5f;
    
    configs["mistral3"] = config;
}

void ModelConfigManager::createQwen25vlConfig() {
    auto config = std::make_shared<ModelConfig>();
    config->architecture = "qwen25vl";
    config->hasVision = true;
    config->requiresSpecialTokenHandling = true;
    
    // Vision parameters
    config->customIntParams["max_pixels"] = 28 * 28 * 1280;
    config->customIntParams["temporal_patch_size"] = 2;
    config->customIntParams["num_channels"] = 3;
    
    // Special tokens for vision
    config->specialTokenIds = {151655, 151652, 151653}; // image, vision_start, vision_end
    
    // 注册两个架构名称：GGUF文件中的实际名称和别名
    configs["qwen25vl"] = config;
    configs["qwen2.5vl"] = config;  // 别名支持
}

void ModelConfigManager::createQwen3Config() {
    auto config = std::make_shared<ModelConfig>();
    config->architecture = "qwen3";
    config->hasVision = false;
    config->requiresSpecialTokenHandling = false;
    
    configs["qwen3"] = config;
}

void ModelConfigManager::createGptossConfig() {
    auto config = std::make_shared<ModelConfig>();
    config->architecture = "gptoss";
    config->hasVision = false;
    config->hasSlidingWindow = true;
    config->hasAttentionSinks = true;
    config->supportsMixtral = true;
    config->requiresSpecialTokenHandling = true;
    
    // Attention parameters with sinks
    config->slidingWindowSize = 4096;
    config->customIntParams["num_experts"] = 8;
    config->customIntParams["num_experts_used"] = 2;
    
    configs["gptoss"] = config;
    configs["gpt-oss"] = config; // Alternative naming
}

void ModelConfigManager::createLlamaConfig() {
    auto config = std::make_shared<ModelConfig>();
    config->architecture = "llama";
    config->hasVision = false;
    config->requiresSpecialTokenHandling = false;
    
    configs["llama"] = config;
}

void ModelConfigManager::createQwen2Config() {
    auto config = std::make_shared<ModelConfig>();
    config->architecture = "qwen2";
    config->hasVision = false;
    config->requiresSpecialTokenHandling = false;
    
    configs["qwen2"] = config;
}

void ModelConfigManager::createGemma2Config() {
    auto config = std::make_shared<ModelConfig>();
    config->architecture = "gemma2";
    config->hasVision = false;
    config->hasSlidingWindow = true;
    config->requiresSpecialTokenHandling = false;
    
    // Attention parameters
    config->slidingWindowSize = 4096;
    config->attentionLogitSoftcap = 50.0f;
    config->finalLogitSoftcap = 30.0f;
    
    configs["gemma2"] = config;
}