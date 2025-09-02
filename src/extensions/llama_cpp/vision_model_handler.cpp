#include "vision_model_handler.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// Static member definitions
std::unordered_map<std::string, std::shared_ptr<VisionModelHandler::VisionConfig>> VisionModelHandler::visionConfigs;
bool VisionModelHandler::initialized = false;

bool VisionModelHandler::hasVisionSupport(const std::string& architecture) {
    if (!initialized) {
        initialize();
    }
    return visionConfigs.find(architecture) != visionConfigs.end();
}

std::shared_ptr<VisionModelHandler::VisionConfig> VisionModelHandler::getVisionConfig(const std::string& architecture) {
    if (!initialized) {
        initialize();
    }
    
    auto it = visionConfigs.find(architecture);
    if (it != visionConfigs.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<int> VisionModelHandler::processVisionTokens(
    const std::string& architecture,
    const std::vector<int>& tokens,
    const std::vector<uint8_t>& imageData
) {
    auto config = getVisionConfig(architecture);
    if (!config) {
        return tokens; // No vision support, return original tokens
    }
    
    std::vector<int> processedTokens = tokens;
    
    // Find image tokens and replace with vision embeddings
    for (size_t i = 0; i < processedTokens.size(); ++i) {
        if (processedTokens[i] == config->imageTokenId) {
            // Replace image token with vision start + vision tokens + vision end
            std::vector<int> visionTokens;
            
            if (config->visionStartTokenId != -1) {
                visionTokens.push_back(config->visionStartTokenId);
            }
            
            // Add placeholder vision tokens (actual vision processing would happen in llama.cpp)
            for (int j = 0; j < config->tokensPerImage; ++j) {
                visionTokens.push_back(config->imageTokenId); // Placeholder
            }
            
            if (config->visionEndTokenId != -1) {
                visionTokens.push_back(config->visionEndTokenId);
            }
            
            // Replace the image token with vision tokens
            processedTokens.erase(processedTokens.begin() + i);
            processedTokens.insert(processedTokens.begin() + i, visionTokens.begin(), visionTokens.end());
            i += visionTokens.size() - 1; // Skip the inserted tokens
        }
    }
    
    return processedTokens;
}

std::shared_ptr<VisionModelHandler::VisionConfig> VisionModelHandler::extractVisionConfig(
    const std::string& architecture,
    const std::unordered_map<std::string, std::string>& metadata
) {
    auto config = getVisionConfig(architecture);
    if (!config) {
        return nullptr;
    }
    
    // Create a copy and update with metadata
    auto extractedConfig = std::make_shared<VisionConfig>(*config);
    
    // Extract vision-specific parameters from metadata
    auto it = metadata.find("vision.image_size");
    if (it != metadata.end()) {
        extractedConfig->imageSize = std::stoi(it->second);
    }
    
    it = metadata.find("vision.patch_size");
    if (it != metadata.end()) {
        extractedConfig->patchSize = std::stoi(it->second);
    }
    
    it = metadata.find("vision.num_channels");
    if (it != metadata.end()) {
        extractedConfig->numChannels = std::stoi(it->second);
    }
    
    it = metadata.find("mm_tokens_per_image");
    if (it != metadata.end()) {
        extractedConfig->tokensPerImage = std::stoi(it->second);
    }
    
    return extractedConfig;
}

bool VisionModelHandler::containsVisionTokens(
    const std::vector<int>& tokens,
    const VisionConfig& config
) {
    for (int token : tokens) {
        if (token == config.imageTokenId || 
            token == config.visionStartTokenId || 
            token == config.visionEndTokenId) {
            return true;
        }
    }
    return false;
}

std::vector<int> VisionModelHandler::calculateVisionDimensions(
    const VisionConfig& config,
    int imageWidth,
    int imageHeight
) {
    std::vector<int> dimensions;
    
    if (config.architecture == "qwen25vl") {
        // Qwen2.5-VL uses dynamic patching
        int maxPixels = config.maxPixels > 0 ? config.maxPixels : 28 * 28 * 1280;
        int numPatches = maxPixels / (config.patchSize * config.patchSize);
        dimensions = {numPatches, config.numChannels, config.patchSize, config.patchSize};
    } else {
        // Standard vision models
        int patchesX = imageWidth / config.patchSize;
        int patchesY = imageHeight / config.patchSize;
        int totalPatches = patchesX * patchesY;
        
        if (config.architecture == "mistral3") {
            // Mistral3 uses spatial merging
            totalPatches = totalPatches / (config.spatialMergeSize * config.spatialMergeSize);
        }
        
        dimensions = {totalPatches, config.numChannels, config.patchSize, config.patchSize};
    }
    
    return dimensions;
}

void VisionModelHandler::initialize() {
    if (initialized) return;
    
    createQwen25vlVisionConfig();
    createGemma3VisionConfig();
    createMistral3VisionConfig();
    
    initialized = true;
}

void VisionModelHandler::createQwen25vlVisionConfig() {
    auto config = std::make_shared<VisionConfig>();
    config->architecture = "qwen25vl";
    config->imageSize = 224;
    config->patchSize = 14;
    config->numChannels = 3;
    config->maxPixels = 28 * 28 * 1280;
    config->temporalPatchSize = 2;
    
    // Qwen2.5-VL special tokens
    config->imageTokenId = 151655;
    config->visionStartTokenId = 151652;
    config->visionEndTokenId = 151653;
    
    visionConfigs["qwen25vl"] = config;
}

void VisionModelHandler::createGemma3VisionConfig() {
    auto config = std::make_shared<VisionConfig>();
    config->architecture = "gemma3";
    config->imageSize = 224;
    config->patchSize = 14;
    config->numChannels = 3;
    config->tokensPerImage = 256;
    
    // Gemma3 uses standard image token
    config->imageTokenId = 256000; // Placeholder, actual value from model
    
    visionConfigs["gemma3"] = config;
}

void VisionModelHandler::createMistral3VisionConfig() {
    auto config = std::make_shared<VisionConfig>();
    config->architecture = "mistral3";
    config->imageSize = 224;
    config->patchSize = 14;
    config->numChannels = 3;
    config->spatialMergeSize = 2;
    
    // Mistral3 vision parameters
    config->customParams["rms_norm_eps"] = 1e-5f;
    
    // Mistral3 uses standard image token
    config->imageTokenId = 32000; // Placeholder, actual value from model
    
    visionConfigs["mistral3"] = config;
}

std::vector<float> VisionModelHandler::preprocessImage(
    const std::vector<uint8_t>& imageData,
    const VisionConfig& config
) {
    // Basic image preprocessing - normalize to [-1, 1]
    std::vector<float> normalized;
    normalized.reserve(imageData.size());
    
    for (uint8_t pixel : imageData) {
        float normalized_pixel = (static_cast<float>(pixel) / 255.0f) * 2.0f - 1.0f;
        normalized.push_back(normalized_pixel);
    }
    
    return normalized;
}

std::vector<int> VisionModelHandler::patchifyImage(
    const std::vector<float>& imagePixels,
    const VisionConfig& config,
    int width,
    int height
) {
    std::vector<int> patches;
    
    int patchesX = width / config.patchSize;
    int patchesY = height / config.patchSize;
    
    for (int py = 0; py < patchesY; ++py) {
        for (int px = 0; px < patchesX; ++px) {
            // Extract patch coordinates
            int startX = px * config.patchSize;
            int startY = py * config.patchSize;
            
            // Add patch index (simplified representation)
            patches.push_back(py * patchesX + px);
        }
    }
    
    return patches;
}