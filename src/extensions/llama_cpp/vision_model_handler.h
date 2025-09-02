#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

/**
 * Vision model handler for processing multimodal models
 * Handles vision-language models like qwen25vl, gemma3, mistral3
 */
class VisionModelHandler {
public:
    struct VisionConfig {
        std::string architecture;
        int imageSize = 224;
        int patchSize = 14;
        int numChannels = 3;
        int tokensPerImage = 256;
        int maxPixels = 0;
        int temporalPatchSize = 1;
        int spatialMergeSize = 2;
        
        // Special tokens for vision processing
        int imageTokenId = -1;
        int visionStartTokenId = -1;
        int visionEndTokenId = -1;
        
        // Model-specific parameters
        std::unordered_map<std::string, float> customParams;
    };
    
    /**
     * Check if a model has vision capabilities
     * @param architecture The model architecture name
     * @return true if model supports vision
     */
    static bool hasVisionSupport(const std::string& architecture);
    
    /**
     * Get vision configuration for a model
     * @param architecture The model architecture name
     * @return Vision configuration or nullptr if not supported
     */
    static std::shared_ptr<VisionConfig> getVisionConfig(const std::string& architecture);
    
    /**
     * Process vision tokens for multimodal input
     * @param architecture The model architecture
     * @param tokens Input tokens
     * @param imageData Raw image data
     * @return Processed tokens with vision embeddings
     */
    static std::vector<int> processVisionTokens(
        const std::string& architecture,
        const std::vector<int>& tokens,
        const std::vector<uint8_t>& imageData
    );
    
    /**
     * Extract vision parameters from GGUF metadata
     * @param architecture The model architecture
     * @param metadata GGUF metadata map
     * @return Vision configuration
     */
    static std::shared_ptr<VisionConfig> extractVisionConfig(
        const std::string& architecture,
        const std::unordered_map<std::string, std::string>& metadata
    );
    
    /**
     * Check if tokens contain vision-related special tokens
     * @param tokens Input tokens
     * @param config Vision configuration
     * @return true if vision tokens are present
     */
    static bool containsVisionTokens(
        const std::vector<int>& tokens,
        const VisionConfig& config
    );
    
    /**
     * Calculate vision tensor dimensions
     * @param config Vision configuration
     * @param imageWidth Image width in pixels
     * @param imageHeight Image height in pixels
     * @return Tensor dimensions (patches, channels, etc.)
     */
    static std::vector<int> calculateVisionDimensions(
        const VisionConfig& config,
        int imageWidth,
        int imageHeight
    );
    
    /**
     * Initialize vision configurations for supported models
     */
    static void initialize();
    
private:
    static std::unordered_map<std::string, std::shared_ptr<VisionConfig>> visionConfigs;
    static bool initialized;
    
    /**
     * Create vision configurations for specific models
     */
    static void createQwen25vlVisionConfig();
    static void createGemma3VisionConfig();
    static void createMistral3VisionConfig();
    
    /**
     * Helper functions for image processing
     */
    static std::vector<float> preprocessImage(
        const std::vector<uint8_t>& imageData,
        const VisionConfig& config
    );
    
    static std::vector<int> patchifyImage(
        const std::vector<float>& imagePixels,
        const VisionConfig& config,
        int width,
        int height
    );
};