#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

/**
 * Model configuration manager for handling ollama-specific model parameters
 * and configurations that need special handling in llama.cpp
 */
class ModelConfigManager {
public:
    struct ModelConfig {
        std::string architecture;
        bool hasVision = false;
        bool hasSlidingWindow = false;
        bool hasAttentionSinks = false;
        bool requiresSpecialTokenHandling = false;
        bool supportsMixtral = false;
        
        // Vision-specific parameters
        int imageSize = 0;
        int patchSize = 0;
        int tokensPerImage = 0;
        
        // Attention parameters
        int slidingWindowSize = 0;
        float attentionLogitSoftcap = 0.0f;
        float finalLogitSoftcap = 0.0f;
        
        // Special tokens
        std::vector<int> specialTokenIds;
        
        // Custom parameters
        std::unordered_map<std::string, float> customFloatParams;
        std::unordered_map<std::string, int> customIntParams;
        std::unordered_map<std::string, std::string> customStringParams;
    };
    
    /**
     * Get configuration for a specific model architecture
     * @param architecture The model architecture name
     * @return Model configuration or nullptr if not found
     */
    static std::shared_ptr<ModelConfig> getConfig(const std::string& architecture);
    
    /**
     * Check if a model requires special handling
     * @param architecture The model architecture name
     * @return true if special handling is required
     */
    static bool requiresSpecialHandling(const std::string& architecture);
    
    /**
     * Check if a model has vision capabilities
     * @param architecture The model architecture name
     * @return true if model supports vision
     */
    static bool hasVisionSupport(const std::string& architecture);
    
    /**
     * Get the list of architectures that require ollama engine
     * @return vector of architecture names
     */
    static std::vector<std::string> getOllamaEngineRequired();
    
    /**
     * Initialize model configurations
     */
    static void initialize();
    
private:
    static std::unordered_map<std::string, std::shared_ptr<ModelConfig>> configs;
    static bool initialized;
    
    /**
     * Create configuration for specific model types
     */
    static void createGemma3Config();
    static void createGemma3nConfig();
    static void createMistral3Config();
    static void createQwen25vlConfig();
    static void createQwen3Config();
    static void createGptossConfig();
    static void createLlamaConfig();
    static void createQwen2Config();
    static void createGemma2Config();
};