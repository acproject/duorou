#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

/**
 * Attention mechanism handler for advanced attention patterns
 * Supports sliding window attention, attention sinks, and other specialized mechanisms
 */
class AttentionHandler {
public:
    enum class AttentionType {
        STANDARD,
        SLIDING_WINDOW,
        ATTENTION_SINKS,
        MIXED_ATTENTION,
        GLOBAL_LOCAL
    };
    
    struct AttentionConfig {
        std::string architecture;
        AttentionType type = AttentionType::STANDARD;
        
        // Sliding window parameters
        int slidingWindowSize = 0;
        bool hasSlidingWindow = false;
        
        // Attention sinks parameters
        bool hasAttentionSinks = false;
        int numSinks = 0;
        std::vector<int> sinkPositions;
        
        // Global-local attention (like Gemma3)
        bool hasGlobalLocal = false;
        int globalLayerInterval = 6; // Every 6th layer is global
        
        // Softcapping parameters
        float attentionLogitSoftcap = 0.0f;
        float finalLogitSoftcap = 0.0f;
        bool hasSoftcapping = false;
        
        // RoPE parameters
        float ropeBase = 10000.0f;
        float ropeScale = 1.0f;
        int originalContextLength = 0;
        bool useNeoXRoPE = false;
        
        // Custom parameters
        std::unordered_map<std::string, float> customParams;
        std::unordered_map<std::string, int> customIntParams;
    };
    
    /**
     * Get attention configuration for a model
     * @param architecture The model architecture name
     * @return Attention configuration or nullptr if not found
     */
    static std::shared_ptr<AttentionConfig> getAttentionConfig(const std::string& architecture);
    
    /**
     * Check if a model uses advanced attention mechanisms
     * @param architecture The model architecture name
     * @return true if advanced attention is used
     */
    static bool hasAdvancedAttention(const std::string& architecture);
    
    /**
     * Check if a model uses sliding window attention
     * @param architecture The model architecture name
     * @return true if sliding window is used
     */
    static bool usesSlidingWindow(const std::string& architecture);
    
    /**
     * Check if a model uses attention sinks
     * @param architecture The model architecture name
     * @return true if attention sinks are used
     */
    static bool usesAttentionSinks(const std::string& architecture);
    
    /**
     * Get the effective context length for a layer
     * @param architecture The model architecture
     * @param layerIndex The layer index
     * @param baseContextLength The base context length
     * @return Effective context length for this layer
     */
    static int getEffectiveContextLength(
        const std::string& architecture,
        int layerIndex,
        int baseContextLength
    );
    
    /**
     * Calculate attention mask for specialized attention patterns
     * @param config Attention configuration
     * @param sequenceLength Current sequence length
     * @param layerIndex Current layer index
     * @return Attention mask parameters
     */
    static std::vector<int> calculateAttentionMask(
        const AttentionConfig& config,
        int sequenceLength,
        int layerIndex
    );
    
    /**
     * Apply softcapping to attention logits
     * @param config Attention configuration
     * @param logits Input logits
     * @param isFinal Whether this is the final layer
     * @return Softcapped logits
     */
    static std::vector<float> applySoftcapping(
        const AttentionConfig& config,
        const std::vector<float>& logits,
        bool isFinal = false
    );
    
    /**
     * Get RoPE parameters for the model
     * @param architecture The model architecture
     * @return RoPE configuration
     */
    static std::unordered_map<std::string, float> getRoPEParams(const std::string& architecture);
    
    /**
     * Initialize attention configurations
     */
    static void initialize();
    
private:
    static std::unordered_map<std::string, std::shared_ptr<AttentionConfig>> attentionConfigs;
    static bool initialized;
    
    /**
     * Create attention configurations for specific models
     */
    static void createGemma2AttentionConfig();
    static void createGemma3AttentionConfig();
    static void createGemma3nAttentionConfig();
    static void createGptossAttentionConfig();
    static void createMistral3AttentionConfig();
    static void createQwen25vlAttentionConfig();
    static void createLlamaAttentionConfig();
    static void createQwen2AttentionConfig();
    static void createQwen3AttentionConfig();
    
    /**
     * Helper functions
     */
    static float softcap(float x, float cap);
    static bool isGlobalLayer(int layerIndex, int interval);
};