#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>

/**
 * Model compatibility checker for Ollama models with llama.cpp
 * Validates model architecture, parameters, and special requirements
 */
class CompatibilityChecker {
public:
    enum class CompatibilityLevel {
        FULLY_COMPATIBLE,    // Direct compatibility
        NEEDS_MAPPING,       // Requires architecture mapping
        NEEDS_MODIFICATION,  // Requires GGUF modification
        PARTIALLY_SUPPORTED, // Some features may not work
        NOT_SUPPORTED        // Cannot be loaded
    };
    
    struct CompatibilityResult {
        CompatibilityLevel level = CompatibilityLevel::NOT_SUPPORTED;
        std::string originalArchitecture;
        std::string mappedArchitecture;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        std::vector<std::string> requiredModifications;
        std::unordered_map<std::string, std::string> recommendations;
        bool needsOllamaEngine = false;
        bool hasVisionSupport = false;
        bool hasAdvancedAttention = false;
    };
    
    struct ModelRequirements {
        std::string architecture;
        std::unordered_set<std::string> requiredTensors;
        std::unordered_set<std::string> optionalTensors;
        std::unordered_map<std::string, std::string> requiredMetadata;
        std::vector<std::string> supportedQuantizations;
        int minContextLength = 0;
        int maxContextLength = 0;
        bool requiresVisionProcessor = false;
        bool requiresSpecialTokenizer = false;
    };
    
    /**
     * Check compatibility of a model with llama.cpp
     * @param architecture The model architecture name
     * @param modelPath Path to the model file (optional)
     * @return Compatibility result with detailed information
     */
    static CompatibilityResult checkCompatibility(
        const std::string& architecture,
        const std::string& modelPath = ""
    );
    
    /**
     * Check compatibility using GGUF metadata
     * @param metadata GGUF metadata key-value pairs
     * @param tensorNames List of tensor names in the model
     * @return Compatibility result
     */
    static CompatibilityResult checkCompatibilityFromMetadata(
        const std::unordered_map<std::string, std::string>& metadata,
        const std::vector<std::string>& tensorNames
    );
    
    /**
     * Get model requirements for a specific architecture
     * @param architecture The model architecture name
     * @return Model requirements or nullptr if not found
     */
    static std::shared_ptr<ModelRequirements> getModelRequirements(const std::string& architecture);
    
    /**
     * Check if an architecture is supported by llama.cpp
     * @param architecture The architecture name
     * @return true if supported (directly or through mapping)
     */
    static bool isArchitectureSupported(const std::string& architecture);
    
    /**
     * Get list of all supported architectures
     * @return Vector of supported architecture names
     */
    static std::vector<std::string> getSupportedArchitectures();
    
    /**
     * Get list of architectures that need Ollama engine
     * @return Vector of architecture names requiring Ollama
     */
    static std::vector<std::string> getOllamaRequiredArchitectures();
    
    /**
     * Validate tensor names against model requirements
     * @param architecture The model architecture
     * @param tensorNames List of tensor names
     * @return Validation result with missing/extra tensors
     */
    static std::pair<std::vector<std::string>, std::vector<std::string>> validateTensors(
        const std::string& architecture,
        const std::vector<std::string>& tensorNames
    );
    
    /**
     * Check if a quantization type is supported for an architecture
     * @param architecture The model architecture
     * @param quantization The quantization type (e.g., "Q4_0", "Q8_0")
     * @return true if supported
     */
    static bool isQuantizationSupported(
        const std::string& architecture,
        const std::string& quantization
    );
    
    /**
     * Get recommended modifications for better compatibility
     * @param architecture The model architecture
     * @return List of recommended modifications
     */
    static std::vector<std::string> getRecommendedModifications(const std::string& architecture);
    
    /**
     * Check if model needs special preprocessing
     * @param architecture The model architecture
     * @return true if special preprocessing is needed
     */
    static bool needsSpecialPreprocessing(const std::string& architecture);
    
    /**
     * Get compatibility score (0-100)
     * @param architecture The model architecture
     * @return Compatibility score
     */
    static int getCompatibilityScore(const std::string& architecture);
    
    /**
     * Initialize compatibility checker
     */
    static void initialize();
    
private:
    static std::unordered_map<std::string, std::shared_ptr<ModelRequirements>> modelRequirements;
    static std::unordered_set<std::string> llamaCppNativeArchitectures;
    static std::unordered_set<std::string> ollamaRequiredArchitectures;
    static bool initialized;
    
    /**
     * Create model requirements for specific architectures
     */
    static void createLlamaRequirements();
    static void createQwen2Requirements();
    static void createQwen3Requirements();
    static void createQwen25vlRequirements();
    static void createGemma2Requirements();
    static void createGemma3Requirements();
    static void createGemma3nRequirements();
    static void createMistral3Requirements();
    static void createGptossRequirements();
    
    /**
     * Helper functions
     */
    static CompatibilityLevel determineCompatibilityLevel(
        const std::string& architecture,
        const ModelRequirements& requirements
    );
    
    static std::vector<std::string> checkArchitectureWarnings(const std::string& architecture);
    static std::vector<std::string> checkArchitectureErrors(const std::string& architecture);
    
    static bool hasRequiredTensors(
        const std::vector<std::string>& tensorNames,
        const std::unordered_set<std::string>& requiredTensors
    );
    
    static std::string normalizeArchitectureName(const std::string& architecture);
};