#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

namespace duorou {
namespace extensions {

/**
 * GGML Incremental Extension for llama.cpp
 * Adds support for new model architectures by extending existing GGML implementations
 * This approach avoids hyperparameter inconsistencies by building on proven architectures
 */
class GGMLIncrementalExtension {
public:
    /**
     * Initialize the extension with llama.cpp integration
     */
    static bool initialize();
    
    /**
     * Check if a model architecture is supported by this extension
     * @param arch_name Architecture name from GGUF file
     * @return true if supported by extension, false otherwise
     */
    static bool isArchitectureSupported(const std::string& arch_name);
    
    /**
     * Get the base architecture for incremental extension
     * @param arch_name Target architecture name
     * @return Base architecture name to extend from
     */
    static std::string getBaseArchitecture(const std::string& arch_name);
    
    /**
     * Apply incremental modifications to model parameters
     * @param arch_name Target architecture name
     * @param base_params Base model parameters
     * @return Modified parameters for target architecture
     */
    static bool applyIncrementalModifications(const std::string& arch_name, void* model_params);
    
    /**
     * Register a new architecture extension
     * @param arch_name New architecture name
     * @param base_arch Base architecture to extend from
     * @param modifications Function to apply specific modifications
     */
    static void registerArchitectureExtension(
        const std::string& arch_name,
        const std::string& base_arch,
        std::function<bool(void*)> modifications
    );
    
private:
    struct ArchitectureExtension {
        std::string base_architecture;
        std::function<bool(void*)> apply_modifications;
    };
    
    static std::unordered_map<std::string, ArchitectureExtension> extensions;
    static bool initialized;
};

/**
 * Qwen2.5VL Extension - extends Qwen2VL for improved vision-language capabilities
 */
class Qwen25VLExtension {
public:
    static bool applyModifications(void* model_params);
    static void registerExtension();
};

/**
 * Gemma3 Extension - extends Gemma2 for improved capabilities
 */
class Gemma3Extension {
public:
    static bool applyModifications(void* model_params);
    static void registerExtension();
};

/**
 * Mistral3 Extension - extends Mistral for improved capabilities
 */
class Mistral3Extension {
public:
    static bool applyModifications(void* model_params);
    static void registerExtension();
};

/**
 * GPToss Extension - extends LLaMA for specialized capabilities
 */
class GptossExtension {
public:
    static bool applyModifications(void* model_params);
    static void registerExtension();
};

} // namespace extensions
} // namespace duorou