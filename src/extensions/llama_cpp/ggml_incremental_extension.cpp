#include "ggml_incremental_extension.h"
#include <iostream>
#include <cassert>

namespace duorou {
namespace extensions {

// Static member definitions
std::unordered_map<std::string, GGMLIncrementalExtension::ArchitectureExtension> 
    GGMLIncrementalExtension::extensions;
bool GGMLIncrementalExtension::initialized = false;

bool GGMLIncrementalExtension::initialize() {
    if (initialized) {
        return true;
    }
    
    // Register built-in extensions
    Qwen25VLExtension::registerExtension();
    Gemma3Extension::registerExtension();
    Mistral3Extension::registerExtension();
    GptossExtension::registerExtension();
    
    initialized = true;
    std::cout << "GGML Incremental Extension initialized successfully" << std::endl;
    return true;
}

bool GGMLIncrementalExtension::isArchitectureSupported(const std::string& arch_name) {
    return extensions.find(arch_name) != extensions.end();
}

std::string GGMLIncrementalExtension::getBaseArchitecture(const std::string& arch_name) {
    auto it = extensions.find(arch_name);
    if (it != extensions.end()) {
        return it->second.base_architecture;
    }
    return "";
}

bool GGMLIncrementalExtension::applyIncrementalModifications(
    const std::string& arch_name, 
    void* model_params) {
    
    auto it = extensions.find(arch_name);
    if (it != extensions.end()) {
        return it->second.apply_modifications(model_params);
    }
    return false;
}

void GGMLIncrementalExtension::registerArchitectureExtension(
    const std::string& arch_name,
    const std::string& base_arch,
    std::function<bool(void*)> modifications) {
    
    ArchitectureExtension ext;
    ext.base_architecture = base_arch;
    ext.apply_modifications = modifications;
    
    extensions[arch_name] = ext;
    std::cout << "Registered architecture extension: " << arch_name 
              << " (base: " << base_arch << ")" << std::endl;
}

// Qwen2.5VL Extension Implementation
bool Qwen25VLExtension::applyModifications(void* model_params) {
    // For now, Qwen2.5VL uses the same parameters as Qwen2VL
    // In the future, we can add specific modifications here
    // such as updated rope sections, attention mechanisms, etc.
    
    std::cout << "Applying Qwen2.5VL modifications..." << std::endl;
    
    // TODO: Add specific Qwen2.5VL modifications when needed
    // For example:
    // - Updated rope sections for better position encoding
    // - Enhanced attention mechanisms
    // - Improved vision-language integration
    
    return true;
}

void Qwen25VLExtension::registerExtension() {
    // 注册qwen25vl架构（GGUF文件中的实际架构名称）
    GGMLIncrementalExtension::registerArchitectureExtension(
        "qwen25vl",
        "qwen2vl",  // Base architecture
        Qwen25VLExtension::applyModifications
    );
    
    // 同时注册qwen2.5vl作为别名，以支持不同的命名方式
    GGMLIncrementalExtension::registerArchitectureExtension(
        "qwen2.5vl",
        "qwen2vl",  // Base architecture
        Qwen25VLExtension::applyModifications
    );
}

// Gemma3 Extension Implementation
bool Gemma3Extension::applyModifications(void* model_params) {
    // Gemma3 extends Gemma2 with improved attention mechanisms
    std::cout << "Applying Gemma3 modifications..." << std::endl;
    
    // TODO: Add specific Gemma3 modifications when needed
    // For example:
    // - Enhanced attention mechanisms
    // - Improved tokenization
    // - Updated model parameters
    
    return true;
}

void Gemma3Extension::registerExtension() {
    GGMLIncrementalExtension::registerArchitectureExtension(
        "gemma3",
        "gemma2",  // Base architecture
        Gemma3Extension::applyModifications
    );
}

// Mistral3 Extension Implementation
bool Mistral3Extension::applyModifications(void* model_params) {
    // Mistral3 extends Mistral with improved capabilities
    std::cout << "Applying Mistral3 modifications..." << std::endl;
    
    // TODO: Add specific Mistral3 modifications when needed
    // For example:
    // - Enhanced sliding window attention
    // - Improved model architecture
    // - Updated parameters
    
    return true;
}

void Mistral3Extension::registerExtension() {
    GGMLIncrementalExtension::registerArchitectureExtension(
        "mistral3",
        "mistral",  // Base architecture
        Mistral3Extension::applyModifications
    );
}

// GPToss Extension Implementation
bool GptossExtension::applyModifications(void* model_params) {
    // GPToss extends LLaMA for specialized capabilities
    std::cout << "Applying GPToss modifications..." << std::endl;
    
    // TODO: Add specific GPToss modifications when needed
    // For example:
    // - Specialized attention patterns
    // - Custom tokenization
    // - Domain-specific optimizations
    
    return true;
}

void GptossExtension::registerExtension() {
    GGMLIncrementalExtension::registerArchitectureExtension(
        "gptoss",
        "llama",  // Base architecture
        GptossExtension::applyModifications
    );
}

} // namespace extensions
} // namespace duorou