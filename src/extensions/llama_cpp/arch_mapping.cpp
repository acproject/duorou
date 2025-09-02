#include "arch_mapping.h"

// Architecture mappings from unsupported to supported names
// Maps ollama-specific architectures to llama.cpp compatible ones
const std::unordered_map<std::string, std::string> ArchMapping::arch_mappings = {
    // Vision-Language Models
    {"qwen25vl", "qwen2vl"},     // Qwen2.5-VL -> Qwen2-VL architecture
    {"gemma3", "gemma2"},       // Gemma3 with vision -> Gemma2 base
    {"mistral3", "llama"},      // Mistral3 with vision -> Llama base
    
    // Text-only Models
    {"gemma3n", "gemma2"},      // Gemma3 text-only -> Gemma2
    {"qwen3", "qwen2"},         // Qwen3 -> Qwen2 architecture
    {"gptoss", "llama"},        // GPT-OSS -> Llama architecture
    {"gpt-oss", "llama"},       // Alternative GPT-OSS naming
    
    // Keep existing mappings
    {"qwen2", "qwen2"},         // Direct mapping for qwen2
    {"gemma2", "gemma2"},       // Direct mapping for gemma2
    {"llama", "llama"},         // Direct mapping for llama
};

std::string ArchMapping::getMappedArchitecture(const std::string& arch_name) {
    auto it = arch_mappings.find(arch_name);
    if (it != arch_mappings.end()) {
        return it->second;
    }
    return arch_name;  // Return original if no mapping found
}

bool ArchMapping::needsMapping(const std::string& arch_name) {
    return arch_mappings.find(arch_name) != arch_mappings.end();
}