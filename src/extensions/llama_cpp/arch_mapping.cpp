#include "arch_mapping.h"

// Architecture mappings from unsupported to supported names
const std::unordered_map<std::string, std::string> ArchMapping::arch_mappings = {
    {"qwen25vl", "qwen2vl"},  // Map Qwen2.5-VL to Qwen2-VL
    // Add more mappings here as needed
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