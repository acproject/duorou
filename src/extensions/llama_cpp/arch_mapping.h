#pragma once

#include <string>
#include <unordered_map>

/**
 * Architecture mapping extension for llama.cpp
 * Maps unsupported architecture names to supported ones
 */
class ArchMapping {
public:
    /**
     * Get the mapped architecture name
     * @param arch_name Original architecture name from GGUF file
     * @return Mapped architecture name that llama.cpp supports
     */
    static std::string getMappedArchitecture(const std::string& arch_name);
    
    /**
     * Check if an architecture needs mapping
     * @param arch_name Architecture name to check
     * @return true if mapping is needed, false otherwise
     */
    static bool needsMapping(const std::string& arch_name);
    
private:
    static const std::unordered_map<std::string, std::string> arch_mappings;
};