#pragma once

#include <string>
#include <memory>

/**
 * GGUF file modifier for architecture compatibility
 * Modifies GGUF files to ensure compatibility with llama.cpp
 */
class GGUFModifier {
public:
    /**
     * Modify GGUF file architecture if needed
     * @param gguf_path Path to the GGUF file
     * @return true if modification was successful or not needed, false on error
     */
    static bool modifyArchitectureIfNeeded(const std::string& gguf_path);
    
    /**
     * Create a temporary modified GGUF file
     * @param original_path Path to original GGUF file
     * @param temp_path Path where temporary file will be created
     * @return true if successful, false on error
     */
    static bool createModifiedGGUF(const std::string& original_path, const std::string& temp_path);
    
    /**
     * Check if GGUF file needs architecture modification
     * @param gguf_path Path to the GGUF file
     * @return true if modification is needed, false otherwise
     */
    static bool needsArchitectureModification(const std::string& gguf_path);
    
    /**
     * Get the architecture from GGUF file
     * @param gguf_path Path to the GGUF file
     * @return Architecture string, empty if error
     */
    static std::string getGGUFArchitecture(const std::string& gguf_path);
    
private:
    /**
     * Modify architecture field in GGUF file
     * @param gguf_path Path to the GGUF file
     * @param new_arch New architecture name
     * @return true if successful, false on error
     */
    static bool modifyArchitectureField(const std::string& gguf_path, const std::string& new_arch);
};