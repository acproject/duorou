#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

/**
 * GGUF file modifier for architecture compatibility
 * Modifies GGUF files to ensure compatibility with llama.cpp
 * Extended to support Ollama models with special requirements
 */
class GGUFModifier {
public:
    /**
     * Modify GGUF file architecture if needed
     * @param gguf_path Path to the GGUF file
     * @return true if modification was successful or not needed, false on error
     */
    static bool modifyArchitectureIfNeeded(const std::string& gguf_path);
    
    // Note: createModifiedGGUF function removed as it's no longer needed
    // Model architecture mapping is now handled through kv_override mechanism
    
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
    
    /**
     * Add missing keys for qwen2.5vl models
     * @param gguf_path Path to the GGUF file
     * @return true if successful, false on error
     */
    static bool addMissingQwen25VLKeys(const std::string& gguf_path);
    
    /**
     * Add missing keys for Gemma3 models
     * @param gguf_path Path to the GGUF file
     * @return true if successful, false on error
     */
    static bool addMissingGemma3Keys(const std::string& gguf_path);
    
    /**
     * Add missing keys for Mistral3 models
     * @param gguf_path Path to the GGUF file
     * @return true if successful, false on error
     */
    static bool addMissingMistral3Keys(const std::string& gguf_path);
    
    /**
     * Add missing keys for GPT-OSS models
     * @param gguf_path Path to the GGUF file
     * @return true if successful, false on error
     */
    static bool addMissingGptossKeys(const std::string& gguf_path);
    
    /**
     * Add vision-related metadata for multimodal models
     * @param gguf_path Path to the GGUF file
     * @param architecture Model architecture
     * @return true if successful, false on error
     */
    static bool addVisionMetadata(const std::string& gguf_path, const std::string& architecture);
    
    /**
     * Add attention mechanism metadata
     * @param gguf_path Path to the GGUF file
     * @param architecture Model architecture
     * @return true if successful, false on error
     */
    static bool addAttentionMetadata(const std::string& gguf_path, const std::string& architecture);
    
    /**
     * Perform comprehensive model-specific modifications
     * @param gguf_path Path to the GGUF file
     * @param architecture Model architecture
     * @return true if successful, false on error
     */
    static bool performModelSpecificModifications(const std::string& gguf_path, const std::string& architecture);
    
    /**
     * Check if a key exists in GGUF file
     * @param gguf_path Path to the GGUF file
     * @param key_name Key name to check
     * @return true if key exists, false otherwise
     */
    static bool hasKey(const std::string& gguf_path, const std::string& key_name);
    
    /**
     * Get all metadata keys from GGUF file
     * @param gguf_path Path to the GGUF file
     * @return Map of key-value pairs
     */
    static std::unordered_map<std::string, std::string> getAllMetadata(const std::string& gguf_path);
    
    /**
     * Set a string value in GGUF file
     * @param gguf_path Path to the GGUF file
     * @param key_name Key name
     * @param value String value
     * @return true if successful, false on error
     */
    static bool setStringValue(const std::string& gguf_path, const std::string& key_name, const std::string& value);
    
    /**
     * Set a float value in GGUF file
     * @param gguf_path Path to the GGUF file
     * @param key_name Key name
     * @param value Float value
     * @return true if successful, false on error
     */
    static bool setFloatValue(const std::string& gguf_path, const std::string& key_name, float value);
    
    /**
     * Set an integer value in GGUF file
     * @param gguf_path Path to the GGUF file
     * @param key_name Key name
     * @param value Integer value
     * @return true if successful, false on error
     */
    static bool setIntValue(const std::string& gguf_path, const std::string& key_name, int value);
    
    /**
     * Set an array value in GGUF file
     * @param gguf_path Path to the GGUF file
     * @param key_name Key name
     * @param values Vector of values
     * @return true if successful, false on error
     */
    static bool setArrayValue(const std::string& gguf_path, const std::string& key_name, const std::vector<uint32_t>& values);
    
private:
    /**
     * Modify architecture field in GGUF file
     * @param gguf_path Path to the GGUF file
     * @param new_arch New architecture name
     * @return true if successful, false on error
     */
    static bool modifyArchitectureField(const std::string& gguf_path, const std::string& new_arch);
};