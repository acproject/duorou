#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "model_path_manager.h"
#include "../../third_party/llama.cpp/vendor/nlohmann/json.hpp"

namespace duorou {
namespace core {

/**
 * @brief LoRA adapter information
 */
struct LoRAAdapter {
    std::string name;           ///< Adapter name
    std::string path;           ///< Adapter file path
    float scale = 1.0f;         ///< Adapter scaling factor
    std::string digest;         ///< SHA256 digest
    size_t size = 0;           ///< File size
    
    LoRAAdapter() = default;
    LoRAAdapter(const std::string& n, const std::string& p, float s = 1.0f)
        : name(n), path(p), scale(s) {}
};

/**
 * @brief Modelfile configuration information
 */
struct ModelfileConfig {
    std::string base_model;                        ///< Base model path
    std::vector<LoRAAdapter> lora_adapters;        ///< LoRA adapter list
    std::unordered_map<std::string, std::string> parameters;  ///< Model parameters
    std::string system_prompt;                     ///< System prompt
    std::string template_format;                   ///< Template format
    
    ModelfileConfig() = default;
};

/**
 * @brief Ollama Modelfile parser
 * Responsible for parsing Ollama model manifest and related configurations, extracting LoRA adapter information
 */
class ModelfileParser {
public:
    /**
     * @brief Constructor
     * @param model_path_manager Model path manager
     */
    explicit ModelfileParser(std::shared_ptr<ModelPathManager> model_path_manager);
    
    /**
     * @brief Destructor
     */
    ~ModelfileParser() = default;
    
    /**
     * @brief Parse Modelfile configuration from manifest
     * @param manifest Model manifest
     * @param config Output configuration information
     * @return Returns true if parsing is successful
     */
    bool parseFromManifest(const ModelManifest& manifest, ModelfileConfig& config);
    
    /**
     * @brief Parse Modelfile configuration from JSON string
     * @param json_str JSON string
     * @param config Output configuration information
     * @return Returns true if parsing is successful
     */
    bool parseFromJson(const std::string& json_str, ModelfileConfig& config);
    
    /**
     * @brief Parse Modelfile configuration from file
     * @param file_path File path
     * @param config Output configuration information
     * @return Returns true if parsing is successful
     */
    bool parseFromFile(const std::string& file_path, ModelfileConfig& config);
    
    /**
     * @brief Validate LoRA adapter file
     * @param adapter LoRA adapter information
     * @return Returns true if validation is successful
     */
    bool validateLoRAAdapter(const LoRAAdapter& adapter);
    
    /**
     * @brief Get list of supported media types
     * @return Supported media types
     */
    static std::vector<std::string> getSupportedMediaTypes();
    
private:
    std::shared_ptr<ModelPathManager> model_path_manager_;
    
    /**
     * @brief Parse template layer
     * @param layer_digest Layer digest
     * @param config Configuration information
     * @return Returns true if parsing is successful
     */
    bool parseTemplateLayer(const std::string& layer_digest, ModelfileConfig& config);
    
    /**
     * @brief Parse system layer
     * @param layer_digest Layer digest
     * @param config Configuration information
     * @return Returns true if parsing is successful
     */
    bool parseSystemLayer(const std::string& layer_digest, ModelfileConfig& config);
    
    /**
     * @brief Parse parameters layer
     * @param layer_digest Layer digest
     * @param config Configuration information
     * @return Returns true if parsing is successful
     */
    bool parseParametersLayer(const std::string& layer_digest, ModelfileConfig& config);
    
    /**
     * @brief Parse adapter layer
     * @param layer_digest Layer digest
     * @param config Configuration information
     * @return Returns true if parsing is successful
     */
    bool parseAdapterLayer(const std::string& layer_digest, ModelfileConfig& config);
    
    /**
     * @brief Read content from blob file
     * @param digest File digest
     * @return File content, returns empty string on failure
     */
    std::string readBlobContent(const std::string& digest);
    
    /**
     * @brief Parse Modelfile instructions
     * @param content Modelfile content
     * @param config Configuration information
     * @return Returns true if parsing is successful
     */
    bool parseModelfileInstructions(const std::string& content, ModelfileConfig& config);
    
    /**
     * @brief Parse FROM instruction
     * @param line Instruction line
     * @param config Configuration information
     * @return Returns true if parsing is successful
     */
    bool parseFromInstruction(const std::string& line, ModelfileConfig& config);
    
    /**
     * @brief Parse ADAPTER instruction
     * @param line Instruction line
     * @param config Configuration information
     * @return Returns true if parsing is successful
     */
    bool parseAdapterInstruction(const std::string& line, ModelfileConfig& config);
    
    /**
     * @brief Parse PARAMETER instruction
     * @param line Instruction line
     * @param config Configuration information
     * @return Returns true if parsing is successful
     */
    bool parseParameterInstruction(const std::string& line, ModelfileConfig& config);
    
    /**
     * @brief Parse TEMPLATE instruction
     * @param line Instruction line
     * @param config Configuration information
     * @return Returns true if parsing is successful
     */
    bool parseTemplateInstruction(const std::string& line, ModelfileConfig& config);
    
    /**
     * @brief Parse SYSTEM instruction
     * @param line Instruction line
     * @param config Configuration object
     * @return Returns true if parsing is successful
     */
    bool parseSystemInstruction(const std::string& line, ModelfileConfig& config);
    
    /**
     * @brief Validate GGUF file header
     * @param file_path File path
     * @return Returns true if validation is successful
     */
    bool validateGGUFHeader(const std::string& file_path);
};

} // namespace core
} // namespace duorou