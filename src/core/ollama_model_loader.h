#pragma once

#include <string>
#include <memory>
#include <cstdint>
#include <iostream>
#include <vector>
#include <map>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include "model_path_manager.h"
#include "modelfile_parser.h"
#include "logger.h"
// #include "llama.h"  // Note: Temporarily disable llama-related functionality

// Forward declarations
// struct llama_model;  // Note: Temporarily disable llama-related functionality
// struct llama_model_params;  // Note: Temporarily disable llama-related functionality

namespace duorou {
namespace core {

/**
 * @brief Ollama model loader
 * Responsible for loading models from ollama downloaded models to llama.cpp
 */
class OllamaModelLoader {
public:
    /**
     * @brief Constructor
     * @param model_path_manager Model path manager
     */
    explicit OllamaModelLoader(std::shared_ptr<ModelPathManager> model_path_manager);
    
    /**
     * @brief Destructor
     */
    ~OllamaModelLoader() = default;
    
    /**
     * @brief Load llama model from ollama model name (temporarily disabled)
     * @param model_name ollama model name (e.g.: "llama3.2", "qwen2.5:7b")
     * @return Returns true if loading succeeds, false if it fails
     */
    // llama_model* loadFromOllamaModel(const std::string& model_name, 
    //                                 const llama_model_params& model_params);
    bool loadFromOllamaModel(const std::string& model_name);
    
    /**
     * @brief Load llama model from ollama model path (temporarily disabled)
     * @param model_path Parsed model path
     * @return Returns true if loading succeeds, false if it fails
     */
    // llama_model* loadFromModelPath(const ModelPath& model_path,
    //                               const llama_model_params& model_params);
    bool loadFromModelPath(const ModelPath& model_path);
    
    /**
     * @brief Load llama model from ollama model, supports LoRA adapters (temporarily disabled)
     * @param model_name ollama model name
     * @param enable_lora Whether to enable LoRA parsing
     * @return Returns true if loading succeeds, false if it fails
     */
    // llama_model* loadFromOllamaModelWithLoRA(const std::string& model_name,
    //                                         const llama_model_params& model_params,
    //                                         bool enable_lora = false);
    bool loadFromOllamaModelWithLoRA(const std::string& model_name,
                                    bool enable_lora = false);
    
    /**
     * @brief Load model from Modelfile configuration (temporarily disabled)
     * @param config Modelfile configuration
     * @return Returns true if loading succeeds, false if it fails
     */
    // llama_model* loadFromModelfileConfig(const ModelfileConfig& config,
    //                                     const llama_model_params& model_params);
    bool loadFromModelfileConfig(const ModelfileConfig& config);
    
    /**
     * @brief Check if ollama model exists
     * @param model_name ollama model name
     * @return Returns true if exists
     */
    bool isOllamaModelAvailable(const std::string& model_name);
    
    /**
     * @brief List all available ollama models
     * @return List of model names
     */
    std::vector<std::string> listAvailableModels();
    
private:
    /**
     * @brief Get GGUF model file path from manifest
     * @param manifest Model manifest
     * @return GGUF file path, returns empty string if failed
     */
    std::string getGGUFPathFromManifest(const ModelManifest& manifest);
    
    /**
     * @brief Parse ollama model name to ModelPath
     * @param model_name ollama model name
     * @param model_path Output model path
     * @return Returns true if parsing succeeds
     */
    bool parseOllamaModelName(const std::string& model_name, ModelPath& model_path);
    
    /**
     * @brief Normalize ollama model name
     * @param model_name Original model name
     * @return Normalized model name
     */
    std::string normalizeOllamaModelName(const std::string& model_name);
    
private:
    std::shared_ptr<ModelPathManager> model_path_manager_;
    std::shared_ptr<ModelfileParser> modelfile_parser_;
    Logger logger_;
    
    /**
     * @brief Parse Modelfile configuration from manifest
     * @param manifest Model manifest
     * @param config Output configuration information
     * @return Returns true if parsing succeeds
     */
    bool parseModelfileFromManifest(const ModelManifest& manifest, ModelfileConfig& config);
};

} // namespace core
} // namespace duorou