#include "ollama_model_loader.h"
#include <iostream>
#include <fstream>
#include <filesystem>

namespace duorou {
namespace extensions {
namespace ollama {

// OllamaModelConfig implementation
bool OllamaModelConfig::loadFromModelfile(const std::string& modelfile_path) {
    ModelfileParser parser;
    ParsedModelfile parsed;
    
    if (!parser.parseFromFile(modelfile_path, parsed)) {
        return false;
    }
    
    // Extract configuration from parsed modelfile
    base_model = parsed.from_model;
    system_prompt = parsed.system_prompt;
    template_content = parsed.template_content;
    
    // Convert parameters
    for (const auto& param : parsed.parameters) {
        parameters[param.name] = param.value;
    }
    
    return true;
}

bool OllamaModelConfig::loadFromManifest(const ModelPath& model_path, ModelPathManager& path_manager) {
    ModelManifest manifest;
    if (!path_manager.readManifest(model_path, manifest)) {
        return false;
    }
    
    // Extract configuration from manifest
    architecture = manifest.architecture;
    
    return true;
}

std::string OllamaModelConfig::getParameter(const std::string& name, const std::string& default_value) const {
    auto it = parameters.find(name);
    return (it != parameters.end()) ? it->second : default_value;
}

void OllamaModelConfig::setParameter(const std::string& name, const std::string& value) {
    parameters[name] = value;
}

// OllamaModelLoader implementation
OllamaModelLoader::OllamaModelLoader(std::shared_ptr<ModelPathManager> model_path_manager)
    : model_path_manager_(model_path_manager),
      modelfile_parser_(std::make_unique<ModelfileParser>()),
      compatibility_checker_(std::make_unique<CompatibilityChecker>()),
      verbose_(false) {
}

OllamaModelLoader::~OllamaModelLoader() = default;

void* OllamaModelLoader::loadFromModelPath(const std::string& model_path, 
                                           const void* model_params) {
    ModelPath parsed_path;
    if (!parsed_path.parseFromString(model_path)) {
        if (verbose_) {
            log("ERROR", "Failed to parse model path: " + model_path);
        }
        return nullptr;
    }
    
    // Get GGUF file path from manifest
    ModelManifest manifest;
    if (!model_path_manager_->readManifest(parsed_path, manifest)) {
        if (verbose_) {
            log("ERROR", "Failed to read manifest for: " + model_path);
        }
        return nullptr;
    }
    
    std::string gguf_path = getGGUFPathFromManifest(manifest);
    if (gguf_path.empty()) {
        if (verbose_) {
            log("ERROR", "No GGUF file found in manifest");
        }
        return nullptr;
    }
    
    return loadFromGGUFPath(gguf_path, model_params);
}

void* OllamaModelLoader::loadFromGGUFPath(const std::string& gguf_path,
                                          const void* model_params) {
    // Check if file exists
    std::ifstream file(gguf_path);
    if (!file.good()) {
        if (verbose_) {
            log("ERROR", "GGUF file not found: " + gguf_path);
        }
        return nullptr;
    }
    file.close();
    
    // Check architecture compatibility
    std::string original_arch, mapped_arch;
    if (!checkArchitectureMapping(gguf_path, original_arch, mapped_arch)) {
        if (verbose_) {
            log("ERROR", "Architecture compatibility check failed for: " + gguf_path);
        }
        return nullptr;
    }
    
    // For now, return a placeholder - actual loading would be implemented
    // when integrating with the main text generator
    if (verbose_) {
        log("INFO", "Model architecture check passed for: " + gguf_path);
    }
    
    // Return a non-null placeholder to indicate success
    return reinterpret_cast<void*>(0x1);
}

bool OllamaModelLoader::checkArchitectureMapping(const std::string& gguf_path,
                                                 std::string& original_arch,
                                                 std::string& mapped_arch) {
    CompatibilityResult result = compatibility_checker_->checkCompatibility(gguf_path);
    
    original_arch = architectureToString(result.detected_arch);
    mapped_arch = result.arch_name;
    
    return result.is_compatible;
}

std::vector<std::string> OllamaModelLoader::getSupportedArchitectures() const {
    auto archs = compatibility_checker_->getSupportedArchitectures();
    std::vector<std::string> result;
    for (auto arch : archs) {
        result.push_back(architectureToString(arch));
    }
    return result;
}

std::string OllamaModelLoader::getGGUFPathFromManifest(const ModelManifest& manifest) {
    // Get the model blob from manifest
    std::string model_blob = manifest.getModelBlob();
    if (!model_blob.empty()) {
        return model_path_manager_->getBlobPath(model_blob);
    }
    
    // Fallback: use the first layer if available
    if (!manifest.layers.empty()) {
        return model_path_manager_->getBlobPath(manifest.layers[0]);
    }
    
    return "";
}

std::vector<void*> OllamaModelLoader::createArchOverrides(
    const std::string& mapped_arch,
    const std::string& model_path) {
    
    std::vector<void*> overrides;
    
    // Placeholder implementation - would create actual overrides
    // when integrating with llama.cpp
    if (verbose_) {
        log("INFO", "Creating architecture overrides for: " + mapped_arch);
    }
    
    return overrides;
}

int OllamaModelLoader::estimateGPULayers(const std::string& gguf_path, size_t available_vram) {
    // Simple estimation based on file size and available VRAM
    // This is a placeholder implementation
    try {
        std::ifstream file(gguf_path, std::ios::binary | std::ios::ate);
        if (!file.good()) {
            return 0;
        }
        size_t file_size = file.tellg();
        file.close();
        
        size_t estimated_layers = (available_vram * 1024 * 1024) / (file_size / 32); // Rough estimate
        return static_cast<int>(std::min(estimated_layers, size_t(100))); // Cap at 100 layers
    } catch (const std::exception&) {
        return 0;
    }
}

void OllamaModelLoader::log(const std::string& level, const std::string& message) {
    if (verbose_) {
        std::cout << "[" << level << "] OllamaModelLoader: " << message << std::endl;
    }
}

} // namespace ollama
} // namespace extensions
} // namespace duorou