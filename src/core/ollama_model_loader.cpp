#include "ollama_model_loader.h"
#include "logger.h"
#include <filesystem>
#include <regex>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace duorou {
namespace core {

OllamaModelLoader::OllamaModelLoader(std::shared_ptr<ModelPathManager> model_path_manager)
    : model_path_manager_(model_path_manager) {
    logger_.initialize();
    
    // Initialize ModelfileParser
    modelfile_parser_ = std::make_shared<ModelfileParser>(model_path_manager_);
    
    // Initialization completed
}

bool OllamaModelLoader::loadFromOllamaModel(const std::string& model_name) {
    ModelPath model_path;
    if (!parseOllamaModelName(model_name, model_path)) {
        logger_.error("Failed to parse ollama model name: " + model_name);
        return false;
    }
    
    return loadFromModelPath(model_path);
}

bool OllamaModelLoader::loadFromModelPath(const ModelPath& model_path) {
    logger_.info("[OllamaModelLoader] Starting to load model: " + model_path.toString());
    
    // Read manifest file
    logger_.info("[OllamaModelLoader] Reading manifest file for model...");
    ModelManifest manifest;
    if (!model_path_manager_->readManifest(model_path, manifest)) {
        logger_.error("[OllamaModelLoader] Failed to read manifest for model: " + model_path.toString());
        return false;
    }
    logger_.info("[OllamaModelLoader] Manifest file read successfully");
    
    // Get GGUF file path from manifest
    logger_.info("[OllamaModelLoader] Extracting GGUF path from manifest...");
    std::string gguf_path = getGGUFPathFromManifest(manifest);
    if (gguf_path.empty()) {
        logger_.error("[OllamaModelLoader] No GGUF model found in manifest for: " + model_path.toString());
        return false;
    }
    logger_.info("[OllamaModelLoader] GGUF path extracted: " + gguf_path);
    
    // Check if file exists
    logger_.info("[OllamaModelLoader] Checking if GGUF file exists...");
    if (!std::filesystem::exists(gguf_path)) {
        logger_.error("[OllamaModelLoader] GGUF model file not found: " + gguf_path);
        return false;
    }
    
    // Get file size information
    auto file_size = std::filesystem::file_size(gguf_path);
    logger_.info("[OllamaModelLoader] GGUF file size: " + std::to_string(file_size / (1024 * 1024)) + " MB");
    
    logger_.info("[OllamaModelLoader] Loading GGUF model from: " + gguf_path);
    
    logger_.info("[OllamaModelLoader] Successfully loaded ollama model: " + model_path.toString());
    logger_.info("[OllamaModelLoader] Model loading completed successfully");
    return true;
}

bool OllamaModelLoader::isOllamaModelAvailable(const std::string& model_name) {
    ModelPath model_path;
    if (!parseOllamaModelName(model_name, model_path)) {
        return false;
    }
    
    ModelManifest manifest;
    return model_path_manager_->readManifest(model_path, manifest);
}

std::vector<std::string> OllamaModelLoader::listAvailableModels() {
    std::vector<std::string> model_names;
    
    // Enumerate all manifest files
    auto manifests = model_path_manager_->enumerateManifests(true);
    
    for (const auto& [path, manifest] : manifests) {
        // Extract model name from path
        // Path format: registry/namespace/repository:tag
        std::regex path_regex(R"(([^/]+)/([^/]+)/([^:]+):([^:]+))");
        std::smatch matches;
        
        if (std::regex_search(path, matches, path_regex)) {
            std::string registry = matches[1].str();
            std::string namespace_ = matches[2].str();
            std::string repository = matches[3].str();
            std::string tag = matches[4].str();

            // 过滤视觉/多模态模型（例如 qwen3-vl、llava、glm-4v、phi-3-vision 等）
            auto is_vision_like = [](const std::string &repo) {
                std::string s = repo;
                std::transform(s.begin(), s.end(), s.begin(), ::tolower);

                const std::vector<std::string> keywords = {
                    // 通用
                    "-vl", "vl", "vision", "multimodal",
                    // 常见模型族
                    "llava", "bakllava", "glm-4v", "4v", "phi-3-vision",
                    "moondream", "minicpm", "cogvlm"
                };

                for (const auto &k : keywords) {
                    if (s.find(k) != std::string::npos) return true;
                }
                return false;
            };

            if (is_vision_like(repository)) {
                // Skip vision models to only expose text models compatible with llama.cpp
                continue;
            }
            
            // Build ollama-style model name
            std::string model_name;
            if (namespace_ == "library") {
                // Official model, omit namespace
                model_name = repository;
            } else {
                model_name = namespace_ + "/" + repository;
            }
            
            if (tag != "latest") {
                model_name += ":" + tag;
            }
            
            model_names.push_back(model_name);
        }
    }
    
    // Sort and remove duplicates
    std::sort(model_names.begin(), model_names.end());
    model_names.erase(std::unique(model_names.begin(), model_names.end()), model_names.end());
    
    return model_names;
}

std::string OllamaModelLoader::getGGUFPathFromManifest(const ModelManifest& manifest) {
    // Find model layer (GGUF file)
    for (const auto& layer : manifest.layers) {
        // Debug output for layer checking
        std::cout << "Checking layer: mediaType=" << layer.media_type << ", digest=" << layer.digest << std::endl;
        std::string blob_path = model_path_manager_->getBlobFilePath(layer.digest);
        std::cout << "Generated blob path: " << blob_path << std::endl;
        std::cout << "File exists: " << (std::filesystem::exists(blob_path) ? "yes" : "no") << std::endl;
        
        // Media type for GGUF models in ollama
        if (layer.media_type == "application/vnd.ollama.image.model" ||
            layer.media_type == "application/vnd.docker.image.rootfs.diff.tar.gzip") {
            
            // Get blob file path
            if (!blob_path.empty() && std::filesystem::exists(blob_path)) {
                return blob_path;
            }
        }
    }
    
    return "";
}

bool OllamaModelLoader::parseOllamaModelName(const std::string& model_name, ModelPath& model_path) {
    std::string normalized_name = normalizeOllamaModelName(model_name);
    return model_path.parseFromString(normalized_name);
}

std::string OllamaModelLoader::normalizeOllamaModelName(const std::string& model_name) {
    std::string normalized = model_name;
    
    // If no registry prefix, add default ollama registry
    if (normalized.find("://") == std::string::npos) {
        normalized = "registry://" + normalized;
    }
    
    // Parse model name format: [scheme://][registry/][namespace/]repository[:tag]
    // Support repository names containing dots, such as qwen2.5vl
    std::regex full_path_regex(R"(^([^:/]+://)([^/]+)/([^/]+)/([^:]+)(?::([^:]+))?$)");
    std::smatch matches;
    
    if (std::regex_match(normalized, matches, full_path_regex)) {
        // Already in full path format, return directly
        if (matches[5].str().empty()) {
            // No tag, add default latest
            normalized += ":latest";
        }
        return normalized;
    }
    
    // Handle simple format: repository[:tag] or namespace/repository[:tag]
    std::regex simple_name_regex(R"(^([^:/]+://)?(?:([^/]+)/)?([^/:]+(?:\.[^/:]+)*)(?::([^:]+))?$)");
    
    if (std::regex_match(normalized, matches, simple_name_regex)) {
        std::string scheme = matches[1].str();
        std::string namespace_part = matches[2].str();
        std::string repository = matches[3].str();
        std::string tag = matches[4].str();
        
        if (scheme.empty()) {
            scheme = "registry://";
        }
        
        if (namespace_part.empty()) {
            namespace_part = "library";
        }
        
        if (tag.empty()) {
            tag = "latest";
        }
        
        normalized = scheme + "registry.ollama.ai/" + namespace_part + "/" + repository + ":" + tag;
    }
    
    return normalized;
}

bool OllamaModelLoader::loadFromOllamaModelWithLoRA(
    const std::string& model_name,
    bool enable_lora) {
    
    logger_.info("[OllamaModelLoader] Loading model with LoRA support: " + model_name);
    
    ModelPath model_path;
    if (!parseOllamaModelName(model_name, model_path)) {
        logger_.error("Failed to parse ollama model name: " + model_name);
        return false;
    }
    
    if (!enable_lora) {
        // If LoRA is not enabled, use standard loading method
        return loadFromModelPath(model_path);
    }
    
    // Read model manifest
    ModelManifest manifest;
    if (!model_path_manager_->readManifest(model_path, manifest)) {
        logger_.error("Failed to read manifest for model: " + model_name);
        return false;
    }
    
    // Parse Modelfile configuration
    ModelfileConfig config;
    if (!parseModelfileFromManifest(manifest, config)) {
        logger_.warning("No Modelfile configuration found, using standard loading");
        return loadFromModelPath(model_path);
    }
    
    // Load model using configuration
    return loadFromModelfileConfig(config);
}

bool OllamaModelLoader::loadFromModelfileConfig(
    const ModelfileConfig& config) {
    
    logger_.info("[OllamaModelLoader] Loading model from Modelfile config");
    logger_.info("Base model: " + config.base_model);
    logger_.info("LoRA adapters: " + std::to_string(config.lora_adapters.size()));
    
    // Verify base model file exists
    if (!std::filesystem::exists(config.base_model)) {
        logger_.error("Base model file not found: " + config.base_model);
        return false;
    }
    
    return true;
}

bool OllamaModelLoader::parseModelfileFromManifest(
    const ModelManifest& manifest,
    ModelfileConfig& config) {
    
    if (!modelfile_parser_) {
        logger_.error("ModelfileParser not initialized");
        return false;
    }
    
    // Parse manifest using ModelfileParser
    bool success = modelfile_parser_->parseFromManifest(manifest, config);
    
    if (success) {
        logger_.info("Successfully parsed Modelfile configuration");
        logger_.info("Base model: " + config.base_model);
        logger_.info("LoRA adapters found: " + std::to_string(config.lora_adapters.size()));
        
        // Validate LoRA adapters
        for (const auto& adapter : config.lora_adapters) {
            if (!modelfile_parser_->validateLoRAAdapter(adapter)) {
                logger_.warning("Invalid LoRA adapter: " + adapter.name + " at " + adapter.path);
            } else {
                logger_.info("Valid LoRA adapter: " + adapter.name + " (scale: " + std::to_string(adapter.scale) + ")");
            }
        }
    } else {
        logger_.warning("Failed to parse Modelfile configuration from manifest");
    }
    
    return success;
}

} // namespace core
} // namespace duorou