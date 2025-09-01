#include "ollama_model_loader.h"
#include "logger.h"
#include "../extensions/llama_cpp/arch_mapping.h"
#include "../extensions/llama_cpp/model_loader_wrapper.h"
#include <filesystem>
#include <regex>
#include <algorithm>
#include <iostream>

namespace duorou {
namespace core {

OllamaModelLoader::OllamaModelLoader(std::shared_ptr<ModelPathManager> model_path_manager)
    : model_path_manager_(model_path_manager) {
    logger_.initialize();
}

struct llama_model* OllamaModelLoader::loadFromOllamaModel(const std::string& model_name, 
                                                         const llama_model_params& model_params) {
    ModelPath model_path;
    if (!parseOllamaModelName(model_name, model_path)) {
        logger_.error("Failed to parse ollama model name: " + model_name);
        return nullptr;
    }
    
    return loadFromModelPath(model_path, model_params);
}

struct llama_model* OllamaModelLoader::loadFromModelPath(const ModelPath& model_path,
                                                       const llama_model_params& model_params) {
    // 读取manifest文件
    ModelManifest manifest;
    if (!model_path_manager_->readManifest(model_path, manifest)) {
        logger_.error("Failed to read manifest for model: " + model_path.toString());
        return nullptr;
    }
    
    // 从manifest获取GGUF文件路径
    std::string gguf_path = getGGUFPathFromManifest(manifest);
    if (gguf_path.empty()) {
        logger_.error("No GGUF model found in manifest for: " + model_path.toString());
        return nullptr;
    }
    
    // 检查文件是否存在
    if (!std::filesystem::exists(gguf_path)) {
        logger_.error("GGUF model file not found: " + gguf_path);
        return nullptr;
    }
    
    logger_.info("Loading GGUF model from: " + gguf_path);
    
    // 使用ModelLoaderWrapper加载模型，支持架构映射
    struct llama_model* model = duorou::extensions::llama_cpp::ModelLoaderWrapper::loadModelWithArchMapping(gguf_path, model_params);
    if (!model) {
        logger_.error("Failed to load GGUF model: " + gguf_path);
        return nullptr;
    }
    
    logger_.info("Successfully loaded ollama model: " + model_path.toString());
    return model;
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
    
    // 枚举所有manifest文件
    auto manifests = model_path_manager_->enumerateManifests(true);
    
    for (const auto& [path, manifest] : manifests) {
        // 从路径提取模型名称
        // 路径格式: registry/namespace/repository:tag
        std::regex path_regex(R"(([^/]+)/([^/]+)/([^:]+):([^:]+))");
        std::smatch matches;
        
        if (std::regex_search(path, matches, path_regex)) {
            std::string registry = matches[1].str();
            std::string namespace_ = matches[2].str();
            std::string repository = matches[3].str();
            std::string tag = matches[4].str();
            
            // 构建ollama风格的模型名称
            std::string model_name;
            if (namespace_ == "library") {
                // 官方模型，省略namespace
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
    
    // 排序并去重
    std::sort(model_names.begin(), model_names.end());
    model_names.erase(std::unique(model_names.begin(), model_names.end()), model_names.end());
    
    return model_names;
}

std::string OllamaModelLoader::getGGUFPathFromManifest(const ModelManifest& manifest) {
    // 查找模型层（GGUF文件）
    for (const auto& layer : manifest.layers) {
        // Debug output for layer checking
        std::cout << "Checking layer: mediaType=" << layer.media_type << ", digest=" << layer.digest << std::endl;
        std::string blob_path = model_path_manager_->getBlobFilePath(layer.digest);
        std::cout << "Generated blob path: " << blob_path << std::endl;
        std::cout << "File exists: " << (std::filesystem::exists(blob_path) ? "yes" : "no") << std::endl;
        
        // ollama中GGUF模型的media type
        if (layer.media_type == "application/vnd.ollama.image.model" ||
            layer.media_type == "application/vnd.docker.image.rootfs.diff.tar.gzip") {
            
            // 获取blob文件路径
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
    
    // 如果没有registry前缀，添加默认的ollama registry
    if (normalized.find("://") == std::string::npos) {
        normalized = "registry://" + normalized;
    }
    
    // 解析模型名称格式：[scheme://][registry/][namespace/]repository[:tag]
    // 支持repository名称包含点号，如qwen2.5vl
    std::regex full_path_regex(R"(^([^:/]+://)([^/]+)/([^/]+)/([^:]+)(?::([^:]+))?$)");
    std::smatch matches;
    
    if (std::regex_match(normalized, matches, full_path_regex)) {
        // 已经是完整路径格式，直接返回
        if (matches[5].str().empty()) {
            // 没有tag，添加默认的latest
            normalized += ":latest";
        }
        return normalized;
    }
    
    // 处理简单格式：repository[:tag] 或 namespace/repository[:tag]
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

} // namespace core
} // namespace duorou