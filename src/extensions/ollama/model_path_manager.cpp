#include "model_path_manager.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include "../../../third_party/llama.cpp/vendor/nlohmann/json.hpp"

namespace duorou {
namespace extensions {
namespace ollama {

// ModelPath 实现
bool ModelPath::parseFromString(const std::string& path_str) {
    // 解析格式: [registry/][namespace/]model[:tag]
    // 例如: "llama3.2:latest" 或 "registry.ollama.ai/library/llama3.2:latest"
    
    std::string remaining = path_str;
    
    // 提取tag
    size_t colon_pos = remaining.find_last_of(':');
    if (colon_pos != std::string::npos) {
        tag = remaining.substr(colon_pos + 1);
        remaining = remaining.substr(0, colon_pos);
    } else {
        tag = "latest";
    }
    
    // 分割路径部分
    std::vector<std::string> parts;
    std::stringstream ss(remaining);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    
    if (parts.empty()) {
        return false;
    }
    
    if (parts.size() == 1) {
        // 只有模型名
        registry = "";
        namespace_ = "library";
        model = parts[0];
    } else if (parts.size() == 2) {
        // namespace/model
        registry = "";
        namespace_ = parts[0];
        model = parts[1];
    } else if (parts.size() == 3) {
        // registry/namespace/model
        registry = parts[0];
        namespace_ = parts[1];
        model = parts[2];
    } else {
        return false;
    }
    
    return true;
}

std::string ModelPath::toString() const {
    std::string result;
    
    if (!registry.empty()) {
        result += registry + "/";
    }
    
    if (!namespace_.empty() && namespace_ != "library") {
        result += namespace_ + "/";
    }
    
    result += model;
    
    if (!tag.empty() && tag != "latest") {
        result += ":" + tag;
    }
    
    return result;
}

// ModelManifest 实现
bool ModelManifest::parseFromJSON(const std::string& json_str) {
    try {
        // 使用nlohmann/json解析JSON
        nlohmann::json json_data = nlohmann::json::parse(json_str);
        
        // 解析schema_version
        if (json_data.contains("schemaVersion")) {
            schema_version = json_data["schemaVersion"].get<std::string>();
        }
        
        // 解析media_type
        if (json_data.contains("mediaType")) {
            media_type = json_data["mediaType"].get<std::string>();
        }
        
        // 解析config
        if (json_data.contains("config")) {
            auto config_obj = json_data["config"];
            if (config_obj.contains("digest")) {
                config["digest"] = config_obj["digest"].get<std::string>();
            }
            if (config_obj.contains("mediaType")) {
                config["mediaType"] = config_obj["mediaType"].get<std::string>();
            }
        }
        
        // 解析layers
        if (json_data.contains("layers") && json_data["layers"].is_array()) {
            layers.clear();
            for (const auto& layer : json_data["layers"]) {
                if (layer.contains("digest")) {
                    layers.push_back(layer["digest"].get<std::string>());
                }
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON manifest: " << e.what() << std::endl;
        return false;
    }
}

std::string ModelManifest::getConfigBlob() const {
    auto it = config.find("config");
    return (it != config.end()) ? it->second : "";
}

std::string ModelManifest::getModelBlob() const {
    if (!layers.empty()) {
        return layers[0]; // 通常第一个layer是模型权重
    }
    return "";
}

// ModelPathManager 实现
ModelPathManager::ModelPathManager(const std::string& ollama_models_dir)
    : verbose_(false) {
    if (ollama_models_dir.empty()) {
        models_dir_ = getDefaultModelsDirectory();
    } else {
        models_dir_ = ollama_models_dir;
    }
    
    if (verbose_) {
        log("INFO", "Initialized ModelPathManager with models directory: " + models_dir_);
    }
}

ModelPathManager::~ModelPathManager() = default;

bool ModelPathManager::readManifest(const ModelPath& model_path, ModelManifest& manifest) {
    std::string manifest_path = getManifestPath(model_path);
    
    if (!fileExists(manifest_path)) {
        if (verbose_) {
            log("ERROR", "Manifest file not found: " + manifest_path);
        }
        return false;
    }
    
    std::string content = readFileContent(manifest_path);
    if (content.empty()) {
        if (verbose_) {
            log("ERROR", "Failed to read manifest file: " + manifest_path);
        }
        return false;
    }
    
    return manifest.parseFromJSON(content);
}

std::string ModelPathManager::getModelDirectory(const ModelPath& model_path) {
    std::string dir = models_dir_ + "/manifests/";
    
    if (!model_path.registry.empty()) {
        dir += model_path.registry + "/";
    }
    
    if (!model_path.namespace_.empty()) {
        dir += model_path.namespace_ + "/";
    }
    
    dir += model_path.model;
    
    return dir;
}

std::string ModelPathManager::getBlobPath(const std::string& blob_sha256) {
    if (blob_sha256.length() < 2) {
        return "";
    }
    
    // Ollama使用前两个字符作为子目录
    std::string prefix = blob_sha256.substr(0, 2);
    return models_dir_ + "/blobs/sha256-" + prefix + "/sha256-" + blob_sha256;
}

std::vector<std::string> ModelPathManager::listAvailableModels() {
    std::vector<std::string> models;
    
    std::string manifests_dir = models_dir_ + "/manifests";
    if (!directoryExists(manifests_dir)) {
        return models;
    }
    
    try {
        // 遍历manifests目录
        for (const auto& entry : std::filesystem::recursive_directory_iterator(manifests_dir)) {
            if (entry.is_regular_file()) {
                std::string relative_path = std::filesystem::relative(entry.path(), manifests_dir);
                
                // 移除文件扩展名（如果有）
                size_t dot_pos = relative_path.find_last_of('.');
                if (dot_pos != std::string::npos) {
                    relative_path = relative_path.substr(0, dot_pos);
                }
                
                // 转换路径分隔符为模型路径格式
                std::replace(relative_path.begin(), relative_path.end(), '/', ':');
                models.push_back(relative_path);
            }
        }
    } catch (const std::exception& e) {
        if (verbose_) {
            log("ERROR", "Failed to list models: " + std::string(e.what()));
        }
    }
    
    return models;
}

bool ModelPathManager::modelExists(const ModelPath& model_path) {
    std::string manifest_path = getManifestPath(model_path);
    return fileExists(manifest_path);
}

std::string ModelPathManager::getDefaultModelsDirectory() {
    // 获取用户主目录
    const char* home = std::getenv("HOME");
    if (home) {
        return std::string(home) + "/.ollama/models";
    }
    
    // 如果无法获取HOME，使用当前目录
    return "./ollama_models";
}

std::string ModelPathManager::getManifestPath(const ModelPath& model_path) {
    std::string path = getModelDirectory(model_path);
    
    if (!model_path.tag.empty()) {
        path += "/" + model_path.tag;
    }
    
    return path;
}

std::string ModelPathManager::readFileContent(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool ModelPathManager::directoryExists(const std::string& dir_path) {
    try {
        return std::filesystem::exists(dir_path) && std::filesystem::is_directory(dir_path);
    } catch (const std::exception&) {
        return false;
    }
}

bool ModelPathManager::fileExists(const std::string& file_path) {
    try {
        return std::filesystem::exists(file_path) && std::filesystem::is_regular_file(file_path);
    } catch (const std::exception&) {
        return false;
    }
}

void ModelPathManager::log(const std::string& level, const std::string& message) {
    if (verbose_) {
        std::cout << "[" << level << "] ModelPathManager: " << message << std::endl;
    }
}

} // namespace ollama
} // namespace extensions
} // namespace duorou