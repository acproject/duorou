#include "ollama_path_resolver.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <algorithm>

namespace duorou {
namespace extensions {
namespace ollama {

OllamaPathResolver::OllamaPathResolver(bool verbose) : verbose_(verbose) {
    log("INFO", "OllamaPathResolver initialized");
}

std::optional<std::string> OllamaPathResolver::resolveModelPath(const std::string& model_name) {
    log("INFO", "Resolving model path for: " + model_name);
    
    // 解析模型名称
    auto model_info = parseModelName(model_name);
    if (!model_info) {
        log("ERROR", "Failed to parse model name: " + model_name);
        return std::nullopt;
    }
    
    // 构建 manifest 文件路径
    std::string manifest_path = buildManifestPath(*model_info);
    log("DEBUG", "Manifest path: " + manifest_path);
    
    // 检查 manifest 文件是否存在
    if (!std::filesystem::exists(manifest_path)) {
        log("ERROR", "Manifest file not found: " + manifest_path);
        return std::nullopt;
    }
    
    // 读取 manifest 文件
    auto manifest = readManifest(manifest_path);
    if (!manifest) {
        log("ERROR", "Failed to read manifest: " + manifest_path);
        return std::nullopt;
    }
    
    // 从 manifest 获取 GGUF 文件路径
    auto gguf_path = getGGUFPathFromManifest(*manifest);
    if (!gguf_path) {
        log("ERROR", "Failed to get GGUF path from manifest");
        return std::nullopt;
    }
    
    log("INFO", "Resolved GGUF path: " + *gguf_path);
    return gguf_path;
}

std::string OllamaPathResolver::getOllamaModelsDir() const {
    if (!custom_models_dir_.empty()) {
        return custom_models_dir_;
    }
    return getDefaultOllamaModelsDir();
}

void OllamaPathResolver::setCustomModelsDir(const std::string& custom_dir) {
    custom_models_dir_ = custom_dir;
    log("INFO", "Custom models directory set to: " + custom_dir);
}

std::optional<OllamaModelInfo> OllamaPathResolver::parseModelName(const std::string& model_name) {
    // 标准化模型名称
    std::string normalized_name = normalizeModelName(model_name);
    
    OllamaModelInfo info;
    info.registry = "registry.ollama.ai";
    info.namespace_name = "library";
    info.tag = "latest";
    
    // 简单的字符串解析，避免使用正则表达式
    std::string remaining = normalized_name;
    
    // 查找标签（:之后的部分）
    size_t tag_pos = remaining.find_last_of(':');
    if (tag_pos != std::string::npos) {
        info.tag = remaining.substr(tag_pos + 1);
        remaining = remaining.substr(0, tag_pos);
    }
    
    // 按 / 分割路径
    std::vector<std::string> parts;
    std::stringstream ss(remaining);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    
    if (parts.empty()) {
        log("ERROR", "Invalid model name format: " + normalized_name);
        return std::nullopt;
    }
    
    if (parts.size() == 1) {
        // 格式：name
        info.name = parts[0];
    } else if (parts.size() == 2) {
        // 格式：namespace/name
        info.namespace_name = parts[0];
        info.name = parts[1];
    } else if (parts.size() == 3) {
        // 格式：registry/namespace/name
        info.registry = parts[0];
        info.namespace_name = parts[1];
        info.name = parts[2];
    } else {
        log("ERROR", "Too many path components in model name: " + normalized_name);
        return std::nullopt;
    }
    
    log("DEBUG", "Parsed model - Registry: " + info.registry + 
                ", Namespace: " + info.namespace_name + 
                ", Name: " + info.name + 
                ", Tag: " + info.tag);
    
    return info;
}

std::optional<nlohmann::json> OllamaPathResolver::readManifest(const std::string& manifest_path) {
    try {
        std::ifstream file(manifest_path);
        if (!file.is_open()) {
            log("ERROR", "Cannot open manifest file: " + manifest_path);
            return std::nullopt;
        }
        
        nlohmann::json manifest;
        file >> manifest;
        
        log("DEBUG", "Successfully read manifest file");
        return manifest;
        
    } catch (const std::exception& e) {
        log("ERROR", "Failed to parse manifest JSON: " + std::string(e.what()));
        return std::nullopt;
    }
}

std::optional<std::string> OllamaPathResolver::getGGUFPathFromManifest(const nlohmann::json& manifest) {
    try {
        if (!manifest.contains("layers") || !manifest["layers"].is_array()) {
            log("ERROR", "Invalid manifest: missing or invalid layers");
            return std::nullopt;
        }
        
        // 查找模型层（通常是最大的文件）
        std::string model_digest;
        size_t max_size = 0;
        
        for (const auto& layer : manifest["layers"]) {
            if (!layer.contains("digest") || !layer.contains("size")) {
                continue;
            }
            
            std::string media_type = layer.value("mediaType", "");
            size_t size = layer["size"].get<size_t>();
            
            // 查找模型文件（通常是最大的文件或特定媒体类型）
            if (media_type == "application/vnd.ollama.image.model" || size > max_size) {
                model_digest = layer["digest"].get<std::string>();
                max_size = size;
            }
        }
        
        if (model_digest.empty()) {
            log("ERROR", "No model layer found in manifest");
            return std::nullopt;
        }
        
        // 构建 blob 文件路径
        std::string blob_path = buildBlobPath(model_digest);
        
        if (!std::filesystem::exists(blob_path)) {
            log("ERROR", "Model blob file not found: " + blob_path);
            return std::nullopt;
        }
        
        return blob_path;
        
    } catch (const std::exception& e) {
        log("ERROR", "Failed to extract GGUF path from manifest: " + std::string(e.what()));
        return std::nullopt;
    }
}

bool OllamaPathResolver::modelExists(const std::string& model_name) {
    auto path = resolveModelPath(model_name);
    return path.has_value();
}

std::vector<std::string> OllamaPathResolver::listAvailableModels() {
    std::vector<std::string> models;
    
    std::string models_dir = getOllamaModelsDir();
    std::string manifests_dir = models_dir + "/manifests";
    
    if (!std::filesystem::exists(manifests_dir)) {
        log("WARNING", "Manifests directory not found: " + manifests_dir);
        return models;
    }
    
    try {
        // 遍历 manifests 目录结构
        for (const auto& registry_entry : std::filesystem::directory_iterator(manifests_dir)) {
            if (!registry_entry.is_directory()) continue;
            
            std::string registry = registry_entry.path().filename().string();
            
            for (const auto& namespace_entry : std::filesystem::directory_iterator(registry_entry)) {
                if (!namespace_entry.is_directory()) continue;
                
                std::string namespace_name = namespace_entry.path().filename().string();
                
                for (const auto& model_entry : std::filesystem::directory_iterator(namespace_entry)) {
                    if (!model_entry.is_directory()) continue;
                    
                    std::string model_name = model_entry.path().filename().string();
                    
                    for (const auto& tag_entry : std::filesystem::directory_iterator(model_entry)) {
                        if (tag_entry.is_regular_file()) {
                            std::string tag = tag_entry.path().filename().string();
                            std::string full_name = registry + "/" + namespace_name + "/" + model_name + ":" + tag;
                            models.push_back(full_name);
                        }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        log("ERROR", "Failed to list models: " + std::string(e.what()));
    }
    
    return models;
}

void OllamaPathResolver::log(const std::string& level, const std::string& message) const {
    if (verbose_ || level == "ERROR") {
        std::cout << "[" << level << "] OllamaPathResolver: " << message << std::endl;
    }
}

std::string OllamaPathResolver::getDefaultOllamaModelsDir() const {
    // 获取用户主目录
    const char* home = std::getenv("HOME");
    if (!home) {
        home = std::getenv("USERPROFILE"); // Windows
    }
    
    if (!home) {
        log("WARNING", "Cannot determine home directory, using current directory");
        return "./models";
    }
    
    return std::string(home) + "/.ollama/models";
}

std::string OllamaPathResolver::normalizeModelName(const std::string& model_name) const {
    // 移除可能的前缀和后缀空格
    std::string normalized = model_name;
    
    // 移除前后空格
    size_t start = normalized.find_first_not_of(" \t\n\r");
    if (start != std::string::npos) {
        normalized = normalized.substr(start);
    }
    
    size_t end = normalized.find_last_not_of(" \t\n\r");
    if (end != std::string::npos) {
        normalized = normalized.substr(0, end + 1);
    }
    
    return normalized;
}

std::string OllamaPathResolver::buildManifestPath(const OllamaModelInfo& model_info) const {
    std::string models_dir = getOllamaModelsDir();
    return models_dir + "/manifests/" + model_info.registry + "/" + 
           model_info.namespace_name + "/" + model_info.name + "/" + model_info.tag;
}

std::string OllamaPathResolver::buildBlobPath(const std::string& digest) const {
    std::string models_dir = getOllamaModelsDir();
    // Ollama 存储 blob 文件时将冒号替换为连字符
    std::string blob_filename = digest;
    std::replace(blob_filename.begin(), blob_filename.end(), ':', '-');
    return models_dir + "/blobs/" + blob_filename;
}

} // namespace ollama
} // namespace extensions
} // namespace duorou