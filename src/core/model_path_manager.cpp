#include "model_path_manager.h"
#include "logger.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <openssl/sha.h>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <unordered_set>

namespace duorou {
namespace core {

// ModelPath implementation
bool ModelPath::parseFromString(const std::string& path) {
    // 正则表达式匹配格式：[scheme://]registry/namespace/repository:tag
    std::regex path_regex(R"(^(?:([^:/]+)://)?([^/]+)/([^/]+)/([^:]+)(?::([^:]+))?$)");
    std::smatch matches;
    
    if (std::regex_match(path, matches, path_regex)) {
        scheme = matches[1].str();
        registry = matches[2].str();
        namespace_ = matches[3].str();
        repository = matches[4].str();
        tag = matches[5].str();
        
        // 设置默认值
        if (scheme.empty()) {
            scheme = "registry";
        }
        if (tag.empty()) {
            tag = "latest";
        }
        
        return true;
    }
    
    return false;
}

std::string ModelPath::toString() const {
    std::ostringstream oss;
    if (!scheme.empty() && scheme != "registry") {
        oss << scheme << "://";
    }
    oss << registry << "/" << namespace_ << "/" << repository;
    if (!tag.empty() && tag != "latest") {
        oss << ":" << tag;
    }
    return oss.str();
}

std::string ModelPath::getBaseURL() const {
    std::ostringstream oss;
    if (!scheme.empty()) {
        oss << scheme << "://";
    }
    oss << registry;
    return oss.str();
}

// ModelPathManager implementation
ModelPathManager::ModelPathManager(const std::string& base_path) 
    : base_path_(base_path), initialized_(false) {
    if (base_path_.empty()) {
        // 使用默认路径
        base_path_ = std::filesystem::current_path() / "models";
    }
}

bool ModelPathManager::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        return true;
    }
    
    try {
        // 确保基础目录存在
        if (!ensureDirectoryExists(base_path_)) {
            std::cerr << "Error: Failed to create base directory: " << base_path_ << std::endl;
            return false;
        }
        
        // 确保manifests和blobs目录存在
        if (!ensureDirectoryExists(getManifestPath()) || 
            !ensureDirectoryExists(getBlobsPath())) {
            std::cerr << "Error: Failed to create manifests or blobs directory" << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "Info: ModelPathManager initialized with base path: " << base_path_ << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to initialize ModelPathManager: " << e.what() << std::endl;
        return false;
    }
}

std::string ModelPathManager::getManifestPath() const {
    return (std::filesystem::path(base_path_) / "manifests").string();
}

std::string ModelPathManager::getManifestFilePath(const ModelPath& model_path) const {
    std::filesystem::path manifest_path = std::filesystem::path(getManifestPath()) / 
                                         model_path.registry / 
                                         model_path.namespace_ / 
                                         model_path.repository / 
                                         model_path.tag;
    return manifest_path.string();
}

std::string ModelPathManager::getBlobsPath() const {
    return (std::filesystem::path(base_path_) / "blobs").string();
}

std::string ModelPathManager::getBlobFilePath(const std::string& digest) const {
    if (!isValidDigest(digest)) {
        return "";
    }
    
    // 使用sha256:前缀的格式
    std::string clean_digest = digest;
    if (clean_digest.substr(0, 7) == "sha256:") {
        clean_digest = clean_digest.substr(7);
    }
    
    return (std::filesystem::path(getBlobsPath()) / clean_digest).string();
}

bool ModelPathManager::isValidDigest(const std::string& digest) {
    // 检查SHA256格式：sha256:64位十六进制字符
    std::regex sha256_regex(R"(^sha256:[a-fA-F0-9]{64}$)");
    return std::regex_match(digest, sha256_regex);
}

bool ModelPathManager::readManifest(const ModelPath& model_path, ModelManifest& manifest) const {
    std::string manifest_file = getManifestFilePath(model_path);
    
    if (!std::filesystem::exists(manifest_file)) {
        std::cerr << "Warning: Manifest file not found: " << manifest_file << std::endl;
        return false;
    }
    
    try {
        std::ifstream file(manifest_file);
        if (!file.is_open()) {
            std::cerr << "Error: Failed to open manifest file: " << manifest_file << std::endl;
            return false;
        }
        
        nlohmann::json json_data;
        file >> json_data;
        
        return loadManifestFromJson(json_data, manifest);
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to read manifest: " << e.what() << std::endl;
        return false;
    }
}

bool ModelPathManager::writeManifest(const ModelPath& model_path, const ModelManifest& manifest) {
    std::string manifest_file = getManifestFilePath(model_path);
    
    try {
        // 确保目录存在
        std::filesystem::path parent_dir = std::filesystem::path(manifest_file).parent_path();
        if (!ensureDirectoryExists(parent_dir.string())) {
            std::cerr << "Error: Failed to create manifest directory: " << parent_dir.string() << std::endl;
            return false;
        }
        
        nlohmann::json json_data;
        if (!saveManifestToJson(manifest, json_data)) {
            return false;
        }
        
        std::ofstream file(manifest_file);
        if (!file.is_open()) {
            std::cerr << "Error: Failed to create manifest file: " << manifest_file << std::endl;
            return false;
        }
        
        file << json_data.dump(2);
        std::cout << "Info: Manifest written to: " << manifest_file << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to write manifest: " << e.what() << std::endl;
        return false;
    }
}

std::unordered_map<std::string, ModelManifest> ModelPathManager::enumerateManifests(bool continue_on_error) const {
    std::unordered_map<std::string, ModelManifest> manifests;
    std::string manifest_root = getManifestPath();
    
    if (!std::filesystem::exists(manifest_root)) {
        std::cerr << "Warning: Manifest directory does not exist: " << manifest_root << std::endl;
        return manifests;
    }
    
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(manifest_root)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                
                // 解析模型路径
                std::string relative_path = std::filesystem::relative(entry.path(), manifest_root).string();
                
                // 将路径转换为模型名称格式
                std::replace(relative_path.begin(), relative_path.end(), '/', ':');
                std::replace(relative_path.begin(), relative_path.end(), '\\', ':');
                
                ModelManifest manifest;
                if (readManifest(ModelPath(), manifest)) {
                    manifests[relative_path] = manifest;
                } else if (!continue_on_error) {
                    std::cerr << "Error: Failed to read manifest: " << file_path << std::endl;
                    break;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to enumerate manifests: " << e.what() << std::endl;
    }
    
    return manifests;
}

bool ModelPathManager::blobExists(const std::string& digest) const {
    std::string blob_file = getBlobFilePath(digest);
    return !blob_file.empty() && std::filesystem::exists(blob_file);
}

size_t ModelPathManager::getBlobSize(const std::string& digest) const {
    std::string blob_file = getBlobFilePath(digest);
    if (blob_file.empty() || !std::filesystem::exists(blob_file)) {
        return 0;
    }
    
    try {
        return std::filesystem::file_size(blob_file);
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to get blob size: " << e.what() << std::endl;
        return 0;
    }
}

size_t ModelPathManager::deleteUnusedLayers(const std::vector<std::string>& used_digests) {
    std::string blobs_dir = getBlobsPath();
    size_t deleted_count = 0;
    
    if (!std::filesystem::exists(blobs_dir)) {
        return 0;
    }
    
    // 创建已使用摘要的集合
    std::unordered_set<std::string> used_set;
    for (const auto& digest : used_digests) {
        std::string clean_digest = digest;
        if (clean_digest.substr(0, 7) == "sha256:") {
            clean_digest = clean_digest.substr(7);
        }
        used_set.insert(clean_digest);
    }
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(blobs_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                
                // 检查是否为有效的SHA256文件名
                if (filename.length() == 64 && 
                    std::all_of(filename.begin(), filename.end(), 
                               [](char c) { return std::isxdigit(c); })) {
                    
                    if (used_set.find(filename) == used_set.end()) {
                        // 未使用的blob，删除它
                        std::filesystem::remove(entry.path());
                        deleted_count++;
                        std::cout << "Info: Deleted unused blob: " << filename << std::endl;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to delete unused layers: " << e.what() << std::endl;
    }
    
    return deleted_count;
}

size_t ModelPathManager::pruneLayers() {
    std::string blobs_dir = getBlobsPath();
    size_t pruned_count = 0;
    
    if (!std::filesystem::exists(blobs_dir)) {
        return 0;
    }
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(blobs_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                
                // 检查文件名是否为有效的SHA256格式
                if (filename.length() != 64 || 
                    !std::all_of(filename.begin(), filename.end(), 
                                [](char c) { return std::isxdigit(c); })) {
                    // 无效的blob文件，删除它
                    std::filesystem::remove(entry.path());
                    pruned_count++;
                    std::cout << "Info: Pruned invalid blob: " << filename << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to prune layers: " << e.what() << std::endl;
    }
    
    return pruned_count;
}

bool ModelPathManager::verifyBlob(const std::string& digest) const {
    std::string blob_file = getBlobFilePath(digest);
    if (blob_file.empty() || !std::filesystem::exists(blob_file)) {
        return false;
    }
    
    std::string calculated_digest = "sha256:" + calculateSHA256(blob_file);
    return calculated_digest == digest;
}

std::string ModelPathManager::calculateSHA256(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }
    
    SHA256_CTX sha256_ctx;
    SHA256_Init(&sha256_ctx);
    
    char buffer[8192];
    while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
        SHA256_Update(&sha256_ctx, buffer, file.gcount());
    }
    
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &sha256_ctx);
    
    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    
    return oss.str();
}

void ModelPathManager::setBasePath(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    base_path_ = path;
    initialized_ = false;  // 需要重新初始化
}

bool ModelPathManager::ensureDirectoryExists(const std::string& path) const {
    try {
        if (!std::filesystem::exists(path)) {
            std::filesystem::create_directories(path);
        }
        return std::filesystem::is_directory(path);
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to create directory " << path << ": " << e.what() << std::endl;
        return false;
    }
}

bool ModelPathManager::loadManifestFromJson(const nlohmann::json& json_data, ModelManifest& manifest) const {
    try {
        manifest.schema_version = json_data.value("schemaVersion", 2);
        manifest.media_type = json_data.value("mediaType", "");
        
        // 加载配置层
        if (json_data.contains("config")) {
            const auto& config_json = json_data["config"];
            manifest.config.digest = config_json.value("digest", "");
            manifest.config.media_type = config_json.value("mediaType", "");
            manifest.config.size = config_json.value("size", 0);
        }
        
        // 加载层列表
        if (json_data.contains("layers")) {
            for (const auto& layer_json : json_data["layers"]) {
                ModelLayer layer;
                layer.digest = layer_json.value("digest", "");
                layer.media_type = layer_json.value("mediaType", "");
                layer.size = layer_json.value("size", 0);
                manifest.layers.push_back(layer);
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to parse manifest JSON: " << e.what() << std::endl;
        return false;
    }
}

bool ModelPathManager::saveManifestToJson(const ModelManifest& manifest, nlohmann::json& json_data) const {
    try {
        json_data["schemaVersion"] = manifest.schema_version;
        json_data["mediaType"] = manifest.media_type;
        
        // 保存配置层
        if (!manifest.config.digest.empty()) {
            json_data["config"] = {
                {"digest", manifest.config.digest},
                {"mediaType", manifest.config.media_type},
                {"size", manifest.config.size}
            };
        }
        
        // 保存层列表
        json_data["layers"] = nlohmann::json::array();
        for (const auto& layer : manifest.layers) {
            json_data["layers"].push_back({
                {"digest", layer.digest},
                {"mediaType", layer.media_type},
                {"size", layer.size}
            });
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to create manifest JSON: " << e.what() << std::endl;
        return false;
    }
}

} // namespace core
} // namespace duorou