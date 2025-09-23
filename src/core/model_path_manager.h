#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <memory>
#include <mutex>
#include "../../third_party/llama.cpp/vendor/nlohmann/json.hpp"

namespace duorou {
namespace core {

/**
 * @brief Model layer information structure
 * Corresponds to the Layer structure in ollama
 */
struct ModelLayer {
    std::string digest;         ///< SHA256 digest
    std::string media_type;     ///< Media type
    size_t size;               ///< Layer size
    
    ModelLayer() : size(0) {}
    ModelLayer(const std::string& d, const std::string& mt, size_t s) 
        : digest(d), media_type(mt), size(s) {}
};

/**
 * @brief Model manifest structure
 * Corresponds to the Manifest structure in ollama
 */
struct ModelManifest {
    int schema_version;                 ///< Manifest version
    std::string media_type;             ///< Media type
    ModelLayer config;                  ///< Configuration layer
    std::vector<ModelLayer> layers;     ///< Model layer list
    
    ModelManifest() : schema_version(2) {}
    
    /**
     * @brief Calculate total manifest size
     * @return Total size (bytes)
     */
    size_t getTotalSize() const {
        size_t total = config.size;
        for (const auto& layer : layers) {
            total += layer.size;
        }
        return total;
    }
    
    /**
     * @brief Get digest list of all layers
     * @return Digest list
     */
    std::vector<std::string> getAllDigests() const {
        std::vector<std::string> digests;
        if (!config.digest.empty()) {
            digests.push_back(config.digest);
        }
        for (const auto& layer : layers) {
            digests.push_back(layer.digest);
        }
        return digests;
    }
};

/**
 * @brief Model path structure
 * Corresponds to the ModelPath structure in ollama
 */
struct ModelPath {
    std::string scheme;         ///< Protocol (e.g., registry)
    std::string registry;       ///< Registry address
    std::string namespace_;     ///< Namespace
    std::string repository;     ///< Repository name
    std::string tag;           ///< Tag
    
    ModelPath() = default;
    
    /**
     * @brief Parse model path from string
     * @param path Path string, format: [scheme://]registry/namespace/repository:tag
     * @return Returns true if parsing succeeds
     */
    bool parseFromString(const std::string& path);
    
    /**
     * @brief Convert to string
     * @return Path string
     */
    std::string toString() const;
    
    /**
     * @brief Get base URL
     * @return Base URL
     */
    std::string getBaseURL() const;
};

/**
 * @brief Model path manager
 * Responsible for managing ollama-compatible model storage structure
 */
class ModelPathManager {
public:
    /**
     * @brief Constructor
     * @param base_path Base storage path
     */
    explicit ModelPathManager(const std::string& base_path = "");
    
    /**
     * @brief Destructor
     */
    ~ModelPathManager() = default;
    
    /**
     * @brief Initialize manager
     * @return Returns true on success
     */
    bool initialize();
    
    /**
     * @brief Get manifest root directory path
     * @return Manifest directory path
     */
    std::string getManifestPath() const;
    
    /**
     * @brief Get manifest file path for specific model
     * @param model_path Model path
     * @return Manifest file path
     */
    std::string getManifestFilePath(const ModelPath& model_path) const;
    
    /**
     * @brief Get blobs root directory path
     * @return Blobs directory path
     */
    std::string getBlobsPath() const;
    
    /**
     * @brief Get blob file path for specific digest
     * @param digest SHA256 digest
     * @return Blob file path, returns empty string if digest is invalid
     */
    std::string getBlobFilePath(const std::string& digest) const;
    
    /**
     * @brief Validate SHA256 digest format
     * @param digest Digest string
     * @return Returns true if format is correct
     */
    static bool isValidDigest(const std::string& digest);
    
    /**
     * @brief Read model manifest
     * @param model_path Model path
     * @param manifest Output manifest
     * @return Returns true on success
     */
    bool readManifest(const ModelPath& model_path, ModelManifest& manifest) const;
    
    /**
     * @brief Write model manifest
     * @param model_path Model path
     * @param manifest Manifest data
     * @return Returns true on success
     */
    bool writeManifest(const ModelPath& model_path, const ModelManifest& manifest);
    
    /**
     * @brief Enumerate all local manifests
     * @param continue_on_error Whether to continue when encountering errors
     * @return Manifest mapping table
     */
    std::unordered_map<std::string, ModelManifest> enumerateManifests(bool continue_on_error = true) const;
    
    /**
     * @brief Check if blob exists
     * @param digest SHA256 digest
     * @return Returns true if exists
     */
    bool blobExists(const std::string& digest) const;
    
    /**
     * @brief Get blob file size
     * @param digest SHA256 digest
     * @return File size, returns 0 if file does not exist
     */
    size_t getBlobSize(const std::string& digest) const;
    
    /**
     * @brief Delete unused layer files
     * @param used_digests List of digests still in use
     * @return Number of deleted files
     */
    size_t deleteUnusedLayers(const std::vector<std::string>& used_digests);
    
    /**
     * @brief Clean up invalid blob files
     * @return Number of cleaned files
     */
    size_t pruneLayers();
    
    /**
     * @brief Verify SHA256 of blob file
     * @param digest Expected digest
     * @return Returns true if verification passes
     */
    bool verifyBlob(const std::string& digest) const;
    
    /**
     * @brief Calculate SHA256 digest of file
     * @param file_path File path
     * @return SHA256 digest string
     */
    static std::string calculateSHA256(const std::string& file_path);
    
    /**
     * @brief Set base storage path
     * @param path New base path
     */
    void setBasePath(const std::string& path);
    
    /**
     * @brief Get base storage path
     * @return Base path
     */
    const std::string& getBasePath() const { return base_path_; }
    
private:
    /**
     * @brief Ensure directory exists
     * @param path Directory path
     * @return Returns true on success
     */
    bool ensureDirectoryExists(const std::string& path) const;
    
    /**
     * @brief Load manifest from JSON
     * @param json_data JSON data
     * @param manifest Output manifest
     * @return Returns true on success
     */
    bool loadManifestFromJson(const nlohmann::json& json_data, ModelManifest& manifest) const;
    
    /**
     * @brief Convert manifest to JSON
     * @param manifest Manifest data
     * @param json_data Output JSON
     * @return Returns true on success
     */
    bool saveManifestToJson(const ModelManifest& manifest, nlohmann::json& json_data) const;
    
private:
    std::string base_path_;         ///< Base storage path
    mutable std::mutex mutex_;      ///< Thread-safe mutex
    bool initialized_;              ///< Whether initialized
};

} // namespace core
} // namespace duorou