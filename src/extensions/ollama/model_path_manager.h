#ifndef DUOROU_EXTENSIONS_OLLAMA_MODEL_PATH_MANAGER_H
#define DUOROU_EXTENSIONS_OLLAMA_MODEL_PATH_MANAGER_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace duorou {
namespace extensions {
namespace ollama {

// 模型路径结构
struct ModelPath {
    std::string registry;    // 注册表
    std::string namespace_;  // 命名空间
    std::string model;       // 模型名称
    std::string tag;         // 标签
    
    bool parseFromString(const std::string& path_str);
    std::string toString() const;
};

// 模型Manifest结构
struct ModelManifest {
    std::string schema_version;
    std::string media_type;
    std::string architecture;
    std::vector<std::string> layers;
    std::unordered_map<std::string, std::string> config;
    
    bool parseFromJSON(const std::string& json_str);
    std::string getConfigBlob() const;
    std::string getModelBlob() const;
};

// Ollama模型路径管理器
class ModelPathManager {
public:
    explicit ModelPathManager(const std::string& ollama_models_dir = "");
    ~ModelPathManager();

    bool readManifest(const ModelPath& model_path, ModelManifest& manifest);
    std::string getModelDirectory(const ModelPath& model_path);
    std::string getBlobPath(const std::string& blob_sha256);
    std::vector<std::string> listAvailableModels();
    bool modelExists(const ModelPath& model_path);
    std::string getModelsDirectory() const { return models_dir_; }
    void setVerbose(bool verbose) { verbose_ = verbose; }

private:
    std::string getDefaultModelsDirectory();
    std::string getManifestPath(const ModelPath& model_path);
    std::string readFileContent(const std::string& file_path);
    bool directoryExists(const std::string& dir_path);
    bool fileExists(const std::string& file_path);
    void log(const std::string& level, const std::string& message);

private:
    std::string models_dir_;
    bool verbose_;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_MODEL_PATH_MANAGER_H