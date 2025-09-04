#ifndef DUOROU_EXTENSIONS_OLLAMA_OLLAMA_PATH_RESOLVER_H
#define DUOROU_EXTENSIONS_OLLAMA_OLLAMA_PATH_RESOLVER_H

#include <string>
#include <vector>
#include <optional>
#include "../../third_party/llama.cpp/vendor/nlohmann/json.hpp"

namespace duorou {
namespace extensions {
namespace ollama {

// Ollama 模型信息结构
struct OllamaModelInfo {
    std::string name;           // 模型名称，如 "qwen2.5vl:7b"
    std::string registry;       // 注册表，如 "registry.ollama.ai"
    std::string namespace_name; // 命名空间，如 "library"
    std::string tag;            // 标签，如 "7b" 或 "latest"
    std::string digest;         // SHA256 摘要
    std::string manifest_path;  // manifest 文件路径
    std::string gguf_path;      // GGUF 文件路径
};

// Ollama 路径解析器
class OllamaPathResolver {
public:
    explicit OllamaPathResolver(bool verbose = false);
    
    // 解析 Ollama 模型名称到实际文件路径
    std::optional<std::string> resolveModelPath(const std::string& model_name);
    
    // 获取 Ollama 模型存储根目录
    std::string getOllamaModelsDir() const;
    
    // 设置自定义 Ollama 模型目录
    void setCustomModelsDir(const std::string& custom_dir);
    
    // 解析模型名称为结构化信息
    std::optional<OllamaModelInfo> parseModelName(const std::string& model_name);
    
    // 读取 manifest 文件
    std::optional<nlohmann::json> readManifest(const std::string& manifest_path);
    
    // 从 manifest 获取 GGUF 文件路径
    std::optional<std::string> getGGUFPathFromManifest(const nlohmann::json& manifest);
    
    // 检查模型是否存在
    bool modelExists(const std::string& model_name);
    
    // 列出所有可用模型
    std::vector<std::string> listAvailableModels();
    
private:
    bool verbose_;
    std::string custom_models_dir_;
    
    // 日志输出
    void log(const std::string& level, const std::string& message) const;
    
    // 获取默认 Ollama 模型目录
    std::string getDefaultOllamaModelsDir() const;
    
    // 标准化模型名称（添加默认标签等）
    std::string normalizeModelName(const std::string& model_name) const;
    
    // 构建 manifest 文件路径
    std::string buildManifestPath(const OllamaModelInfo& model_info) const;
    
    // 构建 blob 文件路径
    std::string buildBlobPath(const std::string& digest) const;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_OLLAMA_PATH_RESOLVER_H