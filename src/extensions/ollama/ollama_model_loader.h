#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstring>

#include "model_path_manager.h"
#include "modelfile_parser.h"
#include "compatibility_checker.h"

namespace duorou {
namespace extensions {
namespace ollama {

// 前向声明
class ModelPathManager;
class CompatibilityChecker;
struct ModelPath;

// Ollama模型配置
struct OllamaModelConfig {
    std::string model_path;           // 模型文件路径
    std::string architecture;         // 模型架构
    std::string base_model;          // 基础模型
    std::string system_prompt;       // 系统提示
    std::string template_content;    // 模板内容
    std::vector<std::string> adapters; // 适配器列表
    std::unordered_map<std::string, std::string> parameters; // 参数映射
    bool verbose = false;
    
    // 从Modelfile加载配置
    bool loadFromModelfile(const std::string& modelfile_path);
    
    // 从Manifest加载配置
    bool loadFromManifest(const ModelPath& model_path, ModelPathManager& path_manager);
    
    // 参数访问
    std::string getParameter(const std::string& name, const std::string& default_value = "") const;
    void setParameter(const std::string& name, const std::string& value);
};

/**
 * @brief Ollama模型加载器
 * 
 * 负责加载ollama格式的模型，支持:
 * - 从ollama模型路径加载模型
 * - 架构兼容性检查和映射
 * - GGUF文件处理
 * - 模型参数配置
 */
class OllamaModelLoader {
public:
    /**
     * @brief 构造函数
     * @param model_path_manager 模型路径管理器
     */
    explicit OllamaModelLoader(std::shared_ptr<ModelPathManager> model_path_manager);
    
    /**
     * @brief 析构函数
     */
    ~OllamaModelLoader();

    /**
     * @brief 从ollama模型路径加载模型
     * @param model_path ollama模型路径 (例如: "llama3.2:latest")
     * @param model_params 模型参数 (void*避免llama.cpp依赖)
     * @return 加载的模型句柄，失败返回nullptr
     */
    void* loadFromModelPath(const std::string& model_path, 
                           const void* model_params);

    /**
     * @brief 从GGUF文件路径直接加载模型
     * @param gguf_path GGUF文件路径
     * @param model_params 模型参数 (void*避免llama.cpp依赖)
     * @return 加载的模型句柄，失败返回nullptr
     */
    void* loadFromGGUFPath(const std::string& gguf_path,
                          const void* model_params);

    /**
     * @brief 检查模型是否需要架构映射
     * @param gguf_path GGUF文件路径
     * @param original_arch 输出原始架构名称
     * @param mapped_arch 输出映射后的架构名称
     * @return 是否需要映射
     */
    bool checkArchitectureMapping(const std::string& gguf_path,
                                  std::string& original_arch,
                                  std::string& mapped_arch);

    /**
     * @brief 获取支持的架构列表
     * @return 支持的架构名称列表
     */
    std::vector<std::string> getSupportedArchitectures() const;

    /**
     * @brief 设置详细日志输出
     * @param verbose 是否启用详细日志
     */
    void setVerbose(bool verbose) { verbose_ = verbose; }

private:
    /**
     * @brief 从manifest获取GGUF文件路径
     * @param manifest 模型manifest
     * @return GGUF文件路径
     */
    std::string getGGUFPathFromManifest(const ModelManifest& manifest);

    /**
     * @brief 创建架构覆盖参数
     * @param mapped_arch 映射后的架构名称
     * @param model_path 模型路径
     * @return kv_override向量
     */
    std::vector<void*> createArchOverrides(
        const std::string& mapped_arch,
        const std::string& model_path);

    /**
     * @brief 估算GPU层数
     * @param gguf_path GGUF文件路径
     * @param available_vram 可用显存(MB)
     * @return 建议的GPU层数
     */
    int estimateGPULayers(const std::string& gguf_path, size_t available_vram);

    /**
     * @brief 日志输出
     * @param level 日志级别
     * @param message 日志消息
     */
    void log(const std::string& level, const std::string& message);

private:
    std::shared_ptr<ModelPathManager> model_path_manager_;
    std::unique_ptr<ModelfileParser> modelfile_parser_;
    std::unique_ptr<CompatibilityChecker> compatibility_checker_;
    bool verbose_;
    
    // 支持的架构映射表
    static const std::unordered_map<std::string, std::string> ARCHITECTURE_MAPPING;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou