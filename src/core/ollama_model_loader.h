#pragma once

#include <string>
#include <memory>
#include "model_path_manager.h"
#include "modelfile_parser.h"
#include "logger.h"
#include "llama.h"

// Forward declarations
struct llama_model;
struct llama_model_params;

namespace duorou {
namespace core {

/**
 * @brief Ollama模型加载器
 * 负责从ollama已下载的模型中加载模型到llama.cpp
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
    ~OllamaModelLoader() = default;
    
    /**
     * @brief 从ollama模型名称加载llama模型
     * @param model_name ollama模型名称 (例如: "llama3.2", "qwen2.5:7b")
     * @param model_params llama模型参数
     * @return 加载成功返回llama_model指针，失败返回nullptr
     */
    llama_model* loadFromOllamaModel(const std::string& model_name, 
                                    const llama_model_params& model_params);
    
    /**
     * @brief 从ollama模型路径加载llama模型
     * @param model_path 解析后的模型路径
     * @param model_params llama模型参数
     * @return 加载成功返回llama_model指针，失败返回nullptr
     */
    llama_model* loadFromModelPath(const ModelPath& model_path,
                                  const llama_model_params& model_params);
    
    /**
     * @brief 从ollama模型加载llama模型，支持LoRA适配器
     * @param model_name ollama模型名称
     * @param model_params llama模型参数
     * @param enable_lora 是否启用LoRA解析
     * @return 加载成功返回llama_model指针，失败返回nullptr
     */
    llama_model* loadFromOllamaModelWithLoRA(const std::string& model_name,
                                            const llama_model_params& model_params,
                                            bool enable_lora = false);
    
    /**
     * @brief 从Modelfile配置加载模型
     * @param config Modelfile配置
     * @param model_params llama模型参数
     * @return 加载成功返回llama_model指针，失败返回nullptr
     */
    llama_model* loadFromModelfileConfig(const ModelfileConfig& config,
                                        const llama_model_params& model_params);
    
    /**
     * @brief 检查ollama模型是否存在
     * @param model_name ollama模型名称
     * @return 存在返回true
     */
    bool isOllamaModelAvailable(const std::string& model_name);
    
    /**
     * @brief 列出所有可用的ollama模型
     * @return 模型名称列表
     */
    std::vector<std::string> listAvailableModels();
    
private:
    /**
     * @brief 从manifest获取GGUF模型文件路径
     * @param manifest 模型manifest
     * @return GGUF文件路径，失败返回空字符串
     */
    std::string getGGUFPathFromManifest(const ModelManifest& manifest);
    
    /**
     * @brief 解析ollama模型名称为ModelPath
     * @param model_name ollama模型名称
     * @param model_path 输出的模型路径
     * @return 解析成功返回true
     */
    bool parseOllamaModelName(const std::string& model_name, ModelPath& model_path);
    
    /**
     * @brief 标准化ollama模型名称
     * @param model_name 原始模型名称
     * @return 标准化后的模型名称
     */
    std::string normalizeOllamaModelName(const std::string& model_name);
    
private:
    std::shared_ptr<ModelPathManager> model_path_manager_;
    std::shared_ptr<ModelfileParser> modelfile_parser_;
    Logger logger_;
    
    /**
     * @brief 解析manifest中的Modelfile配置
     * @param manifest 模型manifest
     * @param config 输出的配置信息
     * @return 解析成功返回true
     */
    bool parseModelfileFromManifest(const ModelManifest& manifest, ModelfileConfig& config);
};

} // namespace core
} // namespace duorou