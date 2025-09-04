#pragma once

#include <string>
#include <memory>
#include <cstdint>
#include <iostream>
#include <vector>
#include <map>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include "model_path_manager.h"
#include "modelfile_parser.h"
#include "logger.h"
// #include "llama.h"  // 注释：暂时禁用llama相关功能

// Forward declarations
// struct llama_model;  // 注释：暂时禁用llama相关功能
// struct llama_model_params;  // 注释：暂时禁用llama相关功能

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
     * @brief 从ollama模型名称加载llama模型 (暂时禁用)
     * @param model_name ollama模型名称 (例如: "llama3.2", "qwen2.5:7b")
     * @return 加载成功返回true，失败返回false
     */
    // llama_model* loadFromOllamaModel(const std::string& model_name, 
    //                                 const llama_model_params& model_params);
    bool loadFromOllamaModel(const std::string& model_name);
    
    /**
     * @brief 从ollama模型路径加载llama模型 (暂时禁用)
     * @param model_path 解析后的模型路径
     * @return 加载成功返回true，失败返回false
     */
    // llama_model* loadFromModelPath(const ModelPath& model_path,
    //                               const llama_model_params& model_params);
    bool loadFromModelPath(const ModelPath& model_path);
    
    /**
     * @brief 从ollama模型加载llama模型，支持LoRA适配器 (暂时禁用)
     * @param model_name ollama模型名称
     * @param enable_lora 是否启用LoRA解析
     * @return 加载成功返回true，失败返回false
     */
    // llama_model* loadFromOllamaModelWithLoRA(const std::string& model_name,
    //                                         const llama_model_params& model_params,
    //                                         bool enable_lora = false);
    bool loadFromOllamaModelWithLoRA(const std::string& model_name,
                                    bool enable_lora = false);
    
    /**
     * @brief 从Modelfile配置加载模型 (暂时禁用)
     * @param config Modelfile配置
     * @return 加载成功返回true，失败返回false
     */
    // llama_model* loadFromModelfileConfig(const ModelfileConfig& config,
    //                                     const llama_model_params& model_params);
    bool loadFromModelfileConfig(const ModelfileConfig& config);
    
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