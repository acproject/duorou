#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "model_path_manager.h"
#include "../../third_party/llama.cpp/vendor/nlohmann/json.hpp"

namespace duorou {
namespace core {

/**
 * @brief LoRA适配器信息
 */
struct LoRAAdapter {
    std::string name;           ///< 适配器名称
    std::string path;           ///< 适配器文件路径
    float scale = 1.0f;         ///< 适配器缩放因子
    std::string digest;         ///< SHA256摘要
    size_t size = 0;           ///< 文件大小
    
    LoRAAdapter() = default;
    LoRAAdapter(const std::string& n, const std::string& p, float s = 1.0f)
        : name(n), path(p), scale(s) {}
};

/**
 * @brief Modelfile配置信息
 */
struct ModelfileConfig {
    std::string base_model;                        ///< 基础模型路径
    std::vector<LoRAAdapter> lora_adapters;        ///< LoRA适配器列表
    std::unordered_map<std::string, std::string> parameters;  ///< 模型参数
    std::string system_prompt;                     ///< 系统提示
    std::string template_format;                   ///< 模板格式
    
    ModelfileConfig() = default;
};

/**
 * @brief Ollama Modelfile解析器
 * 负责解析Ollama模型的manifest和相关配置，提取LoRA适配器信息
 */
class ModelfileParser {
public:
    /**
     * @brief 构造函数
     * @param model_path_manager 模型路径管理器
     */
    explicit ModelfileParser(std::shared_ptr<ModelPathManager> model_path_manager);
    
    /**
     * @brief 析构函数
     */
    ~ModelfileParser() = default;
    
    /**
     * @brief 从manifest解析Modelfile配置
     * @param manifest 模型manifest
     * @param config 输出的配置信息
     * @return 解析成功返回true
     */
    bool parseFromManifest(const ModelManifest& manifest, ModelfileConfig& config);
    
    /**
     * @brief 从JSON字符串解析Modelfile配置
     * @param json_str JSON字符串
     * @param config 输出的配置信息
     * @return 解析成功返回true
     */
    bool parseFromJson(const std::string& json_str, ModelfileConfig& config);
    
    /**
     * @brief 从文件解析Modelfile配置
     * @param file_path 文件路径
     * @param config 输出的配置信息
     * @return 解析成功返回true
     */
    bool parseFromFile(const std::string& file_path, ModelfileConfig& config);
    
    /**
     * @brief 验证LoRA适配器文件
     * @param adapter LoRA适配器信息
     * @return 验证成功返回true
     */
    bool validateLoRAAdapter(const LoRAAdapter& adapter);
    
    /**
     * @brief 获取支持的媒体类型列表
     * @return 支持的媒体类型
     */
    static std::vector<std::string> getSupportedMediaTypes();
    
private:
    std::shared_ptr<ModelPathManager> model_path_manager_;
    
    /**
     * @brief 解析模板层
     * @param layer_digest 层摘要
     * @param config 配置信息
     * @return 解析成功返回true
     */
    bool parseTemplateLayer(const std::string& layer_digest, ModelfileConfig& config);
    
    /**
     * @brief 解析系统层
     * @param layer_digest 层摘要
     * @param config 配置信息
     * @return 解析成功返回true
     */
    bool parseSystemLayer(const std::string& layer_digest, ModelfileConfig& config);
    
    /**
     * @brief 解析参数层
     * @param layer_digest 层摘要
     * @param config 配置信息
     * @return 解析成功返回true
     */
    bool parseParametersLayer(const std::string& layer_digest, ModelfileConfig& config);
    
    /**
     * @brief 解析适配器层
     * @param layer_digest 层摘要
     * @param config 配置信息
     * @return 解析成功返回true
     */
    bool parseAdapterLayer(const std::string& layer_digest, ModelfileConfig& config);
    
    /**
     * @brief 从blob文件读取内容
     * @param digest 文件摘要
     * @return 文件内容，失败返回空字符串
     */
    std::string readBlobContent(const std::string& digest);
    
    /**
     * @brief 解析Modelfile指令
     * @param content Modelfile内容
     * @param config 配置信息
     * @return 解析成功返回true
     */
    bool parseModelfileInstructions(const std::string& content, ModelfileConfig& config);
    
    /**
     * @brief 解析FROM指令
     * @param line 指令行
     * @param config 配置信息
     * @return 解析成功返回true
     */
    bool parseFromInstruction(const std::string& line, ModelfileConfig& config);
    
    /**
     * @brief 解析ADAPTER指令
     * @param line 指令行
     * @param config 配置信息
     * @return 解析成功返回true
     */
    bool parseAdapterInstruction(const std::string& line, ModelfileConfig& config);
    
    /**
     * @brief 解析PARAMETER指令
     * @param line 指令行
     * @param config 配置信息
     * @return 解析成功返回true
     */
    bool parseParameterInstruction(const std::string& line, ModelfileConfig& config);
    
    /**
     * @brief 解析TEMPLATE指令
     * @param line 指令行
     * @param config 配置信息
     * @return 解析成功返回true
     */
    bool parseTemplateInstruction(const std::string& line, ModelfileConfig& config);
    
    /**
     * @brief 解析SYSTEM指令
     * @param line 指令行
     * @param config 配置对象
     * @return 解析成功返回true
     */
    bool parseSystemInstruction(const std::string& line, ModelfileConfig& config);
    
    /**
     * @brief 验证GGUF文件头
     * @param file_path 文件路径
     * @return 验证成功返回true
     */
    bool validateGGUFHeader(const std::string& file_path);
};

} // namespace core
} // namespace duorou