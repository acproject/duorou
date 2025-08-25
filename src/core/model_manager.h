#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <functional>

namespace duorou {
namespace core {

/**
 * @brief 模型类型枚举
 */
enum class ModelType {
    LANGUAGE_MODEL,     ///< 语言模型 (LLaMA)
    DIFFUSION_MODEL     ///< 扩散模型 (Stable Diffusion)
};

/**
 * @brief 模型状态枚举
 */
enum class ModelStatus {
    NOT_LOADED,         ///< 未加载
    LOADING,            ///< 加载中
    LOADED,             ///< 已加载
    ERROR               ///< 错误状态
};

/**
 * @brief 模型信息结构
 */
struct ModelInfo {
    std::string id;                 ///< 模型ID
    std::string name;               ///< 模型名称
    std::string path;               ///< 模型文件路径
    ModelType type;                 ///< 模型类型
    ModelStatus status;             ///< 模型状态
    size_t memory_usage;            ///< 内存使用量（字节）
    std::string description;        ///< 模型描述
    
    ModelInfo() : type(ModelType::LANGUAGE_MODEL), status(ModelStatus::NOT_LOADED), memory_usage(0) {}
};

/**
 * @brief 模型基类
 */
class BaseModel {
public:
    virtual ~BaseModel() = default;
    
    /**
     * @brief 加载模型
     * @param model_path 模型文件路径
     * @return 成功返回true，失败返回false
     */
    virtual bool load(const std::string& model_path) = 0;
    
    /**
     * @brief 卸载模型
     */
    virtual void unload() = 0;
    
    /**
     * @brief 检查模型是否已加载
     * @return 已加载返回true，未加载返回false
     */
    virtual bool isLoaded() const = 0;
    
    /**
     * @brief 获取模型信息
     * @return 模型信息
     */
    virtual ModelInfo getInfo() const = 0;
    
    /**
     * @brief 获取内存使用量
     * @return 内存使用量（字节）
     */
    virtual size_t getMemoryUsage() const = 0;
};

/**
 * @brief 模型管理器类
 * 
 * 负责模型的加载、卸载、管理和资源调度
 */
class ModelManager {
public:
    /**
     * @brief 构造函数
     */
    ModelManager();
    
    /**
     * @brief 析构函数
     */
    ~ModelManager();
    
    /**
     * @brief 初始化模型管理器
     * @return 成功返回true，失败返回false
     */
    bool initialize();
    
    /**
     * @brief 注册模型
     * @param model_info 模型信息
     * @return 成功返回true，失败返回false
     */
    bool registerModel(const ModelInfo& model_info);
    
    /**
     * @brief 加载模型
     * @param model_id 模型ID
     * @return 成功返回true，失败返回false
     */
    bool loadModel(const std::string& model_id);
    
    /**
     * @brief 卸载模型
     * @param model_id 模型ID
     * @return 成功返回true，失败返回false
     */
    bool unloadModel(const std::string& model_id);
    
    /**
     * @brief 卸载所有模型
     */
    void unloadAllModels();
    
    /**
     * @brief 获取模型
     * @param model_id 模型ID
     * @return 模型指针，如果不存在返回nullptr
     */
    std::shared_ptr<BaseModel> getModel(const std::string& model_id) const;
    
    /**
     * @brief 检查模型是否已加载
     * @param model_id 模型ID
     * @return 已加载返回true，未加载返回false
     */
    bool isModelLoaded(const std::string& model_id) const;
    
    /**
     * @brief 获取模型信息
     * @param model_id 模型ID
     * @return 模型信息，如果不存在返回空的ModelInfo
     */
    ModelInfo getModelInfo(const std::string& model_id) const;
    
    /**
     * @brief 获取所有已注册的模型列表
     * @return 模型信息列表
     */
    std::vector<ModelInfo> getAllModels() const;
    
    /**
     * @brief 获取已加载的模型列表
     * @return 已加载模型的ID列表
     */
    std::vector<std::string> getLoadedModels() const;
    
    /**
     * @brief 获取总内存使用量
     * @return 总内存使用量（字节）
     */
    size_t getTotalMemoryUsage() const;
    
    /**
     * @brief 设置内存限制
     * @param limit_bytes 内存限制（字节）
     */
    void setMemoryLimit(size_t limit_bytes);
    
    /**
     * @brief 获取内存限制
     * @return 内存限制（字节）
     */
    size_t getMemoryLimit() const;
    
    /**
     * @brief 检查是否有足够内存加载模型
     * @param model_id 模型ID
     * @return 有足够内存返回true，否则返回false
     */
    bool hasEnoughMemory(const std::string& model_id) const;
    
    /**
     * @brief 设置模型加载回调函数
     * @param callback 回调函数
     */
    void setLoadCallback(std::function<void(const std::string&, bool)> callback);
    
private:
    /**
     * @brief 创建模型实例
     * @param model_info 模型信息
     * @return 模型实例指针
     */
    std::shared_ptr<BaseModel> createModel(const ModelInfo& model_info);
    
    /**
     * @brief 更新模型状态
     * @param model_id 模型ID
     * @param status 新状态
     */
    void updateModelStatus(const std::string& model_id, ModelStatus status);
    
    /**
     * @brief 扫描模型目录
     * @param directory 目录路径
     */
    void scanModelDirectory(const std::string& directory);
    
private:
    std::unordered_map<std::string, ModelInfo> registered_models_;     ///< 已注册的模型
    std::unordered_map<std::string, std::shared_ptr<BaseModel>> loaded_models_;  ///< 已加载的模型
    mutable std::mutex mutex_;                                         ///< 线程安全互斥锁
    size_t memory_limit_;                                              ///< 内存限制
    bool initialized_;                                                 ///< 是否已初始化
    std::function<void(const std::string&, bool)> load_callback_;     ///< 模型加载回调函数
};

} // namespace core
} // namespace duorou