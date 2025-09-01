#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>

namespace duorou {
namespace core {

/**
 * @brief 模型层信息结构
 * 对应ollama中的Layer结构
 */
struct ModelLayer {
    std::string digest;         ///< SHA256摘要
    std::string media_type;     ///< 媒体类型
    size_t size;               ///< 层大小
    
    ModelLayer() : size(0) {}
    ModelLayer(const std::string& d, const std::string& mt, size_t s) 
        : digest(d), media_type(mt), size(s) {}
};

/**
 * @brief 模型清单结构
 * 对应ollama中的Manifest结构
 */
struct ModelManifest {
    int schema_version;                 ///< 清单版本
    std::string media_type;             ///< 媒体类型
    ModelLayer config;                  ///< 配置层
    std::vector<ModelLayer> layers;     ///< 模型层列表
    
    ModelManifest() : schema_version(2) {}
    
    /**
     * @brief 计算清单总大小
     * @return 总大小（字节）
     */
    size_t getTotalSize() const {
        size_t total = config.size;
        for (const auto& layer : layers) {
            total += layer.size;
        }
        return total;
    }
    
    /**
     * @brief 获取所有层的摘要列表
     * @return 摘要列表
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
 * @brief 模型路径结构
 * 对应ollama中的ModelPath结构
 */
struct ModelPath {
    std::string scheme;         ///< 协议（如registry）
    std::string registry;       ///< 注册表地址
    std::string namespace_;     ///< 命名空间
    std::string repository;     ///< 仓库名
    std::string tag;           ///< 标签
    
    ModelPath() = default;
    
    /**
     * @brief 从字符串解析模型路径
     * @param path 路径字符串，格式：[scheme://]registry/namespace/repository:tag
     * @return 解析成功返回true
     */
    bool parseFromString(const std::string& path);
    
    /**
     * @brief 转换为字符串
     * @return 路径字符串
     */
    std::string toString() const;
    
    /**
     * @brief 获取基础URL
     * @return 基础URL
     */
    std::string getBaseURL() const;
};

/**
 * @brief 模型路径管理器
 * 负责管理ollama兼容的模型存储结构
 */
class ModelPathManager {
public:
    /**
     * @brief 构造函数
     * @param base_path 基础存储路径
     */
    explicit ModelPathManager(const std::string& base_path = "");
    
    /**
     * @brief 析构函数
     */
    ~ModelPathManager() = default;
    
    /**
     * @brief 初始化管理器
     * @return 成功返回true
     */
    bool initialize();
    
    /**
     * @brief 获取清单根目录路径
     * @return 清单目录路径
     */
    std::string getManifestPath() const;
    
    /**
     * @brief 获取特定模型的清单文件路径
     * @param model_path 模型路径
     * @return 清单文件路径
     */
    std::string getManifestFilePath(const ModelPath& model_path) const;
    
    /**
     * @brief 获取blobs根目录路径
     * @return blobs目录路径
     */
    std::string getBlobsPath() const;
    
    /**
     * @brief 获取特定摘要的blob文件路径
     * @param digest SHA256摘要
     * @return blob文件路径，摘要无效时返回空字符串
     */
    std::string getBlobFilePath(const std::string& digest) const;
    
    /**
     * @brief 验证SHA256摘要格式
     * @param digest 摘要字符串
     * @return 格式正确返回true
     */
    static bool isValidDigest(const std::string& digest);
    
    /**
     * @brief 读取模型清单
     * @param model_path 模型路径
     * @param manifest 输出清单
     * @return 成功返回true
     */
    bool readManifest(const ModelPath& model_path, ModelManifest& manifest) const;
    
    /**
     * @brief 写入模型清单
     * @param model_path 模型路径
     * @param manifest 清单数据
     * @return 成功返回true
     */
    bool writeManifest(const ModelPath& model_path, const ModelManifest& manifest);
    
    /**
     * @brief 枚举所有本地清单
     * @param continue_on_error 遇到错误时是否继续
     * @return 清单映射表
     */
    std::unordered_map<std::string, ModelManifest> enumerateManifests(bool continue_on_error = true) const;
    
    /**
     * @brief 检查blob是否存在
     * @param digest SHA256摘要
     * @return 存在返回true
     */
    bool blobExists(const std::string& digest) const;
    
    /**
     * @brief 获取blob文件大小
     * @param digest SHA256摘要
     * @return 文件大小，文件不存在返回0
     */
    size_t getBlobSize(const std::string& digest) const;
    
    /**
     * @brief 删除未使用的层文件
     * @param used_digests 仍在使用的摘要列表
     * @return 删除的文件数量
     */
    size_t deleteUnusedLayers(const std::vector<std::string>& used_digests);
    
    /**
     * @brief 清理无效的blob文件
     * @return 清理的文件数量
     */
    size_t pruneLayers();
    
    /**
     * @brief 验证blob文件的SHA256
     * @param digest 预期摘要
     * @return 验证通过返回true
     */
    bool verifyBlob(const std::string& digest) const;
    
    /**
     * @brief 计算文件的SHA256摘要
     * @param file_path 文件路径
     * @return SHA256摘要字符串
     */
    static std::string calculateSHA256(const std::string& file_path);
    
    /**
     * @brief 设置基础存储路径
     * @param path 新的基础路径
     */
    void setBasePath(const std::string& path);
    
    /**
     * @brief 获取基础存储路径
     * @return 基础路径
     */
    const std::string& getBasePath() const { return base_path_; }
    
private:
    /**
     * @brief 确保目录存在
     * @param path 目录路径
     * @return 成功返回true
     */
    bool ensureDirectoryExists(const std::string& path) const;
    
    /**
     * @brief 从JSON加载清单
     * @param json_data JSON数据
     * @param manifest 输出清单
     * @return 成功返回true
     */
    bool loadManifestFromJson(const nlohmann::json& json_data, ModelManifest& manifest) const;
    
    /**
     * @brief 将清单转换为JSON
     * @param manifest 清单数据
     * @param json_data 输出JSON
     * @return 成功返回true
     */
    bool saveManifestToJson(const ModelManifest& manifest, nlohmann::json& json_data) const;
    
private:
    std::string base_path_;         ///< 基础存储路径
    mutable std::mutex mutex_;      ///< 线程安全互斥锁
    bool initialized_;              ///< 是否已初始化
};

} // namespace core
} // namespace duorou