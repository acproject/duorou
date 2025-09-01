#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <future>
#include <unordered_map>
#include "model_path_manager.h"

namespace duorou {

/**
 * @brief 下载进度回调函数
 * @param downloaded 已下载字节数
 * @param total 总字节数
 * @param speed 下载速度（字节/秒）
 */
using DownloadProgressCallback = std::function<void(size_t downloaded, size_t total, double speed)>;

/**
 * @brief 下载结果
 */
struct DownloadResult {
    bool success = false;
    std::string error_message;
    std::string local_path;
    size_t downloaded_bytes = 0;
    double download_time = 0.0;
};

/**
 * @brief 模型信息
 */
struct ModelInfo {
    std::string name;
    std::string tag;
    std::string digest;
    size_t size = 0;
    std::string description;
    std::vector<std::string> families;
    std::string format;
    std::string parameter_size;
    std::string quantization_level;
    std::unordered_map<std::string, std::string> metadata;
};

/**
 * @brief Ollama兼容的模型下载器
 */
class ModelDownloader {
public:
    /**
     * @brief 构造函数
     * @param base_url Ollama服务器基础URL
     * @param model_dir 模型存储目录
     */
    ModelDownloader(const std::string& base_url = "https://registry.ollama.ai",
                   const std::string& model_dir = "~/.ollama/models");
    
    /**
     * @brief 析构函数
     */
    ~ModelDownloader();
    
    /**
     * @brief 设置下载进度回调
     * @param callback 进度回调函数
     */
    void setProgressCallback(DownloadProgressCallback callback);
    
    /**
     * @brief 下载模型
     * @param model_name 模型名称（如 "llama2:7b"）
     * @return 下载结果的Future
     */
    std::future<DownloadResult> downloadModel(const std::string& model_name);
    
    /**
     * @brief 同步下载模型
     * @param model_name 模型名称
     * @return 下载结果
     */
    DownloadResult downloadModelSync(const std::string& model_name);
    
    /**
     * @brief 获取模型信息
     * @param model_name 模型名称
     * @return 模型信息
     */
    ModelInfo getModelInfo(const std::string& model_name);
    
    /**
     * @brief 检查模型是否已下载
     * @param model_name 模型名称
     * @return 是否已下载
     */
    bool isModelDownloaded(const std::string& model_name);
    
    /**
     * @brief 获取本地模型列表
     * @return 本地模型列表
     */
    std::vector<std::string> getLocalModels();
    
    /**
     * @brief 删除本地模型
     * @param model_name 模型名称
     * @return 是否成功删除
     */
    bool deleteModel(const std::string& model_name);
    
    /**
     * @brief 获取模型本地路径
     * @param model_name 模型名称
     * @return 本地路径
     */
    std::string getModelPath(const std::string& model_name);
    
    /**
     * @brief 验证模型完整性
     * @param model_name 模型名称
     * @return 是否完整
     */
    bool verifyModel(const std::string& model_name);
    
    /**
     * @brief 清理未使用的blob
     * @return 清理的字节数
     */
    size_t cleanupUnusedBlobs();
    
    /**
     * @brief 获取缓存使用情况
     * @return 缓存大小（字节）
     */
    size_t getCacheSize();
    
    /**
     * @brief 设置最大缓存大小
     * @param max_size 最大缓存大小（字节）
     */
    void setMaxCacheSize(size_t max_size);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

/**
 * @brief 模型下载器工厂
 */
class ModelDownloaderFactory {
public:
    /**
     * @brief 创建模型下载器
     * @param base_url 基础URL
     * @param model_dir 模型目录
     * @return 模型下载器实例
     */
    static std::unique_ptr<ModelDownloader> create(const std::string& base_url = "https://registry.ollama.ai",
                                                   const std::string& model_dir = "~/.ollama/models");
};

} // namespace duorou