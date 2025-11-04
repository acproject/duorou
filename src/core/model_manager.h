#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <functional>
#include <future>
#include "text_generator.h"
#include "image_generator.h"
#include "model_downloader.h"
#include "ollama_model_loader.h"
#include "model_path_manager.h"

namespace duorou {
namespace core {

/**
 * @brief Model type enumeration
 */
enum class ModelType {
    LANGUAGE_MODEL,     ///< Language model (LLaMA)
    DIFFUSION_MODEL     ///< Diffusion model (Stable Diffusion)
};

/**
 * @brief Model status enumeration
 */
enum class ModelStatus {
    NOT_LOADED,         ///< Not loaded
    LOADING,            ///< Loading
    LOADED,             ///< Loaded
    LOAD_ERROR,              ///< Error status
};

/**
 * @brief Model information structure
 */
struct ModelManagerInfo {
    std::string id;                 ///< Model ID
    std::string name;               ///< Model name
    std::string path;               ///< Model file path
    ModelType type;                 ///< Model type
    ModelStatus status;             ///< Model status
    size_t memory_usage;            ///< Memory usage (bytes)
    std::string description;        ///< Model description
    
    ModelManagerInfo() : type(ModelType::LANGUAGE_MODEL), status(ModelStatus::NOT_LOADED), memory_usage(0) {}
};

/**
 * @brief Base model class
 */
class BaseModel {
public:
    virtual ~BaseModel() = default;
    
    /**
     * @brief Load model
     * @param model_path Model file path
     * @return Returns true on success, false on failure
     */
    virtual bool load(const std::string& model_path) = 0;
    
    /**
     * @brief Unload model
     */
    virtual void unload() = 0;
    
    /**
     * @brief Check if model is loaded
     * @return Returns true if loaded, false if not loaded
     */
    virtual bool isLoaded() const = 0;
    
    /**
     * @brief Get model information
     * @return Model information
     */
    virtual ModelManagerInfo getInfo() const = 0;
    
    /**
     * @brief Get memory usage
     * @return Memory usage (bytes)
     */
    virtual size_t getMemoryUsage() const = 0;
};

/**
 * @brief Model manager class
 * 
 * Responsible for model loading, unloading, management and resource scheduling
 */
class ModelManager {
public:
    /**
     * @brief Constructor
     */
    ModelManager();
    
    /**
     * @brief Destructor
     */
    ~ModelManager();
    
    /**
     * @brief Initialize model manager
     * @return Returns true on success, false on failure
     */
    bool initialize();
    
    /**
     * @brief Register model
     * @param model_info Model information
     * @return Returns true on success, false on failure
     */
    bool registerModel(const ModelManagerInfo& model_info);
    
    /**
     * @brief Load model
     * @param model_id Model ID
     * @return Returns true on success, false on failure
     */
    bool loadModel(const std::string& model_id);
    
    /**
     * @brief Unload model
     * @param model_id Model ID
     * @return Returns true on success, false on failure
     */
    bool unloadModel(const std::string& model_id);
    
    /**
     * @brief Unload all models
     */
    void unloadAllModels();
    
    /**
     * @brief Get model
     * @param model_id Model ID
     * @return Model pointer, returns nullptr if not found
     */
    std::shared_ptr<BaseModel> getModel(const std::string& model_id) const;
    
    /**
     * @brief Check if model is loaded
     * @param model_id Model ID
     * @return Returns true if loaded, false if not loaded
     */
    bool isModelLoaded(const std::string& model_id) const;
    
    /**
     * @brief Get model information
     * @param model_id Model ID
     * @return Model information, returns empty ModelManagerInfo if not found
     */
    ModelManagerInfo getModelInfo(const std::string& model_id) const;
    
    /**
     * @brief Get list of all registered models
     * @return List of model information
     */
    std::vector<ModelManagerInfo> getAllModels() const;
    
    /**
     * @brief Get list of loaded models
     * @return List of loaded model IDs
     */
    std::vector<std::string> getLoadedModels() const;
    
    /**
     * @brief Get total memory usage
     * @return Total memory usage (bytes)
     */
    size_t getTotalMemoryUsage() const;
    
    /**
     * @brief Set memory limit
     * @param limit_bytes Memory limit (bytes)
     */
    void setMemoryLimit(size_t limit_bytes);
    
    /**
     * @brief Get memory limit
     * @return Memory limit (bytes)
     */
    size_t getMemoryLimit() const;
    
    /**
     * @brief Check if there is enough memory to load model
     * @param model_id Model ID
     * @return Returns true if enough memory, false otherwise
     */
    bool hasEnoughMemory(const std::string& model_id) const;
    
    /**
     * @brief Set model load callback function
     * @param callback Callback function
     */
    void setLoadCallback(std::function<void(const std::string&, bool)> callback);
    
    /**
     * @brief Get text generator
     * @param model_id Model ID
     * @return Text generator pointer
     */
    duorou::core::TextGenerator* getTextGenerator(const std::string& model_id) const;
    
    /**
     * @brief Get image generator
     * @param model_id Model ID
     * @return Image generator pointer
     */
    ImageGenerator* getImageGenerator(const std::string& model_id) const;
    
    /**
     * @brief Optimize memory usage
     * @return Size of freed memory (bytes)
     */
    size_t optimizeMemory();
    
    /**
     * @brief Enable automatic memory management
     * @param enable Whether to enable
     */
    void enableAutoMemoryManagement(bool enable);
    
    /**
     * @brief Download model
     * @param model_name Model name (e.g., "llama2:7b")
     * @param progress_callback Progress callback function
     * @return Future of download result
     */
    std::future<duorou::DownloadResult> downloadModel(const std::string& model_name, 
                                             duorou::DownloadProgressCallback progress_callback = nullptr);
    
    /**
     * @brief Download model synchronously
     * @param model_name Model name
     * @param progress_callback Progress callback function
     * @return Download result
     */
    duorou::DownloadResult downloadModelSync(const std::string& model_name,
                                    duorou::DownloadProgressCallback progress_callback = nullptr);
    
    /**
     * @brief Get model information
     * @param model_name Model name
     * @return Model information
     */
    ModelManagerInfo getModelInfo(const std::string& model_name);
    
    /**
     * @brief Check if model is downloaded
     * @param model_name Model name
     * @return Whether downloaded
     */
    bool isModelDownloaded(const std::string& model_name);
    
    /**
     * @brief Get local model list
     * @return Local model list
     */
    std::vector<std::string> getLocalModels();
    
    /**
     * @brief Delete local model
     * @param model_name Model name
     * @return Whether successfully deleted
     */
    bool deleteLocalModel(const std::string& model_name);
    
    /**
     * @brief Verify model integrity
     * @param model_name Model name
     * @return Whether complete
     */
    bool verifyModel(const std::string& model_name);
    
    /**
     * @brief Clean up unused model cache
     * @return Number of bytes cleaned
     */
    size_t cleanupModelCache();
    
    /**
     * @brief Get model cache size
     * @return Cache size (bytes)
     */
    size_t getModelCacheSize();
    
    /**
     * @brief Set maximum model cache size
     * @param max_size Maximum cache size (bytes)
     */
    void setMaxModelCacheSize(size_t max_size);

    /**
     * @brief 设置 Ollama 模型目录
     * @param path 模型目录路径（可包含 ~）
     */
    void setOllamaModelsPath(const std::string& path);

    /**
     * @brief 重新扫描指定本地模型目录
     * @param directory 目录路径
     */
    void rescanModelDirectory(const std::string& directory);
    
private:
    /**
     * @brief Create model instance
     * @param model_info Model information
     * @return Model instance pointer
     */
    std::shared_ptr<BaseModel> createModel(const ModelManagerInfo& model_info);
    
    /**
     * @brief Update model status
     * @param model_id Model ID
     * @param status New status
     */
    void updateModelStatus(const std::string& model_id, ModelStatus status);
    
    /**
     * @brief Scan model directory
     * @param directory Directory path
     */
    void scanModelDirectory(const std::string& directory);
    
private:
    std::unordered_map<std::string, ModelManagerInfo> registered_models_;     ///< Registered models
    std::unordered_map<std::string, std::shared_ptr<BaseModel>> loaded_models_;  ///< Loaded models
    std::unordered_map<std::string, std::unique_ptr<duorou::core::TextGenerator>> text_generators_; ///< Text generator mapping
    mutable std::mutex mutex_;                                         ///< Thread-safe mutex
    size_t memory_limit_;                                              ///< Memory limit
    bool initialized_;                                                 ///< Whether initialized
    bool auto_memory_management_;                                      ///< Automatic memory management
    std::unique_ptr<ModelDownloader> model_downloader_;               ///< Model downloader
    std::function<void(const std::string&, bool)> load_callback_;     ///< Model load callback function
};

} // namespace core
} // namespace duorou