#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>

namespace duorou {
namespace invokeai {

// 模型类型枚举
enum class ModelType {
    CHECKPOINT,     // .ckpt files
    SAFETENSORS,    // .safetensors files
    DIFFUSERS,      // diffusers format
    LORA,           // LoRA adapters
    CONTROLNET,     // ControlNet models
    VAE,            // VAE models
    TEXTUAL_INVERSION, // Textual inversion embeddings
    UNKNOWN
};

// 模型配置结构
struct ModelConfig {
    std::string name;
    std::string path;
    ModelType type;
    std::string description;
    std::string version;
    std::string author;
    std::string license;
    std::map<std::string, std::string> metadata;
    
    // 模型参数
    int width = 512;
    int height = 512;
    std::string base_model; // 基础模型名称
    std::vector<std::string> tags;
    
    // 性能相关
    size_t estimated_memory_mb = 0;
    bool supports_fp16 = true;
    bool supports_cpu = true;
    bool supports_gpu = true;
    
    // 文件信息
    size_t file_size_bytes = 0;
    std::string checksum;
    std::string created_date;
    std::string modified_date;
};

// 模型加载状态
enum class ModelLoadState {
    UNLOADED,
    LOADING,
    LOADED,
    ERROR
};

// 模型运行时信息
struct ModelRuntimeInfo {
    ModelLoadState state = ModelLoadState::UNLOADED;
    std::string error_message;
    size_t actual_memory_usage_mb = 0;
    std::string device; // "cpu", "cuda:0", "mps", etc.
    std::string precision; // "fp32", "fp16", "int8"
    double load_time_seconds = 0.0;
    std::chrono::system_clock::time_point load_timestamp;
};

// 模型加载进度回调
using ModelLoadProgressCallback = std::function<void(const std::string& stage, int progress, int total)>;

// 模型管理器类
class ModelManager {
public:
    ModelManager();
    ~ModelManager();
    
    // 初始化模型管理器
    bool initialize(const std::string& models_root_path);
    void shutdown();
    
    // 模型发现和扫描
    void scan_models();
    void add_model_path(const std::string& path);
    void remove_model_path(const std::string& path);
    std::vector<std::string> get_model_paths() const;
    
    // 模型查询
    std::vector<ModelConfig> get_all_models() const;
    std::vector<ModelConfig> get_models_by_type(ModelType type) const;
    std::vector<ModelConfig> search_models(const std::string& query) const;
    ModelConfig get_model_config(const std::string& model_name) const;
    bool has_model(const std::string& model_name) const;
    
    // 模型加载和卸载
    bool load_model(const std::string& model_name, 
                   const std::string& device = "auto",
                   const std::string& precision = "auto",
                   ModelLoadProgressCallback progress_cb = nullptr);
    bool unload_model(const std::string& model_name = ""); // 空字符串表示卸载当前模型
    bool switch_model(const std::string& model_name);
    
    // 模型状态查询
    std::vector<std::string> get_loaded_models() const;
    std::string get_current_model() const;
    ModelRuntimeInfo get_model_runtime_info(const std::string& model_name) const;
    bool is_model_loaded(const std::string& model_name) const;
    
    // 模型验证和修复
    bool validate_model(const std::string& model_name) const;
    bool repair_model(const std::string& model_name);
    std::vector<std::string> get_missing_dependencies(const std::string& model_name) const;
    
    // 模型安装和下载
    bool install_model_from_url(const std::string& url, const std::string& model_name);
    bool install_model_from_file(const std::string& file_path, const std::string& model_name);
    bool uninstall_model(const std::string& model_name);
    
    // 模型配置管理
    bool save_model_config(const ModelConfig& config);
    bool load_model_config(const std::string& model_name, ModelConfig& config);
    bool update_model_metadata(const std::string& model_name, 
                              const std::map<std::string, std::string>& metadata);
    
    // 缓存和优化
    void clear_model_cache();
    void optimize_memory_usage();
    size_t get_total_memory_usage() const;
    size_t get_available_memory() const;
    
    // 事件回调
    using ModelEventCallback = std::function<void(const std::string& model_name, const std::string& event)>;
    void set_model_event_callback(ModelEventCallback callback);
    
    // 配置设置
    void set_max_loaded_models(int max_models);
    void set_memory_limit(size_t limit_mb);
    void set_auto_unload_enabled(bool enabled);
    void set_cache_enabled(bool enabled);
    
    // 统计信息
    std::map<std::string, std::string> get_statistics() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// 工具函数
ModelType detect_model_type(const std::string& file_path);
std::string model_type_to_string(ModelType type);
ModelType string_to_model_type(const std::string& type_str);
bool is_valid_model_file(const std::string& file_path);
std::string calculate_file_checksum(const std::string& file_path);
size_t get_file_size(const std::string& file_path);

} // namespace invokeai
} // namespace duorou