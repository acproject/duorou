#include "model_manager.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <sstream>
#include <regex>

namespace duorou {
namespace invokeai {

// ModelManager的私有实现类
class ModelManager::Impl {
public:
    Impl() : is_initialized_(false), max_loaded_models_(3), memory_limit_mb_(8192),
             auto_unload_enabled_(true), cache_enabled_(true) {}
    
    bool initialize(const std::string& models_root_path) {
        if (is_initialized_) {
            return true;
        }
        
        models_root_path_ = models_root_path;
        
        // 创建模型根目录
        std::filesystem::create_directories(models_root_path_);
        
        // 添加默认模型路径
        model_paths_.push_back(models_root_path_);
        model_paths_.push_back(models_root_path_ + "/checkpoints");
        model_paths_.push_back(models_root_path_ + "/diffusers");
        model_paths_.push_back(models_root_path_ + "/lora");
        model_paths_.push_back(models_root_path_ + "/controlnet");
        model_paths_.push_back(models_root_path_ + "/vae");
        model_paths_.push_back(models_root_path_ + "/embeddings");
        
        // 创建子目录
        for (const auto& path : model_paths_) {
            std::filesystem::create_directories(path);
        }
        
        // 扫描现有模型
        scan_models();
        
        is_initialized_ = true;
        
        std::cout << "ModelManager initialized successfully" << std::endl;
        std::cout << "Models root path: " << models_root_path_ << std::endl;
        std::cout << "Found " << model_configs_.size() << " models" << std::endl;
        
        return true;
    }
    
    void shutdown() {
        if (!is_initialized_) {
            return;
        }
        
        // 卸载所有已加载的模型
        for (const auto& model_name : loaded_models_) {
            unload_model(model_name);
        }
        
        loaded_models_.clear();
        model_configs_.clear();
        model_runtime_info_.clear();
        
        is_initialized_ = false;
        
        std::cout << "ModelManager shutdown" << std::endl;
    }
    
    void scan_models() {
        model_configs_.clear();
        
        for (const auto& search_path : model_paths_) {
            if (!std::filesystem::exists(search_path)) {
                continue;
            }
            
            scan_directory(search_path);
        }
        
        std::cout << "Model scan completed. Found " << model_configs_.size() << " models" << std::endl;
    }
    
    void add_model_path(const std::string& path) {
        if (std::find(model_paths_.begin(), model_paths_.end(), path) == model_paths_.end()) {
            model_paths_.push_back(path);
            std::filesystem::create_directories(path);
            scan_directory(path);
        }
    }
    
    void remove_model_path(const std::string& path) {
        model_paths_.erase(
            std::remove(model_paths_.begin(), model_paths_.end(), path),
            model_paths_.end());
    }
    
    std::vector<std::string> get_model_paths() const {
        return model_paths_;
    }
    
    std::vector<ModelConfig> get_all_models() const {
        std::vector<ModelConfig> models;
        for (const auto& pair : model_configs_) {
            models.push_back(pair.second);
        }
        return models;
    }
    
    std::vector<ModelConfig> get_models_by_type(ModelType type) const {
        std::vector<ModelConfig> models;
        for (const auto& pair : model_configs_) {
            if (pair.second.type == type) {
                models.push_back(pair.second);
            }
        }
        return models;
    }
    
    std::vector<ModelConfig> search_models(const std::string& query) const {
        std::vector<ModelConfig> results;
        std::string lower_query = to_lower(query);
        
        for (const auto& pair : model_configs_) {
            const auto& config = pair.second;
            
            // 搜索名称、描述和标签
            if (to_lower(config.name).find(lower_query) != std::string::npos ||
                to_lower(config.description).find(lower_query) != std::string::npos) {
                results.push_back(config);
                continue;
            }
            
            // 搜索标签
            for (const auto& tag : config.tags) {
                if (to_lower(tag).find(lower_query) != std::string::npos) {
                    results.push_back(config);
                    break;
                }
            }
        }
        
        return results;
    }
    
    ModelConfig get_model_config(const std::string& model_name) const {
        auto it = model_configs_.find(model_name);
        if (it != model_configs_.end()) {
            return it->second;
        }
        return ModelConfig{}; // 返回空配置
    }
    
    bool has_model(const std::string& model_name) const {
        return model_configs_.find(model_name) != model_configs_.end();
    }
    
    bool load_model(const std::string& model_name, 
                   const std::string& device,
                   const std::string& precision,
                   ModelLoadProgressCallback progress_cb) {
        if (!has_model(model_name)) {
            std::cerr << "Model not found: " << model_name << std::endl;
            return false;
        }
        
        if (is_model_loaded(model_name)) {
            std::cout << "Model already loaded: " << model_name << std::endl;
            return true;
        }
        
        // 检查内存限制
        if (!check_memory_availability(model_name)) {
            if (auto_unload_enabled_) {
                auto_unload_models();
            } else {
                std::cerr << "Insufficient memory to load model: " << model_name << std::endl;
                return false;
            }
        }
        
        // 检查最大加载模型数量
        if (loaded_models_.size() >= max_loaded_models_) {
            if (auto_unload_enabled_) {
                unload_oldest_model();
            } else {
                std::cerr << "Maximum number of loaded models reached" << std::endl;
                return false;
            }
        }
        
        auto config = get_model_config(model_name);
        ModelRuntimeInfo runtime_info;
        runtime_info.state = ModelLoadState::LOADING;
        runtime_info.device = (device == "auto") ? select_best_device() : device;
        runtime_info.precision = (precision == "auto") ? select_best_precision() : precision;
        
        model_runtime_info_[model_name] = runtime_info;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            std::cout << "Loading model: " << model_name << std::endl;
            std::cout << "Device: " << runtime_info.device << ", Precision: " << runtime_info.precision << std::endl;
            
            // 模拟加载过程
            std::vector<std::string> stages = {
                "Validating model files",
                "Loading model weights",
                "Initializing model",
                "Optimizing for device",
                "Finalizing"
            };
            
            for (size_t i = 0; i < stages.size(); ++i) {
                if (progress_cb) {
                    progress_cb(stages[i], i + 1, stages.size());
                }
                
                // 模拟加载时间
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            runtime_info.load_time_seconds = 
                std::chrono::duration<double>(end_time - start_time).count();
            runtime_info.load_timestamp = std::chrono::system_clock::now();
            runtime_info.state = ModelLoadState::LOADED;
            runtime_info.actual_memory_usage_mb = config.estimated_memory_mb;
            
            model_runtime_info_[model_name] = runtime_info;
            loaded_models_.push_back(model_name);
            current_model_ = model_name;
            
            std::cout << "Model loaded successfully: " << model_name << std::endl;
            std::cout << "Load time: " << runtime_info.load_time_seconds << " seconds" << std::endl;
            
            // 触发事件回调
            if (event_callback_) {
                event_callback_(model_name, "loaded");
            }
            
            return true;
            
        } catch (const std::exception& e) {
            runtime_info.state = ModelLoadState::ERROR;
            runtime_info.error_message = e.what();
            model_runtime_info_[model_name] = runtime_info;
            
            std::cerr << "Failed to load model " << model_name << ": " << e.what() << std::endl;
            return false;
        }
    }
    
    bool unload_model(const std::string& model_name) {
        std::string target_model = model_name.empty() ? current_model_ : model_name;
        
        if (target_model.empty() || !is_model_loaded(target_model)) {
            return true; // 已经卸载或不存在
        }
        
        std::cout << "Unloading model: " << target_model << std::endl;
        
        // 从加载列表中移除
        loaded_models_.erase(
            std::remove(loaded_models_.begin(), loaded_models_.end(), target_model),
            loaded_models_.end());
        
        // 更新运行时信息
        auto it = model_runtime_info_.find(target_model);
        if (it != model_runtime_info_.end()) {
            it->second.state = ModelLoadState::UNLOADED;
            it->second.actual_memory_usage_mb = 0;
        }
        
        // 如果是当前模型，切换到其他模型或清空
        if (current_model_ == target_model) {
            current_model_ = loaded_models_.empty() ? "" : loaded_models_.back();
        }
        
        std::cout << "Model unloaded: " << target_model << std::endl;
        
        // 触发事件回调
        if (event_callback_) {
            event_callback_(target_model, "unloaded");
        }
        
        return true;
    }
    
    bool switch_model(const std::string& model_name) {
        if (!is_model_loaded(model_name)) {
            return load_model(model_name, "auto", "auto", nullptr);
        }
        
        current_model_ = model_name;
        std::cout << "Switched to model: " << model_name << std::endl;
        
        if (event_callback_) {
            event_callback_(model_name, "switched");
        }
        
        return true;
    }
    
    std::vector<std::string> get_loaded_models() const {
        return loaded_models_;
    }
    
    std::string get_current_model() const {
        return current_model_;
    }
    
    ModelRuntimeInfo get_model_runtime_info(const std::string& model_name) const {
        auto it = model_runtime_info_.find(model_name);
        if (it != model_runtime_info_.end()) {
            return it->second;
        }
        return ModelRuntimeInfo{};
    }
    
    bool is_model_loaded(const std::string& model_name) const {
        return std::find(loaded_models_.begin(), loaded_models_.end(), model_name) != loaded_models_.end();
    }
    
    bool validate_model(const std::string& model_name) const {
        if (!has_model(model_name)) {
            return false;
        }
        
        auto config = get_model_config(model_name);
        
        // 检查文件是否存在
        if (!std::filesystem::exists(config.path)) {
            return false;
        }
        
        // 检查文件大小
        if (config.file_size_bytes > 0) {
            auto actual_size = get_file_size(config.path);
            if (actual_size != config.file_size_bytes) {
                return false;
            }
        }
        
        // 检查校验和（如果有）
        if (!config.checksum.empty()) {
            auto actual_checksum = calculate_file_checksum(config.path);
            if (actual_checksum != config.checksum) {
                return false;
            }
        }
        
        return true;
    }
    
    bool repair_model(const std::string& model_name) {
        // 简化实现：重新扫描模型
        std::cout << "Attempting to repair model: " << model_name << std::endl;
        scan_models();
        return has_model(model_name);
    }
    
    std::vector<std::string> get_missing_dependencies(const std::string& model_name) const {
        // 简化实现：返回空列表
        return std::vector<std::string>();
    }
    
    bool install_model_from_url(const std::string& url, const std::string& model_name) {
        // 简化实现：模拟安装
        std::cout << "Installing model from URL: " << url << " as " << model_name << std::endl;
        return false; // 暂不实现
    }
    
    bool install_model_from_file(const std::string& file_path, const std::string& model_name) {
        // 简化实现：复制文件到模型目录
        std::cout << "Installing model from file: " << file_path << " as " << model_name << std::endl;
        return false; // 暂不实现
    }
    
    bool uninstall_model(const std::string& model_name) {
        if (!has_model(model_name)) {
            return false;
        }
        
        // 先卸载模型（如果已加载）
        if (is_model_loaded(model_name)) {
            unload_model(model_name);
        }
        
        // 从配置中移除
        model_configs_.erase(model_name);
        
        std::cout << "Model uninstalled: " << model_name << std::endl;
        return true;
    }
    
    bool save_model_config(const ModelConfig& config) {
        model_configs_[config.name] = config;
        return true;
    }
    
    bool load_model_config(const std::string& model_name, ModelConfig& config) {
        if (has_model(model_name)) {
            config = get_model_config(model_name);
            return true;
        }
        return false;
    }
    
    bool update_model_metadata(const std::string& model_name, 
                              const std::map<std::string, std::string>& metadata) {
        if (!has_model(model_name)) {
            return false;
        }
        
        auto& config = model_configs_[model_name];
        for (const auto& pair : metadata) {
            config.metadata[pair.first] = pair.second;
        }
        
        return true;
    }
    
    void clear_model_cache() {
        std::cout << "Clearing model cache" << std::endl;
        // 简化实现：什么都不做
    }
    
    void optimize_memory_usage() {
        if (auto_unload_enabled_) {
            auto_unload_models();
        }
    }
    
    size_t get_available_memory() const {
        return memory_limit_mb_ - get_total_memory_usage();
    }
    
    void set_model_event_callback(ModelEventCallback callback) {
        event_callback_ = callback;
    }
    
    void set_max_loaded_models(int max_models) {
        max_loaded_models_ = max_models;
    }
    
    void set_memory_limit(size_t limit_mb) {
        memory_limit_mb_ = limit_mb;
    }
    
    void set_auto_unload_enabled(bool enabled) {
        auto_unload_enabled_ = enabled;
    }
    
    void set_cache_enabled(bool enabled) {
        cache_enabled_ = enabled;
    }
    
    size_t get_total_memory_usage() const {
        size_t total = 0;
        for (const auto& model_name : loaded_models_) {
            auto runtime_info = get_model_runtime_info(model_name);
            total += runtime_info.actual_memory_usage_mb;
        }
        return total;
    }
    
    std::map<std::string, std::string> get_statistics() const {
        std::map<std::string, std::string> stats;
        stats["total_models"] = std::to_string(model_configs_.size());
        stats["loaded_models"] = std::to_string(loaded_models_.size());
        stats["current_model"] = current_model_;
        stats["memory_usage_mb"] = std::to_string(get_total_memory_usage());
        stats["memory_limit_mb"] = std::to_string(memory_limit_mb_);
        stats["max_loaded_models"] = std::to_string(max_loaded_models_);
        stats["auto_unload_enabled"] = auto_unload_enabled_ ? "true" : "false";
        stats["cache_enabled"] = cache_enabled_ ? "true" : "false";
        return stats;
    }
    
private:
    void scan_directory(const std::string& dir_path) {
        if (!std::filesystem::exists(dir_path)) {
            return;
        }
        
        for (const auto& entry : std::filesystem::recursive_directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                
                if (is_valid_model_file(file_path)) {
                    auto config = create_model_config(file_path);
                    if (!config.name.empty()) {
                        model_configs_[config.name] = config;
                    }
                }
            }
        }
    }
    
    ModelConfig create_model_config(const std::string& file_path) {
        ModelConfig config;
        
        std::filesystem::path path(file_path);
        config.name = path.stem().string();
        config.path = file_path;
        config.type = detect_model_type(file_path);
        config.file_size_bytes = get_file_size(file_path);
        
        // 估算内存使用量（简化计算）
        config.estimated_memory_mb = config.file_size_bytes / (1024 * 1024) * 2; // 大约是文件大小的2倍
        
        // 设置默认值
        config.description = "Model: " + config.name;
        config.version = "unknown";
        config.supports_fp16 = true;
        config.supports_cpu = true;
        config.supports_gpu = true;
        
        // 根据文件名推断一些属性
        std::string lower_name = to_lower(config.name);
        if (lower_name.find("xl") != std::string::npos) {
            config.width = 1024;
            config.height = 1024;
            config.tags.push_back("xl");
        }
        if (lower_name.find("inpaint") != std::string::npos) {
            config.tags.push_back("inpainting");
        }
        if (lower_name.find("anime") != std::string::npos) {
            config.tags.push_back("anime");
        }
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        config.created_date = std::to_string(time_t);
        config.modified_date = config.created_date;
        
        return config;
    }
    
    bool check_memory_availability(const std::string& model_name) const {
        auto config = get_model_config(model_name);
        size_t current_usage = get_total_memory_usage();
        return (current_usage + config.estimated_memory_mb) <= memory_limit_mb_;
    }
    
    void auto_unload_models() {
        while (loaded_models_.size() > 0 && get_total_memory_usage() > memory_limit_mb_ * 0.8) {
            unload_oldest_model();
        }
    }
    
    void unload_oldest_model() {
        if (loaded_models_.empty()) {
            return;
        }
        
        // 找到最早加载的模型
        std::string oldest_model = loaded_models_[0];
        auto oldest_time = std::chrono::system_clock::time_point::min();
        
        for (const auto& model_name : loaded_models_) {
            auto runtime_info = get_model_runtime_info(model_name);
            if (runtime_info.load_timestamp > oldest_time) {
                oldest_time = runtime_info.load_timestamp;
                oldest_model = model_name;
            }
        }
        
        unload_model(oldest_model);
    }
    
    std::string select_best_device() const {
        // 简化的设备选择逻辑
        return "cpu"; // 默认使用CPU
    }
    
    std::string select_best_precision() const {
        return "fp32"; // 默认使用fp32
    }
    
    std::string to_lower(const std::string& str) const {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
    
    bool is_initialized_;
    std::string models_root_path_;
    std::vector<std::string> model_paths_;
    std::map<std::string, ModelConfig> model_configs_;
    std::vector<std::string> loaded_models_;
    std::string current_model_;
    std::map<std::string, ModelRuntimeInfo> model_runtime_info_;
    
    // 配置
    size_t max_loaded_models_;
    size_t memory_limit_mb_;
    bool auto_unload_enabled_;
    bool cache_enabled_;
    
    // 回调
    ModelEventCallback event_callback_;
};

// ModelManager公共接口实现
ModelManager::ModelManager() : pimpl_(std::make_unique<Impl>()) {}

ModelManager::~ModelManager() = default;

bool ModelManager::initialize(const std::string& models_root_path) {
    return pimpl_->initialize(models_root_path);
}

void ModelManager::shutdown() {
    pimpl_->shutdown();
}

void ModelManager::scan_models() {
    pimpl_->scan_models();
}

void ModelManager::add_model_path(const std::string& path) {
    pimpl_->add_model_path(path);
}

void ModelManager::remove_model_path(const std::string& path) {
    pimpl_->remove_model_path(path);
}

std::vector<std::string> ModelManager::get_model_paths() const {
    return pimpl_->get_model_paths();
}

std::vector<ModelConfig> ModelManager::get_all_models() const {
    return pimpl_->get_all_models();
}

std::vector<ModelConfig> ModelManager::get_models_by_type(ModelType type) const {
    return pimpl_->get_models_by_type(type);
}

std::vector<ModelConfig> ModelManager::search_models(const std::string& query) const {
    return pimpl_->search_models(query);
}

ModelConfig ModelManager::get_model_config(const std::string& model_name) const {
    return pimpl_->get_model_config(model_name);
}

bool ModelManager::has_model(const std::string& model_name) const {
    return pimpl_->has_model(model_name);
}

bool ModelManager::load_model(const std::string& model_name, 
                             const std::string& device,
                             const std::string& precision,
                             ModelLoadProgressCallback progress_cb) {
    return pimpl_->load_model(model_name, device, precision, progress_cb);
}

bool ModelManager::unload_model(const std::string& model_name) {
    return pimpl_->unload_model(model_name);
}

bool ModelManager::switch_model(const std::string& model_name) {
    return pimpl_->switch_model(model_name);
}

std::vector<std::string> ModelManager::get_loaded_models() const {
    return pimpl_->get_loaded_models();
}

std::string ModelManager::get_current_model() const {
    return pimpl_->get_current_model();
}

ModelRuntimeInfo ModelManager::get_model_runtime_info(const std::string& model_name) const {
    return pimpl_->get_model_runtime_info(model_name);
}

bool ModelManager::is_model_loaded(const std::string& model_name) const {
    return pimpl_->is_model_loaded(model_name);
}

bool ModelManager::validate_model(const std::string& model_name) const {
    return pimpl_->validate_model(model_name);
}

bool ModelManager::repair_model(const std::string& model_name) {
    return pimpl_->repair_model(model_name);
}

std::vector<std::string> ModelManager::get_missing_dependencies(const std::string& model_name) const {
    return pimpl_->get_missing_dependencies(model_name);
}

bool ModelManager::install_model_from_url(const std::string& url, const std::string& model_name) {
    return pimpl_->install_model_from_url(url, model_name);
}

bool ModelManager::install_model_from_file(const std::string& file_path, const std::string& model_name) {
    return pimpl_->install_model_from_file(file_path, model_name);
}

bool ModelManager::uninstall_model(const std::string& model_name) {
    return pimpl_->uninstall_model(model_name);
}

bool ModelManager::save_model_config(const ModelConfig& config) {
    return pimpl_->save_model_config(config);
}

bool ModelManager::load_model_config(const std::string& model_name, ModelConfig& config) {
    return pimpl_->load_model_config(model_name, config);
}

bool ModelManager::update_model_metadata(const std::string& model_name, 
                                        const std::map<std::string, std::string>& metadata) {
    return pimpl_->update_model_metadata(model_name, metadata);
}

void ModelManager::clear_model_cache() {
    pimpl_->clear_model_cache();
}

void ModelManager::optimize_memory_usage() {
    pimpl_->optimize_memory_usage();
}

size_t ModelManager::get_available_memory() const {
    return pimpl_->get_available_memory();
}

void ModelManager::set_model_event_callback(ModelEventCallback callback) {
    pimpl_->set_model_event_callback(callback);
}

void ModelManager::set_max_loaded_models(int max_models) {
    pimpl_->set_max_loaded_models(max_models);
}

void ModelManager::set_memory_limit(size_t limit_mb) {
    pimpl_->set_memory_limit(limit_mb);
}

void ModelManager::set_auto_unload_enabled(bool enabled) {
    pimpl_->set_auto_unload_enabled(enabled);
}

void ModelManager::set_cache_enabled(bool enabled) {
    pimpl_->set_cache_enabled(enabled);
}

size_t ModelManager::get_total_memory_usage() const {
    return pimpl_->get_total_memory_usage();
}

std::map<std::string, std::string> ModelManager::get_statistics() const {
    return pimpl_->get_statistics();
}

// 工具函数实现
ModelType detect_model_type(const std::string& file_path) {
    std::filesystem::path path(file_path);
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == ".ckpt") {
        return ModelType::CHECKPOINT;
    } else if (extension == ".safetensors") {
        return ModelType::SAFETENSORS;
    } else if (extension == ".pt" || extension == ".pth") {
        std::string filename = path.filename().string();
        std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);
        
        if (filename.find("lora") != std::string::npos) {
            return ModelType::LORA;
        } else if (filename.find("controlnet") != std::string::npos) {
            return ModelType::CONTROLNET;
        } else if (filename.find("vae") != std::string::npos) {
            return ModelType::VAE;
        } else if (filename.find("embedding") != std::string::npos) {
            return ModelType::TEXTUAL_INVERSION;
        }
        return ModelType::CHECKPOINT;
    }
    
    // 检查是否是diffusers格式目录
    if (std::filesystem::is_directory(file_path)) {
        if (std::filesystem::exists(file_path + "/model_index.json")) {
            return ModelType::DIFFUSERS;
        }
    }
    
    return ModelType::UNKNOWN;
}

std::string model_type_to_string(ModelType type) {
    switch (type) {
        case ModelType::CHECKPOINT: return "checkpoint";
        case ModelType::SAFETENSORS: return "safetensors";
        case ModelType::DIFFUSERS: return "diffusers";
        case ModelType::LORA: return "lora";
        case ModelType::CONTROLNET: return "controlnet";
        case ModelType::VAE: return "vae";
        case ModelType::TEXTUAL_INVERSION: return "textual_inversion";
        default: return "unknown";
    }
}

ModelType string_to_model_type(const std::string& type_str) {
    std::string lower_str = type_str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
    
    if (lower_str == "checkpoint") return ModelType::CHECKPOINT;
    if (lower_str == "safetensors") return ModelType::SAFETENSORS;
    if (lower_str == "diffusers") return ModelType::DIFFUSERS;
    if (lower_str == "lora") return ModelType::LORA;
    if (lower_str == "controlnet") return ModelType::CONTROLNET;
    if (lower_str == "vae") return ModelType::VAE;
    if (lower_str == "textual_inversion") return ModelType::TEXTUAL_INVERSION;
    
    return ModelType::UNKNOWN;
}

bool is_valid_model_file(const std::string& file_path) {
    return detect_model_type(file_path) != ModelType::UNKNOWN;
}

std::string calculate_file_checksum(const std::string& file_path) {
    // 简化实现：返回文件大小作为"校验和"
    // 实际实现应该使用MD5或SHA256
    return std::to_string(get_file_size(file_path));
}

size_t get_file_size(const std::string& file_path) {
    try {
        return std::filesystem::file_size(file_path);
    } catch (const std::exception&) {
        return 0;
    }
}

} // namespace invokeai
} // namespace duorou