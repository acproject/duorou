#include "model_manager.h"
#include "logger.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <memory>
#include <vector>
#include <string>
#include <chrono>

// Include llama.cpp headers
#include "llama.h"
#include "stable-diffusion.h"
#include "image_generator.h"
#include "model_downloader.h"

namespace duorou {
namespace core {

// LlamaModel implementation using llama.cpp
class LlamaModel : public BaseModel {
public:
    LlamaModel(const std::string& path) : model_path_(path), loaded_(false), model_(nullptr), ctx_(nullptr), text_generator_(nullptr) {}
    
    ~LlamaModel() {
        unload();
    }
    
    bool load(const std::string& model_path) override {
        if (loaded_) {
            return true;
        }
        
        std::cout << "Loading llama model from: " << model_path << std::endl;
        
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file not found: " << model_path << std::endl;
            return false;
        }
        
        // Initialize llama backend
        llama_backend_init();
        
        // Set model parameters
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0; // Use CPU for now
        model_params.use_mmap = true;
        model_params.use_mlock = false;
        
        // Load model
        model_ = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model_) {
            std::cerr << "Failed to load llama model: " << model_path << std::endl;
            return false;
        }
        
        // Set context parameters
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048; // Context size
        ctx_params.n_threads = std::thread::hardware_concurrency();
        ctx_params.n_threads_batch = std::thread::hardware_concurrency();
        
        // Create context
        ctx_ = llama_init_from_model(model_, ctx_params);
        if (!ctx_) {
            std::cerr << "Failed to create llama context" << std::endl;
            llama_model_free(model_);
            model_ = nullptr;
            return false;
        }
        
        // Create text generator
        text_generator_ = TextGeneratorFactory::create(model_, ctx_);
        if (!text_generator_) {
            std::cerr << "Failed to create text generator" << std::endl;
            llama_free(ctx_);
            llama_model_free(model_);
            ctx_ = nullptr;
            model_ = nullptr;
            return false;
        }
        
        loaded_ = true;
        
        std::cout << "Llama model loaded successfully" << std::endl;
        return true;
    }
    
    void unload() override {
        if (loaded_) {
            std::cout << "Unloading llama model" << std::endl;
            
            text_generator_.reset();
            
            if (ctx_) {
                llama_free(ctx_);
                ctx_ = nullptr;
            }
            
            if (model_) {
                llama_model_free(model_);
                model_ = nullptr;
            }
            
            llama_backend_free();
             
             loaded_ = false;
        }
    }
    
    bool isLoaded() const override {
        return loaded_;
    }
    
    ModelInfo getInfo() const override {
        ModelInfo info;
        info.name = std::filesystem::path(model_path_).filename().string();
        info.path = model_path_;
        info.type = ModelType::LANGUAGE_MODEL;
        info.status = loaded_ ? ModelStatus::LOADED : ModelStatus::NOT_LOADED;
        info.memory_usage = getMemoryUsage();
        return info;
    }
    
    size_t getMemoryUsage() const override {
        if (!loaded_ || !model_) {
            return 0;
        }
        return llama_model_size(model_);
    }
    
    // Additional methods for text generation
    GenerationResult generate(const std::string& prompt, const GenerationParams& params = GenerationParams()) {
        if (!loaded_ || !text_generator_) {
            GenerationResult result;
            result.stop_reason = "Model not loaded";
            result.finished = true;
            return result;
        }
        
        return text_generator_->generate(prompt, params);
    }
    
    GenerationResult generateStream(const std::string& prompt, 
                                   StreamCallback callback,
                                   const GenerationParams& params = GenerationParams()) {
        if (!loaded_ || !text_generator_) {
            GenerationResult result;
            result.stop_reason = "Model not loaded";
            result.finished = true;
            return result;
        }
        
        return text_generator_->generateStream(prompt, callback, params);
    }
    
    TextGenerator* getTextGenerator() const {
        return text_generator_.get();
    }
    
private:
    std::string model_path_;
    bool loaded_;
    llama_model* model_;
    llama_context* ctx_;
    std::unique_ptr<TextGenerator> text_generator_;
};

// StableDiffusionModel implementation using stable-diffusion.cpp
class StableDiffusionModel : public BaseModel {
public:
    StableDiffusionModel(const std::string& path) : model_path_(path), loaded_(false), sd_ctx_(nullptr), image_generator_(nullptr) {}
    
    ~StableDiffusionModel() {
        unload();
    }
    
    bool load(const std::string& model_path) override {
        if (loaded_) {
            return true;
        }
        
        std::cout << "Loading stable diffusion model from: " << model_path << std::endl;
        
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file not found: " << model_path << std::endl;
            return false;
        }
        
        // Initialize stable diffusion context parameters
        sd_ctx_params_t params;
        sd_ctx_params_init(&params);
        
        // Set model parameters
        params.model_path = model_path.c_str();
        params.n_threads = std::thread::hardware_concurrency();
        params.wtype = SD_TYPE_F16; // Use FP16 for efficiency
        params.rng_type = STD_DEFAULT_RNG;
        params.schedule = DEFAULT;
        params.vae_decode_only = false;
        params.vae_tiling = false;
        params.free_params_immediately = false;
        params.keep_clip_on_cpu = false;
        params.keep_control_net_on_cpu = false;
        params.keep_vae_on_cpu = false;
        
        // Create stable diffusion context
        sd_ctx_ = new_sd_ctx(&params);
        if (!sd_ctx_) {
            std::cerr << "Failed to create stable diffusion context" << std::endl;
            return false;
        }
        
        // Create image generator
        image_generator_ = ImageGeneratorFactory::create(sd_ctx_);
        if (!image_generator_) {
            std::cerr << "Failed to create image generator" << std::endl;
            free_sd_ctx(sd_ctx_);
            sd_ctx_ = nullptr;
            return false;
        }
        
        loaded_ = true;
        std::cout << "Stable diffusion model loaded successfully" << std::endl;
        return true;
    }
    
    void unload() override {
        if (loaded_) {
            std::cout << "Unloading stable diffusion model" << std::endl;
            
            image_generator_.reset();
            
            if (sd_ctx_) {
                free_sd_ctx(sd_ctx_);
                sd_ctx_ = nullptr;
            }
            
            loaded_ = false;
        }
    }
    
    bool isLoaded() const override {
        return loaded_;
    }
    
    ModelInfo getInfo() const override {
        ModelInfo info;
        info.name = std::filesystem::path(model_path_).filename().string();
        info.path = model_path_;
        info.type = ModelType::DIFFUSION_MODEL;
        info.status = loaded_ ? ModelStatus::LOADED : ModelStatus::NOT_LOADED;
        info.memory_usage = getMemoryUsage();
        return info;
    }
    
    size_t getMemoryUsage() const override {
        // Estimate memory usage for stable diffusion models
        return loaded_ ? (2ULL * 1024 * 1024 * 1024) : 0; // 2GB estimate
    }
    
    // Additional methods for image generation
    ImageGenerationResult generateImage(const std::string& prompt, 
                                       const ImageGenerationParams& params = ImageGenerationParams()) {
        if (!loaded_ || !image_generator_) {
            ImageGenerationResult result;
            result.error_message = "Model not loaded";
            return result;
        }
        
        return image_generator_->textToImage(prompt, params);
    }
    
    ImageGenerationResult generateImageWithProgress(const std::string& prompt,
                                                   ProgressCallback callback,
                                                   const ImageGenerationParams& params = ImageGenerationParams()) {
        if (!loaded_ || !image_generator_) {
            ImageGenerationResult result;
            result.error_message = "Model not loaded";
            return result;
        }
        
        return image_generator_->textToImageWithProgress(prompt, callback, params);
    }
    
    ImageGenerationResult imageToImage(const std::string& prompt,
                                      const std::vector<uint8_t>& input_image,
                                      int input_width,
                                      int input_height,
                                      const ImageGenerationParams& params = ImageGenerationParams()) {
        if (!loaded_ || !image_generator_) {
            ImageGenerationResult result;
            result.error_message = "Model not loaded";
            return result;
        }
        
        return image_generator_->imageToImage(prompt, input_image, input_width, input_height, params);
    }
    
    ImageGenerator* getImageGenerator() const {
        return image_generator_.get();
    }
    
private:
    std::string model_path_;
    bool loaded_;
    sd_ctx_t* sd_ctx_;
    std::unique_ptr<ImageGenerator> image_generator_;
};

// ModelManager实现
ModelManager::ModelManager()
    : memory_limit_(4ULL * 1024 * 1024 * 1024)  // 默认4GB内存限制
    , initialized_(false)
    , auto_memory_management_(false) {
    // 初始化模型下载器
    model_downloader_ = ModelDownloaderFactory::create();
}

ModelManager::~ModelManager() {
    unloadAllModels();
}

bool ModelManager::initialize() {
    if (initialized_) {
        return true;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        // 扫描默认模型目录
        std::string models_dir = "./models";
        if (std::filesystem::exists(models_dir)) {
            scanModelDirectory(models_dir);
        }
        
        initialized_ = true;
        std::cout << "ModelManager initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ModelManager: " << e.what() << std::endl;
        return false;
    }
}

bool ModelManager::registerModel(const ModelInfo& model_info) {
    if (!initialized_) {
        std::cerr << "ModelManager not initialized" << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (registered_models_.find(model_info.id) != registered_models_.end()) {
        std::cerr << "Model already registered: " << model_info.id << std::endl;
        return false;
    }
    
    registered_models_[model_info.id] = model_info;
    std::cout << "Model registered: " << model_info.id << " (" << model_info.name << ")" << std::endl;
    return true;
}

bool ModelManager::loadModel(const std::string& model_id) {
    if (!initialized_) {
        std::cerr << "ModelManager not initialized" << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查模型是否已注册
    auto it = registered_models_.find(model_id);
    if (it == registered_models_.end()) {
        std::cerr << "Model not registered: " << model_id << std::endl;
        return false;
    }
    
    // 检查模型是否已加载
    if (loaded_models_.find(model_id) != loaded_models_.end()) {
        std::cout << "Model already loaded: " << model_id << std::endl;
        return true;
    }
    
    // 检查内存限制
    if (!hasEnoughMemory(model_id)) {
        std::cerr << "Not enough memory to load model: " << model_id << std::endl;
        return false;
    }
    
    // 创建模型实例
    auto model = createModel(it->second);
    if (!model) {
        std::cerr << "Failed to create model instance: " << model_id << std::endl;
        return false;
    }
    
    // 更新状态为加载中
    updateModelStatus(model_id, ModelStatus::LOADING);
    
    // 加载模型
    bool success = model->load(it->second.path);
    if (success) {
        loaded_models_[model_id] = model;
        updateModelStatus(model_id, ModelStatus::LOADED);
        
        // 调用回调函数
        if (load_callback_) {
            load_callback_(model_id, true);
        }
        
        std::cout << "Model loaded successfully: " << model_id << std::endl;
    } else {
        updateModelStatus(model_id, ModelStatus::ERROR);
        
        // 调用回调函数
        if (load_callback_) {
            load_callback_(model_id, false);
        }
        
        std::cerr << "Failed to load model: " << model_id << std::endl;
    }
    
    return success;
}

bool ModelManager::unloadModel(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = loaded_models_.find(model_id);
    if (it == loaded_models_.end()) {
        std::cout << "Model not loaded: " << model_id << std::endl;
        return false;
    }
    
    // 卸载模型
    it->second->unload();
    loaded_models_.erase(it);
    
    // 更新状态
    updateModelStatus(model_id, ModelStatus::NOT_LOADED);
    
    std::cout << "Model unloaded: " << model_id << std::endl;
    return true;
}

void ModelManager::unloadAllModels() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& pair : loaded_models_) {
        pair.second->unload();
        updateModelStatus(pair.first, ModelStatus::NOT_LOADED);
    }
    
    loaded_models_.clear();
    std::cout << "All models unloaded" << std::endl;
}

std::shared_ptr<BaseModel> ModelManager::getModel(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = loaded_models_.find(model_id);
    if (it != loaded_models_.end()) {
        return it->second;
    }
    
    return nullptr;
}

bool ModelManager::isModelLoaded(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return loaded_models_.find(model_id) != loaded_models_.end();
}

ModelInfo ModelManager::getModelInfo(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = registered_models_.find(model_id);
    if (it != registered_models_.end()) {
        return it->second;
    }
    
    return ModelInfo(); // 返回空的ModelInfo
}

std::vector<ModelInfo> ModelManager::getAllModels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<ModelInfo> models;
    for (const auto& pair : registered_models_) {
        models.push_back(pair.second);
    }
    
    return models;
}

std::vector<std::string> ModelManager::getLoadedModels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> loaded_ids;
    for (const auto& pair : loaded_models_) {
        loaded_ids.push_back(pair.first);
    }
    
    return loaded_ids;
}

size_t ModelManager::getTotalMemoryUsage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t total = 0;
    for (const auto& pair : loaded_models_) {
        total += pair.second->getMemoryUsage();
    }
    
    return total;
}

void ModelManager::setMemoryLimit(size_t limit_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    memory_limit_ = limit_bytes;
    std::cout << "Memory limit set to: " << (limit_bytes / 1024 / 1024) << " MB" << std::endl;
}

size_t ModelManager::getMemoryLimit() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return memory_limit_;
}

bool ModelManager::hasEnoughMemory(const std::string& model_id) const {
    auto it = registered_models_.find(model_id);
    if (it == registered_models_.end()) {
        return false;
    }
    
    // 估算模型内存使用量（这里使用简单的估算）
    size_t estimated_usage = 512 * 1024 * 1024; // 默认512MB
    if (it->second.type == ModelType::DIFFUSION_MODEL) {
        estimated_usage = 1024 * 1024 * 1024; // 扩散模型1GB
    }
    
    size_t current_usage = getTotalMemoryUsage();
    return (current_usage + estimated_usage) <= memory_limit_;
}

void ModelManager::setLoadCallback(std::function<void(const std::string&, bool)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    load_callback_ = callback;
}

TextGenerator* ModelManager::getTextGenerator(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = loaded_models_.find(model_id);
    if (it == loaded_models_.end()) {
        return nullptr;
    }
    
    // Try to cast to LlamaModel
    auto llama_model = std::dynamic_pointer_cast<LlamaModel>(it->second);
    if (llama_model) {
        return llama_model->getTextGenerator();
    }
    
    return nullptr;
}

ImageGenerator* ModelManager::getImageGenerator(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = loaded_models_.find(model_id);
    if (it == loaded_models_.end()) {
        return nullptr;
    }
    
    // Try to cast to StableDiffusionModel
    auto sd_model = std::dynamic_pointer_cast<StableDiffusionModel>(it->second);
    if (sd_model) {
        return sd_model->getImageGenerator();
    }
    
    return nullptr;
}

size_t ModelManager::optimizeMemory() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t freed_memory = 0;
    
    // 获取当前内存使用情况
    size_t current_usage = getTotalMemoryUsage();
    
    // 如果内存使用超过限制的80%，开始优化
    if (current_usage > memory_limit_ * 0.8) {
        // 按最后使用时间排序，卸载最久未使用的模型
        std::vector<std::pair<std::string, std::chrono::steady_clock::time_point>> model_usage;
        
        for (const auto& pair : loaded_models_) {
            // 这里可以添加模型最后使用时间的跟踪
            // 暂时使用当前时间作为占位符
            model_usage.emplace_back(pair.first, std::chrono::steady_clock::now());
        }
        
        // 按使用时间排序（最久未使用的在前）
        std::sort(model_usage.begin(), model_usage.end(),
                 [](const auto& a, const auto& b) {
                     return a.second < b.second;
                 });
        
        // 卸载模型直到内存使用降到限制的60%以下
        for (const auto& pair : model_usage) {
            if (current_usage <= memory_limit_ * 0.6) {
                break;
            }
            
            size_t model_memory = loaded_models_[pair.first]->getMemoryUsage();
            if (unloadModel(pair.first)) {
                freed_memory += model_memory;
                current_usage -= model_memory;
            }
        }
    }
    
    return freed_memory;
}

void ModelManager::enableAutoMemoryManagement(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto_memory_management_ = enable;
    
    if (enable) {
        // 立即执行一次内存优化
        optimizeMemory();
    }
}

std::future<DownloadResult> ModelManager::downloadModel(const std::string& model_name, 
                                                       DownloadProgressCallback progress_callback) {
    if (model_downloader_) {
        model_downloader_->setProgressCallback(progress_callback);
        return model_downloader_->downloadModel(model_name);
    }
    
    // 返回失败的Future
    std::promise<DownloadResult> promise;
    DownloadResult result;
    result.error_message = "Model downloader not initialized";
    promise.set_value(result);
    return promise.get_future();
}

DownloadResult ModelManager::downloadModelSync(const std::string& model_name,
                                              DownloadProgressCallback progress_callback) {
    if (model_downloader_) {
        model_downloader_->setProgressCallback(progress_callback);
        return model_downloader_->downloadModelSync(model_name);
    }
    
    DownloadResult result;
    result.error_message = "Model downloader not initialized";
    return result;
}

ModelInfo ModelManager::getModelInfo(const std::string& model_name) {
    if (model_downloader_) {
        // 获取下载器的ModelInfo（duorou命名空间）
        auto downloader_info = model_downloader_->getModelInfo(model_name);
        
        // 转换为ModelManager的ModelInfo（duorou::core命名空间）
        ModelInfo manager_info;
        manager_info.id = downloader_info.name;
        manager_info.name = downloader_info.name;
        manager_info.description = downloader_info.description;
        manager_info.type = ModelType::LANGUAGE_MODEL; // 默认为语言模型
        manager_info.status = ModelStatus::NOT_LOADED;
        manager_info.memory_usage = 0;
        
        return manager_info;
    }
    
    return ModelInfo{};
}

bool ModelManager::isModelDownloaded(const std::string& model_name) {
    if (model_downloader_) {
        return model_downloader_->isModelDownloaded(model_name);
    }
    
    return false;
}

std::vector<std::string> ModelManager::getLocalModels() {
    if (model_downloader_) {
        return model_downloader_->getLocalModels();
    }
    
    return {};
}

bool ModelManager::deleteLocalModel(const std::string& model_name) {
    if (model_downloader_) {
        return model_downloader_->deleteModel(model_name);
    }
    
    return false;
}

bool ModelManager::verifyModel(const std::string& model_name) {
    if (model_downloader_) {
        return model_downloader_->verifyModel(model_name);
    }
    
    return false;
}

size_t ModelManager::cleanupModelCache() {
    if (model_downloader_) {
        return model_downloader_->cleanupUnusedBlobs();
    }
    
    return 0;
}

size_t ModelManager::getModelCacheSize() {
    if (model_downloader_) {
        return model_downloader_->getCacheSize();
    }
    
    return 0;
}

void ModelManager::setMaxModelCacheSize(size_t max_size) {
    if (model_downloader_) {
        model_downloader_->setMaxCacheSize(max_size);
    }
}

std::shared_ptr<BaseModel> ModelManager::createModel(const ModelInfo& model_info) {
    switch (model_info.type) {
        case ModelType::LANGUAGE_MODEL:
            return std::make_shared<LlamaModel>(model_info.path);
        case ModelType::DIFFUSION_MODEL:
            return std::make_shared<StableDiffusionModel>(model_info.path);
        default:
            std::cerr << "Unknown model type" << std::endl;
            return nullptr;
    }
}

void ModelManager::updateModelStatus(const std::string& model_id, ModelStatus status) {
    auto it = registered_models_.find(model_id);
    if (it != registered_models_.end()) {
        it->second.status = status;
    }
}

void ModelManager::scanModelDirectory(const std::string& directory) {
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string path = entry.path().string();
                std::string extension = entry.path().extension().string();
                
                // 检查文件扩展名以确定模型类型
                ModelInfo info;
                info.path = path;
                info.name = entry.path().stem().string();
                info.status = ModelStatus::NOT_LOADED;
                
                if (extension == ".gguf" || extension == ".bin") {
                    // LLaMA模型文件
                    info.id = "llama_" + info.name;
                    info.type = ModelType::LANGUAGE_MODEL;
                    info.description = "Language model (LLaMA)";
                } else if (extension == ".safetensors" || extension == ".ckpt") {
                    // Stable Diffusion模型文件
                    info.id = "sd_" + info.name;
                    info.type = ModelType::DIFFUSION_MODEL;
                    info.description = "Diffusion model (Stable Diffusion)";
                } else {
                    continue; // 跳过不支持的文件类型
                }
                
                // 注册模型（不加锁，因为调用者已经加锁）
                if (registered_models_.find(info.id) == registered_models_.end()) {
                    registered_models_[info.id] = info;
                    std::cout << "Auto-discovered model: " << info.id << " at " << path << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning model directory: " << e.what() << std::endl;
    }
}

} // namespace core
} // namespace duorou