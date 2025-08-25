#include "model_manager.h"
#include "logger.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <memory>
#include <vector>
#include <string>

// Include llama.cpp headers
#include "llama.h"
#include "stable-diffusion.h"

namespace duorou {
namespace core {

// LlamaModel implementation using llama.cpp
class LlamaModel : public BaseModel {
public:
    LlamaModel(const std::string& path) : model_path_(path), loaded_(false), model_(nullptr), ctx_(nullptr) {}
    
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
        
        loaded_ = true;
        
        std::cout << "Llama model loaded successfully" << std::endl;
        return true;
    }
    
    void unload() override {
        if (loaded_) {
            std::cout << "Unloading llama model" << std::endl;
            
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
    std::string generate(const std::string& prompt, int max_tokens = 100) {
        if (!loaded_ || !ctx_ || !model_) {
            std::cerr << "Model not loaded for generation" << std::endl;
            return "";
        }
        
        // Get vocabulary
        const llama_vocab* vocab = llama_model_get_vocab(model_);
        
        // Tokenize input
        std::vector<llama_token> tokens(prompt.length() + 1);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), 
                                     tokens.data(), tokens.size(), true, false);
        
        if (n_tokens < 0) {
            std::cerr << "Failed to tokenize prompt" << std::endl;
            return "";
        }
        
        tokens.resize(n_tokens);
        
        // Create batch
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        
        // Fill batch with tokens
        for (int i = 0; i < n_tokens; i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = new llama_seq_id[1]{0};
            batch.logits[i] = (i == n_tokens - 1) ? 1 : 0; // Only compute logits for last token
        }
        batch.n_tokens = n_tokens;
        
        // Process prompt
        if (llama_decode(ctx_, batch) != 0) {
            std::cerr << "Failed to decode prompt" << std::endl;
            llama_batch_free(batch);
            return "";
        }
        
        // Generate tokens
        std::string result = prompt;
        
        for (int i = 0; i < max_tokens; i++) {
            // Get logits
            float* logits = llama_get_logits_ith(ctx_, -1);
            if (!logits) {
                break;
            }
            
            // Simple greedy sampling - select token with highest probability
             int n_vocab = llama_vocab_n_tokens(vocab);
             llama_token next_token = 0;
             float max_logit = logits[0];
            
            for (int j = 1; j < n_vocab; j++) {
                if (logits[j] > max_logit) {
                    max_logit = logits[j];
                    next_token = j;
                }
            }
            
            // Check for end of sequence
            if (next_token == llama_vocab_eos(vocab)) {
                break;
            }
            
            // Convert token to text
            char piece[256];
            int piece_len = llama_token_to_piece(vocab, next_token, piece, sizeof(piece), 0, false);
            if (piece_len > 0) {
                result.append(piece, piece_len);
            }
            
            // Prepare next batch
            batch.n_tokens = 1;
            batch.token[0] = next_token;
            batch.pos[0] = n_tokens + i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            
            // Decode next token
            if (llama_decode(ctx_, batch) != 0) {
                break;
            }
        }
        
        // Cleanup
        for (int i = 0; i < batch.n_tokens; i++) {
            delete[] batch.seq_id[i];
        }
        llama_batch_free(batch);
        
        return result;
    }
    
private:
    std::string model_path_;
    bool loaded_;
    llama_model* model_;
    llama_context* ctx_;
};

// StableDiffusionModel implementation using stable-diffusion.cpp
class StableDiffusionModel : public BaseModel {
public:
    StableDiffusionModel(const std::string& path) : model_path_(path), loaded_(false), sd_ctx_(nullptr) {}
    
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
        
        loaded_ = true;
        std::cout << "Stable diffusion model loaded successfully" << std::endl;
        return true;
    }
    
    void unload() override {
        if (loaded_) {
            std::cout << "Unloading stable diffusion model" << std::endl;
            
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
    sd_image_t* generateImage(const std::string& prompt, 
                             const std::string& negative_prompt = "",
                             int width = 512, 
                             int height = 512,
                             int steps = 20,
                             float cfg_scale = 7.5,
                             int64_t seed = -1) {
        if (!loaded_ || !sd_ctx_) {
            std::cerr << "Model not loaded for generation" << std::endl;
            return nullptr;
        }
        
        // Initialize image generation parameters
        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);
        
        // Set generation parameters
        gen_params.prompt = prompt.c_str();
        gen_params.negative_prompt = negative_prompt.c_str();
        gen_params.width = width;
        gen_params.height = height;
        gen_params.sample_method = EULER_A;
        gen_params.sample_steps = steps;
        gen_params.guidance.txt_cfg = cfg_scale;
        gen_params.seed = seed;
        gen_params.batch_count = 1;
        gen_params.clip_skip = -1;
        gen_params.strength = 0.75f;
        
        // Generate image
        sd_image_t* result = generate_image(sd_ctx_, &gen_params);
        
        if (!result) {
            std::cerr << "Failed to generate image" << std::endl;
            return nullptr;
        }
        
        std::cout << "Image generated successfully: " << width << "x" << height << std::endl;
        return result;
    }
    
private:
    std::string model_path_;
    bool loaded_;
    sd_ctx_t* sd_ctx_;
};

// ModelManager实现
ModelManager::ModelManager()
    : memory_limit_(4ULL * 1024 * 1024 * 1024)  // 默认4GB内存限制
    , initialized_(false) {
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
    load_callback_ = std::move(callback);
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