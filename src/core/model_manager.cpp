#include "model_manager.h"
#include "logger.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <thread>
#include <vector>

// Include necessary headers for model loading
#include "image_generator.h"
#include "model_downloader.h"
#include "model_path_manager.h"
#include "../extensions/ollama/ollama_model_loader.h"
#include "../extensions/ollama/model_path_manager.h"
#include "stable-diffusion.h"
#include "text_generator.h"

// Simple Ollama model implementation using extensions
class OllamaModelImpl : public duorou::core::BaseModel {
public:
    explicit OllamaModelImpl(const std::string& model_path) 
        : model_path_(model_path), loaded_(false), memory_usage_(0) {
        model_info_.name = model_path;
        model_info_.type = duorou::core::ModelType::LANGUAGE_MODEL;
        model_info_.status = duorou::core::ModelStatus::NOT_LOADED;
        model_info_.memory_usage = 0;
        model_info_.path = model_path;
    }
    
    bool load(const std::string& model_path) override {
        duorou::core::Logger logger;
        logger.info("[OLLAMA] Loading model: " + model_path);
        loaded_ = true;
        memory_usage_ = 1024 * 1024 * 1024; // 1GB estimate
        model_info_.status = duorou::core::ModelStatus::LOADED;
        model_info_.memory_usage = memory_usage_;
        return true;
    }
    
    void unload() override {
        loaded_ = false;
        memory_usage_ = 0;
        model_info_.status = duorou::core::ModelStatus::NOT_LOADED;
        model_info_.memory_usage = 0;
    }
    
    bool isLoaded() const override { return loaded_; }
    duorou::core::ModelInfo getInfo() const override { return model_info_; }
    size_t getMemoryUsage() const override { return memory_usage_; }
    
private:
    std::string model_path_;
    bool loaded_;
    size_t memory_usage_;
    duorou::core::ModelInfo model_info_;
};



namespace duorou {
namespace core {

// LlamaModel implementation using llama.cpp - DISABLED due to missing llama.h
/*
class LlamaModel : public BaseModel {
public:
  LlamaModel(const std::string &path)
      : model_path_(path), loaded_(false), model_(nullptr), ctx_(nullptr),
        text_generator_(nullptr) {}

  ~LlamaModel() { unload(); }

  bool load(const std::string &model_path) override {
    if (loaded_) {
      return true;
    }

    std::cout << "\n=== Loading Llama Model with Extensions ===" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;

    // Initialize llama backend
    llama_backend_init();

    // Set model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // Use CPU for now
    model_params.use_mmap = true;
    model_params.use_mlock = false;
    model_params.vocab_only = false; // Load full model for text generation

    std::cout << "Model parameters configured:" << std::endl;
    std::cout << "  - GPU layers: " << model_params.n_gpu_layers << std::endl;
    std::cout << "  - Use mmap: " << (model_params.use_mmap ? "true" : "false") << std::endl;
    std::cout << "  - Vocab only: " << (model_params.vocab_only ? "true" : "false") << std::endl;

    // Check if this is an Ollama model (starts with registry.ollama.ai)
    if (model_path.find("registry.ollama.ai") != std::string::npos) {
      std::cout << "\n=== Detected Ollama Model ===" << std::endl;
      std::cout << "Using OllamaModelLoader for: " << model_path << std::endl;
      
      // Extract model name for extension testing
      std::string model_name = model_path;
      if (model_name.find("/") != std::string::npos) {
        model_name = model_name.substr(model_name.find_last_of("/") + 1);
      }
      std::cout << "Extracted model name: " << model_name << std::endl;
      
      // Test extensions with detected model
      std::cout << "\n=== Testing Extensions ===" << std::endl;
      
      // 1. Test GGML incremental extension
      std::string arch = "qwen25vl"; // Default for qwen2.5vl models
      if (model_name.find("qwen") != std::string::npos) {
        arch = "qwen25vl";
      } else if (model_name.find("gemma") != std::string::npos) {
        arch = "gemma3";
      } else if (model_name.find("mistral") != std::string::npos) {
        arch = "mistral3";
      } else if (model_name.find("llama") != std::string::npos) {
        arch = "llama";
      }
      
      std::cout << "Detected architecture: " << arch << std::endl;
      
      std::cout << "\n=== Loading Ollama Model ===" << std::endl;
      auto path_manager = std::make_shared<ModelPathManager>();
      path_manager->initialize();
      OllamaModelLoader ollama_loader(path_manager);
      model_ = ollama_loader.loadFromOllamaModel(model_path, model_params);
    } else {
      std::cout << "\n=== Detected Regular File Model ===" << std::endl;
      // Regular file-based model loading
      if (!std::filesystem::exists(model_path)) {
        std::cerr << "❌ Model file not found: " << model_path << std::endl;
        return false;
      }
      std::cout << "Loading model directly with llama.cpp" << std::endl;
      // Use standard llama.cpp loading
      model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    }

    if (!model_) {
      std::cerr << "❌ Failed to load llama model: " << model_path << std::endl;
      return false;
    }
    
    std::cout << "✅ Model loaded successfully!" << std::endl;

    std::cout << "\n=== Creating Model Context ===" << std::endl;
    // Set context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048; // Context size
    ctx_params.n_threads = std::thread::hardware_concurrency();
    ctx_params.n_threads_batch = std::thread::hardware_concurrency();
    
    std::cout << "Context parameters:" << std::endl;
    std::cout << "  - Context size: " << ctx_params.n_ctx << std::endl;
    std::cout << "  - Threads: " << ctx_params.n_threads << std::endl;
    std::cout << "  - Batch threads: " << ctx_params.n_threads_batch << std::endl;

    // Create context
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
      std::cerr << "❌ Failed to create llama context" << std::endl;
      llama_model_free(model_);
      model_ = nullptr;
      return false;
    }
    std::cout << "✅ Model context created successfully" << std::endl;

    std::cout << "\n=== Creating Text Generator ===" << std::endl;
    // Create text generator - 暂时禁用
    // text_generator_ = TextGeneratorFactory::create(model_, ctx_);
    // if (!text_generator_) {
    //   std::cerr << "❌ Failed to create text generator" << std::endl;
    //   llama_free(ctx_);
    //   llama_model_free(model_);
    //   ctx_ = nullptr;
    //   model_ = nullptr;
    //   return false;
    // }
    // std::cout << "✅ Text generator created successfully" << std::endl;

    loaded_ = true;

    std::cout << "\n🎉 === Model Loading Complete ===" << std::endl;
    std::cout << "✅ Model is ready for text generation!" << std::endl;
    std::cout << "📊 Model memory usage: " << (getMemoryUsage() / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "=== End Model Loading ===\n" << std::endl;
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

  bool isLoaded() const override { return loaded_; }

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
  GenerationResult
  generate(const std::string &prompt,
           const GenerationParams &params = GenerationParams()) {
    if (!loaded_ || !text_generator_) {
      GenerationResult result;
      result.stop_reason = "Model not loaded";
      result.finished = true;
      return result;
    }

    return text_generator_->generate(prompt, params);
  }

  // GenerationResult
  // generateStream(const std::string &prompt, StreamCallback callback,
  //                const GenerationParams &params = GenerationParams()) {
  //   if (!loaded_ || !text_generator_) {
  //     GenerationResult result;
  //     result.stop_reason = "Model not loaded";
  //     result.finished = true;
  //     return result;
  //   }
  //
  //   return text_generator_->generateStream(prompt, callback, params);
  // }  // 暂时禁用

  TextGenerator *getTextGenerator() const { return text_generator_.get(); }

private:
  std::string model_path_;
  bool loaded_;
  llama_model *model_;
  llama_context *ctx_;
  std::unique_ptr<TextGenerator> text_generator_;
};
*/

// StableDiffusionModel implementation using stable-diffusion.cpp
class StableDiffusionModel : public BaseModel {
public:
  StableDiffusionModel(const std::string &path)
      : model_path_(path), loaded_(false), sd_ctx_(nullptr),
        image_generator_(nullptr) {}

  ~StableDiffusionModel() { unload(); }

  bool load(const std::string &model_path) override {
    if (loaded_) {
      return true;
    }

    std::cout << "Loading stable diffusion model from: " << model_path
              << std::endl;

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

  bool isLoaded() const override { return loaded_; }

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
  ImageGenerationResult
  generateImage(const std::string &prompt,
                const ImageGenerationParams &params = ImageGenerationParams()) {
    if (!loaded_ || !image_generator_) {
      ImageGenerationResult result;
      result.error_message = "Model not loaded";
      return result;
    }

    return image_generator_->textToImage(prompt, params);
  }

  ImageGenerationResult generateImageWithProgress(
      const std::string &prompt, ProgressCallback callback,
      const ImageGenerationParams &params = ImageGenerationParams()) {
    if (!loaded_ || !image_generator_) {
      ImageGenerationResult result;
      result.error_message = "Model not loaded";
      return result;
    }

    return image_generator_->textToImageWithProgress(prompt, callback, params);
  }

  ImageGenerationResult
  imageToImage(const std::string &prompt,
               const std::vector<uint8_t> &input_image, int input_width,
               int input_height,
               const ImageGenerationParams &params = ImageGenerationParams()) {
    if (!loaded_ || !image_generator_) {
      ImageGenerationResult result;
      result.error_message = "Model not loaded";
      return result;
    }

    return image_generator_->imageToImage(prompt, input_image, input_width,
                                          input_height, params);
  }

  ImageGenerator *getImageGenerator() const { return image_generator_.get(); }

private:
  std::string model_path_;
  bool loaded_;
  sd_ctx_t *sd_ctx_;
  std::unique_ptr<ImageGenerator> image_generator_;
};

// ModelManager实现
ModelManager::ModelManager()
    : memory_limit_(4ULL * 1024 * 1024 * 1024) // 默认4GB内存限制
      ,
      initialized_(false), auto_memory_management_(false) {
  // 初始化模型下载器
  std::cout << "[DEBUG] Creating ModelDownloader..." << std::endl;
  model_downloader_ = ModelDownloaderFactory::create();
  if (model_downloader_) {
    std::cout << "[DEBUG] ModelDownloader created successfully" << std::endl;
  } else {
    std::cout << "[DEBUG] Failed to create ModelDownloader!" << std::endl;
  }
}

ModelManager::~ModelManager() {
  // 使用try_lock避免析构时的死锁
  std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
  
  if (lock.try_lock()) {
    // 成功获取锁，正常卸载所有模型
    for (auto &pair : loaded_models_) {
      try {
        pair.second->unload();
        updateModelStatus(pair.first, ModelStatus::NOT_LOADED);
      } catch (const std::exception& e) {
        std::cerr << "Error unloading model " << pair.first << ": " << e.what() << std::endl;
      }
    }
    loaded_models_.clear();
    std::cout << "All models unloaded in destructor" << std::endl;
  } else {
    // 无法获取锁，可能存在死锁风险，强制清理
    std::cerr << "Warning: Could not acquire lock in destructor, forcing cleanup" << std::endl;
    // 不加锁直接清理，避免死锁
    for (auto &pair : loaded_models_) {
      try {
        pair.second->unload();
      } catch (const std::exception& e) {
        std::cerr << "Error force-unloading model " << pair.first << ": " << e.what() << std::endl;
      }
    }
    loaded_models_.clear();
  }
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

  } catch (const std::exception &e) {
    std::cerr << "Failed to initialize ModelManager: " << e.what() << std::endl;
    return false;
  }
}

bool ModelManager::registerModel(const ModelInfo &model_info) {
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
  std::cout << "Model registered: " << model_info.id << " (" << model_info.name
            << ")" << std::endl;
  return true;
}

bool ModelManager::loadModel(const std::string &model_id) {
  if (!initialized_) {
    std::cerr << "ModelManager not initialized" << std::endl;
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // 检查模型是否已注册
  auto it = registered_models_.find(model_id);
  if (it == registered_models_.end()) {
    // 如果模型未注册，检查是否为Ollama模型
    if (model_downloader_) {
      auto local_models = model_downloader_->getLocalModels();
      bool is_ollama_model = std::find(local_models.begin(), local_models.end(),
                                       model_id) != local_models.end();

      if (is_ollama_model) {
        std::cout << "[DEBUG] Dynamically registering Ollama model: "
                  << model_id << std::endl;
        // 动态注册Ollama模型
        ModelInfo ollama_model;
        ollama_model.id = model_id;
        ollama_model.name = model_id;
        ollama_model.type = ModelType::LANGUAGE_MODEL;
        ollama_model.path = model_id; // 对于Ollama模型，path就是model_id
        ollama_model.memory_usage = 0;
        ollama_model.status = ModelStatus::NOT_LOADED;
        ollama_model.description = "Ollama model: " + model_id;

        registered_models_[model_id] = ollama_model;
        it = registered_models_.find(model_id);
      } else {
        std::cerr << "Model not registered: " << model_id << std::endl;
        return false;
      }
    } else {
      std::cerr << "Model not registered: " << model_id << std::endl;
      return false;
    }
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
  std::cout << "[DEBUG] Creating model instance for: " << model_id << std::endl;
  auto model = createModel(it->second);
  if (!model) {
    std::cerr << "[ERROR] Failed to create model instance: " << model_id 
              << " (type: " << static_cast<int>(it->second.type) << ")" << std::endl;
    updateModelStatus(model_id, ModelStatus::ERROR);
    return false;
  }

  // 更新状态为加载中
  std::cout << "[DEBUG] Starting model load for: " << model_id << std::endl;
  updateModelStatus(model_id, ModelStatus::LOADING);

  // 记录开始时间
  auto start_time = std::chrono::steady_clock::now();
  
  // 加载模型（带超时处理）
  bool success = false;
  std::string error_message;
  
  try {
     // 直接同步加载，但添加详细的错误信息
     // 注意：异步加载在这个上下文中可能不安全，因为涉及到共享状态
     success = model->load(it->second.path);
   } catch (const std::exception& e) {
    error_message = "Exception during model loading: " + std::string(e.what());
    std::cerr << "[ERROR] " << error_message << " for model: " << model_id << std::endl;
  } catch (...) {
    error_message = "Unknown exception during model loading";
    std::cerr << "[ERROR] " << error_message << " for model: " << model_id << std::endl;
  }
  
  // 计算加载时间
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  if (success) {
    loaded_models_[model_id] = model;
    updateModelStatus(model_id, ModelStatus::LOADED);

    // 调用回调函数
    if (load_callback_) {
      load_callback_(model_id, true);
    }

    std::cout << "[SUCCESS] Model loaded successfully: " << model_id 
              << " (took " << duration.count() << "ms)" << std::endl;
  } else {
    updateModelStatus(model_id, ModelStatus::ERROR);

    // 调用回调函数
    if (load_callback_) {
      load_callback_(model_id, false);
    }

    std::cerr << "[ERROR] Failed to load model: " << model_id 
              << " (took " << duration.count() << "ms)";
    if (!error_message.empty()) {
      std::cerr << " - " << error_message;
    }
    std::cerr << std::endl;
    
    // 记录详细的错误信息
    std::cerr << "[DEBUG] Model details - Path: " << it->second.path 
              << ", Type: " << static_cast<int>(it->second.type)
              << ", Memory limit: " << memory_limit_ / (1024*1024) << "MB" << std::endl;
  }

  return success;
}

bool ModelManager::unloadModel(const std::string &model_id) {
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

  for (auto &pair : loaded_models_) {
    try {
      pair.second->unload();
      updateModelStatus(pair.first, ModelStatus::NOT_LOADED);
    } catch (const std::exception& e) {
      std::cerr << "Error unloading model " << pair.first << ": " << e.what() << std::endl;
    }
  }

  loaded_models_.clear();
  std::cout << "All models unloaded" << std::endl;
}

std::shared_ptr<BaseModel>
ModelManager::getModel(const std::string &model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = loaded_models_.find(model_id);
  if (it != loaded_models_.end()) {
    return it->second;
  }

  return nullptr;
}

bool ModelManager::isModelLoaded(const std::string &model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return loaded_models_.find(model_id) != loaded_models_.end();
}

ModelInfo ModelManager::getModelInfo(const std::string &model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = registered_models_.find(model_id);
  if (it != registered_models_.end()) {
    return it->second;
  }

  return ModelInfo(); // 返回空的ModelInfo
}

std::vector<ModelInfo> ModelManager::getAllModels() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::cout << "[DEBUG] ModelManager::getAllModels() called" << std::endl;

  std::vector<ModelInfo> models;

  // 添加已注册的模型
  std::cout << "[DEBUG] Registered models count: " << registered_models_.size()
            << std::endl;
  for (const auto &pair : registered_models_) {
    models.push_back(pair.second);
  }

  // 添加ollama模型
  if (model_downloader_) {
    std::cout << "[DEBUG] ModelDownloader exists, getting local models..."
              << std::endl;
    try {
      auto local_models = model_downloader_->getLocalModels();
      std::cout << "[DEBUG] Found " << local_models.size()
                << " local ollama models" << std::endl;
      for (const auto &model_name : local_models) {
        std::cout << "[DEBUG] Processing ollama model: " << model_name << std::endl;
        ModelInfo ollama_model;
        ollama_model.id = model_name;
        ollama_model.name = model_name;
        ollama_model.type = ModelType::LANGUAGE_MODEL;
        ollama_model.path = "";
        ollama_model.memory_usage = 0;
        ollama_model.status = ModelStatus::NOT_LOADED;
        ollama_model.description = "Ollama model: " + model_name;
        models.push_back(ollama_model);
        std::cout << "[DEBUG] Added ollama model: " << model_name << std::endl;
      }
      std::cout << "[DEBUG] Finished processing all ollama models" << std::endl;
    } catch (const std::exception& e) {
      std::cout << "[DEBUG] Exception in getLocalModels: " << e.what() << std::endl;
    }
  } else {
    std::cout << "[DEBUG] ModelDownloader is null!" << std::endl;
  }

  // 如果没有任何模型，返回一些默认的示例模型
  if (models.empty()) {
    ModelInfo llama_example;
    llama_example.id = "llama-7b-example";
    llama_example.name = "LLaMA 7B (Example)";
    llama_example.type = ModelType::LANGUAGE_MODEL;
    llama_example.path = "";
    llama_example.memory_usage = 0;
    llama_example.status = ModelStatus::NOT_LOADED;
    llama_example.description = "Example LLaMA model - download required";
    models.push_back(llama_example);

    ModelInfo gpt_example;
    gpt_example.id = "gpt-3.5-turbo";
    gpt_example.name = "GPT-3.5 Turbo";
    gpt_example.type = ModelType::LANGUAGE_MODEL;
    gpt_example.path = "";
    gpt_example.memory_usage = 0;
    gpt_example.status = ModelStatus::NOT_LOADED;
    gpt_example.description = "OpenAI GPT-3.5 Turbo model";
    models.push_back(gpt_example);

    ModelInfo claude_example;
    claude_example.id = "claude-3-sonnet";
    claude_example.name = "Claude 3 Sonnet";
    claude_example.type = ModelType::LANGUAGE_MODEL;
    claude_example.path = "";
    claude_example.memory_usage = 0;
    claude_example.status = ModelStatus::NOT_LOADED;
    claude_example.description = "Anthropic Claude 3 Sonnet model";
    models.push_back(claude_example);

    ModelInfo llama2_example;
    llama2_example.id = "llama2";
    llama2_example.name = "Llama2";
    llama2_example.type = ModelType::LANGUAGE_MODEL;
    llama2_example.path = "";
    llama2_example.memory_usage = 0;
    llama2_example.status = ModelStatus::NOT_LOADED;
    llama2_example.description = "Meta Llama 2 model";
    models.push_back(llama2_example);
  }

  return models;
}

std::vector<std::string> ModelManager::getLoadedModels() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<std::string> loaded_ids;
  for (const auto &pair : loaded_models_) {
    loaded_ids.push_back(pair.first);
  }

  return loaded_ids;
}

size_t ModelManager::getTotalMemoryUsage() const {
  std::lock_guard<std::mutex> lock(mutex_);

  size_t total = 0;
  for (const auto &pair : loaded_models_) {
    total += pair.second->getMemoryUsage();
  }

  return total;
}

void ModelManager::setMemoryLimit(size_t limit_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  memory_limit_ = limit_bytes;
  std::cout << "Memory limit set to: " << (limit_bytes / 1024 / 1024) << " MB"
            << std::endl;
}

size_t ModelManager::getMemoryLimit() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return memory_limit_;
}

bool ModelManager::hasEnoughMemory(const std::string &model_id) const {
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

void ModelManager::setLoadCallback(
    std::function<void(const std::string &, bool)> callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  load_callback_ = callback;
}

TextGenerator *
ModelManager::getTextGenerator(const std::string &model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = loaded_models_.find(model_id);
  if (it == loaded_models_.end()) {
    return nullptr;
  }

  // Try to cast to LlamaModel - DISABLED due to missing llama.h
  /*
  auto llama_model = std::dynamic_pointer_cast<LlamaModel>(it->second);
  if (llama_model) {
    return llama_model->getTextGenerator();
  }
  */

  return nullptr;
}

ImageGenerator *
ModelManager::getImageGenerator(const std::string &model_id) const {
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
    std::vector<std::pair<std::string, std::chrono::steady_clock::time_point>>
        model_usage;

    for (const auto &pair : loaded_models_) {
      // 这里可以添加模型最后使用时间的跟踪
      // 暂时使用当前时间作为占位符
      model_usage.emplace_back(pair.first, std::chrono::steady_clock::now());
    }

    // 按使用时间排序（最久未使用的在前）
    std::sort(model_usage.begin(), model_usage.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });

    // 卸载模型直到内存使用降到限制的60%以下
    for (const auto &pair : model_usage) {
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

std::future<DownloadResult>
ModelManager::downloadModel(const std::string &model_name,
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

DownloadResult
ModelManager::downloadModelSync(const std::string &model_name,
                                DownloadProgressCallback progress_callback) {
  if (model_downloader_) {
    model_downloader_->setProgressCallback(progress_callback);
    return model_downloader_->downloadModelSync(model_name);
  }

  DownloadResult result;
  result.error_message = "Model downloader not initialized";
  return result;
}

ModelInfo ModelManager::getModelInfo(const std::string &model_name) {
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

bool ModelManager::isModelDownloaded(const std::string &model_name) {
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

bool ModelManager::deleteLocalModel(const std::string &model_name) {
  if (model_downloader_) {
    return model_downloader_->deleteModel(model_name);
  }

  return false;
}

bool ModelManager::verifyModel(const std::string &model_name) {
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

std::shared_ptr<BaseModel>
ModelManager::createModel(const ModelInfo &model_info) {
  if (model_info.type == ModelType::LANGUAGE_MODEL) {
    // 检查是否为Ollama模型
    if (model_downloader_ && model_downloader_->isOllamaModel(model_info.name)) {
      Logger logger;
      logger.info("Creating Ollama model for: " + model_info.name);
      return std::make_shared<OllamaModelImpl>(model_info.name);
    } else {
      // return std::make_shared<LlamaModel>(model_info.path);  // Disabled - llama.h not found
      Logger logger;
      logger.warning("LlamaModel creation disabled - llama.h not found");
      return nullptr;
    }
  } else if (model_info.type == ModelType::DIFFUSION_MODEL) {
    return std::make_shared<StableDiffusionModel>(model_info.path);
  }
  
  Logger logger;
  logger.error("Unknown model type");
  return nullptr;
}

void ModelManager::updateModelStatus(const std::string &model_id,
                                     ModelStatus status) {
  auto it = registered_models_.find(model_id);
  if (it != registered_models_.end()) {
    it->second.status = status;
  }
}

void ModelManager::scanModelDirectory(const std::string &directory) {
  try {
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(directory)) {
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
          std::cout << "Auto-discovered model: " << info.id << " at " << path
                    << std::endl;
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Error scanning model directory: " << e.what() << std::endl;
  }
}

} // namespace core
} // namespace duorou