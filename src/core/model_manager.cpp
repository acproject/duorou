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
#include "../extensions/ollama/ollama_model_manager.h"
// Removed llama.h dependency - using new ollama extension architecture
#include "image_generator.h"
#include "model_downloader.h"
#include "model_path_manager.h"
#include "stable-diffusion.h"
#include "text_generator.h"

// OllamaTextGenerator adapter class removed - now using integrated
// TextGenerator with Ollama support

// Simple Ollama model implementation using extensions
class OllamaModelImpl : public duorou::core::BaseModel {
public:
  explicit OllamaModelImpl(const std::string &model_path)
      : model_path_(model_path), loaded_(false), memory_usage_(0) {
    model_info_.name = model_path;
    model_info_.type = duorou::core::ModelType::LANGUAGE_MODEL;
    model_info_.status = duorou::core::ModelStatus::NOT_LOADED;
    model_info_.memory_usage = 0;
    model_info_.path = model_path;

    // Use global OllamaModelManager instance
    // model_manager_ will be obtained through GlobalModelManager in load() method
  }

  bool load(const std::string &model_path) override {
    duorou::core::Logger logger;
  logger.info("[OLLAMA] Loading model: " + model_path);

    // Use new ollama extension architecture with global manager
    // First register the model by name (supports Ollama model names), then load it
    // Detect whether input is an Ollama model name or a direct GGUF file path

    // Get global OllamaModelManager instance
    auto& global_manager = duorou::extensions::ollama::GlobalModelManager::getInstance();
    bool registered = false;
    std::string normalized_id;

    // If it's a GGUF file path, register directly with file path
    try {
      std::filesystem::path p(model_path);
      std::string ext = p.extension().string();
      std::string filename = p.filename().string();
      bool is_mmproj = filename.find("mmproj") != std::string::npos;
      if (!ext.empty() && ext == ".gguf" && std::filesystem::exists(p) && !is_mmproj) {
        // Generate a friendly model id from file name, then normalize
        std::string base_id = std::filesystem::path(model_path).stem().string();
        normalized_id = global_manager.normalizeModelId(base_id);
        registered = global_manager.registerModel(normalized_id, model_path);
        if (!registered) {
          std::cerr << "[ERROR] Failed to register GGUF model: " << model_path << std::endl;
          return false;
        }
      } else {
        // Treat as Ollama model name and register by name
        registered = global_manager.registerModelByName(model_path);
        if (!registered) {
          std::cerr << "[ERROR] Failed to register Ollama model: " << model_path
                    << std::endl;
          return false;
        }
        // Normalize the id based on model name
        normalized_id = global_manager.normalizeModelId(model_path);
      }
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] Exception while registering model: " << e.what() << std::endl;
      return false;
    }
    
    bool success = global_manager.loadModel(normalized_id);
    if (!success) {
      std::cerr << "[ERROR] Failed to load Ollama model: " << normalized_id
                << std::endl;
      return false;
    }

    // Store normalized id for future unload and generation
    model_id_ = normalized_id;
    loaded_ = true;
    memory_usage_ = 1024 * 1024 * 1024; // 1GB estimate
    model_info_.status = duorou::core::ModelStatus::LOADED;
    model_info_.memory_usage = memory_usage_;

    // Create text generator using global manager
     // Create a shared_ptr that won't delete the global manager
     auto shared_manager = std::shared_ptr<duorou::extensions::ollama::OllamaModelManager>(
         &global_manager, [](duorou::extensions::ollama::OllamaModelManager*) {
           // Empty deleter, because global manager is managed by GlobalModelManager
         });
     text_generator_ = std::make_unique<duorou::core::TextGenerator>(
         shared_manager, model_id_);

    return true;
  }

  void unload() override {
    if (!model_id_.empty()) {
      try {
        auto& global_manager = duorou::extensions::ollama::GlobalModelManager::getInstance();
        global_manager.unloadModel(model_id_);
      } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to unload model from global manager: " << e.what() << std::endl;
      }
    }
    text_generator_.reset();
    loaded_ = false;
    memory_usage_ = 0;
    model_info_.status = duorou::core::ModelStatus::NOT_LOADED;
    model_info_.memory_usage = 0;
  }

  bool isLoaded() const override { return loaded_; }
  duorou::core::ModelManagerInfo getInfo() const override { return model_info_; }
  size_t getMemoryUsage() const override { return memory_usage_; }

  // Get the model manager for text generation
  duorou::extensions::ollama::OllamaModelManager *getModelManager() const {
    return &duorou::extensions::ollama::GlobalModelManager::getInstance();
  }

  // Get text generator for this model
  duorou::core::TextGenerator *getTextGenerator() const {
    if (!loaded_ || !text_generator_) {
      return nullptr;
    }
    return text_generator_.get();
  }

private:
  std::string model_path_;
  std::string model_id_;
  bool loaded_;
  size_t memory_usage_;
  mutable std::unique_ptr<duorou::core::TextGenerator> text_generator_;
  duorou::core::ModelManagerInfo model_info_;
};

namespace duorou {
namespace core {

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
    params.vae_decode_only = false;
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

  ModelManagerInfo getInfo() const override {
        ModelManagerInfo info;
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

// ModelManager implementation
ModelManager::ModelManager()
    : memory_limit_(4ULL * 1024 * 1024 * 1024) // Default 4GB memory limit
      ,
      initialized_(false), auto_memory_management_(false) {
  // Initialize model downloader
  model_downloader_ = ModelDownloaderFactory::create();
  // ModelDownloader creation handled without verbose logging
}

ModelManager::~ModelManager() {
  // Use try_lock to avoid deadlock during destruction
  std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);

  if (lock.try_lock()) {
    // Successfully acquired lock, normally unload all models
    for (auto &pair : loaded_models_) {
      try {
        pair.second->unload();
        updateModelStatus(pair.first, duorou::core::ModelStatus::NOT_LOADED);
      } catch (const std::exception &e) {
        std::cerr << "Error unloading model " << pair.first << ": " << e.what()
                  << std::endl;
      }
    }
    loaded_models_.clear();
    std::cout << "All models unloaded in destructor" << std::endl;
  } else {
    // Cannot acquire lock, possible deadlock risk, force cleanup
    std::cerr
        << "Warning: Could not acquire lock in destructor, forcing cleanup"
        << std::endl;
    // Clean up directly without lock to avoid deadlock
    for (auto &pair : loaded_models_) {
      try {
        pair.second->unload();
      } catch (const std::exception &e) {
        std::cerr << "Error force-unloading model " << pair.first << ": "
                  << e.what() << std::endl;
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
    // Scan default model directory
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

bool ModelManager::registerModel(const ModelManagerInfo &model_info) {
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

  // Check if model is already registered
  auto it = registered_models_.find(model_id);
  if (it == registered_models_.end()) {
    // If model is not registered, check if it's an Ollama model
    if (model_downloader_) {
      auto local_models = model_downloader_->getLocalModels();
      bool is_ollama_model = std::find(local_models.begin(), local_models.end(),
                                       model_id) != local_models.end();

      if (is_ollama_model) {
        // Dynamically registering Ollama model
        // Dynamically register Ollama model
        duorou::core::ModelManagerInfo ollama_model;
        ollama_model.id = model_id;
        ollama_model.name = model_id;
        ollama_model.type = duorou::core::ModelType::LANGUAGE_MODEL;
        ollama_model.path = model_id; // For Ollama models, path is the model_id
        ollama_model.memory_usage = 0;
        ollama_model.status = duorou::core::ModelStatus::NOT_LOADED;
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

  // Check if model is already loaded
  if (loaded_models_.find(model_id) != loaded_models_.end()) {
    std::cout << "Model already loaded: " << model_id << std::endl;
    return true;
  }

  // Check memory limits
  // Checking memory availability for model
  if (!hasEnoughMemory(model_id)) {
    std::cerr << "Not enough memory to load model: " << model_id << std::endl;
    return false;
  }
  // Memory check passed for model

  // Create model instance
  // Creating model instance
  auto model = createModel(it->second);
  if (!model) {
    std::cerr << "[ERROR] Failed to create model instance: " << model_id
              << " (type: " << static_cast<int>(it->second.type) << ")"
              << std::endl;
    updateModelStatus(model_id, duorou::core::ModelStatus::LOAD_ERROR);
    return false;
  }

  // Update status to loading
  // Starting model load
  updateModelStatus(model_id, duorou::core::ModelStatus::LOADING);

  // Record start time
  auto start_time = std::chrono::steady_clock::now();

  // Load model (with timeout handling)
  bool success = false;
  std::string error_message;

  try {
    // Direct synchronous loading, but add detailed error information
    // Note: Asynchronous loading may not be safe in this context due to shared state
    success = model->load(it->second.path);
  } catch (const std::exception &e) {
    error_message = "Exception during model loading: " + std::string(e.what());
    std::cerr << "[ERROR] " << error_message << " for model: " << model_id
              << std::endl;
  } catch (...) {
    error_message = "Unknown exception during model loading";
    std::cerr << "[ERROR] " << error_message << " for model: " << model_id
              << std::endl;
  }

  // Calculate loading time
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  if (success) {
    loaded_models_[model_id] = model;
    updateModelStatus(model_id, duorou::core::ModelStatus::LOADED);

    // Call callback function
    if (load_callback_) {
      load_callback_(model_id, true);
    }

    std::cout << "[SUCCESS] Model loaded successfully: " << model_id
              << " (took " << duration.count() << "ms)" << std::endl;
  } else {
    updateModelStatus(model_id, duorou::core::ModelStatus::LOAD_ERROR);

    // Call callback function
    if (load_callback_) {
      load_callback_(model_id, false);
    }

    std::cerr << "[ERROR] Failed to load model: " << model_id << " (took "
              << duration.count() << "ms)";
    if (!error_message.empty()) {
      std::cerr << " - " << error_message;
    }
    std::cerr << std::endl;

    // Log detailed error information
    // Model details logged at error above; suppress extra debug info
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

  // Unload model
  it->second->unload();
  loaded_models_.erase(it);

  // Update status
  updateModelStatus(model_id, duorou::core::ModelStatus::NOT_LOADED);

  std::cout << "Model unloaded: " << model_id << std::endl;
  return true;
}

void ModelManager::unloadAllModels() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto &pair : loaded_models_) {
    try {
      pair.second->unload();
      updateModelStatus(pair.first, duorou::core::ModelStatus::NOT_LOADED);
    } catch (const std::exception &e) {
      std::cerr << "Error unloading model " << pair.first << ": " << e.what()
                << std::endl;
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

ModelManagerInfo ModelManager::getModelInfo(const std::string &model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = registered_models_.find(model_id);
  if (it != registered_models_.end()) {
    return it->second;
  }

  return ModelManagerInfo(); // Return empty ModelManagerInfo
}

std::vector<ModelManagerInfo> ModelManager::getAllModels() const {
  std::lock_guard<std::mutex> lock(mutex_);


  std::vector<ModelManagerInfo> models;

  // Add registered models
  
  for (const auto &pair : registered_models_) {
    ModelManagerInfo info = pair.second;
    // 检查是否已有 mmproj 关联（只读，不写缓存，以避免在 const 方法中修改状态）
    bool has_mmproj = (mmproj_paths_.find(pair.first) != mmproj_paths_.end());
    if (!has_mmproj) {
      // 回退：在同目录尝试发现 mmproj（不更新缓存）
      try {
        if (!info.path.empty()) {
          std::filesystem::path mp(info.path);
          std::filesystem::path dir = mp.parent_path();
          if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
            for (const auto &entry : std::filesystem::directory_iterator(dir)) {
              if (!entry.is_regular_file()) continue;
              if (entry.path().extension() != ".gguf") continue;
              std::string fname = entry.path().filename().string();
              if (fname.rfind("mmproj-", 0) == 0 || fname.find("-mmproj-") != std::string::npos) {
                std::string stem = info.name;
                if (!stem.empty() && fname.find(stem) != std::string::npos) {
                  has_mmproj = true;
                  break;
                }
              }
            }
          }
        }
      } catch (...) {}
    }
    if (has_mmproj) {
      if (info.description.empty()) {
        info.description = "mmproj detected";
      } else {
        info.description += " | mmproj detected";
      }
    }
    models.push_back(info);
  }

  // Add ollama models
  if (model_downloader_) {
    
    try {
      auto local_models = model_downloader_->getLocalModels();
      
      for (const auto &model_name : local_models) {
        
        ModelManagerInfo ollama_model;
        ollama_model.id = model_name;
        ollama_model.name = model_name;
        ollama_model.type = duorou::core::ModelType::LANGUAGE_MODEL;
        ollama_model.path = "";
        ollama_model.memory_usage = 0;
        ollama_model.status = duorou::core::ModelStatus::NOT_LOADED;
        ollama_model.description = "Ollama model: " + model_name;
        // 尝试以注册时的ID格式检查 mmproj（扫描阶段为 llm_<stem>），只读检查
        std::string registered_id = std::string("llm_") + model_name;
        if (mmproj_paths_.find(registered_id) != mmproj_paths_.end()) {
          ollama_model.description += " | mmproj detected";
        } else {
          // 如果已注册模型存在，回退扫描同目录（不更新缓存）
          auto mit = registered_models_.find(registered_id);
          if (mit != registered_models_.end()) {
            const auto &info = mit->second;
            try {
              if (!info.path.empty()) {
                std::filesystem::path mp(info.path);
                std::filesystem::path dir = mp.parent_path();
                if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
                  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
                    if (!entry.is_regular_file()) continue;
                    if (entry.path().extension() != ".gguf") continue;
                    std::string fname = entry.path().filename().string();
                    if (fname.rfind("mmproj-", 0) == 0 || fname.find("-mmproj-") != std::string::npos) {
                      std::string stem = info.name;
                      if (!stem.empty() && fname.find(stem) != std::string::npos) {
                        ollama_model.description += " | mmproj detected";
                        break;
                      }
                    }
                  }
                }
              }
            } catch (...) {}
          }
        }
        models.push_back(ollama_model);
        
      }
      
    } catch (const std::exception &e) {
      
    }
  } else {
    
  }

  // If no models exist, return some default example models
  if (models.empty()) {
    duorou::core::ModelManagerInfo llama_example;
    llama_example.id = "llama-7b-example";
    llama_example.name = "LLaMA 7B (Example)";
    llama_example.type = duorou::core::ModelType::LANGUAGE_MODEL;
    llama_example.path = "";
    llama_example.memory_usage = 0;
    llama_example.status = duorou::core::ModelStatus::NOT_LOADED;
    llama_example.description = "Example LLaMA model - download required";
    models.push_back(llama_example);

    duorou::core::ModelManagerInfo gpt_example;
    gpt_example.id = "gpt-3.5-turbo";
    gpt_example.name = "GPT-3.5 Turbo";
    gpt_example.type = duorou::core::ModelType::LANGUAGE_MODEL;
    gpt_example.path = "";
    gpt_example.memory_usage = 0;
    gpt_example.status = duorou::core::ModelStatus::NOT_LOADED;
    gpt_example.description = "OpenAI GPT-3.5 Turbo model";
    models.push_back(gpt_example);

    duorou::core::ModelManagerInfo claude_example;
    claude_example.id = "claude-3-sonnet";
    claude_example.name = "Claude 3 Sonnet";
    claude_example.type = duorou::core::ModelType::LANGUAGE_MODEL;
    claude_example.path = "";
    claude_example.memory_usage = 0;
    claude_example.status = duorou::core::ModelStatus::NOT_LOADED;
    claude_example.description = "Anthropic Claude 3 Sonnet model";
    models.push_back(claude_example);

    duorou::core::ModelManagerInfo llama2_example;
    llama2_example.id = "llama2";
    llama2_example.name = "Llama2";
    llama2_example.type = duorou::core::ModelType::LANGUAGE_MODEL;
    llama2_example.path = "";
    llama2_example.memory_usage = 0;
    llama2_example.status = duorou::core::ModelStatus::NOT_LOADED;
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

  // Estimate model memory usage (using simple estimation here)
  size_t estimated_usage = 512 * 1024 * 1024; // Default 512MB
  if (it->second.type == ModelType::DIFFUSION_MODEL) {
    estimated_usage = 1024 * 1024 * 1024; // Diffusion model 1GB
  }
  

  // Calculate current memory usage directly to avoid deadlock (don't call getTotalMemoryUsage)
  
  size_t current_usage = 0;
  for (const auto &pair : loaded_models_) {
    current_usage += pair.second->getMemoryUsage();
  }
  
  bool result = (current_usage + estimated_usage) <= memory_limit_;
  
  return result;
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

  // Try to cast to OllamaModelImpl first
  auto ollama_model = std::dynamic_pointer_cast<OllamaModelImpl>(it->second);
  if (ollama_model) {
    auto text_generator = ollama_model->getTextGenerator();
    return text_generator;
  }
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

  // Get current memory usage
  size_t current_usage = getTotalMemoryUsage();

  // If memory usage exceeds 80% of the limit, start optimization
  if (current_usage > memory_limit_ * 0.8) {
    // Sort by last usage time, unload the least recently used models
    std::vector<std::pair<std::string, std::chrono::steady_clock::time_point>>
        model_usage;

    for (const auto &pair : loaded_models_) {
      // Model last usage time tracking can be added here
      // Using current time as placeholder for now
      model_usage.emplace_back(pair.first, std::chrono::steady_clock::now());
    }

    // Sort by usage time (least recently used first)
    std::sort(model_usage.begin(), model_usage.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });

    // Unload models until memory usage drops below 60% of the limit
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
    // Immediately perform memory optimization
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

  // Return failed Future
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

ModelManagerInfo ModelManager::getModelInfo(const std::string &model_name) {
  if (model_downloader_) {
    // Get downloader's ModelInfo (duorou namespace)
    auto downloader_info = model_downloader_->getModelInfo(model_name);

    // Convert to ModelManager's ModelManagerInfo (duorou::core namespace)
    ModelManagerInfo manager_info;
    manager_info.id = downloader_info.name;
    manager_info.name = downloader_info.name;
    manager_info.description = downloader_info.description;
    manager_info.type = ModelType::LANGUAGE_MODEL; // Default to language model
    manager_info.status = ModelStatus::NOT_LOADED;
    manager_info.memory_usage = 0;

    return manager_info;
  }

  return ModelManagerInfo{};
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

void ModelManager::setOllamaModelsPath(const std::string &path) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!model_downloader_) {
    model_downloader_ = ModelDownloaderFactory::create();
  }

  try {
    model_downloader_->setModelDirectory(path);
  } catch (const std::exception &e) {
    std::cerr << "Failed to set Ollama models path: " << e.what()
              << std::endl;
  }
}

void ModelManager::rescanModelDirectory(const std::string &directory) {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    if (std::filesystem::exists(directory)) {
      scanModelDirectory(directory);
    } else {
      std::cerr << "Rescan skipped: directory not found: " << directory
                << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during rescan: " << e.what() << std::endl;
  }
}

std::shared_ptr<BaseModel>
ModelManager::createModel(const ModelManagerInfo &model_info) {
  

  if (model_info.type == ModelType::LANGUAGE_MODEL) {
    // Check if it's an Ollama model
    

    bool isOllama = false;
    if (model_downloader_) {
      isOllama = model_downloader_->isOllamaModel(model_info.name);
      
    }

    // Also check if model name contains ollama characteristics
    bool hasOllamaPattern =
        model_info.name.find("registry.ollama.ai") != std::string::npos ||
        model_info.name.find("ollama") != std::string::npos;
    
    // Additionally, support direct .gguf file path
    bool isGGUFPath = false;
    bool is_mmproj = false;
    if (!model_info.path.empty()) {
      std::filesystem::path p(model_info.path);
      isGGUFPath = (p.extension().string() == ".gguf");
      is_mmproj = p.filename().string().find("mmproj") != std::string::npos;
    }

    if ((model_downloader_ && isOllama) || hasOllamaPattern || (isGGUFPath && !is_mmproj)) {
      
      duorou::core::Logger logger;
    logger.info("Creating Ollama model for: " + model_info.name);
      // For GGUF direct path, pass the path; otherwise pass the name
      auto model = std::make_shared<OllamaModelImpl>(isGGUFPath ? model_info.path : model_info.name);
      
      return model;
    } else {
      // return std::make_shared<LlamaModel>(model_info.path);  // Disabled -
      // llama.h not found
      
      duorou::core::Logger logger;
    logger.warning("LlamaModel creation disabled - llama.h not found");
      return nullptr;
    }
  } else if (model_info.type == ModelType::DIFFUSION_MODEL) {
    
    return std::make_shared<StableDiffusionModel>(model_info.path);
  }

  
  duorou::core::Logger logger;
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

        // Check file extension to determine model type
        ModelManagerInfo info;
        info.path = path;
        info.name = entry.path().stem().string();
        info.status = ModelStatus::NOT_LOADED;

        // Register diffusion and local GGUF language models here
        if (extension == ".safetensors" || extension == ".ckpt") {
          // Stable Diffusion model file
          info.id = "sd_" + info.name;
          info.type = ModelType::DIFFUSION_MODEL;
          info.description = "Diffusion model (Stable Diffusion)";
        } else if (extension == ".gguf") {
          // 处理 .gguf：既包含 LLM，也可能是 mmproj 投影权重
          std::string filename = entry.path().filename().string();
          bool is_mmproj = (filename.find("mmproj-") == 0) || (filename.find("-mmproj-") != std::string::npos);

          if (is_mmproj) {
            // 解析 mmproj 关联的 LLM 名称：
            // 约定文件名形如：mmproj-<stem>.gguf 或 mmproj-<stem>-<dtype>.gguf
            std::string stem = info.name; // 带有前缀的文件名去掉扩展
            // 去除前缀 "mmproj-"
            if (stem.rfind("mmproj-", 0) == 0) {
              stem = stem.substr(std::string("mmproj-").size());
            } else {
              // 尝试从中间形态提取，如：qwen3-vl-4b-instruct-mmproj-f16
              // 简化处理：去掉 "mmproj-" 及其前缀部分
              auto pos = stem.find("mmproj-");
              if (pos != std::string::npos) {
                // 取 mmproj- 前面的部分作为 LLM 名称基干
                stem = stem.substr(0, pos - 1); // 去掉连接符
              }
            }

            std::string llm_model_id = "llm_" + stem;
            // 注意：调用方已持有 mutex_，此处不要重复加锁，避免死锁
            mmproj_paths_[llm_model_id] = path;
            std::cout << "Associated mmproj found for " << llm_model_id << ": " << path << std::endl;
            // mmproj 不注册为模型条目
            continue;
          } else {
            // 本地 LLM GGUF 文件
            info.id = "llm_" + info.name;
            info.type = ModelType::LANGUAGE_MODEL;
            info.description = "Local GGUF language model";
          }
        } else {
          continue; // Skip unsupported file types
        }

        // Register model (no locking, as caller already locked)
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

std::optional<std::string> ModelManager::getAssociatedMmprojPath(const std::string &model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  // 1) 直接命中缓存（只读）
  if (auto it = mmproj_paths_.find(model_id); it != mmproj_paths_.end()) {
    return it->second;
  }

  // 2) 回退：基于已注册模型的路径在同级目录查找 mmproj gguf（不更新缓存）
  auto mit = registered_models_.find(model_id);
  if (mit == registered_models_.end()) {
    return std::nullopt;
  }

  const auto &info = mit->second;
  if (info.path.empty()) {
    return std::nullopt;
  }

  try {
    std::filesystem::path mp(info.path);
    std::filesystem::path dir = mp.parent_path();
    if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
      for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".gguf") continue;
        std::string fname = entry.path().filename().string();
        if (fname.rfind("mmproj-", 0) == 0 || fname.find("-mmproj-") != std::string::npos) {
          std::string stem = info.name;
          if (!stem.empty() && fname.find(stem) != std::string::npos) {
            std::string found = entry.path().string();
            std::cout << "Associated mmproj discovered via fallback for " << model_id << ": " << found << std::endl;
            return found;
          }
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Error locating mmproj via fallback: " << e.what() << std::endl;
  }

  return std::nullopt;
}

} // namespace core
} // namespace duorou