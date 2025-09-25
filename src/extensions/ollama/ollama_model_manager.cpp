#include "ollama_model_manager.h"
#include "inference_engine.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>

namespace duorou {
namespace extensions {
namespace ollama {

// OllamaModelManager实现
OllamaModelManager::OllamaModelManager(bool verbose)
    : verbose_(verbose), max_concurrent_models_(3), path_resolver_(verbose),
      total_memory_usage_(0), active_models_count_(0) {
  log("INFO", "OllamaModelManager initialized");
}

OllamaModelManager::~OllamaModelManager() {
  clearAllModels();
  log("INFO", "OllamaModelManager destroyed");
}

bool OllamaModelManager::registerModel(const std::string &model_id,
                                       const std::string &gguf_file_path) {
  if (!isValidModelId(model_id)) {
    log("ERROR", "Invalid model ID: " + model_id);
    return false;
  }

  if (!std::filesystem::exists(gguf_file_path)) {
    log("ERROR", "GGUF file does not exist: " + gguf_file_path);
    return false;
  }

  // 检查是否已经注册
  if (registered_models_.find(model_id) != registered_models_.end()) {
    log("WARNING", "Model already registered: " + model_id);
    return true;
  }

  // 解析模型信息
  ModelInfo model_info;
  model_info.model_id = model_id;
  model_info.file_path = gguf_file_path;

  std::cout << "[DEBUG] OllamaModelManager: About to call parseModelInfo for: "
            << gguf_file_path << std::endl;
  if (!parseModelInfo(gguf_file_path, model_info)) {
    std::cout << "[DEBUG] OllamaModelManager: parseModelInfo returned false"
              << std::endl;
    log("ERROR", "Failed to parse model info for: " + gguf_file_path);
    return false;
  }
  std::cout << "[DEBUG] OllamaModelManager: parseModelInfo succeeded"
            << std::endl;

  // 注册模型
  registered_models_[model_id] = model_info;
  model_states_[model_id] = ModelLoadState::UNLOADED;

  log("INFO",
      "Model registered: " + model_id + " (" + model_info.architecture + ")");
  return true;
}

bool OllamaModelManager::registerModelByName(const std::string &model_name) {
  std::cout << "[DEBUG] OllamaModelManager::registerModelByName called with: "
            << model_name << std::endl;
  log("INFO", "Registering model by name: " + model_name);

  // 使用 OllamaPathResolver 解析模型路径
  auto gguf_path = path_resolver_.resolveModelPath(model_name);
  std::cout << "[DEBUG] OllamaModelManager: Resolved GGUF path result: "
            << (gguf_path ? *gguf_path : "(null)") << std::endl;
  if (!gguf_path) {
    log("ERROR", "Failed to resolve model path for: " + model_name);
    return false;
  }

  // 生成有效的模型 ID（使用统一的转换函数）
  std::string model_id = normalizeModelId(model_name);

  std::cout << "[DEBUG] OllamaModelManager: Generated model ID: " << model_id
            << " for model: " << model_name << std::endl;
  log("DEBUG", "Generated model ID: " + model_id + " for model: " + model_name);

  // 调用原有的 registerModel 方法
  bool result = registerModel(model_id, *gguf_path);
  std::cout << "[DEBUG] OllamaModelManager: registerModel returned: "
            << (result ? "true" : "false") << std::endl;
  return result;
}

bool OllamaModelManager::loadModel(const std::string &model_id) {
  auto it = registered_models_.find(model_id);
  if (it == registered_models_.end()) {
    log("ERROR", "Model not registered: " + model_id);
    return false;
  }

  // 检查是否已经加载
  if (isModelLoaded(model_id)) {
    log("INFO", "Model already loaded: " + model_id);
    return true;
  }

  // 检查资源可用性
  if (!checkResourceAvailability()) {
    log("ERROR", "Insufficient resources to load model: " + model_id);
    return false;
  }

  // 设置加载状态
  model_states_[model_id] = ModelLoadState::LOADING;

  // 执行加载
  bool success = loadModelInternal(model_id);

  if (success) {
    model_states_[model_id] = ModelLoadState::LOADED;
    registered_models_[model_id].is_loaded = true;
    active_models_count_++;
    log("INFO", "Model loaded successfully: " + model_id);
  } else {
    model_states_[model_id] = ModelLoadState::LOAD_ERROR;
    log("ERROR", "Failed to load model: " + model_id);
  }

  return success;
}

bool OllamaModelManager::unloadModel(const std::string &model_id) {
  if (!isModelLoaded(model_id)) {
    log("WARNING", "Model not loaded: " + model_id);
    return true;
  }

  bool success = unloadModelInternal(model_id);

  if (success) {
    model_states_[model_id] = ModelLoadState::UNLOADED;
    registered_models_[model_id].is_loaded = false;
    active_models_count_--;
    log("INFO", "Model unloaded: " + model_id);
  } else {
    log("ERROR", "Failed to unload model: " + model_id);
  }

  return success;
}

bool OllamaModelManager::isModelLoaded(const std::string &model_id) const {
  // model_id should already be normalized when passed to this method
  auto state_it = model_states_.find(model_id);
  auto engine_it = inference_engines_.find(model_id);
  
  return state_it != model_states_.end() && 
         state_it->second == ModelLoadState::LOADED &&
         engine_it != inference_engines_.end();
}

std::vector<std::string> OllamaModelManager::getRegisteredModels() const {
  std::vector<std::string> models;
  models.reserve(registered_models_.size());

  for (const auto &pair : registered_models_) {
    models.push_back(pair.first);
  }

  return models;
}

std::vector<std::string> OllamaModelManager::getLoadedModels() const {
  std::vector<std::string> models;

  for (const auto &pair : model_states_) {
    if (pair.second == ModelLoadState::LOADED) {
      models.push_back(pair.first);
    }
  }

  return models;
}

const ModelInfo *
OllamaModelManager::getModelInfo(const std::string &model_id) const {
  std::string normalized_id = normalizeModelId(model_id);
  auto it = registered_models_.find(normalized_id);
  return (it != registered_models_.end()) ? &it->second : nullptr;
}

ModelLoadState
OllamaModelManager::getModelLoadState(const std::string &model_id) const {
  std::string normalized_id = normalizeModelId(model_id);
  auto it = model_states_.find(normalized_id);
  return (it != model_states_.end()) ? it->second : ModelLoadState::UNLOADED;
}

InferenceResponse
OllamaModelManager::generateText(const InferenceRequest &request) {
  std::cout << "[DEBUG] OllamaModelManager::generateText called with model: "
            << request.model_id << std::endl;
  InferenceResponse response;
  auto start_time = std::chrono::high_resolution_clock::now();

  // 转换模型ID以匹配注册时的格式
  std::string normalized_model_id = normalizeModelId(request.model_id);
  std::cout << "[DEBUG] Normalized model ID: " << normalized_model_id
            << std::endl;

  // 打印当前注册的所有模型
  std::cout << "[DEBUG] Currently registered models:" << std::endl;
  auto registered_models = getRegisteredModels();
  for (const auto &model : registered_models) {
    std::cout << "[DEBUG]   - " << model << std::endl;
  }

  // 先检查模型是否已注册
  if (registered_models_.find(normalized_model_id) == registered_models_.end()) {
    std::cout << "[ERROR] Model not registered: " << normalized_model_id << std::endl;
    response.error_message = "Model not registered: " + normalized_model_id;
    response.success = false;
    return response;
  }

  // 检查模型加载状态并在 ERROR 场景下返回更明确错误
  auto load_state = getModelLoadState(normalized_model_id);
  std::cout << "[DEBUG] Model load state for " << normalized_model_id << ": "
            << static_cast<int>(load_state) << std::endl;

  if (load_state != ModelLoadState::LOADED) {
    if (load_state == ModelLoadState::LOAD_ERROR) {
      std::cout << "[ERROR] Model initialization failed previously: "
                << normalized_model_id << std::endl;
      response.error_message =
          "Model initialization failed: " + normalized_model_id;
    } else {
      std::cout << "[DEBUG] Model not loaded: " << normalized_model_id << std::endl;
      response.error_message = "Model not loaded: " + normalized_model_id;
    }
    response.success = false;
    return response;
  }

  try {
    // 获取模型信息
    const ModelInfo *model_info = getModelInfo(normalized_model_id);
    if (!model_info) {
      response.error_message = "Model info not found: " + normalized_model_id;
      response.success = false;
      return response;
    }

    // 获取已加载的推理引擎
    auto engine_it = inference_engines_.find(normalized_model_id);
    if (engine_it == inference_engines_.end()) {
      response.error_message =
          "Inference engine not found for: " + normalized_model_id;
      response.success = false;
      return response;
    }

    InferenceEngine* inference_engine = engine_it->second.get();

    // 双重校验：引擎存在但未就绪（理论上不应发生，但为了更好错误提示）
    if (!inference_engine || !inference_engine->isReady()) {
      std::cout << "[ERROR] Inference engine not ready: "
                << normalized_model_id << std::endl;
      response.error_message =
          "Inference engine not ready: " + normalized_model_id;
      response.success = false;
      return response;
    }

    // 执行文本生成
    std::string generated_text = inference_engine->generateText(
        request.prompt, request.max_tokens, request.temperature, request.top_p);

    // 计算推理时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // 设置响应
    response.success = true;
    response.generated_text = generated_text;
    response.tokens_generated =
        static_cast<int>(generated_text.length() / 4); // 粗略估算
    response.inference_time_ms = static_cast<float>(duration.count());

    std::cout << "[DEBUG] Text generation completed successfully" << std::endl;

  } catch (const std::exception &e) {
    response.error_message = "Inference error: " + std::string(e.what());
    response.success = false;
  }

  return response;
}

InferenceResponse
OllamaModelManager::generateTextWithImages(const InferenceRequest &request) {
  InferenceResponse response;

  // 检查模型是否加载
  if (!isModelLoaded(request.model_id)) {
    response.error_message = "Model not loaded: " + request.model_id;
    return response;
  }

  // 检查模型是否支持视觉
  const ModelInfo *model_info = getModelInfo(request.model_id);
  if (!model_info || !model_info->has_vision) {
    response.error_message =
        "Model does not support vision: " + request.model_id;
    return response;
  }

  // TODO: 重新实现多模态推理引擎集成
  response.error_message = "Multimodal inference engine not implemented yet";
  response.success = false;
  response.tokens_generated = 0;
  response.inference_time_ms = 0.0f;

  return response;
}

std::vector<InferenceResponse> OllamaModelManager::generateTextBatch(
    const std::vector<InferenceRequest> &requests) {

  std::vector<InferenceResponse> responses;
  responses.reserve(requests.size());

  for (const auto &request : requests) {
    if (request.image_features.empty()) {
      responses.push_back(generateText(request));
    } else {
      responses.push_back(generateTextWithImages(request));
    }
  }

  return responses;
}

bool OllamaModelManager::validateModel(const std::string &gguf_file_path,
                                       std::string &error_message) {
  try {
    GGUFParser parser(false); // 不输出详细日志

    if (!parser.parseFile(gguf_file_path)) {
      error_message = "Failed to parse GGUF file";
      return false;
    }

    if (!parser.validateFile()) {
      error_message = "GGUF file validation failed";
      return false;
    }

    const auto &architecture = parser.getArchitecture();
    if (!GGUFParser::isSupportedArchitecture(architecture.name)) {
      error_message = "Unsupported architecture: " + architecture.name;
      return false;
    }

    return true;

  } catch (const std::exception &e) {
    error_message = "Validation error: " + std::string(e.what());
    return false;
  }
}

void OllamaModelManager::clearAllModels() {
  log("INFO", "Clearing all models...");

  // 卸载所有已加载的模型
  auto loaded_models = getLoadedModels();
  for (const std::string &model_id : loaded_models) {
    unloadModel(model_id);
  }

  // 清空所有数据结构
  registered_models_.clear();
  // inference_engines_.clear(); // TODO: 重新实现
  model_states_.clear();

  total_memory_usage_ = 0;
  active_models_count_ = 0;

  log("INFO", "All models cleared");
}

size_t OllamaModelManager::getMemoryUsage() const {
  // 简化的内存使用计算
  total_memory_usage_ = 0;

  // TODO: 重新实现内存使用计算
  // for (const auto& pair : inference_engines_) {
  //     // 估算每个模型的内存使用（简化实现）
  //     const ModelInfo* info = getModelInfo(pair.first);
  //     if (info) {
  //         // 粗略估算：参数数量 * 4字节（float32）
  //         size_t estimated_size = static_cast<size_t>(info->vocab_size) *
  //         1024; // 简化估算 total_memory_usage_ += estimated_size;
  //     }
  // }

  return total_memory_usage_;
}

// 私有方法实现
bool OllamaModelManager::loadModelInternal(const std::string &model_id) {
  // 检查模型是否已注册
  auto model_it = registered_models_.find(model_id);
  if (model_it == registered_models_.end()) {
    log("ERROR", "Model not registered: " + model_id);
    return false;
  }

  // 检查模型是否已加载
  if (inference_engines_.find(model_id) != inference_engines_.end()) {
    log("INFO", "Model already loaded: " + model_id);
    return true;
  }

  // 设置加载状态
  model_states_[model_id] = ModelLoadState::LOADING;
  
  try {
    // 创建推理引擎
    auto engine = createInferenceEngine(model_id);
    if (!engine) {
      log("ERROR", "Failed to create inference engine for: " + model_id);
      model_states_[model_id] = ModelLoadState::LOAD_ERROR;
      return false;
    }

    // 推理引擎会在initialize时自动加载模型文件

    // 存储推理引擎
    inference_engines_[model_id] = std::move(engine);
    model_states_[model_id] = ModelLoadState::LOADED;
    active_models_count_++;

    log("INFO", "Model loaded successfully: " + model_id);
    return true;

  } catch (const std::exception &e) {
    log("ERROR", "Exception during model loading: " + std::string(e.what()));
    model_states_[model_id] = ModelLoadState::LOAD_ERROR;
    return false;
  }
}

bool OllamaModelManager::unloadModelInternal(const std::string &model_id) {
  // 查找推理引擎
  auto engine_it = inference_engines_.find(model_id);
  if (engine_it == inference_engines_.end()) {
    log("WARNING", "Model not loaded: " + model_id);
    return true; // 已经卸载，返回成功
  }

  try {
    // 移除推理引擎（析构函数会自动清理资源）
    inference_engines_.erase(engine_it);
    model_states_[model_id] = ModelLoadState::UNLOADED;
    active_models_count_--;

    log("INFO", "Model unloaded successfully: " + model_id);
    return true;

  } catch (const std::exception &e) {
    log("ERROR", "Exception during model unloading: " + std::string(e.what()));
    return false;
  }
}

bool OllamaModelManager::parseModelInfo(const std::string &gguf_file_path,
                                        ModelInfo &model_info) {
  try {
    log("DEBUG", "Creating GGUFParser for: " + gguf_file_path);
    GGUFParser parser(true); // 启用详细日志

    log("DEBUG", "Calling parseFile...");
    if (!parser.parseFile(gguf_file_path)) {
      log("ERROR", "parseFile returned false");
      return false;
    }

    log("DEBUG", "Getting architecture...");
    const auto &architecture = parser.getArchitecture();
    log("DEBUG", "Architecture name: " + architecture.name);
    log("DEBUG",
        "Context length: " + std::to_string(architecture.context_length));
    log("DEBUG", "Has vision: " +
                     std::string(architecture.has_vision ? "true" : "false"));

    model_info.architecture = architecture.name;
    model_info.context_length = architecture.context_length;
    model_info.has_vision = architecture.has_vision;

    // 获取词汇表大小（简化实现）
    model_info.vocab_size = 151936; // Qwen2.5VL默认值

    log("DEBUG", "Model info parsed successfully");
    return true;

  } catch (const std::exception &e) {
    log("ERROR", "Failed to parse model info: " + std::string(e.what()));
    return false;
  }
}

std::unique_ptr<InferenceEngine>
OllamaModelManager::createInferenceEngine(const std::string &model_id) {
  const ModelInfo *model_info = getModelInfo(model_id);
  if (!model_info) {
    log("ERROR", "Model info not found for: " + model_id);
    return nullptr;
  }

  try {
    auto engine = std::make_unique<MLInferenceEngine>(model_id);
    if (!engine->initialize()) {
      log("ERROR", "Failed to initialize inference engine for: " + model_id);
      return nullptr;
    }

    // 进一步校验引擎就绪状态
    if (!engine->isReady()) {
      log("ERROR", "Inference engine not ready after initialization for: " + model_id);
      return nullptr;
    }

    log("INFO", "Inference engine created successfully for: " + model_id);
    return std::move(engine);

  } catch (const std::exception &e) {
    log("ERROR", "Failed to create inference engine: " + std::string(e.what()));
    return nullptr;
  }
}

bool OllamaModelManager::checkResourceAvailability() const {
  // 检查是否超过最大并发模型数
  if (active_models_count_ >= max_concurrent_models_) {
    return false;
  }

  // 可以添加更多资源检查（内存、GPU等）
  return true;
}

void OllamaModelManager::cleanupUnusedResources() {
  // 清理未使用的资源（简化实现）
  // 在实际应用中，可以实现LRU缓存等策略
}

std::string
OllamaModelManager::generateModelId(const std::string &file_path) const {
  // 简单的文件名提取，不使用filesystem
  size_t last_slash = file_path.find_last_of("/\\");
  std::string filename = (last_slash != std::string::npos)
                             ? file_path.substr(last_slash + 1)
                             : file_path;

  // 移除扩展名
  size_t last_dot = filename.find_last_of('.');
  if (last_dot != std::string::npos) {
    filename = filename.substr(0, last_dot);
  }

  return filename;
}

std::string
OllamaModelManager::normalizeModelId(const std::string &model_name) const {
  // 修剪前后空白字符
  auto trim = [](const std::string &s) {
    auto begin = std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); });
    auto end = std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base();
    if (begin >= end) return std::string();
    return std::string(begin, end);
  };
  std::string model_id = trim(model_name);
  // 仅替换不允许的字符，允许字符集: 字母数字、下划线、横杠、点、冒号、斜杠
  for (char &c : model_id) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-' && c != '.' && c != ':' && c != '/') {
      c = '_';
    }
  }
  return model_id;
}

bool OllamaModelManager::isValidModelId(const std::string &model_id) const {
  if (model_id.empty() || model_id.length() > 100) {
    return false;
  }

  // 检查是否包含有效字符
  for (char c : model_id) {
    if (!std::isalnum(c) && c != '_' && c != '-' && c != '.' && c != ':' && c != '/') {
      return false;
    }
  }

  return true;
}

void OllamaModelManager::log(const std::string &level,
                             const std::string &message) const {
  if (verbose_ || level == "ERROR") {
    std::cout << "[" << level << "] OllamaModelManager: " << message
              << std::endl;
  }
}

// 工厂函数
std::unique_ptr<OllamaModelManager> createOllamaModelManager(bool verbose) {
  return std::make_unique<OllamaModelManager>(verbose);
}

// 全局模型管理器实现
std::unique_ptr<OllamaModelManager> GlobalModelManager::instance_ = nullptr;
bool GlobalModelManager::initialized_ = false;

OllamaModelManager &GlobalModelManager::getInstance() {
  if (!initialized_ || !instance_) {
    throw std::runtime_error("GlobalModelManager not initialized");
  }
  return *instance_;
}

void GlobalModelManager::initialize(bool verbose) {
  if (!initialized_) {
    instance_ = createOllamaModelManager(verbose);
    initialized_ = true;
  }
}

void GlobalModelManager::shutdown() {
  if (initialized_) {
    instance_.reset();
    initialized_ = false;
  }
}

} // namespace ollama
} // namespace extensions
} // namespace duorou