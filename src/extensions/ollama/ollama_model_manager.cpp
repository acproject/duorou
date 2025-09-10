#include "ollama_model_manager.h"
#include "text_processor.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace duorou {
namespace extensions {
namespace ollama {

OllamaModelManager::OllamaModelManager(bool verbose)
    : verbose_(verbose), max_concurrent_models_(3), path_resolver_(verbose),
      total_memory_usage_(0), active_models_count_(0) {
  // 初始化BPE文本处理器
  auto vocab = std::make_shared<Vocabulary>();
  text_processor_ = createTextProcessor("bpe", vocab);
  log("DEBUG", "OllamaModelManager initialized with BPE processor");
}

OllamaModelManager::~OllamaModelManager() {
  clearAllModels();
  log("INFO", "OllamaModelManager destroyed");
}

bool OllamaModelManager::registerModel(const std::string &model_id,
                                       const std::string &gguf_file_path) {
  if (model_id.empty() || gguf_file_path.empty()) {
    log("ERROR", "Model ID or file path is empty");
    return false;
  }

  if (!isValidModelId(model_id)) {
    log("ERROR", "Invalid model ID: " + model_id);
    return false;
  }

  if (!std::filesystem::exists(gguf_file_path)) {
    log("ERROR", "GGUF file does not exist: " + gguf_file_path);
    return false;
  }

  std::string error_message;
  if (!validateModel(gguf_file_path, error_message)) {
    log("ERROR", "Model validation failed: " + error_message);
    return false;
  }

  ModelInfo model_info;
  if (!parseModelInfo(gguf_file_path, model_info)) {
    log("ERROR", "Failed to parse model info from: " + gguf_file_path);
    return false;
  }

  model_info.model_id = model_id;
  model_info.file_path = gguf_file_path;
  model_info.is_loaded = false;

  registered_models_[model_id] = model_info;
  model_states_[model_id] = ModelLoadState::UNLOADED;

  log("INFO", "Model registered: " + model_id + " -> " + gguf_file_path);
  return true;
}

bool OllamaModelManager::registerModelByName(const std::string &model_name) {
  auto gguf_path_opt = path_resolver_.resolveModelPath(model_name);
  if (!gguf_path_opt) {
    log("ERROR", "Failed to resolve model path for: " + model_name);
    return false;
  }

  std::string gguf_path = *gguf_path_opt;
  std::string model_id = generateModelId(gguf_path);
  if (model_id.empty()) {
    log("ERROR", "Failed to generate model ID for: " + model_name);
    return false;
  }

  bool success = registerModel(model_id, gguf_path);

  // Also register with the original model name as an alias for easier lookup
  if (success && model_name != model_id) {
    ModelInfo model_info = registered_models_[model_id];
    model_info.model_id = model_name;
    registered_models_[model_name] = model_info;
    model_states_[model_name] = ModelLoadState::UNLOADED;
    log("INFO", "Model alias registered: " + model_name + " -> " + gguf_path);
  }

  return success;
}

bool OllamaModelManager::loadModel(const std::string &model_id) {
  if (model_id.empty()) {
    log("ERROR", "Model ID is empty");
    return false;
  }

  if (registered_models_.find(model_id) == registered_models_.end()) {
    log("ERROR", "Model not registered: " + model_id);
    return false;
  }

  if (isModelLoaded(model_id)) {
    log("INFO", "Model already loaded: " + model_id);
    return true;
  }

  if (!checkResourceAvailability()) {
    log("ERROR", "Insufficient resources to load model: " + model_id);
    return false;
  }

  model_states_[model_id] = ModelLoadState::LOADING;
  bool success = loadModelInternal(model_id);

  if (success) {
    model_states_[model_id] = ModelLoadState::LOADED;
    registered_models_[model_id].is_loaded = true;
    active_models_count_++;
    log("INFO", "Model loaded successfully: " + model_id);
  } else {
    model_states_[model_id] = ModelLoadState::ERROR;
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
  }

  return success;
}

bool OllamaModelManager::isModelLoaded(const std::string &model_id) const {
  return model_states_.find(model_id) != model_states_.end() &&
         model_states_.at(model_id) == ModelLoadState::LOADED;
}

std::vector<std::string> OllamaModelManager::getRegisteredModels() const {
  std::vector<std::string> models;
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
  auto it = registered_models_.find(model_id);
  return (it != registered_models_.end()) ? &it->second : nullptr;
}

ModelLoadState
OllamaModelManager::getModelLoadState(const std::string &model_id) const {
  auto it = model_states_.find(model_id);
  return (it != model_states_.end()) ? it->second : ModelLoadState::UNLOADED;
}

InferenceResponse
OllamaModelManager::generateText(const InferenceRequest &request) {
  InferenceResponse response;

  // 增强输入验证
  if (request.model_id.empty()) {
    response.error_message = "Model ID is empty";
    log("ERROR", response.error_message);
    return response;
  }

  if (request.prompt.empty()) {
    response.error_message = "Prompt is empty";
    log("ERROR", response.error_message);
    return response;
  }

  // 检查prompt长度的合理性
  if (request.prompt.length() > 100000) {
    response.error_message = "Prompt is too long (" +
                             std::to_string(request.prompt.length()) +
                             " characters, max 100000)";
    log("ERROR", response.error_message);
    return response;
  }

  // 检查max_tokens的合理性
  if (request.max_tokens > 10000) {
    log("WARNING",
        "max_tokens is very large: " + std::to_string(request.max_tokens));
  }

  // 验证模型是否已注册
  if (registered_models_.find(request.model_id) == registered_models_.end()) {
    response.error_message = "Model not registered: " + request.model_id;
    log("ERROR", response.error_message);
    return response;
  }

  // 验证模型是否已加载
  if (!isModelLoaded(request.model_id)) {
    response.error_message = "Model not loaded: " + request.model_id;
    log("ERROR", response.error_message);
    return response;
  }

  Qwen25VLModularEngine *engine = getInferenceEngine(request.model_id);
  if (!engine) {
    response.error_message =
        "Failed to get inference engine for model: " + request.model_id;
    log("ERROR", response.error_message);
    return response;
  }

  // 验证文本处理器是否可用
  if (!text_processor_) {
    response.error_message = "Text processor not initialized";
    log("ERROR", response.error_message);
    return response;
  }

  try {
    auto start = std::chrono::high_resolution_clock::now();

    // 分词处理，带详细错误信息
    std::vector<uint32_t> input_tokens;
    try {
      input_tokens = tokenize(request.prompt);
    } catch (const std::exception &e) {
      response.error_message =
          "Tokenization failed: " + std::string(e.what()) + " for prompt: '" +
          (request.prompt.length() > 100 ? request.prompt.substr(0, 100) + "..."
                                         : request.prompt) +
          "'";
      log("ERROR", response.error_message);
      return response;
    }

    // 检查tokenization是否成功
    if (input_tokens.empty()) {
      response.error_message =
          "Failed to tokenize input prompt (empty result): '" +
          (request.prompt.length() > 100 ? request.prompt.substr(0, 100) + "..."
                                         : request.prompt) +
          "'";
      log("ERROR", response.error_message);
      return response;
    }

    // 检查token数量的合理性
    if (input_tokens.size() > 50000) {
      response.error_message =
          "Input tokens too many: " + std::to_string(input_tokens.size()) +
          " (max 50000)";
      log("ERROR", response.error_message);
      return response;
    }

    log("INFO", "Tokenized prompt into " + std::to_string(input_tokens.size()) +
                    " tokens");

    // 推理处理，带详细错误信息
    std::vector<uint32_t> output_tokens;
    try {
      output_tokens = engine->generateText(input_tokens, request.max_tokens);
    } catch (const std::exception &e) {
      response.error_message =
          "Inference engine failed: " + std::string(e.what());
      log("ERROR", response.error_message);
      return response;
    }

    // 检查推理结果
    if (output_tokens.empty()) {
      response.error_message = "Inference engine returned empty output";
      log("ERROR", response.error_message);
      return response;
    }

    // 解码处理，带详细错误信息
    try {
      response.generated_text = detokenize(output_tokens);
    } catch (const std::exception &e) {
      response.error_message =
          "Detokenization failed: " + std::string(e.what());
      log("ERROR", response.error_message);
      return response;
    }

    auto end = std::chrono::high_resolution_clock::now();
    response.success = true;
    response.tokens_generated = static_cast<uint32_t>(output_tokens.size());
    response.inference_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    log("INFO", "Generated " + std::to_string(output_tokens.size()) +
                    " tokens in " + std::to_string(response.inference_time_ms) +
                    "ms");

  } catch (const std::exception &e) {
    response.error_message =
        std::string("Unexpected exception during text generation: ") + e.what();
    log("ERROR", response.error_message);
  } catch (...) {
    response.error_message = "Unknown exception during text generation";
    log("ERROR", response.error_message);
  }

  return response;
}

InferenceResponse
OllamaModelManager::generateTextWithImages(const InferenceRequest &request) {
  InferenceResponse response;

  if (request.model_id.empty() || request.prompt.empty()) {
    response.error_message = "Model ID or prompt is empty";
    return response;
  }

  Qwen25VLModularEngine *engine = getInferenceEngine(request.model_id);
  if (!engine) {
    response.error_message =
        "Failed to get inference engine for model: " + request.model_id;
    return response;
  }

  try {
    auto start = std::chrono::high_resolution_clock::now();
    auto input_tokens = tokenize(request.prompt);
    // 将image_features转换为Tensor格式
    algorithms::Tensor image_tensor; // 这里需要实际的转换逻辑
    auto output_tokens = engine->generateMultimodal(input_tokens, image_tensor,
                                                    request.max_tokens);
    response.generated_text = detokenize(output_tokens);
    auto end = std::chrono::high_resolution_clock::now();
    response.success = true;
    response.tokens_generated =
        static_cast<uint32_t>(response.generated_text.size());
    response.inference_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
  } catch (const std::exception &e) {
    response.error_message =
        std::string("Exception during image inference: ") + e.what();
  }

  return response;
}

std::vector<InferenceResponse> OllamaModelManager::generateTextBatch(
    const std::vector<InferenceRequest> &requests) {
  std::vector<InferenceResponse> responses;
  for (const auto &request : requests) {
    responses.push_back(generateText(request));
  }
  return responses;
}

bool OllamaModelManager::validateModel(const std::string &gguf_file_path,
                                       std::string &error_message) {
  if (!std::filesystem::exists(gguf_file_path)) {
    error_message = "File does not exist: " + gguf_file_path;
    return false;
  }

  std::ifstream file(gguf_file_path, std::ios::binary);
  if (!file.is_open()) {
    error_message = "Cannot open file: " + gguf_file_path;
    return false;
  }

  // 检查GGUF魔数
  char magic[4];
  file.read(magic, 4);
  if (file.gcount() != 4 || std::string(magic, 4) != "GGUF") {
    error_message = "Invalid GGUF file format";
    return false;
  }

  file.close();
  return true;
}

void OllamaModelManager::clearAllModels() {
  std::vector<std::string> loaded_models = getLoadedModels();
  for (const std::string &model_id : loaded_models) {
    unloadModel(model_id);
  }

  registered_models_.clear();
  inference_engines_.clear();
  model_states_.clear();
  total_memory_usage_ = 0;
  active_models_count_ = 0;

  log("INFO", "All models cleared");
}

size_t OllamaModelManager::getMemoryUsage() const {
  return total_memory_usage_;
}

bool OllamaModelManager::loadModelInternal(const std::string &model_id) {
  auto model_it = registered_models_.find(model_id);
  if (model_it == registered_models_.end()) {
    log("ERROR", "Model not found in registry: " + model_id);
    return false;
  }

  const ModelInfo &model_info = model_it->second;

  // 创建推理引擎
  if (!createInferenceEngine(model_id)) {
    log("ERROR", "Failed to create inference engine for model: " + model_id);
    return false;
  }

  // 获取推理引擎
  Qwen25VLModularEngine *engine = getInferenceEngine(model_id);
  if (!engine) {
    log("ERROR", "Failed to get inference engine for model: " + model_id);
    destroyInferenceEngine(model_id);
    return false;
  }

  // 加载权重
  try {
    if (!engine->loadWeights(model_info.file_path)) {
      log("ERROR",
          "engine->loadWeights() returned false for model: " + model_id);
      destroyInferenceEngine(model_id);
      return false;
    }
    log("DEBUG", "engine->loadWeights() completed successfully");
  } catch (const std::exception &e) {
    log("ERROR",
        "Exception in engine->loadWeights(): " + std::string(e.what()));
    destroyInferenceEngine(model_id);
    return false;
  }

  return true;
}

bool OllamaModelManager::unloadModelInternal(const std::string &model_id) {
  destroyInferenceEngine(model_id);
  return true;
}

bool OllamaModelManager::parseModelInfo(const std::string &gguf_file_path,
                                        ModelInfo &model_info) {
  try {
    GGUFParser parser(verbose_);
    if (!parser.parseFile(gguf_file_path)) {
      return false;
    }

    const auto &architecture = parser.getArchitecture();
    model_info.architecture = architecture.name;
    model_info.context_length = architecture.context_length;
    
    // 使用从GGUF文件中解析的vocab_size，如果没有则使用默认值
    if (architecture.vocab_size > 0) {
      model_info.vocab_size = architecture.vocab_size;
      log("INFO", "Using vocab_size from GGUF: " + std::to_string(architecture.vocab_size));
    } else {
      model_info.vocab_size = 151643; // Qwen2.5VL默认值
      log("WARNING", "Using default vocab_size: 151643");
    }
    
    model_info.has_vision = architecture.has_vision;

    // 从GGUF文件中加载词汇表数据
    if (!loadVocabularyFromGGUF(parser, gguf_file_path)) {
      log("WARNING",
          "Failed to load vocabulary from GGUF file, using empty vocabulary");
    }

    return true;
  } catch (const std::exception &e) {
    log("ERROR", "Exception parsing model info: " + std::string(e.what()));
    return false;
  }
}

Qwen25VLModularEngine *
OllamaModelManager::getInferenceEngine(const std::string &model_id) {
  auto it = inference_engines_.find(model_id);
  return (it != inference_engines_.end()) ? it->second.get() : nullptr;
}

bool OllamaModelManager::createInferenceEngine(const std::string &model_id) {
  if (inference_engines_.find(model_id) != inference_engines_.end()) {
    return true; // 已存在
  }

  try {
    auto engine = std::make_unique<Qwen25VLModularEngine>();

    // 创建默认配置
    Qwen25VLConfig config;
    // 可以根据模型信息调整配置
    auto model_it = registered_models_.find(model_id);
    if (model_it != registered_models_.end()) {
      const ModelInfo &model_info = model_it->second;
      if (model_info.vocab_size > 0) {
        config.vocab_size = model_info.vocab_size;
      }
      // 可以根据需要设置其他配置参数
    }

    // 初始化引擎
    if (!engine->initialize(config)) {
      log("ERROR",
          "Failed to initialize inference engine for model: " + model_id);
      return false;
    }

    inference_engines_[model_id] = std::move(engine);
    log("DEBUG",
        "Inference engine created and initialized for model: " + model_id);
    return true;
  } catch (const std::exception &e) {
    log("ERROR", "Failed to create inference engine: " + std::string(e.what()));
    return false;
  }
}

void OllamaModelManager::destroyInferenceEngine(const std::string &model_id) {
  auto it = inference_engines_.find(model_id);
  if (it != inference_engines_.end()) {
    inference_engines_.erase(it);
  }
}

bool OllamaModelManager::checkResourceAvailability() const {
  return active_models_count_ < max_concurrent_models_;
}

void OllamaModelManager::cleanupUnusedResources() {
  // 实现资源清理逻辑
}

std::string
OllamaModelManager::generateModelId(const std::string &file_path) const {
  std::filesystem::path path(file_path);
  std::string filename = path.stem().string();

  // 移除常见的模型文件后缀
  const std::vector<std::string> suffixes = {".gguf", ".bin", ".safetensors"};
  for (const std::string &suffix : suffixes) {
    if (filename.size() >= suffix.size() &&
        filename.substr(filename.size() - suffix.size()) == suffix) {
      filename = filename.substr(0, filename.length() - suffix.length());
      break;
    }
  }

  return filename;
}

bool OllamaModelManager::isValidModelId(const std::string &model_id) const {
  return !model_id.empty() && model_id.find('/') == std::string::npos;
}

void OllamaModelManager::log(const std::string &level,
                             const std::string &message) const {
  if (verbose_ || level == "ERROR") {
    std::cout << "[" << level << "] OllamaModelManager: " << message
              << std::endl;
  }
}

// Token转换辅助方法实现
std::vector<uint32_t> OllamaModelManager::tokenize(const std::string &text) {
  if (!text_processor_) {
    log("ERROR", "Text processor not initialized");
    return {};
  }

  if (text.empty()) {
    log("WARNING", "Empty text provided for tokenization");
    return {};
  }

  try {
    // 使用BPE处理器进行分词
    auto tokens_int32 = text_processor_->encode(text, true);

    // 转换为uint32_t
    std::vector<uint32_t> tokens;
    tokens.reserve(tokens_int32.size());
    for (int32_t token : tokens_int32) {
      if (token >= 0) {
        tokens.push_back(static_cast<uint32_t>(token));
      }
    }

    // 如果分词结果为空，尝试字符级回退
    if (tokens.empty()) {
      log("WARNING", "Primary tokenization failed, attempting character-level "
                     "fallback for text: '" +
                         text + "'");

      // 字符级回退：将每个字符转换为字节级token
      const auto *vocab = text_processor_->getVocabulary();
      if (vocab) {
        for (unsigned char c : text) {
          // 尝试查找字节级token（格式如 "<0xXX>"）
          std::stringstream ss;
          ss << "<0x" << std::hex << std::uppercase << std::setfill('0')
             << std::setw(2) << static_cast<unsigned int>(c) << ">";
          int32_t byte_token = vocab->encode(ss.str());

          if (byte_token >= 0) {
            tokens.push_back(static_cast<uint32_t>(byte_token));
          } else {
            // 如果连字节级token都找不到，使用UNK token
            int32_t unk_token = vocab->encode("<unk>");
            if (unk_token >= 0) {
              tokens.push_back(static_cast<uint32_t>(unk_token));
            } else {
              // 最后的回退：使用token ID 0（通常是UNK或PAD）
              tokens.push_back(0);
            }
          }
        }

        if (!tokens.empty()) {
          log("INFO", "Character-level fallback succeeded, generated " +
                          std::to_string(tokens.size()) + " tokens");
        } else {
          log("ERROR",
              "All tokenization methods failed for text: '" + text + "'");
        }
      } else {
        log("ERROR", "Cannot access vocabulary for fallback tokenization");
      }
    } else {
      log("DEBUG", "Tokenized '" + text + "' to " +
                       std::to_string(tokens.size()) + " tokens");
    }

    return tokens;
  } catch (const std::exception &e) {
    log("ERROR", "Tokenization failed: " + std::string(e.what()));

    // 异常情况下的最后回退
    log("WARNING", "Attempting emergency character-level tokenization");
    std::vector<uint32_t> emergency_tokens;
    for (size_t i = 0; i < std::min(text.length(), size_t(512)); ++i) {
      emergency_tokens.push_back(
          static_cast<uint32_t>(static_cast<unsigned char>(text[i])));
    }

    if (!emergency_tokens.empty()) {
      log("INFO", "Emergency tokenization generated " +
                      std::to_string(emergency_tokens.size()) + " tokens");
    }

    return emergency_tokens;
  }
}

std::string
OllamaModelManager::detokenize(const std::vector<uint32_t> &tokens) {
  if (!text_processor_) {
    log("ERROR", "Text processor not initialized");
    return "";
  }

  try {
    // 转换为int32_t
    std::vector<int32_t> tokens_int32;
    tokens_int32.reserve(tokens.size());
    for (uint32_t token : tokens) {
      tokens_int32.push_back(static_cast<int32_t>(token));
    }

    // 使用BPE处理器进行解码
    std::string text = text_processor_->decode(tokens_int32);

    log("DEBUG", "Detokenized " + std::to_string(tokens.size()) +
                     " tokens to '" + text + "'");
    return text;
  } catch (const std::exception &e) {
    log("ERROR", "Detokenization failed: " + std::string(e.what()));
    return "";
  }
}

void OllamaModelManager::setTextProcessor(
    std::unique_ptr<TextProcessor> processor) {
  text_processor_ = std::move(processor);
  log("DEBUG", "Text processor updated");
}

void OllamaModelManager::setMaxConcurrentModels(uint32_t max_models) {
  max_concurrent_models_ = max_models;
  log("DEBUG", "Max concurrent models set to " + std::to_string(max_models));
}

uint32_t OllamaModelManager::getMaxConcurrentModels() const {
  return max_concurrent_models_;
}

bool OllamaModelManager::loadVocabularyFromGGUF(
    const GGUFParser &parser, const std::string &gguf_file_path) {
  try {
    // 获取词汇表tokens
    const auto *tokens_metadata = parser.getMetadata("tokenizer.ggml.tokens");
    if (!tokens_metadata) {
      log("WARNING", "No tokenizer.ggml.tokens found in GGUF file");
      return false;
    }

    // 获取token分数
    const auto *scores_metadata = parser.getMetadata("tokenizer.ggml.scores");

    // 获取token类型
    const auto *types_metadata =
        parser.getMetadata("tokenizer.ggml.token_type");

    // 解析tokens数组
    std::vector<std::string> token_strings = tokens_metadata->asStringArray();
    if (token_strings.empty()) {
      log("ERROR", "Empty token list in GGUF file: " + gguf_file_path);
      return false;
    }

    // 验证词汇表大小的合理性
    if (token_strings.size() < 100) {
      log("WARNING", "Vocabulary size is suspiciously small: " +
                         std::to_string(token_strings.size()) + " tokens");
    } else if (token_strings.size() > 1000000) {
      log("WARNING", "Vocabulary size is very large: " +
                         std::to_string(token_strings.size()) + " tokens");
    }

    // 检查是否有空字符串token
    size_t empty_tokens = 0;
    for (size_t i = 0; i < token_strings.size(); ++i) {
      if (token_strings[i].empty()) {
        empty_tokens++;
        if (empty_tokens <= 5) { // 只记录前5个空token的位置
          log("WARNING", "Empty token found at index: " + std::to_string(i));
        }
      }
    }
    if (empty_tokens > 0) {
      log("WARNING",
          "Total empty tokens found: " + std::to_string(empty_tokens));
    }

    // 解析scores数组（如果存在）
    std::vector<float> token_scores;
    if (scores_metadata) {
      // scores通常是float32数组，需要从原始数据中解析
      const auto &scores_data = scores_metadata->data;
      size_t num_scores = scores_data.size() / sizeof(float);
      token_scores.resize(num_scores);
      std::memcpy(token_scores.data(), scores_data.data(), scores_data.size());
    } else {
      // 如果没有scores，使用默认值
      token_scores.resize(token_strings.size(), 0.0f);
    }

    // 解析token类型（如果存在）
    std::vector<int32_t> token_types;
    if (types_metadata) {
      token_types = types_metadata->asInt32Array();
    } else {
      // 如果没有类型信息，使用默认类型
      token_types.resize(token_strings.size(), 1); // TOKEN_TYPE_NORMAL
    }

    // 确保所有数组大小一致
    size_t vocab_size = token_strings.size();
    if (token_scores.size() != vocab_size) {
      token_scores.resize(vocab_size, 0.0f);
    }
    if (token_types.size() != vocab_size) {
      token_types.resize(vocab_size, 1);
    }

    // 创建新的词汇表并初始化
    auto vocab = std::make_shared<Vocabulary>();
    vocab->initialize(token_strings, token_types, token_scores);

    // 验证词汇表初始化是否成功
    if (vocab->size() == 0) {
      log("ERROR", "Vocabulary initialization failed - size is 0");
      return false;
    }

    if (vocab->size() != token_strings.size()) {
      log("WARNING", "Vocabulary size mismatch: expected " +
                         std::to_string(token_strings.size()) + ", got " +
                         std::to_string(vocab->size()));
    }

    // 测试基本的编码/解码功能
    try {
      // 测试单个token编码
      int32_t test_token = vocab->encode("test");
      if (test_token >= 0) {
        std::string decoded = vocab->decode(test_token);
        log("INFO", "Vocabulary test passed: 'test' -> token " +
                        std::to_string(test_token) + " -> '" + decoded + "'");
      } else {
        log("WARNING", "Vocabulary test: 'test' produced invalid token: " +
                           std::to_string(test_token));
      }
    } catch (const std::exception &e) {
      log("WARNING", "Vocabulary test failed: " + std::string(e.what()));
    }

    // 设置特殊tokens（如果存在）
    const auto *bos_metadata =
        parser.getMetadata("tokenizer.ggml.bos_token_id");
    const auto *eos_metadata =
        parser.getMetadata("tokenizer.ggml.eos_token_id");
    const auto *add_bos_metadata =
        parser.getMetadata("tokenizer.ggml.add_bos_token");
    const auto *add_eos_metadata =
        parser.getMetadata("tokenizer.ggml.add_eos_token");

    if (bos_metadata) {
      int32_t bos_token = bos_metadata->asInt32();
      bool add_bos = add_bos_metadata ? add_bos_metadata->asBool() : false;
      vocab->setBOS({bos_token}, add_bos);
    }

    if (eos_metadata) {
      int32_t eos_token = eos_metadata->asInt32();
      bool add_eos = add_eos_metadata ? add_eos_metadata->asBool() : false;
      vocab->setEOS({eos_token}, add_eos);
    }

    // 获取tokenizer模型类型
    std::string tokenizer_model = "bpe"; // 默认使用BPE
    const auto *model_metadata = parser.getMetadata("tokenizer.ggml.model");
    if (model_metadata) {
      std::string original_model = model_metadata->asString();
      // 只支持 "sentencepiece" 和 "bpe"，其他类型使用 "bpe" 作为fallback
      if (original_model == "sentencepiece" || original_model == "bpe") {
        tokenizer_model = original_model;
      } else {
        log("WARNING", "Unsupported tokenizer model type: " + original_model +
                           ", using BPE as fallback");
        tokenizer_model = "bpe";
      }
    }

    // 重新创建text processor
    text_processor_ = createTextProcessor(tokenizer_model, vocab);

    // 确保text processor创建成功
    if (!text_processor_) {
      log("ERROR",
          "Failed to create text processor with type: " + tokenizer_model);
      return false;
    }

    log("INFO", "Successfully loaded vocabulary from GGUF: " +
                    std::to_string(vocab_size) +
                    " tokens, model: " + tokenizer_model);
    return true;

  } catch (const std::exception &e) {
    log("ERROR",
        "Exception loading vocabulary from GGUF: " + std::string(e.what()));
    return false;
  }
}

std::unique_ptr<OllamaModelManager> createOllamaModelManager(bool verbose) {
  return std::make_unique<OllamaModelManager>(verbose);
}

std::unique_ptr<OllamaModelManager> GlobalModelManager::instance_ = nullptr;
bool GlobalModelManager::initialized_ = false;

OllamaModelManager &GlobalModelManager::getInstance() {
  if (!initialized_) {
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