#ifndef DUOROU_EXTENSIONS_OLLAMA_OLLAMA_MODEL_MANAGER_H
#define DUOROU_EXTENSIONS_OLLAMA_OLLAMA_MODEL_MANAGER_H

#ifdef __cplusplus

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// 注意：避免在命名空间内包含头文件，以防标准库命名空间被嵌套。
// 所有包含应在命名空间之外进行。

#if !defined(DUOROU_ENABLE_OLLAMA)

namespace duorou {
namespace extensions {
namespace ollama {

// ---------- Stub declarations when Ollama extension is disabled ----------

// 模型信息结构体
struct ModelInfo {
  std::string model_id;
  std::string file_path;
  std::string architecture;
  unsigned int context_length = 0;
  unsigned int vocab_size = 0;
  bool has_vision = false;
  bool is_loaded = false;
};

// 模型加载状态枚举
enum class ModelLoadState { UNLOADED, LOADING, LOADED, LOAD_ERROR };

// 推理请求结构体
struct InferenceRequest {
  std::string model_id;
  std::string prompt;
  std::vector<std::vector<float>> image_features;
  unsigned int max_tokens = 100;
  float temperature = 0.7f;
  float top_p = 0.9f;
};

// 推理响应结构体
struct InferenceResponse {
  bool success = false;
  std::string generated_text;
  std::string error_message;
  unsigned int tokens_generated = 0;
  float inference_time_ms = 0.0f;
};

class OllamaModelManager {
public:
  explicit OllamaModelManager(bool /*verbose*/ = false) {}
  ~OllamaModelManager() = default;

  bool registerModel(const std::string &/*model_id*/, const std::string &/*gguf_file_path*/) { return false; }
  bool registerModelByName(const std::string &/*model_name*/) { return false; }
  bool loadModel(const std::string &/*model_id*/) { return false; }
  bool unloadModel(const std::string &/*model_id*/) { return false; }
  bool isModelLoaded(const std::string &/*model_id*/) const { return false; }

  std::vector<std::string> getRegisteredModels() const { return {}; }
  std::vector<std::string> getLoadedModels() const { return {}; }
  const ModelInfo *getModelInfo(const std::string &/*model_id*/) const { return nullptr; }
  ModelLoadState getModelLoadState(const std::string &/*model_id*/) const { return ModelLoadState::UNLOADED; }

  InferenceResponse generateText(const InferenceRequest &/*request*/) { return {}; }
  InferenceResponse generateTextWithImages(const InferenceRequest &/*request*/) { return {}; }
  std::vector<InferenceResponse> generateTextBatch(const std::vector<InferenceRequest> &/*requests*/) { return {}; }

  bool validateModel(const std::string &/*gguf_file_path*/, std::string &/*error_message*/) { return false; }

  void clearAllModels() {}
  size_t getMemoryUsage() const { return 0; }

  std::string normalizeModelId(const std::string &model_name) const { return model_name; }
};

inline std::unique_ptr<OllamaModelManager> createOllamaModelManager(bool verbose = false) {
  return std::make_unique<OllamaModelManager>(verbose);
}

class GlobalModelManager {
public:
  static OllamaModelManager &getInstance() {
    static OllamaModelManager instance;
    return instance;
  }
  static void initialize(bool /*verbose*/ = false) {}
  static void shutdown() {}
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#else // DUOROU_ENABLE_OLLAMA

// ---------- Real declarations when Ollama extension is enabled ----------

#include "gguf_parser.h"
#include "ollama_path_resolver.h"

namespace duorou {
namespace extensions {
namespace ollama {

// Forward declaration
class InferenceEngine;

// 模型信息结构体
struct ModelInfo {
  std::string model_id;
  std::string file_path;
  std::string architecture;
  unsigned int context_length;
  unsigned int vocab_size;
  bool has_vision;
  bool is_loaded;

  ModelInfo()
      : context_length(0), vocab_size(0), has_vision(false), is_loaded(false) {}
};

// 模型加载状态枚举
enum class ModelLoadState { UNLOADED, LOADING, LOADED, LOAD_ERROR };

// 推理请求结构体
struct InferenceRequest {
  std::string model_id;
  std::string prompt;
  std::vector<std::vector<float>> image_features;
  unsigned int max_tokens;
  float temperature;
  float top_p;

  InferenceRequest() : max_tokens(100), temperature(0.7f), top_p(0.9f) {}
};

// 推理响应结构体
struct InferenceResponse {
  bool success;
  std::string generated_text;
  std::string error_message;
  unsigned int tokens_generated;
  float inference_time_ms;

  InferenceResponse()
      : success(false), tokens_generated(0), inference_time_ms(0.0f) {}
};

class OllamaModelManager {
public:
  explicit OllamaModelManager(bool verbose = false);
  ~OllamaModelManager();

  bool registerModel(const std::string &model_id,
                     const std::string &gguf_file_path);
  bool registerModelByName(const std::string &model_name);
  bool loadModel(const std::string &model_id);
  bool unloadModel(const std::string &model_id);
  bool isModelLoaded(const std::string &model_id) const;

  std::vector<std::string> getRegisteredModels() const;
  std::vector<std::string> getLoadedModels() const;
  const ModelInfo *getModelInfo(const std::string &model_id) const;
  ModelLoadState getModelLoadState(const std::string &model_id) const;

  InferenceResponse generateText(const InferenceRequest &request);
  InferenceResponse generateTextWithImages(const InferenceRequest &request);

  std::vector<InferenceResponse>
  generateTextBatch(const std::vector<InferenceRequest> &requests);

  bool validateModel(const std::string &gguf_file_path,
                     std::string &error_message);

  void clearAllModels();
  size_t getMemoryUsage() const;

  // 公共工具方法
  std::string normalizeModelId(const std::string &model_name) const;

private:
  bool loadModelInternal(const std::string &model_id);
  bool unloadModelInternal(const std::string &model_id);

  bool parseModelInfo(const std::string &gguf_file_path, ModelInfo &model_info);

  std::unique_ptr<InferenceEngine>
  createInferenceEngine(const std::string &model_id);

  bool checkResourceAvailability() const;
  void cleanupUnusedResources();

  std::string generateModelId(const std::string &file_path) const;
  bool isValidModelId(const std::string &model_id) const;
  void log(const std::string &level, const std::string &message) const;

private:
  bool verbose_;
  unsigned int max_concurrent_models_;

  OllamaPathResolver path_resolver_;

  std::unordered_map<std::string, ModelInfo> registered_models_;

  std::unordered_map<std::string, ModelLoadState> model_states_;

  // 存储已加载的推理引擎
  std::unordered_map<std::string, std::unique_ptr<InferenceEngine>>
      inference_engines_;

  mutable size_t total_memory_usage_;
  mutable unsigned int active_models_count_;
};

std::unique_ptr<OllamaModelManager>
createOllamaModelManager(bool verbose = false);

class GlobalModelManager {
public:
  static OllamaModelManager &getInstance();
  static void initialize(bool verbose = false);
  static void shutdown();

private:
  static std::unique_ptr<OllamaModelManager> instance_;
  static bool initialized_;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_ENABLE_OLLAMA

#endif // DUOROU_EXTENSIONS_OLLAMA_OLLAMA_MODEL_MANAGER_H

#endif // __cplusplus
