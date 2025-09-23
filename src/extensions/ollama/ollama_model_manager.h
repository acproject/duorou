#ifndef DUOROU_EXTENSIONS_OLLAMA_OLLAMA_MODEL_MANAGER_H
#define DUOROU_EXTENSIONS_OLLAMA_OLLAMA_MODEL_MANAGER_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
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
    
    ModelInfo() : context_length(0), vocab_size(0), has_vision(false), is_loaded(false) {}
};

// 模型加载状态枚举
enum class ModelLoadState {
    UNLOADED,
    LOADING,
    LOADED,
    LOAD_ERROR
};

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
    
    InferenceResponse() : success(false), tokens_generated(0), inference_time_ms(0.0f) {}
};

class OllamaModelManager {
public:
    explicit OllamaModelManager(bool verbose = false);
    ~OllamaModelManager();

    bool registerModel(const std::string& model_id, const std::string& gguf_file_path);
    bool registerModelByName(const std::string& model_name);
    bool loadModel(const std::string& model_id);
    bool unloadModel(const std::string& model_id);
    bool isModelLoaded(const std::string& model_id) const;

    std::vector<std::string> getRegisteredModels() const;
    std::vector<std::string> getLoadedModels() const;
    const ModelInfo* getModelInfo(const std::string& model_id) const;
    ModelLoadState getModelLoadState(const std::string& model_id) const;

    InferenceResponse generateText(const InferenceRequest& request);
    InferenceResponse generateTextWithImages(const InferenceRequest& request);

    std::vector<InferenceResponse> generateTextBatch(const std::vector<InferenceRequest>& requests);

    bool validateModel(const std::string& gguf_file_path, std::string& error_message);

    void clearAllModels();
    size_t getMemoryUsage() const;

    // 公共工具方法
    std::string normalizeModelId(const std::string& model_name) const;

private:
    bool loadModelInternal(const std::string& model_id);
    bool unloadModelInternal(const std::string& model_id);

    bool parseModelInfo(const std::string& gguf_file_path, ModelInfo& model_info);

    std::unique_ptr<InferenceEngine> createInferenceEngine(const std::string& model_id);

    bool checkResourceAvailability() const;
    void cleanupUnusedResources();

    std::string generateModelId(const std::string& file_path) const;
    bool isValidModelId(const std::string& model_id) const;
    void log(const std::string& level, const std::string& message) const;

private:
    bool verbose_;
    unsigned int max_concurrent_models_;

    OllamaPathResolver path_resolver_;

    std::unordered_map<std::string, ModelInfo> registered_models_;

    std::unordered_map<std::string, ModelLoadState> model_states_;
    
    // 存储已加载的推理引擎
    std::unordered_map<std::string, std::unique_ptr<InferenceEngine>> inference_engines_;

    mutable size_t total_memory_usage_;
    mutable unsigned int active_models_count_;
};

std::unique_ptr<OllamaModelManager> createOllamaModelManager(bool verbose = false);

class GlobalModelManager {
public:
    static OllamaModelManager& getInstance();
    static void initialize(bool verbose = false);
    static void shutdown();

private:
    static std::unique_ptr<OllamaModelManager> instance_;
    static bool initialized_;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_OLLAMA_MODEL_MANAGER_H