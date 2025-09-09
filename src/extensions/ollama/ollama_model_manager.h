#ifndef DUOROU_EXTENSIONS_OLLAMA_OLLAMA_MODEL_MANAGER_H
#define DUOROU_EXTENSIONS_OLLAMA_OLLAMA_MODEL_MANAGER_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "gguf_parser.h"
#include "qwen25vl_modular_engine.h"
#include "ollama_path_resolver.h"

namespace duorou {
namespace extensions {
namespace ollama {

// 模型信息结构
struct ModelInfo {
    std::string model_id;
    std::string file_path;
    std::string architecture;
    uint32_t context_length;
    uint32_t vocab_size;
    bool has_vision;
    bool is_loaded;
    
    ModelInfo() : context_length(0), vocab_size(0), has_vision(false), is_loaded(false) {}
};

// 模型加载状态
enum class ModelLoadState {
    UNLOADED,
    LOADING,
    LOADED,
    ERROR
};

// 推理请求结构
struct InferenceRequest {
    std::string model_id;
    std::string prompt;
    std::vector<std::vector<float>> image_features; // 可选的图像特征
    uint32_t max_tokens;
    float temperature;
    float top_p;
    
    InferenceRequest() : max_tokens(512), temperature(0.7f), top_p(0.9f) {}
};

// 推理响应结构
struct InferenceResponse {
    bool success;
    std::string generated_text;
    std::string error_message;
    uint32_t tokens_generated;
    float inference_time_ms;
    
    InferenceResponse() : success(false), tokens_generated(0), inference_time_ms(0.0f) {}
};

// Ollama模型管理器
class OllamaModelManager {
public:
    explicit OllamaModelManager(bool verbose = false);
    ~OllamaModelManager();
    
    // 模型管理
    bool registerModel(const std::string& model_id, const std::string& gguf_file_path);
    bool registerModelByName(const std::string& model_name); // 从 Ollama 模型名称注册
    bool loadModel(const std::string& model_id);
    bool unloadModel(const std::string& model_id);
    bool isModelLoaded(const std::string& model_id) const;
    
    // 模型信息
    std::vector<std::string> getRegisteredModels() const;
    std::vector<std::string> getLoadedModels() const;
    const ModelInfo* getModelInfo(const std::string& model_id) const;
    ModelLoadState getModelLoadState(const std::string& model_id) const;
    
    // 推理接口
    InferenceResponse generateText(const InferenceRequest& request);
    InferenceResponse generateTextWithImages(const InferenceRequest& request);
    
    // 批量推理
    std::vector<InferenceResponse> generateTextBatch(const std::vector<InferenceRequest>& requests);
    
    // 模型验证
    bool validateModel(const std::string& gguf_file_path, std::string& error_message);
    
    // 资源管理
    void clearAllModels();
    size_t getMemoryUsage() const;
    
    // 配置
    void setVerbose(bool verbose) { verbose_ = verbose; }
    void setMaxConcurrentModels(uint32_t max_models) { max_concurrent_models_ = max_models; }
    void setCustomModelsDir(const std::string& custom_dir) { path_resolver_.setCustomModelsDir(custom_dir); }
    
private:
    // 内部模型加载
    bool loadModelInternal(const std::string& model_id);
    bool unloadModelInternal(const std::string& model_id);
    
    // 模型信息解析
    bool parseModelInfo(const std::string& gguf_file_path, ModelInfo& model_info);
    
    // 推理引擎管理
    Qwen25VLModularEngine* getInferenceEngine(const std::string& model_id);
    bool createInferenceEngine(const std::string& model_id);
    void destroyInferenceEngine(const std::string& model_id);
    
    // 资源检查
    bool checkResourceAvailability() const;
    void cleanupUnusedResources();
    
    // 工具函数
    std::string generateModelId(const std::string& file_path) const;
    bool isValidModelId(const std::string& model_id) const;
    void log(const std::string& level, const std::string& message) const;
    
    // Token转换辅助方法
    std::vector<uint32_t> tokenize(const std::string& text);
    std::string detokenize(const std::vector<uint32_t>& tokens);
    
private:
    bool verbose_;
    uint32_t max_concurrent_models_;
    
    // Ollama 路径解析器
    OllamaPathResolver path_resolver_;
    
    // 模型注册表
    std::unordered_map<std::string, ModelInfo> registered_models_;
    
    // 推理引擎映射
    std::unordered_map<std::string, std::unique_ptr<Qwen25VLModularEngine>> inference_engines_;
    
    // 模型加载状态
    std::unordered_map<std::string, ModelLoadState> model_states_;
    
    // 统计信息
    mutable size_t total_memory_usage_;
    mutable uint32_t active_models_count_;
};

// 工厂函数
std::unique_ptr<OllamaModelManager> createOllamaModelManager(bool verbose = false);

// 全局模型管理器实例（可选）
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