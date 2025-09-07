#include "ollama_model_manager.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <cstdlib>

namespace duorou {
namespace extensions {
namespace ollama {

// OllamaModelManager实现
OllamaModelManager::OllamaModelManager(bool verbose)
    : verbose_(verbose), max_concurrent_models_(3), path_resolver_(verbose), total_memory_usage_(0), active_models_count_(0) {
    log("INFO", "OllamaModelManager initialized");
}

OllamaModelManager::~OllamaModelManager() {
    clearAllModels();
    log("INFO", "OllamaModelManager destroyed");
}

bool OllamaModelManager::registerModel(const std::string& model_id, const std::string& gguf_file_path) {
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
    
    std::cout << "[DEBUG] OllamaModelManager: About to call parseModelInfo for: " << gguf_file_path << std::endl;
    if (!parseModelInfo(gguf_file_path, model_info)) {
        std::cout << "[DEBUG] OllamaModelManager: parseModelInfo returned false" << std::endl;
        log("ERROR", "Failed to parse model info for: " + gguf_file_path);
        return false;
    }
    std::cout << "[DEBUG] OllamaModelManager: parseModelInfo succeeded" << std::endl;
    
    // 注册模型
    registered_models_[model_id] = model_info;
    model_states_[model_id] = ModelLoadState::UNLOADED;
    
    log("INFO", "Model registered: " + model_id + " (" + model_info.architecture + ")");
    return true;
}

bool OllamaModelManager::registerModelByName(const std::string& model_name) {
    std::cout << "[DEBUG] OllamaModelManager::registerModelByName called with: " << model_name << std::endl;
    log("INFO", "Registering model by name: " + model_name);
    
    // 使用 OllamaPathResolver 解析模型路径
    auto gguf_path = path_resolver_.resolveModelPath(model_name);
    std::cout << "[DEBUG] OllamaModelManager: Resolved GGUF path result: " << (gguf_path ? *gguf_path : "(null)") << std::endl;
    if (!gguf_path) {
        log("ERROR", "Failed to resolve model path for: " + model_name);
        return false;
    }
    
    // 生成有效的模型 ID（将特殊字符替换为下划线）
    std::string model_id = model_name;
    for (char& c : model_id) {
        if (!std::isalnum(c) && c != '_' && c != '-' && c != '.') {
            c = '_';
        }
    }
    
    std::cout << "[DEBUG] OllamaModelManager: Generated model ID: " << model_id << " for model: " << model_name << std::endl;
    log("DEBUG", "Generated model ID: " + model_id + " for model: " + model_name);
    
    // 调用原有的 registerModel 方法
    bool result = registerModel(model_id, *gguf_path);
    std::cout << "[DEBUG] OllamaModelManager: registerModel returned: " << (result ? "true" : "false") << std::endl;
    return result;
}

bool OllamaModelManager::loadModel(const std::string& model_id) {
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
        model_states_[model_id] = ModelLoadState::ERROR;
        log("ERROR", "Failed to load model: " + model_id);
    }
    
    return success;
}

bool OllamaModelManager::unloadModel(const std::string& model_id) {
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

bool OllamaModelManager::isModelLoaded(const std::string& model_id) const {
    auto it = inference_engines_.find(model_id);
    return it != inference_engines_.end();
}

std::vector<std::string> OllamaModelManager::getRegisteredModels() const {
    std::vector<std::string> models;
    models.reserve(registered_models_.size());
    for (const auto& pair : registered_models_) {
        models.push_back(pair.first);
    }
    return models;
}

std::vector<std::string> OllamaModelManager::getLoadedModels() const {
    std::vector<std::string> loaded_models;
    for (const auto& pair : inference_engines_) {
        loaded_models.push_back(pair.first);
    }
    return loaded_models;
}

const ModelInfo* OllamaModelManager::getModelInfo(const std::string& model_id) const {
    auto it = registered_models_.find(model_id);
    return (it != registered_models_.end()) ? &it->second : nullptr;
}

ModelLoadState OllamaModelManager::getModelLoadState(const std::string& model_id) const {
    auto it = model_states_.find(model_id);
    return (it != model_states_.end()) ? it->second : ModelLoadState::UNLOADED;
}

InferenceResponse OllamaModelManager::generateText(const InferenceRequest& request) {
    InferenceResponse response;
    response.success = false;
    
    // 验证请求
    if (request.model_id.empty()) {
        response.error_message = "Model ID is empty";
        return response;
    }
    
    auto engine = getInferenceEngine(request.model_id);
    if (!engine) {
        response.error_message = "Model not loaded: " + request.model_id;
        return response;
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "[DEBUG] Calling engine->generateText with prompt: " << request.prompt.substr(0, 20) << "..." << std::endl;
        response.generated_text = engine->generateText(request.prompt, request.max_tokens);
        auto end = std::chrono::high_resolution_clock::now();
        response.success = true;
        response.tokens_generated = static_cast<uint32_t>(response.generated_text.size());
        response.inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    } catch (const std::exception& e) {
        response.error_message = std::string("Exception during inference: ") + e.what();
    }
    
    return response;
}

InferenceResponse OllamaModelManager::generateTextWithImages(const InferenceRequest& request) {
    InferenceResponse response;
    response.success = false;
    
    if (request.model_id.empty()) {
        response.error_message = "Model ID is empty";
        return response;
    }
    
    auto engine = getInferenceEngine(request.model_id);
    if (!engine) {
        response.error_message = "Model not loaded: " + request.model_id;
        return response;
    }
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        response.generated_text = engine->generateTextWithImages(request.prompt, request.image_features, request.max_tokens);
        auto end = std::chrono::high_resolution_clock::now();
        response.success = true;
        response.tokens_generated = static_cast<uint32_t>(response.generated_text.size());
        response.inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    } catch (const std::exception& e) {
        response.error_message = std::string("Exception during image inference: ") + e.what();
    }
    
    return response;
}

std::vector<InferenceResponse> OllamaModelManager::generateTextBatch(
    const std::vector<InferenceRequest>& requests) {
    std::vector<InferenceResponse> responses;
    responses.reserve(requests.size());
    for (const auto& request : requests) {
        responses.push_back(generateText(request));
    }
    return responses;
}

bool OllamaModelManager::validateModel(const std::string& gguf_file_path, std::string& error_message) {
    try {
        log("DEBUG", "Creating GGUFParser for: " + gguf_file_path);
        GGUFParser parser(true); // 启用详细日志
        
        log("DEBUG", "Calling parseFile...");
        if (!parser.parseFile(gguf_file_path)) {
            error_message = "Failed to parse GGUF file";
            return false;
        }
        
        if (!parser.validateFile()) {
            error_message = "GGUF file validation failed";
            return false;
        }
        
        const auto& architecture = parser.getArchitecture();
        if (!GGUFParser::isSupportedArchitecture(architecture.name)) {
            error_message = std::string("Unsupported architecture: ") + architecture.name;
            return false;
        }
        
        // 验证通过
        return true;
        
    } catch (const std::exception& e) {
        error_message = std::string("Validation error: ") + e.what();
        return false;
    }
}

void OllamaModelManager::clearAllModels() {
    // 卸载所有已加载的模型
    for (auto it = inference_engines_.begin(); it != inference_engines_.end(); ) {
        destroyInferenceEngine(it->first);
        it = inference_engines_.erase(it);
    }
    
    // 重置状态
    model_states_.clear();
    registered_models_.clear();
    active_models_count_ = 0;
    total_memory_usage_ = 0;
}

size_t OllamaModelManager::getMemoryUsage() const {
    // 简化估算内存使用
    total_memory_usage_ = 0;
    
    for (const auto& pair : inference_engines_) {
        // 估算每个模型的内存使用（简化实现）
        const ModelInfo* info = getModelInfo(pair.first);
        if (info) {
            // 粗略估算：参数数量 * 4字节（float32）
            size_t estimated_size = static_cast<size_t>(info->vocab_size) * 1024; // 简化估算
            total_memory_usage_ += estimated_size;
        }
    }
    
    return total_memory_usage_;
}

// 私有方法实现
bool OllamaModelManager::loadModelInternal(const std::string& model_id) {
    const ModelInfo& model_info = registered_models_[model_id];
    
    log("DEBUG", "Starting loadModelInternal for model: " + model_id);
    log("DEBUG", "Model file path: " + model_info.file_path);
    
    // 创建推理引擎
    log("DEBUG", "Creating inference engine...");
    if (!createInferenceEngine(model_id)) {
        log("ERROR", "Failed to create inference engine for model: " + model_id);
        return false;
    }
    log("DEBUG", "Inference engine created successfully");
    
    // 加载模型
    log("DEBUG", "Getting inference engine...");
    Qwen25VLInferenceEngine* engine = getInferenceEngine(model_id);
    if (!engine) {
        log("ERROR", "Failed to get inference engine for model: " + model_id);
        destroyInferenceEngine(model_id);
        return false;
    }
    log("DEBUG", "Got inference engine, calling loadModel...");
    
    try {
        if (!engine->loadModel(model_info.file_path)) {
            log("ERROR", "engine->loadModel() returned false for model: " + model_id);
            destroyInferenceEngine(model_id);
            return false;
        }
        log("DEBUG", "engine->loadModel() completed successfully");
    } catch (const std::exception& e) {
        log("ERROR", "Exception in engine->loadModel(): " + std::string(e.what()));
        destroyInferenceEngine(model_id);
        return false;
    } catch (...) {
        log("ERROR", "Unknown exception in engine->loadModel()");
        destroyInferenceEngine(model_id);
        return false;
    }

    // 配置推理引擎：根据模型上下文长度设置序列长度并启用KV缓存
    if (engine) {
        // 从环境变量读取最大序列长度上限，默认使用较保守的 2048，防止占用过多内存
        uint32_t max_cap = 2048;
        if (const char* env_cap = std::getenv("DUOROU_MAX_SEQ_LEN")) {
            try {
                long v = std::strtol(env_cap, nullptr, 10);
                if (v > 0 && v < static_cast<long>(UINT32_MAX)) {
                    max_cap = static_cast<uint32_t>(v);
                } else {
                    log("WARNING", std::string("Invalid DUOROU_MAX_SEQ_LEN value: ") + env_cap + ", falling back to 2048");
                }
            } catch (...) {
                log("WARNING", std::string("Failed to parse DUOROU_MAX_SEQ_LEN: ") + env_cap + ", falling back to 2048");
            }
        }

        const uint32_t requested_ctx = model_info.context_length > 0 ? model_info.context_length : 2048;
        const uint32_t ctx_len = std::min(requested_ctx, max_cap);
        if (requested_ctx > max_cap) {
            log("WARNING", "Requested context length " + std::to_string(requested_ctx) +
                           " exceeds cap " + std::to_string(max_cap) + ", clamping to cap to avoid OOM");
        }
        log("INFO", "Setting max sequence length: requested=" + std::to_string(requested_ctx) +
                     ", using=" + std::to_string(ctx_len));
        engine->setMaxSequenceLength(ctx_len);
        engine->enableKVCache(true);
        
        // 模型预热：在加载后立即进行一次轻量推理，触发KV缓存初始化与相关日志
        if (verbose_) {
            log("INFO", "Warming up inference engine to initialize KV cache...");
        }
        engine->warmupModel();
    }
    
    return true;
}

bool OllamaModelManager::unloadModelInternal(const std::string& model_id) {
    destroyInferenceEngine(model_id);
    return true;
}

bool OllamaModelManager::parseModelInfo(const std::string& gguf_file_path, ModelInfo& model_info) {
    try {
        log("DEBUG", "Creating GGUFParser for: " + gguf_file_path);
        GGUFParser parser(true); // 启用详细日志
        
        log("DEBUG", "Calling parseFile...");
        if (!parser.parseFile(gguf_file_path)) {
            log("ERROR", "parseFile returned false");
            return false;
        }
        
        log("DEBUG", "Getting architecture...");
        const auto& architecture = parser.getArchitecture();
        log("DEBUG", "Architecture name: " + architecture.name);
        log("DEBUG", "Context length: " + std::to_string(architecture.context_length));
        log("DEBUG", "Has vision: " + std::string(architecture.has_vision ? "true" : "false"));
        
        model_info.architecture = architecture.name;
        model_info.context_length = architecture.context_length;
        model_info.has_vision = architecture.has_vision;
        
        // 获取词汇表大小（简化实现）
        model_info.vocab_size = 151936; // Qwen2.5VL默认值
        
        log("DEBUG", "Model info parsed successfully");
        return true;
        
    } catch (const std::exception& e) {
        log("ERROR", "Failed to parse model info: " + std::string(e.what()));
        return false;
    }
}

Qwen25VLInferenceEngine* OllamaModelManager::getInferenceEngine(const std::string& model_id) {
    auto it = inference_engines_.find(model_id);
    return (it != inference_engines_.end()) ? it->second.get() : nullptr;
}

bool OllamaModelManager::createInferenceEngine(const std::string& model_id) {
    if (inference_engines_.find(model_id) != inference_engines_.end()) {
        return true; // 已存在
    }
    
    try {
        auto engine = std::make_unique<Qwen25VLInferenceEngine>(verbose_);
        inference_engines_[model_id] = std::move(engine);
        return true;
    } catch (const std::exception& e) {
        log("ERROR", "Failed to create inference engine: " + std::string(e.what()));
        return false;
    }
}

void OllamaModelManager::destroyInferenceEngine(const std::string& model_id) {
    auto it = inference_engines_.find(model_id);
    if (it != inference_engines_.end()) {
        inference_engines_.erase(it);
    }
}

bool OllamaModelManager::checkResourceAvailability() const {
    // 简化的资源检查：这里可以根据系统内存、显存等进行判断
    return true;
}

void OllamaModelManager::cleanupUnusedResources() {
    // 占位符：清理未使用的资源
}

std::string OllamaModelManager::generateModelId(const std::string& file_path) const {
    // 根据文件名生成简单的模型ID
    try {
        std::filesystem::path p(file_path);
        std::string stem = p.stem().string();
        for (char& c : stem) {
            if (!std::isalnum(c) && c != '_' && c != '-' && c != '.') {
                c = '_';
            }
        }
        return stem;
    } catch (...) {
        return file_path;
    }
}

bool OllamaModelManager::isValidModelId(const std::string& model_id) const {
    return !model_id.empty();
}

void OllamaModelManager::log(const std::string& level, const std::string& message) const {
    std::cout << "[" << level << "] OllamaModelManager: " << message << std::endl;
}

std::unique_ptr<OllamaModelManager> createOllamaModelManager(bool verbose) {
    return std::make_unique<OllamaModelManager>(verbose);
}

std::unique_ptr<OllamaModelManager> GlobalModelManager::instance_ = nullptr;
bool GlobalModelManager::initialized_ = false;

OllamaModelManager& GlobalModelManager::getInstance() {
    if (!initialized_) {
        initialize();
    }
    return *instance_;
}

void GlobalModelManager::initialize(bool verbose) {
    if (!initialized_) {
        instance_ = std::make_unique<OllamaModelManager>(verbose);
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