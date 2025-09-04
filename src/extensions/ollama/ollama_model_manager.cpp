#include "ollama_model_manager.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>

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
    auto it = model_states_.find(model_id);
    return it != model_states_.end() && it->second == ModelLoadState::LOADED;
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
    std::vector<std::string> models;
    
    for (const auto& pair : model_states_) {
        if (pair.second == ModelLoadState::LOADED) {
            models.push_back(pair.first);
        }
    }
    
    return models;
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
    
    // 检查模型是否加载
    if (!isModelLoaded(request.model_id)) {
        response.error_message = "Model not loaded: " + request.model_id;
        return response;
    }
    
    // 获取推理引擎
    Qwen25VLInferenceEngine* engine = getInferenceEngine(request.model_id);
    if (!engine) {
        response.error_message = "Failed to get inference engine for: " + request.model_id;
        return response;
    }
    
    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 执行推理
        response.generated_text = engine->generateText(request.prompt, request.max_tokens);
        response.success = true;
        
        // 计算生成的token数量（简化实现）
        auto tokens = engine->tokenize(response.generated_text);
        response.tokens_generated = static_cast<uint32_t>(tokens.size());
        
    } catch (const std::exception& e) {
        response.error_message = "Inference error: " + std::string(e.what());
        response.success = false;
    }
    
    // 计算推理时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    response.inference_time_ms = static_cast<float>(duration.count());
    
    return response;
}

InferenceResponse OllamaModelManager::generateTextWithImages(const InferenceRequest& request) {
    InferenceResponse response;
    
    // 检查模型是否加载
    if (!isModelLoaded(request.model_id)) {
        response.error_message = "Model not loaded: " + request.model_id;
        return response;
    }
    
    // 检查模型是否支持视觉
    const ModelInfo* model_info = getModelInfo(request.model_id);
    if (!model_info || !model_info->has_vision) {
        response.error_message = "Model does not support vision: " + request.model_id;
        return response;
    }
    
    // 获取推理引擎
    Qwen25VLInferenceEngine* engine = getInferenceEngine(request.model_id);
    if (!engine) {
        response.error_message = "Failed to get inference engine for: " + request.model_id;
        return response;
    }
    
    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 执行多模态推理
        response.generated_text = engine->generateTextWithImages(
            request.prompt, request.image_features, request.max_tokens);
        response.success = true;
        
        // 计算生成的token数量
        auto tokens = engine->tokenize(response.generated_text);
        response.tokens_generated = static_cast<uint32_t>(tokens.size());
        
    } catch (const std::exception& e) {
        response.error_message = "Multimodal inference error: " + std::string(e.what());
        response.success = false;
    }
    
    // 计算推理时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    response.inference_time_ms = static_cast<float>(duration.count());
    
    return response;
}

std::vector<InferenceResponse> OllamaModelManager::generateTextBatch(
    const std::vector<InferenceRequest>& requests) {
    
    std::vector<InferenceResponse> responses;
    responses.reserve(requests.size());
    
    for (const auto& request : requests) {
        if (request.image_features.empty()) {
            responses.push_back(generateText(request));
        } else {
            responses.push_back(generateTextWithImages(request));
        }
    }
    
    return responses;
}

bool OllamaModelManager::validateModel(const std::string& gguf_file_path, std::string& error_message) {
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
        
        const auto& architecture = parser.getArchitecture();
        if (!GGUFParser::isSupportedArchitecture(architecture.name)) {
            error_message = "Unsupported architecture: " + architecture.name;
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        error_message = "Validation error: " + std::string(e.what());
        return false;
    }
}

void OllamaModelManager::clearAllModels() {
    log("INFO", "Clearing all models...");
    
    // 卸载所有已加载的模型
    auto loaded_models = getLoadedModels();
    for (const std::string& model_id : loaded_models) {
        unloadModel(model_id);
    }
    
    // 清空所有数据结构
    registered_models_.clear();
    inference_engines_.clear();
    model_states_.clear();
    
    total_memory_usage_ = 0;
    active_models_count_ = 0;
    
    log("INFO", "All models cleared");
}

size_t OllamaModelManager::getMemoryUsage() const {
    // 简化的内存使用计算
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
    
    // 创建推理引擎
    if (!createInferenceEngine(model_id)) {
        return false;
    }
    
    // 加载模型
    Qwen25VLInferenceEngine* engine = getInferenceEngine(model_id);
    if (!engine || !engine->loadModel(model_info.file_path)) {
        destroyInferenceEngine(model_id);
        return false;
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

std::string OllamaModelManager::generateModelId(const std::string& file_path) const {
    std::filesystem::path path(file_path);
    return path.stem().string();
}

bool OllamaModelManager::isValidModelId(const std::string& model_id) const {
    if (model_id.empty() || model_id.length() > 100) {
        return false;
    }
    
    // 检查是否包含有效字符
    for (char c : model_id) {
        if (!std::isalnum(c) && c != '_' && c != '-' && c != '.') {
            return false;
        }
    }
    
    return true;
}

void OllamaModelManager::log(const std::string& level, const std::string& message) const {
    if (verbose_ || level == "ERROR") {
        std::cout << "[" << level << "] OllamaModelManager: " << message << std::endl;
    }
}

// 工厂函数
std::unique_ptr<OllamaModelManager> createOllamaModelManager(bool verbose) {
    return std::make_unique<OllamaModelManager>(verbose);
}

// 全局模型管理器实现
std::unique_ptr<OllamaModelManager> GlobalModelManager::instance_ = nullptr;
bool GlobalModelManager::initialized_ = false;

OllamaModelManager& GlobalModelManager::getInstance() {
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