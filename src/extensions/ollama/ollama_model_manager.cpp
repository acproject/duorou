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

OllamaModelManager::OllamaModelManager(bool verbose)
    : verbose_(verbose), max_concurrent_models_(3), path_resolver_(verbose), total_memory_usage_(0), active_models_count_(0) {
    log("INFO", "OllamaModelManager initialized");
}

OllamaModelManager::~OllamaModelManager() {
    clearAllModels();
    log("INFO", "OllamaModelManager destroyed");
}

bool OllamaModelManager::registerModel(const std::string& model_id, const std::string& gguf_file_path) {
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

bool OllamaModelManager::registerModelByName(const std::string& model_name) {
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

bool OllamaModelManager::loadModel(const std::string& model_id) {
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
    }

    return success;
}

bool OllamaModelManager::isModelLoaded(const std::string& model_id) const {
    return model_states_.find(model_id) != model_states_.end() && 
           model_states_.at(model_id) == ModelLoadState::LOADED;
}

std::vector<std::string> OllamaModelManager::getRegisteredModels() const {
    std::vector<std::string> models;
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
    
    if (request.model_id.empty() || request.prompt.empty()) {
        response.error_message = "Model ID or prompt is empty";
        return response;
    }

    Qwen25VLModularEngine* engine = getInferenceEngine(request.model_id);
    if (!engine) {
        response.error_message = "Failed to get inference engine for model: " + request.model_id;
        return response;
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();
        auto input_tokens = tokenize(request.prompt);
        auto output_tokens = engine->generateText(input_tokens, request.max_tokens);
        response.generated_text = detokenize(output_tokens);
        auto end = std::chrono::high_resolution_clock::now();
        response.success = true;
        response.tokens_generated = static_cast<uint32_t>(response.generated_text.size());
        response.inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    } catch (const std::exception& e) {
        response.error_message = std::string("Exception during text generation: ") + e.what();
    }

    return response;
}

InferenceResponse OllamaModelManager::generateTextWithImages(const InferenceRequest& request) {
    InferenceResponse response;
    
    if (request.model_id.empty() || request.prompt.empty()) {
        response.error_message = "Model ID or prompt is empty";
        return response;
    }

    Qwen25VLModularEngine* engine = getInferenceEngine(request.model_id);
    if (!engine) {
        response.error_message = "Failed to get inference engine for model: " + request.model_id;
        return response;
    }
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        auto input_tokens = tokenize(request.prompt);
        // 将image_features转换为Tensor格式
        algorithms::Tensor image_tensor; // 这里需要实际的转换逻辑
        auto output_tokens = engine->generateMultimodal(input_tokens, image_tensor, request.max_tokens);
        response.generated_text = detokenize(output_tokens);
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
    for (const auto& request : requests) {
        responses.push_back(generateText(request));
    }
    return responses;
}

bool OllamaModelManager::validateModel(const std::string& gguf_file_path, std::string& error_message) {
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
    for (const std::string& model_id : loaded_models) {
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

bool OllamaModelManager::loadModelInternal(const std::string& model_id) {
    auto model_it = registered_models_.find(model_id);
    if (model_it == registered_models_.end()) {
        log("ERROR", "Model not found in registry: " + model_id);
        return false;
    }

    const ModelInfo& model_info = model_it->second;
    
    // 创建推理引擎
    if (!createInferenceEngine(model_id)) {
        log("ERROR", "Failed to create inference engine for model: " + model_id);
        return false;
    }
    
    // 获取推理引擎
    Qwen25VLModularEngine* engine = getInferenceEngine(model_id);
    if (!engine) {
        log("ERROR", "Failed to get inference engine for model: " + model_id);
        destroyInferenceEngine(model_id);
        return false;
    }
    
    // 加载权重
    try {
        if (!engine->loadWeights(model_info.file_path)) {
            log("ERROR", "engine->loadWeights() returned false for model: " + model_id);
            destroyInferenceEngine(model_id);
            return false;
        }
        log("DEBUG", "engine->loadWeights() completed successfully");
    } catch (const std::exception& e) {
        log("ERROR", "Exception in engine->loadWeights(): " + std::string(e.what()));
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
        GGUFParser parser(verbose_);
        if (!parser.parseFile(gguf_file_path)) {
            return false;
        }
        
        const auto& architecture = parser.getArchitecture();
        model_info.architecture = architecture.name;
        model_info.context_length = architecture.context_length;
        model_info.vocab_size = 151936; // Qwen2.5VL默认值
        model_info.has_vision = architecture.has_vision;
        
        return true;
    } catch (const std::exception& e) {
        log("ERROR", "Exception parsing model info: " + std::string(e.what()));
        return false;
    }
}

Qwen25VLModularEngine* OllamaModelManager::getInferenceEngine(const std::string& model_id) {
    auto it = inference_engines_.find(model_id);
    return (it != inference_engines_.end()) ? it->second.get() : nullptr;
}

bool OllamaModelManager::createInferenceEngine(const std::string& model_id) {
    if (inference_engines_.find(model_id) != inference_engines_.end()) {
        return true; // 已存在
    }
    
    try {
        auto engine = std::make_unique<Qwen25VLModularEngine>();
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
    return active_models_count_ < max_concurrent_models_;
}

void OllamaModelManager::cleanupUnusedResources() {
    // 实现资源清理逻辑
}

std::string OllamaModelManager::generateModelId(const std::string& file_path) const {
    std::filesystem::path path(file_path);
    std::string filename = path.stem().string();
    
    // 移除常见的模型文件后缀
    const std::vector<std::string> suffixes = {".gguf", ".bin", ".safetensors"};
    for (const std::string& suffix : suffixes) {
        if (filename.size() >= suffix.size() && 
            filename.substr(filename.size() - suffix.size()) == suffix) {
            filename = filename.substr(0, filename.length() - suffix.length());
            break;
        }
    }
    
    return filename;
}

bool OllamaModelManager::isValidModelId(const std::string& model_id) const {
    return !model_id.empty() && model_id.find('/') == std::string::npos;
}

void OllamaModelManager::log(const std::string& level, const std::string& message) const {
    if (verbose_ || level == "ERROR") {
        std::cout << "[" << level << "] OllamaModelManager: " << message << std::endl;
    }
}

// Token转换辅助方法实现
std::vector<uint32_t> OllamaModelManager::tokenize(const std::string& text) {
    // 简单的tokenize实现，实际应该使用真正的tokenizer
    std::vector<uint32_t> tokens;
    for (char c : text) {
        tokens.push_back(static_cast<uint32_t>(c));
    }
    return tokens;
}

std::string OllamaModelManager::detokenize(const std::vector<uint32_t>& tokens) {
    // 简单的detokenize实现，实际应该使用真正的detokenizer
    std::string text;
    for (uint32_t token : tokens) {
        if (token < 256) {
            text += static_cast<char>(token);
        }
    }
    return text;
}

std::unique_ptr<OllamaModelManager> createOllamaModelManager(bool verbose) {
    return std::make_unique<OllamaModelManager>(verbose);
}

std::unique_ptr<OllamaModelManager> GlobalModelManager::instance_ = nullptr;
bool GlobalModelManager::initialized_ = false;

OllamaModelManager& GlobalModelManager::getInstance() {
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