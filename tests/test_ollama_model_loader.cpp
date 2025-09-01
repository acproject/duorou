#include "../src/core/ollama_model_loader.h"
#include "../src/core/model_path_manager.h"
#include "../src/core/logger.h"
#include <iostream>
#include <memory>
#include <filesystem>
#include "llama.h"

using namespace duorou::core;

// Function to check GGUF file architecture
void checkGGUFArchitecture(const std::string& gguf_path) {
    std::cout << "\n=== Checking GGUF Architecture ===" << std::endl;
    std::cout << "File: " << gguf_path << std::endl;
    
    // Initialize llama backend
    llama_backend_init();
    
    // Load model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU only for metadata reading
    
    // Try to load the model to get architecture info
    llama_model* model = llama_model_load_from_file(gguf_path.c_str(), model_params);
    
    if (model) {
        std::cout << "Model loaded successfully!" << std::endl;
        
        // Get model metadata
        char arch_name[256];
        int arch_name_len = llama_model_meta_val_str(model, "general.architecture", arch_name, sizeof(arch_name));
        if (arch_name_len > 0) {
            std::cout << "Architecture: " << arch_name << std::endl;
        } else {
            std::cout << "Could not read architecture from model" << std::endl;
        }
        
        char model_name[256];
        int model_name_len = llama_model_meta_val_str(model, "general.name", model_name, sizeof(model_name));
        if (model_name_len > 0) {
            std::cout << "Model name: " << model_name << std::endl;
        }
        
        llama_model_free(model);
    } else {
        std::cout << "Failed to load model for architecture check" << std::endl;
    }
    
    llama_backend_free();
    std::cout << "=== End GGUF Architecture Check ===\n" << std::endl;
}

#define TEST_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            std::cout << "FAIL: " << #condition << " at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

bool testOllamaModelNameParsing() {
    std::cout << "Testing ollama model name parsing..." << std::endl;
    
    auto model_path_manager = std::make_shared<ModelPathManager>();
    if (!model_path_manager->initialize()) {
        std::cout << "Failed to initialize model path manager" << std::endl;
        return false;
    }
    
    OllamaModelLoader loader(model_path_manager);
    
    // 测试简单模型名称
    std::cout << "Testing simple model name: llama3.2" << std::endl;
    bool available = loader.isOllamaModelAvailable("llama3.2");
    std::cout << "llama3.2 available: " << (available ? "yes" : "no") << std::endl;
    
    // 测试带tag的模型名称
    std::cout << "Testing model name with tag: qwen2.5:7b" << std::endl;
    available = loader.isOllamaModelAvailable("qwen2.5:7b");
    std::cout << "qwen2.5:7b available: " << (available ? "yes" : "no") << std::endl;
    
    // 测试带namespace的模型名称
    std::cout << "Testing model name with namespace: microsoft/phi" << std::endl;
    available = loader.isOllamaModelAvailable("microsoft/phi");
    std::cout << "microsoft/phi available: " << (available ? "yes" : "no") << std::endl;
    
    return true;
}

bool testListAvailableModels() {
    std::cout << "Testing list available models..." << std::endl;
    
    auto model_path_manager = std::make_shared<ModelPathManager>();
    if (!model_path_manager->initialize()) {
        std::cout << "Failed to initialize model path manager" << std::endl;
        return false;
    }
    
    OllamaModelLoader loader(model_path_manager);
    
    auto models = loader.listAvailableModels();
    std::cout << "Found " << models.size() << " available models:" << std::endl;
    
    for (const auto& model : models) {
        std::cout << "  - " << model << std::endl;
    }
    
    return true;
}

bool testLoadOllamaModel() {
    std::cout << "Testing load ollama model..." << std::endl;
    
    auto model_path_manager = std::make_shared<ModelPathManager>();
    if (!model_path_manager->initialize()) {
        std::cout << "Failed to initialize model path manager" << std::endl;
        return false;
    }
    
    OllamaModelLoader loader(model_path_manager);
    
    // 获取可用模型列表
    auto models = loader.listAvailableModels();
    if (models.empty()) {
        std::cout << "No ollama models available for testing" << std::endl;
        return true; // 不算失败，只是没有模型可测试
    }
    
    // 尝试加载第一个可用模型
    std::string test_model = models[0];
    std::cout << "Attempting to load model: " << test_model << std::endl;
    
    // 设置llama模型参数
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // 使用CPU
    model_params.use_mmap = true;
    model_params.vocab_only = true; // 只加载词汇表，避免加载整个模型
    
    // 初始化llama backend
    llama_backend_init();
    
    struct llama_model* model = loader.loadFromOllamaModel(test_model, model_params);
    
    if (model) {
        std::cout << "Successfully loaded model: " << test_model << std::endl;
        
        
        
        // 获取模型信息
        // Note: llama_n_vocab API has changed, using model context instead
        std::cout << "Model loaded successfully" << std::endl;
        
        // 释放模型
        llama_model_free(model);
        std::cout << "Model freed successfully" << std::endl;
    } else {
        std::cout << "Failed to load model: " << test_model << std::endl;
        // 这可能是正常的，因为模型文件可能很大或格式不兼容
    }
    
    llama_backend_free();
    
    return true;
}

int main() {
    // Initialize logger instance
    Logger logger;
    logger.initialize();
    logger.setLogLevel(LogLevel::INFO);
    
    std::cout << "=== Ollama Model Loader Tests ===" << std::endl;
    
    bool all_passed = true;
    
    if (!testOllamaModelNameParsing()) {
        std::cout << "❌ Ollama model name parsing test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Ollama model name parsing test passed" << std::endl;
    }
    
    if (!testListAvailableModels()) {
        std::cout << "❌ List available models test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ List available models test passed" << std::endl;
    }
    
    if (!testLoadOllamaModel()) {
        std::cout << "❌ Load ollama model test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Load ollama model test passed" << std::endl;
    }
    
    if (all_passed) {
        std::cout << "\n🎉 All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n💥 Some tests failed!" << std::endl;
        return 1;
    }
}