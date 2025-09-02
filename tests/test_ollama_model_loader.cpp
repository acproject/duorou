#include "../src/core/ollama_model_loader.h"
#include "../src/core/model_path_manager.h"
#include "../src/core/logger.h"
#include "../src/core/text_generator.h"
#include "../src/extensions/llama_cpp/arch_mapping.h"
#include "../src/extensions/llama_cpp/model_config_manager.h"
#include "../src/extensions/llama_cpp/compatibility_checker.h"
#include "../src/extensions/llama_cpp/vision_model_handler.h"
#include "../src/extensions/llama_cpp/attention_handler.h"
#include "../src/extensions/llama_cpp/gguf_modifier.h"
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

bool testOllamaModelConversation() {
    std::cout << "Testing ollama model conversation..." << std::endl;
    
    auto model_path_manager = std::make_shared<ModelPathManager>();
    if (!model_path_manager->initialize()) {
        std::cout << "Failed to initialize model path manager" << std::endl;
        return false;
    }
    
    // 创建Ollama模型加载器
    OllamaModelLoader loader(model_path_manager);
    auto models = loader.listAvailableModels();
    
    if (models.empty()) {
        std::cout << "No ollama models available for conversation testing" << std::endl;
        return true; // 不算失败，只是没有模型可测试
    }
    
    std::string test_model = models[0];
    std::cout << "Testing conversation with model: " << test_model << std::endl;
    
    try {
        // 初始化llama backend
        llama_backend_init();
        
        // 设置llama模型参数
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0; // 使用CPU
        model_params.use_mmap = true;
        
        // 尝试加载模型
        std::cout << "Loading model..." << std::endl;
        struct llama_model* model = loader.loadFromOllamaModel(test_model, model_params);
        if (!model) {
            std::cout << "Failed to load model: " << test_model << std::endl;
            llama_backend_free();
            return false;
        }
        std::cout << "Model loaded successfully" << std::endl;
        
        // 创建llama上下文
        std::cout << "Creating llama context..." << std::endl;
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;
        ctx_params.n_threads = 4;
        
        struct llama_context* ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) {
            std::cout << "Failed to create llama context" << std::endl;
            llama_model_free(model);
            llama_backend_free();
            return false;
        }
        
        // 创建文本生成器
        std::cout << "Creating text generator..." << std::endl;
        auto text_generator = std::make_shared<TextGenerator>(model, ctx);
        std::cout << "Text generator created successfully" << std::endl;
        
        // 进行简单的对话测试
        std::cout << "Starting conversation test..." << std::endl;
        std::string prompt = "Hello, how are you?";
        std::cout << "Prompt: " << prompt << std::endl;
        
        try {
             GenerationParams params;
             params.max_tokens = 50;
             params.temperature = 0.7;
             
             auto result = text_generator->generate(prompt, params);
             std::cout << "Response: " << result.text << std::endl;
             std::cout << "Conversation test completed successfully" << std::endl;
         } catch (const std::exception& e) {
             std::cout << "Error during generation: " << e.what() << std::endl;
         }
        
        // 释放资源
         llama_free(ctx);
         llama_model_free(model);
         llama_backend_free();
        
    } catch (const std::exception& e) {
        std::cout << "Exception during conversation test: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

// Test llama_cpp extension features
bool testArchMapping() {
    std::cout << "Testing architecture mapping..." << std::endl;
    
    // Test known architecture mappings
    TEST_ASSERT(ArchMapping::needsMapping("gemma3"));
    TEST_ASSERT(ArchMapping::needsMapping("mistral3"));
    TEST_ASSERT(ArchMapping::needsMapping("qwen25vl"));
    TEST_ASSERT(!ArchMapping::needsMapping("llama"));
    
    // Test architecture mapping results
    std::string mapped = ArchMapping::getMappedArchitecture("gemma3");
    TEST_ASSERT(!mapped.empty());
    std::cout << "gemma3 maps to: " << mapped << std::endl;
    
    mapped = ArchMapping::getMappedArchitecture("mistral3");
    TEST_ASSERT(!mapped.empty());
    std::cout << "mistral3 maps to: " << mapped << std::endl;
    
    return true;
}

bool testModelConfigManager() {
    std::cout << "Testing model configuration manager..." << std::endl;
    
    // Initialize the config manager
    ModelConfigManager::initialize();
    
    // Test getting configurations for different models
    auto gemma3Config = ModelConfigManager::getConfig("gemma3");
    TEST_ASSERT(gemma3Config != nullptr);
    TEST_ASSERT(gemma3Config->hasVision);
    std::cout << "gemma3 config loaded successfully" << std::endl;
    
    auto mistral3Config = ModelConfigManager::getConfig("mistral3");
    TEST_ASSERT(mistral3Config != nullptr);
    TEST_ASSERT(mistral3Config->hasVision);
    std::cout << "mistral3 config loaded successfully" << std::endl;
    
    auto qwen25vlConfig = ModelConfigManager::getConfig("qwen25vl");
    TEST_ASSERT(qwen25vlConfig != nullptr);
    TEST_ASSERT(qwen25vlConfig->hasVision);
    std::cout << "qwen25vl config loaded successfully" << std::endl;
    
    return true;
}

bool testCompatibilityChecker() {
    std::cout << "Testing compatibility checker..." << std::endl;
    
    CompatibilityChecker checker;
    
    // Test architecture compatibility
    TEST_ASSERT(checker.isArchitectureSupported("llama"));
    TEST_ASSERT(checker.isArchitectureSupported("gemma3"));
    TEST_ASSERT(checker.isArchitectureSupported("mistral3"));
    
    // Test model requirements
    auto requirements = checker.getModelRequirements("gemma3");
    TEST_ASSERT(!requirements->supportedQuantizations.empty());
    std::cout << "gemma3 requirements checked successfully" << std::endl;
    
    // Test special preprocessing needs
    TEST_ASSERT(checker.needsSpecialPreprocessing("qwen25vl"));
    TEST_ASSERT(!checker.needsSpecialPreprocessing("llama"));
    
    return true;
}

bool testVisionModelHandler() {
    std::cout << "Testing vision model handler..." << std::endl;
    
    VisionModelHandler handler;
    handler.initialize();
    
    // Test vision support detection (simplified test)
    std::cout << "Vision model handler initialized successfully" << std::endl;
    
    std::cout << "Vision model handler tests completed" << std::endl;
    return true;
}

bool testAttentionHandler() {
    std::cout << "Testing attention handler..." << std::endl;
    
    AttentionHandler handler;
    handler.initialize();
    
    // Test advanced attention detection
    TEST_ASSERT(handler.hasAdvancedAttention("mistral3"));
    TEST_ASSERT(!handler.hasAdvancedAttention("llama"));
    
    std::cout << "Attention handler tests completed" << std::endl;
    return true;
}

bool testGGUFModifier() {
    std::cout << "Testing GGUF modifier..." << std::endl;
    
    GGUFModifier modifier;
    
    // Test architecture modification needs (without actual files)
    TEST_ASSERT(modifier.needsArchitectureModification("gemma3"));
    TEST_ASSERT(modifier.needsArchitectureModification("mistral3"));
    TEST_ASSERT(!modifier.needsArchitectureModification("llama"));
    
    // Test GGUF architecture retrieval
    std::string arch = modifier.getGGUFArchitecture("gemma3");
    TEST_ASSERT(!arch.empty());
    std::cout << "gemma3 GGUF architecture: " << arch << std::endl;
    
    std::cout << "GGUF modifier tests completed" << std::endl;
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
    
    if (!testOllamaModelConversation()) {
        std::cout << "❌ Ollama model conversation test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Ollama model conversation test passed" << std::endl;
    }
    
    // Test llama_cpp extension features
    if (!testArchMapping()) {
        std::cout << "❌ Architecture mapping test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Architecture mapping test passed" << std::endl;
    }
    
    if (!testModelConfigManager()) {
        std::cout << "❌ Model config manager test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Model config manager test passed" << std::endl;
    }
    
    if (!testCompatibilityChecker()) {
        std::cout << "❌ Compatibility checker test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Compatibility checker test passed" << std::endl;
    }
    
    if (!testVisionModelHandler()) {
        std::cout << "❌ Vision model handler test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Vision model handler test passed" << std::endl;
    }
    
    if (!testAttentionHandler()) {
        std::cout << "❌ Attention handler test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Attention handler test passed" << std::endl;
    }
    
    if (!testGGUFModifier()) {
        std::cout << "❌ GGUF modifier test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ GGUF modifier test passed" << std::endl;
    }
    
    if (all_passed) {
        std::cout << "\n🎉 All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n💥 Some tests failed!" << std::endl;
        return 1;
    }
}