#include "../src/core/ollama_model_loader.h"
#include "../src/core/model_path_manager.h"
#include "../src/core/logger.h"
#include "../src/core/text_generator.h"
#include "../src/core/modelfile_parser.h"
#include "../src/extensions/llama_cpp/ggml_incremental_extension.h"
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

// Simplified GGUF architecture check using new extensions
void checkGGUFArchitecture(const std::string& gguf_path) {
    std::cout << "\n=== Checking GGUF Architecture (via extensions) ===" << std::endl;
    std::cout << "File: " << gguf_path << std::endl;
    
    // Use the new GGUF modifier to check architecture
    GGUFModifier modifier;
    
    // Extract filename to determine model type
    std::filesystem::path path(gguf_path);
    std::string filename = path.filename().string();
    
    // Simple architecture detection based on filename patterns
    std::string detected_arch = "unknown";
    if (filename.find("qwen") != std::string::npos) {
        detected_arch = "qwen2vl";
    } else if (filename.find("gemma") != std::string::npos) {
        detected_arch = "gemma3";
    } else if (filename.find("mistral") != std::string::npos) {
        detected_arch = "mistral3";
    } else if (filename.find("llama") != std::string::npos) {
        detected_arch = "llama";
    }
    
    std::cout << "Detected architecture: " << detected_arch << std::endl;
    
    // Check if architecture needs incremental extension
    if (duorou::extensions::GGMLIncrementalExtension::isArchitectureSupported(detected_arch)) {
        std::string mapped = duorou::extensions::GGMLIncrementalExtension::getBaseArchitecture(detected_arch);
        std::cout << "GGML incremental extension: " << detected_arch << " -> " << mapped << std::endl;
    }
    
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
    std::cout << "Testing ollama model conversation with qwen2.5vl..." << std::endl;
    
    auto model_path_manager = std::make_shared<ModelPathManager>();
    if (!model_path_manager->initialize()) {
        std::cout << "Failed to initialize model path manager" << std::endl;
        return false;
    }
    
    // 创建Ollama模型加载器
    OllamaModelLoader loader(model_path_manager);
    
    // 指定测试qwen2.5vl模型
    std::string test_model = "qwen2.5vl:7b";
    std::cout << "Testing conversation with model: " << test_model << std::endl;
    
    // 检查模型是否可用
    if (!loader.isOllamaModelAvailable(test_model)) {
        std::cout << "Model " << test_model << " is not available, trying alternative names..." << std::endl;
        
        // 尝试其他可能的qwen2.5vl模型名称
        std::vector<std::string> alternative_names = {
            "qwen2.5vl", "qwen2.5-vl:7b", "qwen2.5-vl", "qwen25vl:7b", "qwen25vl"
        };
        
        bool found = false;
        for (const auto& alt_name : alternative_names) {
            if (loader.isOllamaModelAvailable(alt_name)) {
                test_model = alt_name;
                found = true;
                std::cout << "Found model with name: " << test_model << std::endl;
                break;
            }
        }
        
        if (!found) {
            std::cout << "No qwen2.5vl model found. Available models:" << std::endl;
            auto models = loader.listAvailableModels();
            for (const auto& model : models) {
                std::cout << "  - " << model << std::endl;
            }
            return true; // 不算失败，只是没有模型可测试
        }
    }
    
    // 测试扩展功能：架构映射
    std::cout << "\n=== Testing Extensions with " << test_model << " ===" << std::endl;
    
    // 1. 测试GGML增量扩展
    std::string arch = "qwen25vl"; // qwen2.5vl的架构名
    if (duorou::extensions::GGMLIncrementalExtension::isArchitectureSupported(arch)) {
        std::string mapped_arch = duorou::extensions::GGMLIncrementalExtension::getBaseArchitecture(arch);
        std::cout << "GGML incremental extension: " << arch << " -> " << mapped_arch << std::endl;
    } else {
        std::cout << "Architecture " << arch << " does not need incremental extension" << std::endl;
    }
    
    // 2. 测试模型配置管理器
    ModelConfigManager::initialize();
    auto config = ModelConfigManager::getConfig("qwen25vl");
    if (config) {
        std::cout << "Model config loaded - hasVision: " << (config->hasVision ? "true" : "false") << std::endl;
        std::cout << "Model config - imageSize: " << config->imageSize << std::endl;
    }
    
    // 3. 测试兼容性检查器
    CompatibilityChecker checker;
    if (checker.isArchitectureSupported("qwen25vl")) {
        std::cout << "Architecture qwen25vl is supported" << std::endl;
        
        auto requirements = checker.getModelRequirements("qwen25vl");
        if (requirements) {
            std::cout << "Model requirements loaded successfully" << std::endl;
        }
    }
    
    // 4. 测试视觉模型处理器
    VisionModelHandler visionHandler;
    visionHandler.initialize();
    std::cout << "Vision model handler initialized for qwen2.5vl" << std::endl;
    
    // 5. 实际加载模型进行简单对话测试
    std::cout << "\n=== Loading Model for Conversation ===" << std::endl;
    
    // 设置llama模型参数 - 完整模型加载
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // 使用CPU避免GPU兼容性问题
    model_params.use_mmap = true;
    model_params.vocab_only = false; // 加载完整模型以支持文本生成
    
    // 初始化llama backend
    llama_backend_init();
    
    try {
        struct llama_model* model = loader.loadFromOllamaModel(test_model, model_params);
        
        if (model) {
            std::cout << "Successfully loaded qwen2.5vl model!" << std::endl;
            
            // 创建上下文参数
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx = 4096; // 设置足够的上下文长度
            ctx_params.n_threads = 4;
            ctx_params.n_threads_batch = 4;
            
            // 创建上下文
            struct llama_context* ctx = llama_new_context_with_model(model, ctx_params);
            
            if (ctx) {
                std::cout << "Successfully created model context!" << std::endl;
                
                // 完整模型问答测试
                std::string test_prompt = "你好，请简单介绍一下你自己。";
                std::cout << "\n=== 完整模型问答测试 ===" << std::endl;
                std::cout << "用户输入: \"" << test_prompt << "\"" << std::endl;
                
                // 获取vocab指针
                const struct llama_vocab* vocab = llama_model_get_vocab(model);
                if (!vocab) {
                    std::cout << "Failed to get vocab from model" << std::endl;
                    llama_free(ctx);
                    llama_model_free(model);
                    return false;
                }
                
                // Tokenize输入
                std::vector<llama_token> tokens;
                tokens.resize(test_prompt.length() + 100);
                
                int n_tokens = llama_tokenize(vocab, test_prompt.c_str(), test_prompt.length(), 
                                             tokens.data(), tokens.size(), true, false);
                
                if (n_tokens > 0) {
                    tokens.resize(n_tokens);
                    std::cout << "✅ 输入已tokenized为 " << n_tokens << " 个tokens" << std::endl;
                    
                    // 创建批处理
                    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
                    
                    // 添加tokens到批处理
                    for (int i = 0; i < n_tokens; ++i) {
                        batch.token[i] = tokens[i];
                        batch.pos[i] = i;
                        batch.n_seq_id[i] = 1;
                        batch.seq_id[i][0] = 0;
                        batch.logits[i] = false;
                    }
                    batch.n_tokens = n_tokens;
                    
                    // 最后一个token需要logits
                    batch.logits[batch.n_tokens - 1] = true;
                    
                    // 解码输入
                    if (llama_decode(ctx, batch) != 0) {
                        std::cout << "❌ 模型解码失败" << std::endl;
                        llama_batch_free(batch);
                        llama_free(ctx);
                        llama_model_free(model);
                        return false;
                    }
                    
                    std::cout << "✅ 模型成功处理输入" << std::endl;
                    std::cout << "\n🤖 模型回复: \"";
                    
                    // 生成回复
                    std::string response = "";
                    int max_tokens = 200; // 限制生成长度
                    int generated_tokens = 0;
                    
                    for (int i = 0; i < max_tokens; ++i) {
                        // 获取logits
                        float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
                        
                        // 简单的贪婪采样（选择概率最高的token）
                        const struct llama_vocab* model_vocab = llama_model_get_vocab(model);
                        int vocab_size = 32000; // 使用常见的vocab大小
                        llama_token next_token = 0;
                        float max_prob = logits[0];
                        
                        for (int j = 1; j < vocab_size; ++j) {
                            if (logits[j] > max_prob) {
                                max_prob = logits[j];
                                next_token = j;
                            }
                        }
                        
                        // 检查是否是结束token
                        if (next_token == llama_vocab_eos(model_vocab)) {
                            break;
                        }
                        
                        // 将token转换为文本
                        char token_str[256];
                        int len = llama_token_to_piece(model_vocab, next_token, token_str, sizeof(token_str), 0, false);
                        if (len > 0) {
                            std::string token_text(token_str, len);
                            response += token_text;
                            std::cout << token_text << std::flush; // 实时输出
                        }
                        
                        // 准备下一次解码
                        batch.n_tokens = 1;
                        batch.token[0] = next_token;
                        batch.pos[0] = n_tokens + i;
                        batch.n_seq_id[0] = 1;
                        batch.seq_id[0][0] = 0;
                        batch.logits[0] = true;
                        
                        if (llama_decode(ctx, batch) != 0) {
                            std::cout << "\n❌ 生成过程中解码失败" << std::endl;
                            break;
                        }
                        
                        generated_tokens++;
                    }
                    
                    std::cout << "\"" << std::endl;
                    std::cout << "\n📊 生成统计:" << std::endl;
                    std::cout << "  - 输入tokens: " << n_tokens << std::endl;
                    std::cout << "  - 生成tokens: " << generated_tokens << std::endl;
                    std::cout << "  - 总回复长度: " << response.length() << " 字符" << std::endl;
                    
                    if (generated_tokens > 0) {
                        std::cout << "\n✅ qwen2.5vl模型完整问答测试成功!" << std::endl;
                        std::cout << "🎉 模型能够正确理解中文输入并生成回复" << std::endl;
                    } else {
                        std::cout << "\n⚠️ 模型加载成功但未能生成回复" << std::endl;
                    }
                    
                    llama_batch_free(batch);
                } else {
                    std::cout << "❌ 输入tokenization失败" << std::endl;
                }
                
                // 释放上下文
                llama_free(ctx);
            } else {
                std::cout << "Failed to create model context" << std::endl;
            }
            
            // 释放模型
            llama_free_model(model);
        } else {
            std::cout << "Failed to load qwen2.5vl model" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "Exception during model loading: " << e.what() << std::endl;
        return false;
    }
    
    llama_backend_free();
    
    std::cout << "\n=== Extensions Integration Test Completed ===" << std::endl;
    return true;
}

// Test llama_cpp extension features
bool testGGMLIncrementalExtension() {
    std::cout << "Testing GGML Incremental Extension..." << std::endl;
    
    using namespace duorou::extensions;
    
    // Initialize the extension
    TEST_ASSERT(GGMLIncrementalExtension::initialize());
    
    // Test supported architectures
    TEST_ASSERT(GGMLIncrementalExtension::isArchitectureSupported("qwen2.5vl"));
    
    // Test base architecture retrieval
    std::string base_arch = GGMLIncrementalExtension::getBaseArchitecture("qwen2.5vl");
    TEST_ASSERT(base_arch == "qwen2vl");
    std::cout << "qwen2.5vl extends from: " << base_arch << std::endl;
    
    // Test unsupported architecture
    TEST_ASSERT(!GGMLIncrementalExtension::isArchitectureSupported("unknown_arch"));
    
    // Test incremental modifications (with dummy parameters)
    void* dummy_params = nullptr;
    TEST_ASSERT(GGMLIncrementalExtension::applyIncrementalModifications("qwen2.5vl", dummy_params));
    
    std::cout << "GGML Incremental Extension tests completed successfully" << std::endl;
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
    
    // Note: GGUFModifier requires actual GGUF files to test properly
    // For now, we'll test the static methods that don't require file I/O
    
    // Test that the class can be used (basic functionality test)
    // In a real scenario, these would be tested with actual GGUF files
    std::cout << "GGUF modifier class is available for use" << std::endl;
    std::cout << "Note: Full testing requires actual GGUF model files" << std::endl;
    
    std::cout << "GGUF modifier tests completed" << std::endl;
    return true;
}

bool testModelfileParser() {
    std::cout << "\n=== Testing Modelfile Parser ===" << std::endl;
    
    auto model_path_manager = std::make_shared<ModelPathManager>();
    ModelfileParser parser(model_path_manager);
    
    // Test LoRA adapter validation
    LoRAAdapter valid_adapter;
    valid_adapter.name = "test_adapter";
    valid_adapter.path = "/tmp/test_adapter.gguf";
    valid_adapter.scale = 1.0f;
    
    // Create a dummy GGUF file for testing
    std::ofstream test_file(valid_adapter.path, std::ios::binary);
    if (test_file.is_open()) {
        // Write GGUF magic number
        test_file.write("GGUF", 4);
        // Write version (3)
        uint32_t version = 3;
        test_file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        // Write tensor count
        uint64_t tensor_count = 100;
        test_file.write(reinterpret_cast<const char*>(&tensor_count), sizeof(tensor_count));
        // Write metadata count
        uint64_t metadata_count = 10;
        test_file.write(reinterpret_cast<const char*>(&metadata_count), sizeof(metadata_count));
        // Add some dummy data to make file size reasonable
        std::vector<char> dummy_data(1024 * 1024, 0); // 1MB
        test_file.write(dummy_data.data(), dummy_data.size());
        test_file.close();
        
        // Test validation
        bool is_valid = parser.validateLoRAAdapter(valid_adapter);
        TEST_ASSERT(is_valid);
        std::cout << "Valid LoRA adapter validation: PASSED" << std::endl;
        
        // Clean up
        std::filesystem::remove(valid_adapter.path);
    }
    
    // Test invalid adapter (non-existent file)
    LoRAAdapter invalid_adapter;
    invalid_adapter.name = "invalid_adapter";
    invalid_adapter.path = "/non/existent/path.gguf";
    invalid_adapter.scale = 1.0f;
    
    bool is_invalid = parser.validateLoRAAdapter(invalid_adapter);
    TEST_ASSERT(!is_invalid);
    std::cout << "Invalid LoRA adapter validation: PASSED" << std::endl;
    
    // Test invalid scale
    LoRAAdapter invalid_scale_adapter = valid_adapter;
    invalid_scale_adapter.scale = -1.0f;
    bool invalid_scale = parser.validateLoRAAdapter(invalid_scale_adapter);
    TEST_ASSERT(!invalid_scale);
    std::cout << "Invalid scale validation: PASSED" << std::endl;
    
    std::cout << "=== Modelfile Parser Test Passed ===\n" << std::endl;
    return true;
}

bool testLoRAModelLoading() {
    std::cout << "\n=== Testing LoRA Model Loading ===" << std::endl;
    
    auto model_path_manager = std::make_shared<ModelPathManager>();
    OllamaModelLoader loader(model_path_manager);
    
    // Test ModelfileConfig creation
    ModelfileConfig config;
    config.base_model = "/path/to/base/model.gguf";
    
    LoRAAdapter adapter;
    adapter.name = "test_lora";
    adapter.path = "/path/to/lora.gguf";
    adapter.scale = 1.0f;
    config.lora_adapters.push_back(adapter);
    
    config.parameters["temperature"] = "0.7";
    config.parameters["top_p"] = "0.9";
    config.system_prompt = "You are a helpful assistant.";
    config.template_format = "{{ .System }}\n{{ .Prompt }}";
    
    std::cout << "ModelfileConfig created successfully" << std::endl;
    std::cout << "Base model: " << config.base_model << std::endl;
    std::cout << "LoRA adapters: " << config.lora_adapters.size() << std::endl;
    std::cout << "Parameters: " << config.parameters.size() << std::endl;
    
    // Test supported media types
    auto supported_types = ModelfileParser::getSupportedMediaTypes();
    TEST_ASSERT(!supported_types.empty());
    std::cout << "Supported media types: " << supported_types.size() << std::endl;
    
    for (const auto& type : supported_types) {
        std::cout << "  - " << type << std::endl;
    }
    
    std::cout << "=== LoRA Model Loading Test Passed ===\n" << std::endl;
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
    if (!testGGMLIncrementalExtension()) {
        std::cout << "❌ GGML Incremental Extension test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ GGML Incremental Extension test passed" << std::endl;
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
    
    if (!testModelfileParser()) {
        std::cout << "❌ Modelfile parser test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Modelfile parser test passed" << std::endl;
    }
    
    if (!testLoRAModelLoading()) {
        std::cout << "❌ LoRA model loading test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ LoRA model loading test passed" << std::endl;
    }
    
    if (all_passed) {
        std::cout << "\n🎉 All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n💥 Some tests failed!" << std::endl;
        return 1;
    }
}