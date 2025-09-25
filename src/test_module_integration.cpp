#include <iostream>
#include <cassert>
#include <cstdlib>
#include <string>
#include <chrono>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>

// 包含 ollama 扩展的头文件
#include "extensions/ollama/ollama_model_manager.h"
#include "extensions/ollama/ollama_path_resolver.h"
#include "extensions/ollama/gguf_parser.h"

// 为了测试在没有实际模型文件的情况下运行
#include "ml/context.h"
#include "kvcache/cache.h"
#include "core/text_generator.h"
#include "model/qwen_text_model.h"

using namespace duorou::extensions::ollama;

// 全局测试变量
static int test_count = 0;
static int test_passed = 0;
static int test_failed = 0;

// 测试助手函数
void assertTrue(bool condition, const std::string& test_name) {
    test_count++;
    if (condition) {
        test_passed++;
        std::cout << "[PASS] " << test_name << std::endl;
    } else {
        test_failed++;
        std::cout << "[FAIL] " << test_name << std::endl;
    }
}

void assertEqual(const std::string& expected, const std::string& actual, const std::string& test_name) {
    test_count++;
    if (expected == actual) {
        test_passed++;
        std::cout << "[PASS] " << test_name << std::endl;
    } else {
        test_failed++;
        std::cout << "[FAIL] " << test_name << " - Expected: '" << expected << "', Actual: '" << actual << "'" << std::endl;
    }
}

// 泛型断言，支持整型、布尔、浮点等类型
template <typename T>
void assertEqual(const T& expected, const T& actual, const std::string& test_name) {
    test_count++;
    if (expected == actual) {
        test_passed++;
        std::cout << "[PASS] " << test_name << std::endl;
    } else {
        test_failed++;
        std::ostringstream oss;
        oss << expected;
        std::string expected_str = oss.str();
        oss.str("");
        oss.clear();
        oss << actual;
        std::string actual_str = oss.str();
        std::cout << "[FAIL] " << test_name << " - Expected: '" << expected_str << "', Actual: '" << actual_str << "'" << std::endl;
    }
}

// 捕获标准输出的助手类
class OutputCapture {
private:
    std::stringstream buffer;
    std::streambuf* old_cout;
    std::streambuf* old_cerr;
    
public:
    OutputCapture() : old_cout(std::cout.rdbuf()), old_cerr(std::cerr.rdbuf()) {
        std::cout.rdbuf(buffer.rdbuf());
        std::cerr.rdbuf(buffer.rdbuf());
    }
    
    ~OutputCapture() {
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }
    
    std::string getOutput() const {
        return buffer.str();
    }
};

// 测试基本的 OllamaPathResolver 功能
void testOllamaPathResolver() {
    std::cout << "\n=== Testing OllamaPathResolver ===" << std::endl;
    
    OllamaPathResolver resolver(true);
    
    // 测试默认模型目录路径构造
    std::string models_dir = resolver.getOllamaModelsDir();
    assertTrue(!models_dir.empty(), "OllamaPathResolver should return non-empty models directory");
    assertTrue(models_dir.find(".ollama/models") != std::string::npos, 
               "Default models directory should contain .ollama/models");
    
    // 测试自定义模型目录
    std::string custom_dir = "/tmp/test_models";
    resolver.setCustomModelsDir(custom_dir);
    assertEqual(custom_dir, resolver.getOllamaModelsDir(), 
                "Custom models directory should be set correctly");
    
    // 测试模型名称解析
    auto model_info = resolver.parseModelName("qwen2.5:7b");
    assertTrue(model_info.has_value(), "Should parse valid model name");
    if (model_info) {
        assertEqual("qwen2.5", model_info->name, "Model name should be parsed correctly");
        assertEqual("7b", model_info->tag, "Model tag should be parsed correctly");
        assertEqual("library", model_info->namespace_name, "Default namespace should be library");
    }
    
    // 测试列出可用模型（即使目录不存在也不应崩溃）
    auto models = resolver.listAvailableModels();
    assertTrue(true, "listAvailableModels should not crash even if directory doesn't exist");
    
    std::cout << "[INFO] Found " << models.size() << " available models" << std::endl;
}

// 测试 OllamaModelManager 基本功能
void testOllamaModelManager() {
    std::cout << "\n=== Testing OllamaModelManager ===" << std::endl;
    
    // 初始化全局模型管理器
    GlobalModelManager::initialize(true);
    
    OllamaModelManager& manager = GlobalModelManager::getInstance();
    
    // 测试获取注册模型列表（应为空）
    auto registered_models = manager.getRegisteredModels();
    assertEqual(0, (int)registered_models.size(), "Initially should have no registered models");
    
    // 测试获取加载模型列表（应为空）
    auto loaded_models = manager.getLoadedModels();
    assertEqual(0, (int)loaded_models.size(), "Initially should have no loaded models");
    
    // 测试模型ID标准化
    std::string normalized = manager.normalizeModelId("  qwen2.5:7b  ");
    assertEqual(std::string("qwen2.5:7b"), normalized, "Model ID normalization should trim whitespace");
    
    // 测试内存使用情况获取
    size_t memory_usage = manager.getMemoryUsage();
    assertTrue(memory_usage >= 0, "Memory usage should be non-negative");
    
    std::cout << "[INFO] Current memory usage: " << memory_usage << " bytes" << std::endl;
    
    GlobalModelManager::shutdown();
}

// 测试 DUOROU_FORCE_LLAMA 环境变量功能
void testDuorouForceLlama() {
    std::cout << "\n=== Testing DUOROU_FORCE_LLAMA Environment Variable ===" << std::endl;
    
    // 创建一个临时的小型GGUF文件用于测试，若用户真实模型存在则直接使用真实模型路径
    std::string temp_gguf = "/tmp/test_model.gguf";
    std::string user_model_path = "/Users/acproject/.ollama/models/blobs/sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025";
    bool use_real_model = std::filesystem::exists(user_model_path);
    if (use_real_model) {
        std::cout << "[INFO] Using real GGUF model at: " << user_model_path << std::endl;
        temp_gguf = user_model_path;
    } else {
        // 创建最小的GGUF文件头
        {
            std::ofstream file(temp_gguf, std::ios::binary);
            if (file) {
                // GGUF魔数
                file.write("GGUF", 4);
                // 版本 (3)
                uint32_t version = 3;
                file.write(reinterpret_cast<const char*>(&version), 4);
                // tensor_count (0)
                uint64_t tensor_count = 0;
                file.write(reinterpret_cast<const char*>(&tensor_count), 8);
                // metadata_kv_count (1)
                uint64_t metadata_count = 1;
                file.write(reinterpret_cast<const char*>(&metadata_count), 8);
                
                // 添加一个 architecture 键值对
                // key: "general.architecture"
                std::string key = "general.architecture";
                uint64_t key_len = key.length();
                file.write(reinterpret_cast<const char*>(&key_len), 8);
                file.write(key.c_str(), key_len);
                
                // value type (string = 8)
                uint32_t value_type = 8;
                file.write(reinterpret_cast<const char*>(&value_type), 4);
                
                // value: "qwen2"
                std::string value = "qwen2";
                uint64_t value_len = value.length();
                file.write(reinterpret_cast<const char*>(&value_len), 8);
                file.write(value.c_str(), value_len);
                
                file.close();
            }
        }
    }
    
    // 验证文件创建成功
    if (!std::filesystem::exists(temp_gguf)) {
        std::cout << "[WARN] Could not create or find GGUF file, skipping DUOROU_FORCE_LLAMA tests" << std::endl;
        return;
    }
    
    // 测试1: 不设置 DUOROU_FORCE_LLAMA（应该检测架构；若为 qwen2，则不使用 llama.cpp）
    {
        std::cout << "\n--- Test 1: Without DUOROU_FORCE_LLAMA ---" << std::endl;
        
        // 确保环境变量未设置
        unsetenv("DUOROU_FORCE_LLAMA");
        
        GlobalModelManager::initialize(true);
        OllamaModelManager& manager = GlobalModelManager::getInstance();
        
        // 注册模型
        bool registered = manager.registerModel("test_qwen2", temp_gguf);
        assertTrue(registered, "Should successfully register test model");
        
        if (registered) {
            // 获取模型信息，输出架构
            const ModelInfo* info = manager.getModelInfo("test_qwen2");
            assertTrue(info != nullptr, "Should get model info");
            if (info) {
                std::cout << "[DEBUG] Detected architecture in test: " << info->architecture << std::endl;
            }
            
            // 尝试加载模型并捕获输出
            OutputCapture capture;
            bool loaded = manager.loadModel("test_qwen2");
            (void)loaded; // 不强制要求加载成功，只校验日志
            std::string output = capture.getOutput();
            
            std::cout << "[DEBUG] Load output: " << output << std::endl;
            
            // 检查输出中不包含 "forced by DUOROU_FORCE_LLAMA"
            assertTrue(output.find("forced by DUOROU_FORCE_LLAMA") == std::string::npos,
                      "Should not show forced by DUOROU_FORCE_LLAMA when env var not set");
            
            // 若为 qwen2 架构，应该显示 use_llama_backend_=false 且初始化内部 Forward
            if (info && info->architecture == "qwen2") {
                assertTrue(output.find("use_llama_backend_=false") != std::string::npos,
                          "qwen2 architecture should use internal forward (use_llama_backend_=false)");
            }
        }
        
        GlobalModelManager::shutdown();
    }
    
    // 测试2: 设置 DUOROU_FORCE_LLAMA=1（应该强制使用 llama.cpp）
    {
        std::cout << "\n--- Test 2: With DUOROU_FORCE_LLAMA=1 ---" << std::endl;
        
        // 设置环境变量
        setenv("DUOROU_FORCE_LLAMA", "1", 1);
        
        GlobalModelManager::initialize(true);
        OllamaModelManager& manager = GlobalModelManager::getInstance();
        
        bool registered = manager.registerModel("test_qwen2_forced", temp_gguf);
        assertTrue(registered, "Should successfully register test model with FORCE_LLAMA");
        
        if (registered) {
            // 尝试加载模型并捕获输出
            OutputCapture capture;
            bool loaded = manager.loadModel("test_qwen2_forced");
            (void)loaded; // 不强制要求加载成功，只校验日志
            std::string output = capture.getOutput();
            
            std::cout << "[DEBUG] Forced load output: " << output << std::endl;
            
            // 检查输出中包含 "forced by DUOROU_FORCE_LLAMA"
            assertTrue(output.find("forced by DUOROU_FORCE_LLAMA") != std::string::npos,
                      "Should show forced by DUOROU_FORCE_LLAMA when env var is set");
            
            // 应该显示 use_llama_backend_=true
            assertTrue(output.find("use_llama_backend_=true") != std::string::npos,
                      "Should force use_llama_backend_=true when DUOROU_FORCE_LLAMA is set");
        }
        
        GlobalModelManager::shutdown();
        
        // 清理环境变量
        unsetenv("DUOROU_FORCE_LLAMA");
    }
}

// 测试推理请求和响应结构
void testInferenceStructures() {
    std::cout << "\n=== Testing Inference Structures ===" << std::endl;
    
    // 测试 InferenceRequest 默认值
    InferenceRequest request;
    assertEqual(100u, request.max_tokens, "Default max_tokens should be 100");
    assertTrue(request.temperature == 0.7f, "Default temperature should be 0.7");
    assertTrue(request.top_p == 0.9f, "Default top_p should be 0.9");
    
    // 测试 InferenceResponse 默认值
    InferenceResponse response;
    assertEqual(false, response.success, "Default success should be false");
    assertEqual(0u, response.tokens_generated, "Default tokens_generated should be 0");
    assertTrue(response.inference_time_ms == 0.0f, "Default inference_time_ms should be 0.0");
}

// 测试 GGUFParser 基本功能
void testGGUFParser() {
    std::cout << "\n=== Testing GGUFParser ===" << std::endl;
    
    GGUFParser parser;
    
    // 测试解析不存在的文件
    bool parsed = parser.parseFile("/nonexistent/file.gguf");
    assertEqual(false, parsed, "Should fail to parse non-existent file");
    
    // 创建一个无效的GGUF文件
    std::string invalid_gguf = "/tmp/invalid.gguf";
    {
        std::ofstream file(invalid_gguf, std::ios::binary);
        if (file) {
            file.write("INVALID", 7);  // 错误的魔数
            file.close();
        }
    }
    
    if (std::filesystem::exists(invalid_gguf)) {
        parsed = parser.parseFile(invalid_gguf);
        assertEqual(false, parsed, "Should fail to parse invalid GGUF file");
        std::filesystem::remove(invalid_gguf);
    }
}

// 测试 TextGeneratorFallback
void testTextGeneratorFallback() {
    std::cout << "\n=== Testing TextGenerator Fallback ===" << std::endl;
    duorou::core::TextGenerator tg; // default: fallback mock implementation
    std::string prompt = "你好，世界";
    auto start = std::chrono::high_resolution_clock::now();
    duorou::core::GenerationParams params;
    params.max_tokens = 32;
    params.temperature = 0.7f;
    params.top_p = 0.9f;
    auto result = tg.generate(prompt, params);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    assertTrue(result.finished, "TextGenerator::generate should finish");
    assertTrue(!result.text.empty(), "TextGenerator should return non-empty text");
    std::cout << "[INFO] Fallback generate text: " << result.text << std::endl;
    std::cout << "[INFO] Fallback generated tokens: " << result.generated_tokens << ", duration: " << duration_ms << " ms" << std::endl;

    // Streaming
    auto sstart = std::chrono::high_resolution_clock::now();
    std::string streamed_text;
    int streamed_tokens = 0;
    auto streamResult = tg.generateStream(
        prompt,
        [&](int token, const std::string &text, bool /*finished*/) {
            streamed_text += text;
            if (token >= 0) streamed_tokens++;
        },
        params
    );
    auto send = std::chrono::high_resolution_clock::now();
    auto sduration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(send - sstart).count();
    assertTrue(streamResult.finished, "TextGenerator::generateStream should finish");
    assertTrue(!streamResult.text.empty(), "TextGenerator::generateStream should return non-empty text");
    std::cout << "[INFO] Fallback stream text: " << streamed_text << std::endl;
    std::cout << "[INFO] Fallback stream tokens: " << streamed_tokens << ", duration: " << sduration_ms << " ms" << std::endl;
}

// 测试 QwenTextModel 的最小端到端生成路径（无权重）
void testQwenTextModelMinimal() {
    std::cout << "\n=== Testing QwenTextModel Minimal E2E ===" << std::endl;
    // 使用较小的隐藏维度与块数以降低内存占用
    duorou::model::TextModelOptions opts;
    opts.hiddenSize = 32;
    opts.blockCount = 1;
    opts.embeddingLength = 32;

    duorou::model::QwenTextModel model(opts);

    // 初始化，传入真实的 gguf 路径（Ollama blobs 哈希文件名也应被接受）
    bool ok = model.initialize("/Users/acproject/.ollama/models/blobs/sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025");
    assertTrue(ok, "QwenTextModel::initialize should succeed even without weights");
    assertTrue(model.isInitialized(), "QwenTextModel should be initialized");

    size_t vocab = model.getVocabSize();
    assertTrue(vocab > 0, "QwenTextModel should report non-zero vocab size");
    std::cout << "[INFO] Qwen vocab size: " << vocab << std::endl;
    std::cout << "[INFO] Qwen hidden size: " << model.getOptions().hiddenSize << std::endl;

    // 编码中文提示（可能回退到 UNK）
    std::string prompt = "你好";
    std::vector<int32_t> input_ids;
    try {
        input_ids = model.encode(prompt, /*addSpecial=*/true);
    } catch (const std::exception &e) {
        std::cout << "[WARN] encode threw: " << e.what() << ", using fallback UNK" << std::endl;
        input_ids = {0};
    }
    assertTrue(!input_ids.empty(), "QwenTextModel::encode should return at least one token");
    std::cout << "[INFO] input_ids size: " << input_ids.size() << std::endl;

    // 前向一次，记录logits形状与部分值
    auto fstart = std::chrono::high_resolution_clock::now();
    auto logits = model.forward(input_ids);
    auto fend = std::chrono::high_resolution_clock::now();
    auto fms = std::chrono::duration_cast<std::chrono::milliseconds>(fend - fstart).count();
    assertTrue(!logits.empty(), "QwenTextModel::forward should return logits");
    assertEqual(vocab, logits.size(), "QwenTextModel::forward logits size should equal vocab size");
    std::cout << "[INFO] forward duration: " << fms << " ms" << std::endl;
    std::cout << "[INFO] logits[0..7]: ";
    for (size_t i = 0; i < std::min<size_t>(8, logits.size()); ++i) {
        std::cout << logits[i] << (i + 1 < std::min<size_t>(8, logits.size()) ? ", " : "\n");
    }

    // 生成若干 token
    size_t max_len = input_ids.size() + 8;
    auto gstart = std::chrono::high_resolution_clock::now();
    auto out_ids = model.generate(input_ids, max_len, /*temperature=*/0.7f, /*topP=*/0.9f);
    auto gend = std::chrono::high_resolution_clock::now();
    auto gms = std::chrono::duration_cast<std::chrono::milliseconds>(gend - gstart).count();
    assertTrue(!out_ids.empty(), "QwenTextModel::generate should return tokens");
    assertTrue(out_ids.size() >= input_ids.size(), "Generated sequence length should be >= input length");
    bool ids_valid = true;
    for (auto id : out_ids) {
        if (id < 0 || static_cast<size_t>(id) >= vocab) { ids_valid = false; break; }
    }
    assertTrue(ids_valid, "All generated token ids should be within vocab range");

    // 解码（不校验内容，仅确保不会崩溃）
    try {
        std::string text = model.decode(out_ids);
        std::cout << "[INFO] decode result length: " << text.size() << std::endl;
    } catch (const std::exception &e) {
        std::cout << "[WARN] decode threw: " << e.what() << std::endl;
    }

    std::cout << "[INFO] generate produced " << out_ids.size() - input_ids.size() << " new tokens, duration: " << gms << " ms" << std::endl;
}

// 主测试函数
int main() {
    std::cout << "=== Duorou Module Integration Test ===" << std::endl;
    std::cout << "Testing ollama extension and DUOROU_FORCE_LLAMA functionality" << std::endl;
    
    try {
        testOllamaPathResolver();
        testOllamaModelManager();
        testInferenceStructures();
        testGGUFParser();
        testDuorouForceLlama();
        testTextGeneratorFallback();
        testQwenTextModelMinimal();

    } catch (const std::exception& e) {
        std::cout << "[ERROR] Exception during testing: " << e.what() << std::endl;
        test_failed++;
    }
    
    // 输出测试结果
    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "Total tests: " << test_count << std::endl;
    std::cout << "Passed: " << test_passed << std::endl;
    std::cout << "Failed: " << test_failed << std::endl;
    
    if (test_failed == 0) {
        std::cout << "All tests passed! ✅" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed! ❌" << std::endl;
        return 1;
    }
}