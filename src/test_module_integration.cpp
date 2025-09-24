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
    
    // 创建一个临时的小型GGUF文件用于测试
    std::string temp_gguf = "/tmp/test_model.gguf";
    
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
    
    // 验证文件创建成功
    if (!std::filesystem::exists(temp_gguf)) {
        std::cout << "[WARN] Could not create temporary GGUF file, skipping DUOROU_FORCE_LLAMA tests" << std::endl;
        return;
    }
    
    // 测试1: 不设置 DUOROU_FORCE_LLAMA（应该检测为 qwen2，不使用 llama.cpp）
    {
        std::cout << "\n--- Test 1: Without DUOROU_FORCE_LLAMA ---" << std::endl;
        
        // 确保环境变量未设置
        unsetenv("DUOROU_FORCE_LLAMA");
        
        GlobalModelManager::initialize(true);
        OllamaModelManager& manager = GlobalModelManager::getInstance();
        
        // 注册临时模型
        bool registered = manager.registerModel("test_qwen2", temp_gguf);
        assertTrue(registered, "Should successfully register test model");
        
        if (registered) {
            // 获取模型信息，检查架构
            const ModelInfo* info = manager.getModelInfo("test_qwen2");
            assertTrue(info != nullptr, "Should get model info");
            if (info) {
                assertEqual(std::string("qwen2"), info->architecture, "Should detect qwen2 architecture");
            }
            
            // 尝试加载模型并捕获输出
            OutputCapture capture;
            bool loaded = manager.loadModel("test_qwen2");
            std::string output = capture.getOutput();
            
            std::cout << "[DEBUG] Load output: " << output << std::endl;
            
            // 检查输出中不包含 "forced by DUOROU_FORCE_LLAMA"
            assertTrue(output.find("forced by DUOROU_FORCE_LLAMA") == std::string::npos,
                      "Should not show forced by DUOROU_FORCE_LLAMA when env var not set");
            
            // 对于 qwen2 架构，应该显示 use_llama_backend_=false 且初始化内部 Forward
            assertTrue(output.find("use_llama_backend_=false") != std::string::npos,
                      "qwen2 architecture should use internal forward (use_llama_backend_=false)");
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
    
    // 清理临时文件
    std::filesystem::remove(temp_gguf);
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