#include "src/extensions/ollama/ollama_model_manager.h"
#include <iostream>
#include <vector>
#include <string>

using namespace duorou::extensions::ollama;

int main() {
    std::cout << "Testing OllamaModelManager tokenizer..." << std::endl;
    
    // 创建模型管理器
    auto manager = std::make_unique<OllamaModelManager>(true);
    
    // 测试文本
    std::string test_text = "Hello, world! This is a test.";
    std::cout << "Original text: " << test_text << std::endl;
    
    try {
        // 测试分词
        auto tokens = manager->tokenize(test_text);
        std::cout << "Tokens (" << tokens.size() << "): ";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << tokens[i];
            if (i < tokens.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        // 测试解码
        std::string decoded_text = manager->detokenize(tokens);
        std::cout << "Decoded text: " << decoded_text << std::endl;
        
        // 检查是否一致
        if (decoded_text == test_text) {
            std::cout << "✓ Tokenization test PASSED: Original and decoded text match" << std::endl;
        } else {
            std::cout << "✗ Tokenization test FAILED: Text mismatch" << std::endl;
            std::cout << "  Expected: " << test_text << std::endl;
            std::cout << "  Got:      " << decoded_text << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}