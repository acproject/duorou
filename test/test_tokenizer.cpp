#include "../src/extensions/ollama/ollama_model_manager.h"
#include <iostream>
#include <vector>
#include <string>

using namespace duorou::extensions::ollama;

int main() {
    std::cout << "Testing OllamaModelManager tokenizer..." << std::endl;
    
    try {
        // 创建模型管理器
        auto manager = std::make_unique<OllamaModelManager>(true);
        
        // 创建一个简单的测试词汇表
        std::cout << "Creating a simple test vocabulary..." << std::endl;
        std::vector<std::string> test_tokens = {
             "<|endoftext|>", "<|im_start|>", "<|im_end|>", 
             "Hello", " world", " Hello", " world", "!", 
             "This", " is", " a", " test", "."
         };
        std::vector<int32_t> test_types(test_tokens.size(), 1);
        std::vector<float> test_scores(test_tokens.size(), 0.0f);
        std::vector<std::string> test_merges;
        
        // 获取词汇表并初始化
        auto vocab = std::make_shared<duorou::extensions::ollama::Vocabulary>();
        vocab->initialize(test_tokens, test_types, test_scores, test_merges);
        
        // 重新创建文本处理器
        auto text_processor = duorou::extensions::ollama::createTextProcessor("bpe", vocab, "");
        manager->setTextProcessor(std::move(text_processor));
        std::cout << "Test vocabulary initialized with " << test_tokens.size() << " tokens" << std::endl;
        
        // 测试简单的token编码/解码
        std::cout << "\nTesting individual tokens:" << std::endl;
        
        // 测试单个token
        std::vector<std::string> test_cases = {"Hello", " world", "!"};
        
        for (const std::string& test_token : test_cases) {
            std::cout << "\nTesting token: '" << test_token << "'" << std::endl;
            auto tokens = manager->tokenize(test_token);
            std::cout << "Tokens (" << tokens.size() << "): ";
            for (size_t i = 0; i < tokens.size(); ++i) {
                std::cout << tokens[i];
                if (i < tokens.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            
            std::string decoded = manager->detokenize(tokens);
            std::cout << "Decoded: '" << decoded << "'" << std::endl;
            
            if (decoded == test_token) {
                std::cout << "✓ Token test PASSED" << std::endl;
            } else {
                std::cout << "✗ Token test FAILED" << std::endl;
            }
        }
        
        // 测试组合token
        std::cout << "\nTesting combined tokens:" << std::endl;
        std::vector<uint32_t> combined_tokens = {3, 4}; // Hello + " world"
        std::string combined_decoded = manager->detokenize(combined_tokens);
        std::cout << "Combined tokens [3, 4] decoded to: '" << combined_decoded << "'" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}