#include "src/extensions/ollama/ollama_model_manager.h"
#include <iostream>
#include <vector>
#include <string>

using namespace duorou::extensions::ollama;

int main() {
    std::cout << "Testing BPE Processor with realistic scenario..." << std::endl;
    
    try {
        // 创建模型管理器
        auto manager = std::make_unique<OllamaModelManager>(true);
        
        // 创建一个更完整的测试词汇表，包含字节级token
        std::cout << "Creating byte-level vocabulary..." << std::endl;
        std::vector<std::string> test_tokens;
        std::vector<int32_t> test_types;
        std::vector<float> test_scores;
        
        // 添加特殊token
        test_tokens.push_back("<|endoftext|>");
        test_tokens.push_back("<|im_start|>");
        test_tokens.push_back("<|im_end|>");
        test_types.insert(test_types.end(), 3, 1);
        test_scores.insert(test_scores.end(), 3, 0.0f);
        
        // 添加一些常见的字节级token (GPT-2风格)
        // 空格映射到私有使用区域
        test_tokens.push_back("Ġ");  // 空格的GPT-2表示
        test_tokens.push_back("Hello");
        test_tokens.push_back("world");
        test_tokens.push_back("!");
        test_tokens.push_back("This");
        test_tokens.push_back("is");
        test_tokens.push_back("a");
        test_tokens.push_back("test");
        test_tokens.push_back(".");
        
        // 添加字节级回退token
        for (int i = 0; i < 256; i++) {
            char byte_char = static_cast<char>(i);
            std::string byte_token = "<0x" + 
                std::string(i < 16 ? "0" : "") + 
                std::to_string(i) + ">";
            test_tokens.push_back(byte_token);
        }
        
        test_types.resize(test_tokens.size(), 1);
        test_scores.resize(test_tokens.size(), 0.0f);
        
        std::vector<std::string> test_merges;
        
        // 获取词汇表并初始化
        auto vocab = std::make_shared<duorou::extensions::ollama::Vocabulary>();
        vocab->initialize(test_tokens, test_types, test_scores, test_merges);
        
        // 重新创建文本处理器
        auto text_processor = duorou::extensions::ollama::createTextProcessor("bpe", vocab, "");
        manager->setTextProcessor(std::move(text_processor));
        std::cout << "Vocabulary initialized with " << test_tokens.size() << " tokens" << std::endl;
        
        // 测试简单文本
        std::string test_text = "Hello world!";
        std::cout << "\nTesting text: '" << test_text << "'" << std::endl;
        
        auto tokens = manager->tokenize(test_text);
        std::cout << "Tokens (" << tokens.size() << "): ";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << tokens[i];
            if (i < tokens.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        std::string decoded = manager->detokenize(tokens);
        std::cout << "Decoded: '" << decoded << "'" << std::endl;
        
        if (decoded == test_text) {
            std::cout << "✓ BPE test PASSED" << std::endl;
        } else {
            std::cout << "✗ BPE test FAILED" << std::endl;
            std::cout << "  Expected: " << test_text << std::endl;
            std::cout << "  Got:      " << decoded << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}