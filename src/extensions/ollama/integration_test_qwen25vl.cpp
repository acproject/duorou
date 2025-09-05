#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include "qwen25vl_special_tokens.h"
#include "qwen2_preprocessor.h"

using namespace duorou::extensions::ollama;

void testChineseTextProcessing() {
    std::cout << "Testing Chinese text processing..." << std::endl;
    
    Qwen2Preprocessor preprocessor;
    
    // 测试中文文本预处理
    std::string chineseText = "你好，世界！这是一个测试。";
    std::string processed = preprocessor.preprocessText(chineseText);
    
    std::cout << "Original: " << chineseText << std::endl;
    std::cout << "Processed: " << processed << std::endl;
    
    // 测试会话格式化
    std::string userMessage = preprocessor.formatConversation("user", "你好，请介绍一下自己");
    std::string assistantMessage = preprocessor.formatConversation("assistant", "你好！我是Qwen，一个AI助手。");
    
    std::cout << "Formatted user message: " << userMessage << std::endl;
    std::cout << "Formatted assistant message: " << assistantMessage << std::endl;
    
    std::cout << "Chinese text processing test passed." << std::endl;
}

void testSpecialTokenIntegration() {
    std::cout << "Testing special token integration..." << std::endl;
    
    Qwen2Preprocessor preprocessor;
    
    // 测试特殊token字符串识别
    assert(preprocessor.isSpecialTokenString("<|im_start|>"));
    assert(preprocessor.isSpecialTokenString("<|im_end|>"));
    assert(preprocessor.isSpecialTokenString("<|endoftext|>"));
    assert(!preprocessor.isSpecialTokenString("普通文本"));
    
    // 测试特殊token ID获取
    assert(preprocessor.getSpecialTokenId("<|im_start|>") == Qwen25VLTokens::IM_START);
    assert(preprocessor.getSpecialTokenId("<|im_end|>") == Qwen25VLTokens::IM_END);
    assert(preprocessor.getSpecialTokenId("<|endoftext|>") == Qwen25VLTokens::ENDOFTEXT);
    
    // 测试视觉相关token
    assert(Qwen25VLSpecialTokens::isVisionToken(Qwen25VLTokens::VISION_START));
    assert(Qwen25VLSpecialTokens::isVisionToken(Qwen25VLTokens::VISION_END));
    assert(!Qwen25VLSpecialTokens::isVisionToken(Qwen25VLTokens::IM_START));
    
    std::cout << "Special token integration test passed." << std::endl;
}

void testByteEncodingDecoding() {
    std::cout << "Testing byte encoding/decoding..." << std::endl;
    
    Qwen2Preprocessor preprocessor;
    
    // 测试字节编码和解码
    std::string original = "Hello 世界! 🌍";
    std::string encoded = preprocessor.encodeBytes(original);
    std::string decoded = preprocessor.decodeBytes(encoded);
    
    std::cout << "Original: " << original << std::endl;
    std::cout << "Encoded: " << encoded << std::endl;
    std::cout << "Decoded: " << decoded << std::endl;
    
    assert(original == decoded);
    
    std::cout << "Byte encoding/decoding test passed." << std::endl;
}

int main() {
    std::cout << "Running Qwen2.5VL Integration Tests..." << std::endl;
    
    try {
        testChineseTextProcessing();
        testSpecialTokenIntegration();
        testByteEncodingDecoding();
        
        std::cout << "\nAll integration tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Integration test failed: " << e.what() << std::endl;
        return 1;
    }
}