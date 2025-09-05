#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <map>
#include "qwen25vl_special_tokens.h"

using namespace duorou::extensions::ollama;

class Qwen25VLTokenizationTest {
public:
    static void runAllTests() {
        std::cout << "Running Qwen2.5VL Tokenization Tests..." << std::endl;
        
        testSpecialTokens();
        testSpecialTokenMaps();
        testTokenClassification();
        testTokenStringConversion();
        
        std::cout << "All tests passed!" << std::endl;
    }
    
private:
    static void testSpecialTokens() {
        std::cout << "Testing special tokens..." << std::endl;
        
        // 测试特殊token ID常量
        assert(Qwen25VLTokens::ENDOFTEXT == 151643);
        assert(Qwen25VLTokens::IM_START == 151644);
        assert(Qwen25VLTokens::IM_END == 151645);
        assert(Qwen25VLTokens::VISION_START == 151652);
        assert(Qwen25VLTokens::VISION_END == 151653);
        
        // 测试特殊token字符串获取
        assert(Qwen25VLSpecialTokens::getTokenString(151643) == "<|endoftext|>");
        assert(Qwen25VLSpecialTokens::getTokenString(151644) == "<|im_start|>");
        assert(Qwen25VLSpecialTokens::getTokenString(151645) == "<|im_end|>");
        
        // 测试特殊token检查
        assert(Qwen25VLSpecialTokens::isSpecialToken(151643));
        assert(Qwen25VLSpecialTokens::isSpecialToken(151644));
        assert(!Qwen25VLSpecialTokens::isSpecialToken(12345));
        
        std::cout << "Special tokens test passed." << std::endl;
    }
    
    static void testSpecialTokenMaps() {
        std::cout << "Testing special token maps..." << std::endl;
        
        // 测试特殊token映射
        auto special_map = Qwen25VLSpecialTokens::getSpecialTokenMap();
        assert(!special_map.empty());
        assert(special_map.find("<|endoftext|>") != special_map.end());
        assert(special_map.find("<|im_start|>") != special_map.end());
        assert(special_map.find("<|im_end|>") != special_map.end());
        
        // 测试中文token映射
        auto chinese_map = Qwen25VLSpecialTokens::getChineseTokenMap();
        assert(!chinese_map.empty());
        assert(chinese_map.find("你") != chinese_map.end());
        assert(chinese_map.find("好") != chinese_map.end());
        
        // 测试完整token映射
        auto all_map = Qwen25VLSpecialTokens::getAllTokenMap();
        assert(all_map.size() >= special_map.size() + chinese_map.size());
        
        std::cout << "Special token maps test passed." << std::endl;
    }
    
    static void testTokenClassification() {
        std::cout << "Testing token classification..." << std::endl;
        
        // 测试视觉token识别
        assert(Qwen25VLSpecialTokens::isVisionToken(151652)); // VISION_START
        assert(Qwen25VLSpecialTokens::isVisionToken(151653)); // VISION_END
        assert(Qwen25VLSpecialTokens::isVisionToken(151654)); // VISION_PAD
        assert(!Qwen25VLSpecialTokens::isVisionToken(151643)); // ENDOFTEXT
        
        // 测试对话token识别
        assert(Qwen25VLSpecialTokens::isConversationToken(151644)); // IM_START
        assert(Qwen25VLSpecialTokens::isConversationToken(151645)); // IM_END
        assert(!Qwen25VLSpecialTokens::isConversationToken(151652)); // VISION_START
        
        // 测试特殊token识别
        assert(Qwen25VLSpecialTokens::isSpecialToken(151643)); // ENDOFTEXT
        assert(Qwen25VLSpecialTokens::isSpecialToken(151644)); // IM_START
        assert(!Qwen25VLSpecialTokens::isSpecialToken(12345)); // 普通token
        
        std::cout << "Token classification test passed." << std::endl;
    }
    
    static void testTokenStringConversion() {
        std::cout << "Testing token string conversion..." << std::endl;
        
        // 测试已知token的字符串转换
        assert(Qwen25VLSpecialTokens::getTokenString(151643) == "<|endoftext|>");
        assert(Qwen25VLSpecialTokens::getTokenString(151644) == "<|im_start|>");
        assert(Qwen25VLSpecialTokens::getTokenString(151645) == "<|im_end|>");
        assert(Qwen25VLSpecialTokens::getTokenString(151652) == "<|vision_start|>");
        assert(Qwen25VLSpecialTokens::getTokenString(151653) == "<|vision_end|>");
        
        // 测试工具调用token
        assert(Qwen25VLSpecialTokens::getTokenString(151657) == "<|tool_call_start|>");
        assert(Qwen25VLSpecialTokens::getTokenString(151658) == "<|tool_call_end|>");
        
        // 测试未知token返回空字符串
        assert(Qwen25VLSpecialTokens::getTokenString(999999).empty());
        
        std::cout << "Token string conversion test passed." << std::endl;
    }
    

};

int main() {
    try {
        Qwen25VLTokenizationTest::runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}