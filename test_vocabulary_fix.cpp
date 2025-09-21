#include <iostream>
#include <vector>
#include <string>
#include "src/model/vocabulary.h"
#include "src/utils/string_utils.h"

using namespace duorou::model;
using namespace duorou::utils;

int main() {
    std::cout << "=== 测试词汇表十六进制字符解码修复 ===" << std::endl;
    
    // 模拟从GGUF文件读取的包含十六进制编码的token
    std::vector<std::string> rawTokens = {
        "hello",                          // 普通token
        "world",                          // 普通token
        "\\x48\\x65\\x6c\\x6c\\x6f",     // "Hello" 的十六进制编码
        "\\x57\\x6f\\x72\\x6c\\x64",     // "World" 的十六进制编码
        "\\x0a",                         // 换行符
        "\\x20",                         // 空格
        "normal_token",                   // 普通token
        "\\xe4\\xb8\\xad\\xe6\\x96\\x87", // "中文" 的UTF-8十六进制编码
    };
    
    std::vector<int32_t> types(rawTokens.size(), 0); // TOKEN_TYPE_NORMAL
    std::vector<float> scores(rawTokens.size(), 0.0f);
    std::vector<std::string> merges;
    
    std::cout << "\n原始tokens (从GGUF文件读取):" << std::endl;
    for (size_t i = 0; i < rawTokens.size(); ++i) {
        std::cout << "  [" << i << "] \"" << rawTokens[i] << "\"" << std::endl;
    }
    
    // 测试解码函数
    std::cout << "\n测试解码函数:" << std::endl;
    auto decodedTokens = decodeTokenStrings(rawTokens);
    for (size_t i = 0; i < decodedTokens.size(); ++i) {
        std::cout << "  [" << i << "] \"" << rawTokens[i] << "\" -> \"" << decodedTokens[i] << "\"" << std::endl;
    }
    
    // 初始化词汇表
    Vocabulary vocab;
    vocab.initialize(rawTokens, types, scores, merges);
    
    std::cout << "\n测试词汇表编码/解码:" << std::endl;
    
    // 测试编码
    std::vector<std::string> testStrings = {"hello", "world", "Hello", "World", "\n", " ", "normal_token", "中文"};
    
    for (const auto& testStr : testStrings) {
        int32_t id = vocab.encode(testStr);
        if (id >= 0) {
            std::string decoded = vocab.decode(id);
            std::cout << "  \"" << testStr << "\" -> ID:" << id << " -> \"" << decoded << "\"";
            if (testStr == decoded) {
                std::cout << " ✓" << std::endl;
            } else {
                std::cout << " ✗ (不匹配)" << std::endl;
            }
        } else {
            std::cout << "  \"" << testStr << "\" -> 未找到" << std::endl;
        }
    }
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
    
    return 0;
}