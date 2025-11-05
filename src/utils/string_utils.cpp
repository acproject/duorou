#include "string_utils.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace duorou {
namespace utils {

std::string decodeHexEscapes(const std::string& input) {
    std::string result;
    result.reserve(input.length());
    
    for (size_t i = 0; i < input.length(); ++i) {
        // Handle \x## format
        if (i + 3 < input.length() && 
            input[i] == '\\' && 
            input[i + 1] == 'x') {
            
            // 尝试解析十六进制字符
            std::string hexStr = input.substr(i + 2, 2);
            
            // 检查是否为有效的十六进制字符
            if (std::all_of(hexStr.begin(), hexStr.end(), 
                           [](char c) { return std::isxdigit(c); })) {
                
                // 转换十六进制字符串为字节
                unsigned int value;
                std::stringstream ss;
                ss << std::hex << hexStr;
                ss >> value;
                
                result += static_cast<char>(value);
                i += 3; // 跳过 \x## 
            } else {
                result += input[i];
            }
        }
        // Handle <0x###> format (from model output)
        else if (i + 5 < input.length() && 
                 input[i] == '<' && 
                 input[i + 1] == '0' && 
                 input[i + 2] == 'x') {
            
            // Find the closing >
            size_t closePos = input.find('>', i + 3);
            if (closePos != std::string::npos && closePos > i + 3) {
                std::string hexStr = input.substr(i + 3, closePos - i - 3);
                
                // 检查是否为有效的十六进制字符
                if (!hexStr.empty() && std::all_of(hexStr.begin(), hexStr.end(), 
                                   [](char c) { return std::isxdigit(c); })) {
                    
                    // 转换十六进制字符串为字节
                    unsigned int value;
                    std::stringstream ss;
                    ss << std::hex << hexStr;
                    ss >> value;
                    
                    // Convert to UTF-8 if it's a valid Unicode codepoint
                    if (value <= 0x7F) {
                        // ASCII range
                        result += static_cast<char>(value);
                    } else if (value <= 0x7FF) {
                        // 2-byte UTF-8
                        result += static_cast<char>(0xC0 | (value >> 6));
                        result += static_cast<char>(0x80 | (value & 0x3F));
                    } else if (value <= 0xFFFF) {
                        // 3-byte UTF-8
                        result += static_cast<char>(0xE0 | (value >> 12));
                        result += static_cast<char>(0x80 | ((value >> 6) & 0x3F));
                        result += static_cast<char>(0x80 | (value & 0x3F));
                    } else if (value <= 0x10FFFF) {
                        // 4-byte UTF-8
                        result += static_cast<char>(0xF0 | (value >> 18));
                        result += static_cast<char>(0x80 | ((value >> 12) & 0x3F));
                        result += static_cast<char>(0x80 | ((value >> 6) & 0x3F));
                        result += static_cast<char>(0x80 | (value & 0x3F));
                    } else {
                        // Invalid Unicode, keep original
                        result += input.substr(i, closePos - i + 1);
                    }
                    
                    i = closePos; // 跳过整个 <0x###> 序列
                } else {
                    result += input[i];
                }
            } else {
                result += input[i];
            }
        } else {
            result += input[i];
        }
    }
    
    return result;
}

bool containsHexEscapes(const std::string& input) {
    for (size_t i = 0; i + 3 < input.length(); ++i) {
        // Check for \x## format
        if (input[i] == '\\' && 
            input[i + 1] == 'x' &&
            std::isxdigit(input[i + 2]) &&
            std::isxdigit(input[i + 3])) {
            return true;
        }
        
        // Check for <0x###> format
        if (i + 5 < input.length() && 
            input[i] == '<' && 
            input[i + 1] == '0' && 
            input[i + 2] == 'x') {
            
            size_t closePos = input.find('>', i + 3);
            if (closePos != std::string::npos && closePos > i + 3) {
                std::string hexStr = input.substr(i + 3, closePos - i - 3);
                if (!hexStr.empty() && std::all_of(hexStr.begin(), hexStr.end(), 
                                   [](char c) { return std::isxdigit(c); })) {
                    return true;
                }
            }
        }
    }
    return false;
}

std::vector<std::string> decodeTokenStrings(const std::vector<std::string>& tokens) {
    std::vector<std::string> result;
    result.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        if (containsHexEscapes(token)) {
            result.push_back(decodeHexEscapes(token));
        } else {
            result.push_back(token);
        }
    }
    
    return result;
}

} // namespace utils
} // namespace duorou