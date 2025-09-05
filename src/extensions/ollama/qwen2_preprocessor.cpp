#include "qwen2_preprocessor.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>

namespace duorou {
namespace extensions {
namespace ollama {

// 静态常量定义
const std::string Qwen2Preprocessor::CONVERSATION_START = "<|im_start|>";
const std::string Qwen2Preprocessor::CONVERSATION_END = "<|im_end|>";
const std::string Qwen2Preprocessor::SYSTEM_PREFIX = "system";
const std::string Qwen2Preprocessor::USER_PREFIX = "user";
const std::string Qwen2Preprocessor::ASSISTANT_PREFIX = "assistant";

Qwen2Preprocessor::Qwen2Preprocessor() 
    : debug_mode_(false), normalize_unicode_(true), handle_byte_tokens_(true) {
    // 初始化特殊token映射
    special_token_map_ = Qwen25VLSpecialTokens::getAllTokenMap();
    
    // 初始化正则表达式模式
    initializePatterns();
}

void Qwen2Preprocessor::initializePatterns() {
    // 特殊token模式：匹配 <|...| > 格式的token
    special_token_pattern_ = std::regex(R"(<\|[^|]*\|>)");
    
    // 中文字符模式：匹配中文字符
    chinese_pattern_ = std::regex(R"([\u4e00-\u9fff]+)");
    
    // 空白字符模式：匹配多个连续空白字符
    whitespace_pattern_ = std::regex(R"(\s+)");
    
    // 字节token模式：匹配 <0xXX> 格式的字节token
    byte_pattern_ = std::regex(R"(<0x[0-9A-Fa-f]{2}>)");
}

std::string Qwen2Preprocessor::preprocessText(const std::string& text) {
    if (text.empty()) {
        return text;
    }
    
    debugLog("开始预处理文本: " + text.substr(0, std::min(text.length(), size_t(50))) + "...");
    
    std::string result = text;
    
    // 1. 清理控制字符
    result = cleanControlCharacters(result);
    
    // 2. 规范化空白字符
    result = normalizeWhitespace(result);
    
    // 3. 规范化中文文本
    if (normalize_unicode_) {
        result = normalizeChinese(result);
    }
    
    // 4. 处理字节级编码
    if (handle_byte_tokens_) {
        result = encodeBytes(result);
    }
    
    debugLog("预处理完成: " + result.substr(0, std::min(result.length(), size_t(50))) + "...");
    
    return result;
}

std::string Qwen2Preprocessor::postprocessText(const std::string& text) {
    if (text.empty()) {
        return text;
    }
    
    debugLog("开始后处理文本: " + text.substr(0, std::min(text.length(), size_t(50))) + "...");
    
    std::string result = text;
    
    // 1. 解码字节级编码
    if (handle_byte_tokens_) {
        result = decodeBytes(result);
    }
    
    // 2. 清理多余的空白字符
    result = normalizeWhitespace(result);
    
    // 3. 移除特殊token标记
    result = std::regex_replace(result, special_token_pattern_, "");
    
    // 4. 去除首尾空白
    result.erase(0, result.find_first_not_of(" \t\n\r"));
    result.erase(result.find_last_not_of(" \t\n\r") + 1);
    
    debugLog("后处理完成: " + result.substr(0, std::min(result.length(), size_t(50))) + "...");
    
    return result;
}

std::string Qwen2Preprocessor::formatConversation(const std::string& role, const std::string& content) {
    std::ostringstream oss;
    oss << CONVERSATION_START << role << "\n" << content << CONVERSATION_END;
    return oss.str();
}

std::vector<std::string> Qwen2Preprocessor::tokenizeSpecialTokens(const std::string& text) {
    std::vector<std::string> fragments;
    
    if (text.empty()) {
        return fragments;
    }
    
    // 使用正则表达式分割特殊token
    std::sregex_token_iterator iter(text.begin(), text.end(), special_token_pattern_, -1);
    std::sregex_token_iterator end;
    
    std::sregex_iterator special_iter(text.begin(), text.end(), special_token_pattern_);
    std::sregex_iterator special_end;
    
    size_t last_pos = 0;
    
    for (auto it = special_iter; it != special_end; ++it) {
        // 添加特殊token之前的文本
        if (it->position() > last_pos) {
            std::string before = text.substr(last_pos, it->position() - last_pos);
            if (!before.empty()) {
                fragments.push_back(before);
            }
        }
        
        // 添加特殊token
        fragments.push_back(it->str());
        last_pos = it->position() + it->length();
    }
    
    // 添加最后剩余的文本
    if (last_pos < text.length()) {
        std::string remaining = text.substr(last_pos);
        if (!remaining.empty()) {
            fragments.push_back(remaining);
        }
    }
    
    // 如果没有找到特殊token，返回原始文本
    if (fragments.empty()) {
        fragments.push_back(text);
    }
    
    return fragments;
}

std::string Qwen2Preprocessor::normalizeChinese(const std::string& text) {
    // 简单的中文规范化：去除中文字符间的多余空格
    std::string result = text;
    
    // 移除中文字符之间的空格
    std::regex chinese_space_pattern(R"(([\u4e00-\u9fff])\s+([\u4e00-\u9fff]))");
    result = std::regex_replace(result, chinese_space_pattern, "$1$2");
    
    return result;
}

std::string Qwen2Preprocessor::encodeBytes(const std::string& text) {
    std::string result;
    result.reserve(text.length() * 2); // 预分配空间
    
    for (unsigned char c : text) {
        if (c < 32 || c > 126) { // 非可打印ASCII字符
            std::ostringstream oss;
            oss << "<0x" << std::hex << std::uppercase << static_cast<int>(c) << ">";
            result += oss.str();
        } else {
            result += c;
        }
    }
    
    return result;
}

std::string Qwen2Preprocessor::decodeBytes(const std::string& text) {
    std::string result;
    
    std::sregex_iterator iter(text.begin(), text.end(), byte_pattern_);
    std::sregex_iterator end;
    
    size_t last_pos = 0;
    
    for (auto it = iter; it != end; ++it) {
        // 添加字节token之前的文本
        if (it->position() > last_pos) {
            result += text.substr(last_pos, it->position() - last_pos);
        }
        
        // 解码字节token
        std::string byte_str = it->str();
        if (byte_str.length() >= 6) { // <0xXX> 至少6个字符
            std::string hex_str = byte_str.substr(3, 2); // 提取XX部分
            try {
                int byte_val = std::stoi(hex_str, nullptr, 16);
                result += static_cast<char>(byte_val);
            } catch (const std::exception&) {
                // 如果解码失败，保留原始字符串
                result += byte_str;
            }
        } else {
            result += byte_str;
        }
        
        last_pos = it->position() + it->length();
    }
    
    // 添加最后剩余的文本
    if (last_pos < text.length()) {
        result += text.substr(last_pos);
    }
    
    return result;
}

bool Qwen2Preprocessor::isSpecialTokenString(const std::string& token) {
    return special_token_map_.find(token) != special_token_map_.end();
}

int32_t Qwen2Preprocessor::getSpecialTokenId(const std::string& token) {
    auto it = special_token_map_.find(token);
    return (it != special_token_map_.end()) ? it->second : -1;
}

std::string Qwen2Preprocessor::cleanControlCharacters(const std::string& text) {
    std::string result;
    result.reserve(text.length());
    
    for (char c : text) {
        // 保留可打印字符、换行符、制表符和回车符
        if (std::isprint(static_cast<unsigned char>(c)) || c == '\n' || c == '\t' || c == '\r') {
            result += c;
        }
    }
    
    return result;
}

std::string Qwen2Preprocessor::normalizeWhitespace(const std::string& text) {
    // 将多个连续的空白字符替换为单个空格
    std::string result = std::regex_replace(text, whitespace_pattern_, " ");
    
    // 去除首尾空白
    result.erase(0, result.find_first_not_of(" \t\n\r"));
    result.erase(result.find_last_not_of(" \t\n\r") + 1);
    
    return result;
}

std::vector<std::string> Qwen2Preprocessor::splitIntoFragments(const std::string& text) {
    return tokenizeSpecialTokens(text);
}

std::vector<std::string> Qwen2Preprocessor::mergeFragments(const std::vector<std::string>& fragments) {
    std::vector<std::string> result;
    
    for (const auto& fragment : fragments) {
        if (!fragment.empty()) {
            // 如果是特殊token，直接添加
            if (isSpecialTokenString(fragment)) {
                result.push_back(fragment);
            } else {
                // 如果是普通文本，可以与前一个普通文本合并
                if (!result.empty() && !isSpecialTokenString(result.back())) {
                    result.back() += fragment;
                } else {
                    result.push_back(fragment);
                }
            }
        }
    }
    
    return result;
}

void Qwen2Preprocessor::debugLog(const std::string& message) {
    if (debug_mode_) {
        std::cout << "[Qwen2Preprocessor] " << message << std::endl;
    }
}

// 工具函数实现
bool isValidUTF8(const std::string& str) {
    // 简单的UTF-8验证
    for (size_t i = 0; i < str.length(); ) {
        unsigned char c = str[i];
        
        if (c < 0x80) {
            i++; // ASCII字符
        } else if ((c >> 5) == 0x06) {
            if (i + 1 >= str.length() || (str[i + 1] & 0xC0) != 0x80) return false;
            i += 2; // 2字节UTF-8
        } else if ((c >> 4) == 0x0E) {
            if (i + 2 >= str.length() || (str[i + 1] & 0xC0) != 0x80 || (str[i + 2] & 0xC0) != 0x80) return false;
            i += 3; // 3字节UTF-8
        } else if ((c >> 3) == 0x1E) {
            if (i + 3 >= str.length() || (str[i + 1] & 0xC0) != 0x80 || (str[i + 2] & 0xC0) != 0x80 || (str[i + 3] & 0xC0) != 0x80) return false;
            i += 4; // 4字节UTF-8
        } else {
            return false; // 无效的UTF-8序列
        }
    }
    return true;
}

std::string toUTF8(const std::string& str) {
    // 如果已经是有效的UTF-8，直接返回
    if (isValidUTF8(str)) {
        return str;
    }
    
    // 简单的转换：将无效字符替换为问号
    std::string result;
    for (char c : str) {
        if (static_cast<unsigned char>(c) < 128) {
            result += c;
        } else {
            result += '?';
        }
    }
    
    return result;
}

size_t getByteLength(const std::string& str) {
    return str.length();
}

std::string safeTruncate(const std::string& str, size_t max_bytes) {
    if (str.length() <= max_bytes) {
        return str;
    }
    
    // 简单截取，确保不破坏UTF-8字符
    std::string result = str.substr(0, max_bytes);
    
    // 如果最后一个字符是UTF-8多字节序列的一部分，向前回退
    while (!result.empty() && (static_cast<unsigned char>(result.back()) & 0x80) != 0) {
        result.pop_back();
    }
    
    return result;
}

} // namespace ollama
} // namespace extensions
} // namespace duorou