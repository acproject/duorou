#ifndef QWEN2_PREPROCESSOR_H
#define QWEN2_PREPROCESSOR_H

#include <cstdint>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>
#include "qwen25vl_special_tokens.h"

namespace duorou {
namespace extensions {
namespace ollama {

class Qwen2Preprocessor {
public:
    Qwen2Preprocessor();
    ~Qwen2Preprocessor() = default;
    
    std::string preprocessText(const std::string& text);
    std::string postprocessText(const std::string& text);
    std::string formatConversation(const std::string& role, const std::string& content);
    std::vector<std::string> tokenizeSpecialTokens(const std::string& text);
    std::string normalizeChinese(const std::string& text);
    std::string encodeBytes(const std::string& text);
    std::string decodeBytes(const std::string& text);
    bool isSpecialTokenString(const std::string& token);
    int32_t getSpecialTokenId(const std::string& token);
    void setDebugMode(bool enable) { debug_mode_ = enable; }
    
private:
    void initializePatterns();
    std::string cleanControlCharacters(const std::string& text);
    std::string normalizeWhitespace(const std::string& text);
    std::vector<std::string> splitIntoFragments(const std::string& text);
    std::vector<std::string> mergeFragments(const std::vector<std::string>& fragments);
    void debugLog(const std::string& message);
    
    std::unordered_map<std::string, int32_t> special_token_map_;
    std::regex special_token_pattern_;
    std::regex chinese_pattern_;
    std::regex whitespace_pattern_;
    std::regex byte_pattern_;
    bool debug_mode_;
    bool normalize_unicode_;
    bool handle_byte_tokens_;
    
    static const std::string CONVERSATION_START;
    static const std::string CONVERSATION_END;
    static const std::string SYSTEM_PREFIX;
    static const std::string USER_PREFIX;
    static const std::string ASSISTANT_PREFIX;
};

bool isValidUTF8(const std::string& str);
std::string toUTF8(const std::string& str);
size_t getByteLength(const std::string& str);
std::string safeTruncate(const std::string& str, size_t max_bytes);

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // QWEN2_PREPROCESSOR_H