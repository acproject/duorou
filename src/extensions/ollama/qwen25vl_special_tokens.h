#ifndef QWEN25VL_SPECIAL_TOKENS_H
#define QWEN25VL_SPECIAL_TOKENS_H

#include <cstdint>
#include <string>
#include <unordered_map>

namespace duorou {
namespace extensions {
namespace ollama {

// Qwen2.5VL特殊token ID常量
struct Qwen25VLTokens {
    static constexpr int32_t ENDOFTEXT = 151643;
    static constexpr int32_t IM_START = 151644;
    static constexpr int32_t IM_END = 151645;
    static constexpr int32_t OBJECT_REF_START = 151646;
    static constexpr int32_t OBJECT_REF_END = 151647;
    static constexpr int32_t BOX_START = 151648;
    static constexpr int32_t BOX_END = 151649;
    static constexpr int32_t QUAD_START = 151650;
    static constexpr int32_t QUAD_END = 151651;
    static constexpr int32_t VISION_START = 151652;
    static constexpr int32_t VISION_END = 151653;
    static constexpr int32_t VISION_PAD = 151654;
    static constexpr int32_t IMAGE_PAD = 151655;
    static constexpr int32_t VIDEO_PAD = 151656;
    static constexpr int32_t TOOL_CALL_START = 151657;
    static constexpr int32_t TOOL_CALL_END = 151658;
    static constexpr int32_t THINK_START = 151659;
    static constexpr int32_t THINK_END = 151660;
    
    // 常见中文词汇token ID
    static constexpr int32_t CHINESE_HELLO = 104387;     // "你好"
    static constexpr int32_t CHINESE_WORLD = 104388;     // "世界"
    static constexpr int32_t CHINESE_THANK = 104389;     // "谢谢"
    static constexpr int32_t CHINESE_PLEASE = 104390;    // "请"
    static constexpr int32_t CHINESE_QUESTION = 104391;  // "问题"
    static constexpr int32_t CHINESE_ANSWER = 104392;    // "回答"
    static constexpr int32_t CHINESE_HELP = 104393;      // "帮助"
    static constexpr int32_t CHINESE_UNDERSTAND = 104394; // "理解"
};

// Qwen2.5VL特殊token处理类
class Qwen25VLSpecialTokens {
public:
    // 获取特殊token映射
    static std::unordered_map<std::string, int32_t> getSpecialTokenMap() {
        return {
            {"<|endoftext|>", Qwen25VLTokens::ENDOFTEXT},
            {"<|im_start|>", Qwen25VLTokens::IM_START},
            {"<|im_end|>", Qwen25VLTokens::IM_END},
            {"<|object_ref_start|>", Qwen25VLTokens::OBJECT_REF_START},
            {"<|object_ref_end|>", Qwen25VLTokens::OBJECT_REF_END},
            {"<|box_start|>", Qwen25VLTokens::BOX_START},
            {"<|box_end|>", Qwen25VLTokens::BOX_END},
            {"<|quad_start|>", Qwen25VLTokens::QUAD_START},
            {"<|quad_end|>", Qwen25VLTokens::QUAD_END},
            {"<|vision_start|>", Qwen25VLTokens::VISION_START},
            {"<|vision_end|>", Qwen25VLTokens::VISION_END},
            {"<|vision_pad|>", Qwen25VLTokens::VISION_PAD},
            {"<|image_pad|>", Qwen25VLTokens::IMAGE_PAD},
            {"<|video_pad|>", Qwen25VLTokens::VIDEO_PAD},
            {"<|tool_call_start|>", Qwen25VLTokens::TOOL_CALL_START},
            {"<|tool_call_end|>", Qwen25VLTokens::TOOL_CALL_END},
            {"<|think_start|>", Qwen25VLTokens::THINK_START},
            {"<|think_end|>", Qwen25VLTokens::THINK_END}
        };
    }
    
    // 获取中文token映射
    static std::unordered_map<std::string, int32_t> getChineseTokenMap() {
        return {
            {"你好", Qwen25VLTokens::CHINESE_HELLO},
            {"世界", Qwen25VLTokens::CHINESE_WORLD},
            {"谢谢", Qwen25VLTokens::CHINESE_THANK},
            {"请", Qwen25VLTokens::CHINESE_PLEASE},
            {"问题", Qwen25VLTokens::CHINESE_QUESTION},
            {"回答", Qwen25VLTokens::CHINESE_ANSWER},
            {"帮助", Qwen25VLTokens::CHINESE_HELP},
            {"理解", Qwen25VLTokens::CHINESE_UNDERSTAND}
        };
    }
    
    // 获取所有token映射
    static std::unordered_map<std::string, int32_t> getAllTokenMap() {
        auto special_tokens = getSpecialTokenMap();
        auto chinese_tokens = getChineseTokenMap();
        special_tokens.insert(chinese_tokens.begin(), chinese_tokens.end());
        return special_tokens;
    }
    
    // 判断是否为特殊token
    static bool isSpecialToken(int32_t token_id) {
        return (token_id >= Qwen25VLTokens::ENDOFTEXT && token_id <= Qwen25VLTokens::THINK_END);
    }
    
    // 判断是否为视觉相关token
    static bool isVisionToken(int32_t token_id) {
        return (token_id >= Qwen25VLTokens::VISION_START && token_id <= Qwen25VLTokens::VIDEO_PAD);
    }
    
    // 判断是否为会话token
    static bool isConversationToken(int32_t token_id) {
        return (token_id == Qwen25VLTokens::IM_START || token_id == Qwen25VLTokens::IM_END);
    }
    
    // 根据token ID获取token字符串
    static std::string getTokenString(int32_t token_id) {
        auto token_map = getAllTokenMap();
        for (const auto& pair : token_map) {
            if (pair.second == token_id) {
                return pair.first;
            }
        }
        return "";
    }
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // QWEN25VL_SPECIAL_TOKENS_H