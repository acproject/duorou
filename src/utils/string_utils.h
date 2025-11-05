#pragma once

#include <string>
#include <vector>

namespace duorou {
namespace utils {

/**
 * 解码包含十六进制转义序列的字符串
 * 例如: "\\x48\\x65\\x6c\\x6c\\x6f" -> "Hello"
 * @param input 包含十六进制转义序列的字符串
 * @return 解码后的字符串
 */
std::string decodeHexEscapes(const std::string& input);

/**
 * 检查字符串是否包含十六进制转义序列
 * @param input 要检查的字符串
 * @return 如果包含十六进制转义序列则返回true
 */
bool containsHexEscapes(const std::string& input);

/**
 * 批量解码字符串数组中的十六进制转义序列
 * @param tokens 包含可能的十六进制转义序列的字符串数组
 * @return 解码后的字符串数组
 */
std::vector<std::string> decodeTokenStrings(const std::vector<std::string>& tokens);

} // namespace utils
} // namespace duorou