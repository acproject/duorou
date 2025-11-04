#include "RespParser.hpp"

std::vector<std::string> RespParser::parse(std::string &input) {
    std::vector<std::string> args;
    if (input.empty()) return args;

    size_t pos = 0;
    if (input[0] != '*') {
        // 不支持内联协议，等待更多数据或丢弃无效前缀
        return args;
    }

    // 解析数组头部：*<count>\r\n
    size_t newline = input.find("\r\n", pos);
    if (newline == std::string::npos) return args; // 半包

    // 注意：substr 第二个参数是长度，这里应为 newline - 1 - 0
    // 但我们从 1 开始取到 newline 之前的部分
    const std::string count_str = input.substr(1, newline - 1);
    int count = 0;
    try {
        count = std::stoi(count_str);
    } catch (...) {
        return args; // 等待更多或忽略错误
    }
    if (count <= 0) return args;

    pos = newline + 2; // 跳过 \r\n

    for (int i = 0; i < count; ++i) {
        // 期望 $<len>\r\n
        if (pos >= input.size()) return std::vector<std::string>();
        if (input[pos] != '$') {
            return std::vector<std::string>();
        }

        size_t len_end = input.find("\r\n", pos);
        if (len_end == std::string::npos) return std::vector<std::string>();

        int len = 0;
        try {
            len = std::stoi(input.substr(pos + 1, len_end - pos - 1));
        } catch (...) {
            return std::vector<std::string>();
        }
        if (len < 0) return std::vector<std::string>();

        pos = len_end + 2; // 跳过长度行
        // 需要确保 payload + CRLF 完整到达
        if (pos + static_cast<size_t>(len) + 2 > input.size()) {
            return std::vector<std::string>();
        }

        // 提取 payload
        args.emplace_back(input.substr(pos, static_cast<size_t>(len)));
        pos += static_cast<size_t>(len);

        // 校验结尾 CRLF
        if (pos + 1 >= input.size() || input[pos] != '\r' || input[pos + 1] != '\n') {
            return std::vector<std::string>();
        }
        pos += 2; // 跳过 CRLF
    }

    // 到这里表示一条命令完整解析成功，移除已消费的数据
    input.erase(0, pos);
    return args;
}