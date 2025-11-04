#pragma once

#include <string>
#include <vector>

class RespParser {
public:
    // 解析并消费输入缓冲区中的一条完整 RESP 命令。
    // 若缓冲区中不足以组成完整命令，返回空向量且不修改缓冲区。
    // 若解析成功，返回命令参数，并从缓冲区中移除已消费的字节。
    static std::vector<std::string> parse(std::string& buffer);
};