#pragma once

#include "gguf.h"
#include <string>
#include <vector>
#include <memory>

namespace duorou {
namespace extensions {
namespace llama_cpp {

/**
 * GGUF文件包装器，用于在模型加载前动态修改GGUF元数据
 * 主要用于解决qwen2.5vl模型缺失dimension_sections的问题
 */
class GGUFWrapper {
public:
    /**
     * 创建包含dimension_sections的临时GGUF文件
     * @param original_path 原始GGUF文件路径
     * @param temp_path 临时文件路径
     * @return 临时文件路径
     */
    static std::string createTempGGUFWithDimensionSections(
        const std::string& original_path,
        const std::string& temp_path
    );

    /**
     * 检查GGUF文件是否缺失dimension_sections键
     * @param file_path GGUF文件路径
     * @return 是否缺失该键
     */
    static bool isMissingDimensionSections(const std::string& file_path);

    /**
     * 从GGUF文件中读取mrope_section数组
     * @param file_path GGUF文件路径
     * @param mrope_sections 输出数组，包含4个整数
     * @return 是否成功读取
     */
    static bool readMropeSections(
        const std::string& file_path,
        int32_t mrope_sections[4]
    );

private:

};

} // namespace llama_cpp
} // namespace extensions
} // namespace duorou