#pragma once

#include "llama.h"
#include "arch_mapping.h"
#include <string>
#include <vector>
#include <memory>

namespace duorou {
namespace extensions {
namespace llama_cpp {

/**
 * 模型加载包装器，提供架构映射功能
 * 使用llama.cpp的kv_overrides功能来覆盖模型的架构字段
 */
class ModelLoaderWrapper {
public:
    /**
     * 加载模型，自动处理架构映射
     * @param model_path 模型文件路径
     * @param params 模型参数
     * @return 加载的模型指针，失败返回nullptr
     */
    static struct llama_model* loadModelWithArchMapping(
        const std::string& model_path,
        llama_model_params params
    );

private:
    /**
     * 检查模型是否需要架构映射
     * @param model_path 模型文件路径
     * @param original_arch 输出原始架构名称
     * @param mapped_arch 输出映射后的架构名称
     * @return 是否需要映射
     */
    static bool checkArchitectureMapping(
        const std::string& model_path,
        std::string& original_arch,
        std::string& mapped_arch
    );

    /**
     * 创建kv_overrides数组来覆盖架构字段
     * @param mapped_arch 映射后的架构名称
     * @return kv_overrides数组
     */
    static std::vector<llama_model_kv_override> createArchOverrides(
        const std::string& mapped_arch
    );
};

} // namespace llama_cpp
} // namespace extensions
} // namespace duorou