#ifndef DUOROU_EXTENSIONS_OLLAMA_LLAMA_PARAMS_CONVERTER_H
#define DUOROU_EXTENSIONS_OLLAMA_LLAMA_PARAMS_CONVERTER_H

#include "gguf_parser.h"
#include <cstdint>
#include <memory>
#include <string>

// Forward declaration for llama.cpp types
struct llama_model_params;
struct llama_model_kv_override;

namespace duorou {
namespace extensions {
namespace ollama {

/**
 * 将GGUF解析器提取的模型信息转换为llama.cpp的llama_model_params结构体
 */
class LlamaParamsConverter {
public:
    /**
     * 从GGUF解析器创建llama_model_params
     * @param parser 已解析的GGUF文件解析器
     * @param custom_params 可选的自定义参数覆盖
     * @return 配置好的llama_model_params结构体
     */
    static llama_model_params createFromGGUF(
        const GGUFParser& parser,
        const llama_model_params* custom_params = nullptr
    );

    /**
     * 从GGUF文件路径直接创建llama_model_params
     * @param gguf_file_path GGUF文件路径
     * @param custom_params 可选的自定义参数覆盖
     * @return 配置好的llama_model_params结构体
     */
    static llama_model_params createFromFile(
        const std::string& gguf_file_path,
        const llama_model_params* custom_params = nullptr
    );

    /**
     * 获取默认的llama_model_params配置
     * @return 默认配置的llama_model_params
     */
    static llama_model_params getDefaultParams();

    /**
     * 验证llama_model_params配置是否有效
     * @param params 要验证的参数
     * @return 是否有效
     */
    static bool validateParams(const llama_model_params& params);

    /**
     * 打印llama_model_params配置信息（用于调试）
     * @param params 要打印的参数
     */
    static void printParams(const llama_model_params& params);

private:
    /**
     * 从GGUF元数据中提取GPU相关配置
     */
    static void extractGpuConfig(
        const GGUFParser& parser,
        llama_model_params& params
    );

    /**
     * 从GGUF元数据中提取内存映射配置
     */
    static void extractMemoryConfig(
        const GGUFParser& parser,
        llama_model_params& params
    );

    /**
     * 从GGUF元数据中提取键值覆盖配置
     */
    static void extractKvOverrides(
        const GGUFParser& parser,
        llama_model_params& params
    );

    /**
     * 应用自定义参数覆盖
     */
    static void applyCustomOverrides(
        const llama_model_params* custom_params,
        llama_model_params& params
    );
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // LLAMA_PARAMS_CONVERTER_H