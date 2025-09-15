#ifndef DUOROU_EXTENSIONS_OLLAMA_LLAMA_MODEL_LOADER_H
#define DUOROU_EXTENSIONS_OLLAMA_LLAMA_MODEL_LOADER_H

#include "llama_params_converter.h"
#include "gguf_parser.h"
#include <string>
#include <memory>

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_model_params;
struct llama_context;
struct llama_context_params;

namespace duorou {
namespace extensions {
namespace ollama {

/**
 * 集成的llama模型加载器，结合GGUF解析器和llama.cpp
 */
class LlamaModelLoader {
public:
    /**
     * 使用GGUF解析器和自定义参数加载llama模型
     * @param gguf_file_path GGUF文件路径
     * @param custom_params 可选的自定义参数覆盖
     * @return 加载的llama_model指针，失败时返回nullptr
     */
    static llama_model* loadModelWithParams(
        const std::string& gguf_file_path,
        const llama_model_params* custom_params = nullptr
    );

    /**
     * 从已解析的GGUF数据加载llama模型
     * @param parser 已解析的GGUF解析器
     * @param gguf_file_path GGUF文件路径
     * @param custom_params 可选的自定义参数覆盖
     * @return 加载的llama_model指针，失败时返回nullptr
     */
    static llama_model* loadModelFromGGUF(
        const GGUFParser& parser,
        const std::string& gguf_file_path,
        const llama_model_params* custom_params = nullptr
    );

    /**
     * 创建llama上下文
     * @param model 已加载的llama模型
     * @param parser GGUF解析器（用于提取上下文参数）
     * @param custom_ctx_params 可选的自定义上下文参数
     * @return 创建的llama_context指针，失败时返回nullptr
     */
    static llama_context* createContext(
        llama_model* model,
        const GGUFParser& parser,
        const llama_context_params* custom_ctx_params = nullptr
    );

    /**
     * 获取默认的llama上下文参数
     * @param parser GGUF解析器（用于提取模型信息）
     * @return 配置好的llama_context_params
     */
    static llama_context_params getDefaultContextParams(
        const GGUFParser& parser
    );

    /**
     * 验证模型是否成功加载
     * @param model 要验证的模型指针
     * @return 是否有效
     */
    static bool validateModel(llama_model* model);

    /**
     * 安全释放llama模型
     * @param model 要释放的模型指针
     */
    static void freeModel(llama_model* model);

    /**
     * 安全释放llama上下文
     * @param ctx 要释放的上下文指针
     */
    static void freeContext(llama_context* ctx);

    /**
     * 打印模型信息（用于调试）
     * @param model 模型指针
     * @param parser GGUF解析器
     */
    static void printModelInfo(
        llama_model* model,
        const GGUFParser& parser
    );

private:
    /**
     * 初始化llama后端（如果尚未初始化）
     */
    static void ensureBackendInitialized();

    /**
     * 从GGUF解析器提取上下文参数
     */
    static void extractContextParams(
        const GGUFParser& parser,
        llama_context_params& ctx_params
    );

    /**
     * 验证GGUF文件与llama.cpp的兼容性
     */
    static bool validateCompatibility(
        const GGUFParser& parser
    );
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_LLAMA_MODEL_LOADER_H