#ifndef DUOROU_EXTENSIONS_OLLAMA_LLAMA_MODEL_LOADER_H
#define DUOROU_EXTENSIONS_OLLAMA_LLAMA_MODEL_LOADER_H

#include "../third_party/llama.cpp/include/llama.h"
#include "../third_party/llama.cpp/include/ggml.h"
#include "../third_party/llama.cpp/common/common.h"

// 前向声明
namespace duorou {
namespace extensions {
namespace ollama {

class GGUFParser;
struct ModelInfo;

/**
 * LlamaModelLoader - 结合GGUF解析器和llama.cpp的模型加载器
 * 提供统一的模型加载和管理接口
 */
class LlamaModelLoader {
public:
    /**
     * 使用指定参数加载模型
     * @param model_path 模型文件路径
     * @param params llama模型参数
     * @return 加载的模型指针，失败时抛出异常
     */
    static struct llama_model* loadModel(const std::string& model_path, 
                                         const llama_model_params& params);
    
    /**
     * 从GGUF解析器加载模型
     * @param parser GGUF解析器实例
     * @return 加载的模型指针，失败时抛出异常
     */
    static struct llama_model* loadModelWithGGUF(const GGUFParser& parser);
    
    /**
     * 为模型创建上下文
     * @param model 已加载的模型
     * @param model_info 模型信息
     * @return 创建的上下文指针，失败时抛出异常
     */
    static struct llama_context* createContext(struct llama_model* model, 
                                               const ModelInfo& model_info);
    
    /**
     * 获取默认模型参数
     * @return 默认的llama模型参数
     */
    static llama_model_params getDefaultModelParams();
    
    /**
     * 验证模型是否正确加载
     * @param model 要验证的模型
     * @param model_info 期望的模型信息
     * @return 验证是否通过
     */
    static bool validateModel(struct llama_model* model, const ModelInfo& model_info);
    
    /**
     * 释放模型资源
     * @param model 要释放的模型
     */
    static void freeModel(struct llama_model* model);
    
    /**
     * 释放上下文资源
     * @param ctx 要释放的上下文
     */
    static void freeContext(struct llama_context* ctx);
    
    /**
     * 打印模型信息
     * @param model 要打印信息的模型
     */
    static void printModelInfo(struct llama_model* model);

private:
    /**
     * 确保llama后端已初始化
     */
    static void ensureBackendInitialized();
    
    /**
     * 从模型信息提取上下文参数
     * @param model_info 模型信息
     * @return 上下文参数
     */
    static llama_context_params extractContextParams(const ModelInfo& model_info);
    
    /**
     * 验证GGUF解析器与模型信息的兼容性
     * @param parser GGUF解析器
     * @param model_info 模型信息
     * @return 是否兼容
     */
    static bool validateCompatibility(const GGUFParser& parser, 
                                    const ModelInfo& model_info);
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_LLAMA_MODEL_LOADER_H