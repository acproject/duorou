#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "../../core/text_generator.h"
#include "../../../third_party/llama.cpp/include/llama.h"

namespace duorou {
namespace extensions {
namespace ollama {

/**
 * @brief Ollama模型文本生成器
 * 
 * 继承自TextGenerator，使用llama.cpp库实现Ollama模型的文本生成功能
 */
class OllamaTextGenerator : public duorou::core::TextGenerator {
public:
    /**
     * @brief 构造函数
     * @param model llama_model指针，由OllamaModelLoader加载
     */
    explicit OllamaTextGenerator(llama_model* model);
    
    /**
     * @brief 析构函数
     */
    ~OllamaTextGenerator();
    
    /**
     * @brief 生成文本
     * @param prompt 输入提示词
     * @param params 生成参数
     * @return 生成结果
     */
    duorou::core::GenerationResult generate(
        const std::string& prompt,
        const duorou::core::GenerationParams& params = duorou::core::GenerationParams());
    
    /**
     * @brief 流式生成文本
     * @param prompt 输入提示词
     * @param callback 流式回调函数
     * @param params 生成参数
     * @return 生成结果
     */
    duorou::core::GenerationResult generateStream(
        const std::string& prompt,
        duorou::core::StreamCallback callback,
        const duorou::core::GenerationParams& params = duorou::core::GenerationParams());
    
    /**
     * @brief 计算文本的token数量
     * @param text 输入文本
     * @return token数量
     */
    size_t countTokens(const std::string& text) const;
    
    /**
     * @brief 检查是否可以生成
     * @return 是否可以生成
     */
    bool canGenerate() const;
    
    /**
     * @brief 重置生成器状态
     */
    void reset();
    
    /**
     * @brief 获取上下文大小
     * @return 上下文大小
     */
    int getContextSize() const;
    
    /**
     * @brief 获取词汇表大小
     * @return 词汇表大小
     */
    int getVocabSize() const;

private:
    llama_model* model_;           ///< llama模型指针
    llama_context* context_;       ///< llama上下文指针
    
    /**
     * @brief 初始化llama上下文
     * @return 是否成功初始化
     */
    bool initializeContext();
    
    /**
     * @brief 清理资源
     */
    void cleanup();
    
    /**
     * @brief 将文本转换为tokens
     * @param text 输入文本
     * @param add_bos 是否添加开始符号
     * @return token向量
     */
    std::vector<llama_token> tokenize(const std::string& text, bool add_bos = true) const;
    
    /**
     * @brief 将tokens转换为文本
     * @param tokens token向量
     * @return 文本字符串
     */
    std::string detokenize(const std::vector<llama_token>& tokens) const;
    
    /**
     * @brief 应用采样参数
     * @param candidates 候选token
     * @param params 生成参数
     * @return 选中的token
     */
    llama_token sampleToken(llama_token_data_array* candidates, 
                           const duorou::core::GenerationParams& params);
};

} // namespace ollama
} // namespace extensions
} // namespace duorou