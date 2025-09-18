#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>
// #include "../../third_party/llama.cpp/include/llama.h"  //
// 注释：暂时禁用llama相关功能
#include "../extensions/ollama/ollama_model_manager.h"

namespace duorou {
namespace core {

/**
 * @brief 文本生成参数结构
 */
struct GenerationParams {
  int max_tokens = 100;                    ///< 最大生成token数
  float temperature = 0.8f;                ///< 温度参数，控制随机性
  float top_p = 0.9f;                      ///< Top-p采样参数
  int top_k = 40;                          ///< Top-k采样参数
  float repeat_penalty = 1.1f;             ///< 重复惩罚
  int repeat_last_n = 64;                  ///< 重复惩罚考虑的token数
  int64_t seed = -1;                       ///< 随机种子，-1表示随机
  std::vector<std::string> stop_sequences; ///< 停止序列
  bool stream = false;                     ///< 是否流式输出

  GenerationParams() = default;
};

/**
 * @brief 生成结果结构
 */
struct GenerationResult {
  std::string text; ///< 生成的文本
  // std::vector<llama_token> tokens;  ///< 生成的token序列 - 暂时禁用
  bool finished;           ///< 是否完成生成
  std::string stop_reason; ///< 停止原因
  size_t prompt_tokens;    ///< 提示词token数
  size_t generated_tokens; ///< 生成的token数
  double generation_time;  ///< 生成时间（秒）

  GenerationResult()
      : finished(false), prompt_tokens(0), generated_tokens(0),
        generation_time(0.0) {}
};

/**
 * @brief 流式生成回调函数类型
 * @param token 新生成的token
 * @param text 对应的文本片段
 * @param finished 是否完成
 */
// typedef std::function<void(llama_token token, const std::string& text, bool
// finished)> StreamCallback;  // 暂时禁用
typedef std::function<void(int token, const std::string &text, bool finished)>
    StreamCallback;

/**
 * @brief 文本生成器类
 *
 * 负责使用llama模型进行文本生成，支持多种采样策略和参数配置
 */
class TextGenerator {
public:
  /**
   * @brief 构造函数
   * @param model llama模型指针
   * @param context llama上下文指针
   */
  // TextGenerator(llama_model* model, llama_context* context);  // 暂时禁用

  /**
   * @brief 默认构造函数（用于Ollama模型）
   * @param model_path 模型路径
   */
  TextGenerator(const std::string &model_path = "");

  /**
   * @brief 构造函数（使用OllamaModelManager）
   * @param model_manager Ollama模型管理器指针
   * @param model_id 模型ID
   */
  TextGenerator(std::shared_ptr<duorou::extensions::ollama::OllamaModelManager>
                    model_manager,
                const std::string &model_id);

  /**
   * @brief 析构函数
   */
  ~TextGenerator();

  /**
   * @brief 生成文本
   * @param prompt 输入提示词
   * @param params 生成参数
   * @return 生成结果
   */
  GenerationResult
  generate(const std::string &prompt,
           const GenerationParams &params = GenerationParams());

  /**
   * @brief 流式生成文本
   * @param prompt 输入提示词
   * @param callback 流式回调函数
   * @param params 生成参数
   * @return 生成结果
   */
  GenerationResult
  generateStream(const std::string &prompt, StreamCallback callback,
                 const GenerationParams &params = GenerationParams());

  /**
   * @brief 计算文本的token数量
   * @param text 输入文本
   * @return token数量
   */
  size_t countTokens(const std::string &text) const;

  /**
   * @brief 将文本转换为token序列
   * @param text 输入文本
   * @param add_bos 是否添加开始token
   * @return token序列
   */
  // std::vector<llama_token> textToTokens(const std::string& text, bool add_bos
  // = true) const;  // 暂时禁用

  /**
   * @brief 将token序列转换为文本
   * @param tokens token序列
   * @return 文本
   */
  // std::string tokensToText(const std::vector<llama_token>& tokens) const;  //
  // 暂时禁用

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
  /**
   * @brief 应用Top-K采样
   * @param logits logits数组
   * @param k Top-K参数
   */
  void applyTopK(float *logits, int k);

  /**
   * @brief 应用Top-P采样
   * @param logits logits数组
   * @param p Top-P参数
   */
  void applyTopP(float *logits, float p);

  /**
   * @brief 应用温度采样
   * @param logits logits数组
   * @param temperature 温度参数
   */
  void applyTemperature(float *logits, float temperature);

  /**
   * @brief 检查是否应该停止生成
   * @param generated_text 已生成的文本
   * @param stop_sequences 停止序列
   * @return 是否应该停止
   */
  bool shouldStop(const std::string &generated_text,
                  const std::vector<std::string> &stop_sequences) const;

  /**
   * @brief 初始化随机数生成器
   * @param seed 随机种子
   */
  void initializeRNG(int64_t seed);

  /**
   * @brief 归一化模型ID，与OllamaModelManager保持一致
   * @param model_name 原始模型名称
   * @return 归一化后的模型ID
   */
  std::string normalizeModelId(const std::string &model_name) const;

private:
  std::mt19937 rng_; ///< 随机数生成器

  mutable std::mutex mutex_; ///< 线程安全互斥锁

  // 模型信息
  int context_size_; ///< 上下文大小
  int vocab_size_;   ///< 词汇表大小

  // Ollama模型管理器
  std::shared_ptr<duorou::extensions::ollama::OllamaModelManager>
      model_manager_;
  std::string model_id_; ///< 当前使用的模型ID
  bool use_ollama_;      ///< 是否使用Ollama模型
};

/**
 * @brief 文本生成器工厂类
 */
class TextGeneratorFactory {
public:
  /**
   * @brief 创建文本生成器
   * @param model llama模型指针
   * @param context llama上下文指针
   * @return 文本生成器指针
   */
  // static std::unique_ptr<TextGenerator> create(llama_model* model,
  // llama_context* context);  // 暂时禁用
};

} // namespace core
} // namespace duorou