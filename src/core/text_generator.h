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
 * @brief 文本生成结果结构
 */
struct GenerationResult {
  std::string text; ///< 生成的文本
  bool success;            ///< 是否成功
  bool finished;           ///< 是否完成生成
  std::string stop_reason; ///< 停止原因
  size_t prompt_tokens;    ///< 提示词token数
  size_t generated_tokens; ///< 生成的token数
  double generation_time;  ///< 生成时间（秒）

  GenerationResult()
      : finished(false), prompt_tokens(0), generated_tokens(0),
        generation_time(0.0), success(false) {}
};

/**
 * @brief 流式生成回调函数类型
 * @param token 当前生成的token ID
 * @param text 当前生成的文本片段
 * @param finished 是否完成生成
 */
typedef std::function<void(int token, const std::string &text, bool finished)>
    StreamCallback;

/**
 * @brief 文本生成器类
 *
 * 提供基于大语言模型的文本生成功能，支持多种采样策略和流式输出。
 * 可以使用本地llama.cpp模型或远程Ollama服务。
 */
class TextGenerator {
public:
  /**
   * @brief 构造函数（使用本地模型）
   * @param model_path 模型文件路径
   *
   * 注意：当前版本暂时禁用了llama.cpp功能
   */
  TextGenerator(const std::string &model_path = "");

  /**
   * @brief 构造函数（使用Ollama模型管理器）
   * @param model_manager Ollama模型管理器实例
   * @param model_id 要使用的模型ID
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
   * @brief 格式化输入为Qwen ChatML格式
   * @param user_input 用户输入
   * @param system_prompt 系统提示词（可选）
   * @return 格式化后的ChatML字符串
   */
  std::string formatQwenChatML(const std::string &user_input, 
                               const std::string &system_prompt = "") const;

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
   * @brief 设置模型路径
   * @param model_path 新的模型路径
   * @return 是否设置成功
   *
   * 注意：当前版本暂时禁用了llama.cpp功能
   */
  bool setModelPath(const std::string &model_path);

  /**
   * @brief 获取模型路径
   * @return 当前模型路径
   */
  std::string getModelPath() const;

  /**
   * @brief 检查是否可以生成文本
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
   * @param logits 输出概率分布
   * @param k Top-K参数
   */
  void applyTopK(float *logits, int k);

  /**
   * @brief 应用Top-P采样
   * @param logits 输出概率分布
   * @param p Top-P参数
   */
  void applyTopP(float *logits, float p);

  /**
   * @brief 应用温度采样
   * @param logits 输出概率分布
   * @param temperature 温度参数
   */
  void applyTemperature(float *logits, float temperature);

  /**
   * @brief 检查是否应该停止生成
   * @param generated_text 已生成的文本
   * @param stop_sequences 停止序列列表
   * @return 是否应该停止
   */
  bool shouldStop(const std::string &generated_text,
                  const std::vector<std::string> &stop_sequences) const;

  /**
   * @brief 初始化随机数生成器
   * @param seed 随机种子
   */
  void initializeRNG(int64_t seed);

private:
  std::mt19937 rng_; ///< 随机数生成器

  mutable std::mutex mutex_; ///< 线程安全互斥锁

  std::string model_path_; ///< 模型文件路径
  int context_size_; ///< 上下文大小
  int vocab_size_;   ///< 词汇表大小

  // Ollama相关成员
  std::shared_ptr<duorou::extensions::ollama::OllamaModelManager>
      model_manager_;
  std::string model_id_; ///< 当前使用的模型ID
  bool use_ollama_;      ///< 是否使用Ollama模型
};

/**
 * @brief 文本生成器工厂类
 *
 * 提供创建不同类型文本生成器的静态方法
 */
class TextGeneratorFactory {
public:
  /**
   * @brief 创建使用Ollama的文本生成器
   * @param model_manager Ollama模型管理器
   * @param model_id 模型ID
   * @return 文本生成器实例
   */
  static std::unique_ptr<TextGenerator>
  createOllamaGenerator(std::shared_ptr<duorou::extensions::ollama::OllamaModelManager> model_manager,
                       const std::string &model_id);
};

} // namespace core
} // namespace duorou