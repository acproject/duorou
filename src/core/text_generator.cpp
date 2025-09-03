#include "text_generator.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>

namespace duorou {
namespace core {

// 默认构造函数实现
TextGenerator::TextGenerator(const std::string &model_path)
    : context_size_(2048), vocab_size_(32000) {
  // 初始化随机数生成器
  std::random_device rd;
  rng_.seed(rd());
}

// 析构函数
TextGenerator::~TextGenerator() {
  // 空析构函数实现
}

// 生成文本
GenerationResult TextGenerator::generate(const std::string &prompt,
                                         const GenerationParams &params) {
  std::lock_guard<std::mutex> lock(mutex_);

  GenerationResult result;
  result.text = "[DISABLED] Text generation is currently disabled";
  result.finished = true;
  result.stop_reason = "disabled";
  result.prompt_tokens = countTokens(prompt);
  result.generated_tokens = 0;
  result.generation_time = 0.0;

  return result;
}

// 流式生成文本
GenerationResult TextGenerator::generateStream(const std::string &prompt,
                                               StreamCallback callback,
                                               const GenerationParams &params) {
  std::lock_guard<std::mutex> lock(mutex_);

  GenerationResult result;
  result.text = "[DISABLED] Stream generation is currently disabled";
  result.finished = true;
  result.stop_reason = "disabled";
  result.prompt_tokens = countTokens(prompt);
  result.generated_tokens = 0;
  result.generation_time = 0.0;

  // 调用回调函数通知完成
  if (callback) {
    callback(0, result.text, true);
  }

  return result;
}

// 计算文本的token数量
size_t TextGenerator::countTokens(const std::string &text) const {
  // 简单估算：平均每4个字符一个token
  return text.length() / 4 + 1;
}

// 检查是否可以生成
bool TextGenerator::canGenerate() const {
  // 目前返回false，因为功能被禁用
  return false;
}

// 重置生成器状态
void TextGenerator::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  // 重置内部状态
}

// 获取上下文大小
int TextGenerator::getContextSize() const { return context_size_; }

// 获取词汇表大小
int TextGenerator::getVocabSize() const { return vocab_size_; }

// 应用Top-K采样
void TextGenerator::applyTopK(float *logits, int k) {
  if (k <= 0 || !logits)
    return;

  // 简单实现：将除了前k个最大值之外的所有值设为负无穷
  std::vector<std::pair<float, int>> logit_pairs;
  for (int i = 0; i < vocab_size_; ++i) {
    logit_pairs.emplace_back(logits[i], i);
  }

  std::partial_sort(
      logit_pairs.begin(), logit_pairs.begin() + k, logit_pairs.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  for (int i = k; i < vocab_size_; ++i) {
    logits[logit_pairs[i].second] = -INFINITY;
  }
}

// 应用Top-P采样
void TextGenerator::applyTopP(float *logits, float p) {
  if (p <= 0.0f || p >= 1.0f || !logits)
    return;

  // 计算softmax概率
  std::vector<std::pair<float, int>> prob_pairs;
  float max_logit = *std::max_element(logits, logits + vocab_size_);

  float sum = 0.0f;
  for (int i = 0; i < vocab_size_; ++i) {
    float prob = std::exp(logits[i] - max_logit);
    prob_pairs.emplace_back(prob, i);
    sum += prob;
  }

  // 归一化
  for (auto &pair : prob_pairs) {
    pair.first /= sum;
  }

  // 按概率降序排序
  std::sort(prob_pairs.begin(), prob_pairs.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  // 计算累积概率并截断
  float cumulative = 0.0f;
  for (size_t i = 0; i < prob_pairs.size(); ++i) {
    cumulative += prob_pairs[i].first;
    if (cumulative > p) {
      // 将剩余的token概率设为0
      for (size_t j = i + 1; j < prob_pairs.size(); ++j) {
        logits[prob_pairs[j].second] = -INFINITY;
      }
      break;
    }
  }
}

// 应用温度采样
void TextGenerator::applyTemperature(float *logits, float temperature) {
  if (temperature <= 0.0f || !logits)
    return;

  for (int i = 0; i < vocab_size_; ++i) {
    logits[i] /= temperature;
  }
}

// 检查是否应该停止生成
bool TextGenerator::shouldStop(
    const std::string &generated_text,
    const std::vector<std::string> &stop_sequences) const {
  for (const auto &stop_seq : stop_sequences) {
    if (generated_text.find(stop_seq) != std::string::npos) {
      return true;
    }
  }
  return false;
}

// 初始化随机数生成器
void TextGenerator::initializeRNG(int64_t seed) {
  if (seed == -1) {
    std::random_device rd;
    rng_.seed(rd());
  } else {
    rng_.seed(static_cast<unsigned int>(seed));
  }
}

} // namespace core
} // namespace duorou