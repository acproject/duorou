#include "text_generator.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>

namespace duorou {
namespace core {

// 默认构造函数实现
TextGenerator::TextGenerator(const std::string &model_path)
    : context_size_(2048), vocab_size_(32000), use_ollama_(false) {
  // 初始化随机数生成器
  std::random_device rd;
  rng_.seed(rd());
}

// Ollama模型管理器构造函数
TextGenerator::TextGenerator(
    std::shared_ptr<duorou::extensions::ollama::OllamaModelManager>
        model_manager,
    const std::string &model_id)
    : context_size_(2048), vocab_size_(32000), model_manager_(model_manager),
      model_id_(normalizeModelId(model_id)), use_ollama_(true) {
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
  std::cout << "[DEBUG] TextGenerator::generate() called with prompt: "
            << prompt.substr(0, 50) << "..." << std::endl;

  std::lock_guard<std::mutex> lock(mutex_);
  GenerationResult result;

  if (use_ollama_ && model_manager_) {
    std::cout << "[DEBUG] Using Ollama model manager for inference"
              << std::endl;

    try {
      // 创建推理请求
      duorou::extensions::ollama::InferenceRequest request;
      request.model_id = model_id_;
      request.prompt = prompt;
      request.max_tokens = params.max_tokens;
      request.temperature = params.temperature;
      request.top_p = params.top_p;

      // 执行推理
      auto start_time = std::chrono::high_resolution_clock::now();
      auto response = model_manager_->generateText(request);
      auto end_time = std::chrono::high_resolution_clock::now();

      if (response.success) {
        result.text = response.generated_text;
        result.finished = true;
        result.stop_reason = "completed";
        result.prompt_tokens = countTokens(prompt);
        result.generated_tokens = response.tokens_generated;
        result.generation_time =
            std::chrono::duration<double>(end_time - start_time).count();

        std::cout << "[DEBUG] Ollama inference successful: "
                  << result.text.substr(0, 50) << "..." << std::endl;
      } else {
        std::cout << "[DEBUG] Ollama inference failed: "
                  << response.error_message << std::endl;
        result.text = "抱歉，推理过程中出现错误: " + response.error_message;
        result.finished = true;
        result.stop_reason = "error";
        result.prompt_tokens = countTokens(prompt);
        result.generated_tokens = 0;
        result.generation_time = 0.0;
      }
    } catch (const std::exception &e) {
      std::cout << "[DEBUG] Exception during Ollama inference: " << e.what()
                << std::endl;
      result.text = "抱歉，推理过程中出现异常: " + std::string(e.what());
      result.finished = true;
      result.stop_reason = "exception";
      result.prompt_tokens = countTokens(prompt);
      result.generated_tokens = 0;
      result.generation_time = 0.0;
    }
  } else {
    std::cout << "[DEBUG] Using fallback mock implementation" << std::endl;

    // 简单的模拟响应
    if (prompt.find("你好") != std::string::npos ||
        prompt.find("hello") != std::string::npos) {
      result.text =
          "你好！我是 Duorou AI 助手，很高兴为您服务。有什么我可以帮助您的吗？";
    } else {
      result.text = "感谢您的提问。这是一个模拟的文本生成响应。当前版本使用简化"
                    "的实现，未来将集成完整的 llama.cpp 功能。";
    }
    result.finished = true;
    result.stop_reason = "completed";
    result.prompt_tokens = countTokens(prompt);
    result.generated_tokens = countTokens(result.text);
    result.generation_time = 0.5; // 模拟生成时间
  }

  std::cout << "[DEBUG] TextGenerator returning result: "
            << result.text.substr(0, 30) << "..." << std::endl;
  return result;
}

// 流式生成文本
GenerationResult TextGenerator::generateStream(const std::string &prompt,
                                               StreamCallback callback,
                                               const GenerationParams &params) {
  std::cout << "[DEBUG] TextGenerator::generateStream() called with prompt: "
            << prompt.substr(0, 50) << "..." << std::endl;

  std::lock_guard<std::mutex> lock(mutex_);
  GenerationResult result;

  if (!callback) {
    result.text = "Error: No callback provided for streaming";
    result.finished = true;
    result.stop_reason = "error";
    result.prompt_tokens = countTokens(prompt);
    result.generated_tokens = 0;
    result.generation_time = 0.0;
    return result;
  }

  if (use_ollama_ && model_manager_) {
    std::cout << "[DEBUG] Using Ollama model manager for streaming inference"
              << std::endl;

    try {
      // 创建流式推理请求
      duorou::extensions::ollama::InferenceRequest request;
      request.model_id = model_id_;
      request.prompt = prompt;
      request.max_tokens = params.max_tokens;
      request.temperature = params.temperature;
      request.top_p = params.top_p;

      auto start_time = std::chrono::high_resolution_clock::now();
      auto response = model_manager_->generateText(request);
      auto end_time = std::chrono::high_resolution_clock::now();

      if (response.success) {
        // 模拟流式输出，将完整响应分块发送
        std::string full_text = response.generated_text;
        const size_t chunk_size = 10; // 每次发送10个字符

        for (size_t i = 0; i < full_text.length(); i += chunk_size) {
          std::string chunk = full_text.substr(i, chunk_size);
          bool is_final = (i + chunk_size >= full_text.length());
          callback(i / chunk_size, chunk, is_final);

          // 模拟流式延迟
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        result.text = full_text;
        result.finished = true;
        result.stop_reason = "completed";
        result.prompt_tokens = countTokens(prompt);
        result.generated_tokens = response.tokens_generated;
        result.generation_time =
            std::chrono::duration<double>(end_time - start_time).count();

        std::cout << "[DEBUG] Ollama streaming inference successful"
                  << std::endl;
      } else {
        std::string error_msg =
            "抱歉，流式推理过程中出现错误: " + response.error_message;
        callback(0, error_msg, true);

        result.text = error_msg;
        result.finished = true;
        result.stop_reason = "error";
        result.prompt_tokens = countTokens(prompt);
        result.generated_tokens = 0;
        result.generation_time = 0.0;
      }
    } catch (const std::exception &e) {
      std::string error_msg =
          "抱歉，流式推理过程中出现异常: " + std::string(e.what());
      callback(0, error_msg, true);

      result.text = error_msg;
      result.finished = true;
      result.stop_reason = "exception";
      result.prompt_tokens = countTokens(prompt);
      result.generated_tokens = 0;
      result.generation_time = 0.0;
    }
  } else {
    std::cout << "[DEBUG] Using fallback mock streaming implementation"
              << std::endl;

    // 模拟流式生成
    std::string response_text;
    if (prompt.find("你好") != std::string::npos ||
        prompt.find("hello") != std::string::npos) {
      response_text =
          "你好！我是 Duorou AI 助手，很高兴为您服务。有什么我可以帮助您的吗？";
    } else {
      response_text = "感谢您的提问。这是一个模拟的流式文本生成响应。当前版本使"
                      "用简化的实现，未来将集成完整的 llama.cpp 功能。";
    }

    // 分块发送响应
    const size_t chunk_size = 8;
    for (size_t i = 0; i < response_text.length(); i += chunk_size) {
      std::string chunk = response_text.substr(i, chunk_size);
      bool is_final = (i + chunk_size >= response_text.length());
      callback(i / chunk_size, chunk, is_final);

      // 模拟生成延迟
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    result.text = response_text;
    result.finished = true;
    result.stop_reason = "completed";
    result.prompt_tokens = countTokens(prompt);
    result.generated_tokens = countTokens(response_text);
    result.generation_time = 0.5;
  }

  std::cout << "[DEBUG] TextGenerator returning streaming result" << std::endl;
  return result;
}

// 计算文本的token数量
size_t TextGenerator::countTokens(const std::string &text) const {
  // 简单估算：平均每4个字符一个token
  return text.length() / 4 + 1;
}

// 检查是否可以生成
bool TextGenerator::canGenerate() const {
  std::cout << "[DEBUG] TextGenerator::canGenerate() called - returning true "
               "(functionality enabled)"
            << std::endl;
  // 启用文本生成功能
  return true;
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

std::string TextGenerator::normalizeModelId(const std::string &model_name) const {
  std::string model_id = model_name;
  // 将特殊字符替换为下划线（与OllamaModelManager中的逻辑保持一致）
  for (char &c : model_id) {
    if (!std::isalnum(c) && c != '_' && c != '-' && c != '.') {
      c = '_';
    }
  }
  return model_id;
}

} // namespace core
} // namespace duorou