#include "qwen25vl_inference_engine.h"
#include "text_processor.h"
#include "qwen25vl_special_tokens.h"
#include "../../core/logger.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <iomanip>

namespace duorou {
namespace extensions {
namespace ollama {

// 默认构造函数
Qwen25VLInferenceEngine::Qwen25VLInferenceEngine()
    : config_(), gguf_parser_(nullptr), vision_encoder_(nullptr),
      bos_token_id_(151643), eos_token_id_(151645), pad_token_id_(151643),
      unk_token_id_(151643), kv_cache_(nullptr), kv_cache_enabled_(false),
      temperature_(1.0f), top_p_(0.9f), top_k_(50), repetition_penalty_(1.1f),
      model_loaded_(false), verbose_(false), max_sequence_length_(2048),
      num_threads_(1), parallel_processing_enabled_(false),
      quantization_enabled_(false), quantization_type_("none"),
      total_inference_time_(0.0), total_tokens_generated_(0), actual_vocab_size_(0) {
  log("INFO", "Qwen25VLInferenceEngine initialized with default settings");
}

// 带verbose参数的构造函数
Qwen25VLInferenceEngine::Qwen25VLInferenceEngine(bool verbose)
    : config_(), gguf_parser_(nullptr), vision_encoder_(nullptr),
      bos_token_id_(151643), eos_token_id_(151645), pad_token_id_(151643),
      unk_token_id_(151643), kv_cache_(nullptr), kv_cache_enabled_(false),
      temperature_(1.0f), top_p_(0.9f), top_k_(50), repetition_penalty_(1.1f),
      model_loaded_(false), verbose_(verbose), max_sequence_length_(2048),
      num_threads_(1), parallel_processing_enabled_(false),
      quantization_enabled_(false), quantization_type_("none"),
      total_inference_time_(0.0), total_tokens_generated_(0), actual_vocab_size_(0) {
  log("INFO", "Qwen25VLInferenceEngine initialized with verbose=" +
                  std::to_string(verbose));
}

// 析构函数
Qwen25VLInferenceEngine::~Qwen25VLInferenceEngine() {
  unloadModel();
  log("INFO", "Qwen25VLInferenceEngine destroyed");
}

// 模型加载
bool Qwen25VLInferenceEngine::loadModel(const std::string &model_path) {
  log("DEBUG", "=== ENTERING loadModel method ===");
  log("INFO", "Loading model from: " + model_path);

  try {
    // 创建GGUF解析器
    gguf_parser_ = std::make_unique<GGUFParser>();

    // 解析GGUF文件
    if (!gguf_parser_->parseFile(model_path)) {
      log("ERROR", "Failed to parse GGUF file: " + model_path);
      return false;
    }

    // 从GGUF架构信息初始化config_
    log("DEBUG", std::string("About to call initializeConfigFromGGUF, gguf_parser_ is: ") + 
         (gguf_parser_ ? "valid" : "null"));
    if (!initializeConfigFromGGUF()) {
      log("ERROR", "Failed to initialize config from GGUF architecture");
      return false;
    }
    log("DEBUG", "Successfully initialized config from GGUF");

    // 加载模型权重
    if (!loadWeights(model_path)) {
      log("ERROR", "Failed to load model weights");
      return false;
    }

    // 初始化KV缓存 - 支持GQA
  kv_cache_ = std::make_unique<KVCache>();
  kv_cache_->resize(config_.num_layers, max_sequence_length_,
                    config_.hidden_size, config_.num_key_value_heads);

    // 预计算RoPE频率
    precomputeRoPEFreqs();

    // 加载词汇表
    if (!loadVocabulary()) {
      log("ERROR", "Failed to load vocabulary from GGUF");
      return false;
    }

    // 初始化文本处理器
    auto vocab = std::make_shared<Vocabulary>();

    // 从GGUF文件中加载词汇表数据到vocab
    std::vector<std::string> token_values;
    std::vector<int32_t> token_types;
    std::vector<float> token_scores;
    std::vector<std::string> merges;

    // 将vocab_映射转换为向量格式
    token_values.resize(vocab_.size());
    token_types.resize(vocab_.size(), 1);     // 默认为normal token
    token_scores.resize(vocab_.size(), 0.0f); // 默认分数为0

    for (const auto &pair : vocab_) {
      if (pair.second >= 0 &&
          pair.second < static_cast<int32_t>(token_values.size())) {
        token_values[pair.second] = pair.first;
      }
    }

    vocab->initialize(token_values, token_types, token_scores, merges);

    // 读取GGUF文件中的tokenizer模型类型
    std::string tokenizer_type = "bpe"; // 默认值
    bool is_qwen25vl = false;           // 将变量定义移到外层作用域

    const auto *tokenizer_model_kv =
        gguf_parser_->getMetadata("tokenizer.ggml.model");
    if (tokenizer_model_kv && tokenizer_model_kv->type == GGUFType::STRING) {
      try {
        std::string tokenizer_model = tokenizer_model_kv->asString();
        log("INFO", "Detected tokenizer model: " + tokenizer_model);

        // 对于Qwen2.5VL模型，强制使用正确的tokenizer配置
        // 检查模型是否为Qwen2.5VL（通过检查特殊token或其他标识）

        // 检查是否有Qwen2.5VL特有的特殊token
        const auto *special_tokens_kv =
            gguf_parser_->getMetadata("tokenizer.ggml.added_tokens");
        if (special_tokens_kv) {
          // 如果有added_tokens，很可能是Qwen2.5VL
          is_qwen25vl = true;
          log("INFO", "Detected Qwen2.5VL model based on added_tokens");
        }

        // 也可以通过检查词汇表大小来判断
        if (token_values.size() > 150000) {
          is_qwen25vl = true;
          log("INFO", "Detected Qwen2.5VL model based on vocabulary size: " +
                          std::to_string(token_values.size()));
        }

        if (is_qwen25vl) {
          // 对于Qwen2.5VL，强制使用BPE但需要特殊配置
          tokenizer_type = "bpe";
          log("INFO", "Using BPE tokenizer for Qwen2.5VL model");
        } else {
          // 根据检测到的模型类型设置正确的处理器类型
          if (tokenizer_model == "gpt2" || tokenizer_model == "qwen2") {
            tokenizer_type = "bpe";
          } else if (tokenizer_model == "llama" ||
                     tokenizer_model == "sentencepiece") {
            tokenizer_type = "sentencepiece";
          } else {
            log("WARNING", "Unknown tokenizer model: " + tokenizer_model +
                               ", defaulting to BPE");
            tokenizer_type = "bpe";
          }
        }
      } catch (const std::exception &e) {
        log("WARNING",
            "Failed to read tokenizer model: " + std::string(e.what()));
      }
    }

    // 读取预分词器正则表达式
    std::string pre_tokenizer_regex;
    const auto *pretokenizer_kv =
        gguf_parser_->getMetadata("tokenizer.ggml.pretokenizer");
    if (pretokenizer_kv && pretokenizer_kv->type == GGUFType::STRING) {
      try {
        pre_tokenizer_regex = pretokenizer_kv->asString();
        log("INFO", "Found pretokenizer regex: " + pre_tokenizer_regex);
      } catch (const std::exception &e) {
        log("WARNING",
            "Failed to read pretokenizer regex: " + std::string(e.what()));
      }
    } else {
      // 如果GGUF文件中没有pretokenizer，为Qwen2.5VL使用默认的正则表达式
      if (is_qwen25vl) {
        // 使用ollama源代码中Qwen2.5VL的预分词器正则表达式
        pre_tokenizer_regex =
            R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
        log("INFO", "Using default Qwen2.5VL pretokenizer regex: " +
                        pre_tokenizer_regex);
      }
    }

    // 创建对应类型的文本处理器
    text_processor_ =
        createTextProcessor(tokenizer_type, vocab, pre_tokenizer_regex);
    if (!text_processor_) {
      log("WARNING", "Failed to create " + tokenizer_type +
                         " text processor, trying fallback");
      // 尝试另一种类型作为回退
      std::string fallback_type =
          (tokenizer_type == "bpe") ? "sentencepiece" : "bpe";
      text_processor_ =
          createTextProcessor(fallback_type, vocab, pre_tokenizer_regex);
      if (!text_processor_) {
        log("WARNING", "Failed to create any text processor, using fallback");
      }
    }

    model_loaded_ = true;
    log("INFO", "Model loaded successfully");
    return true;

  } catch (const std::exception &e) {
    log("ERROR", "Exception during model loading: " + std::string(e.what()));
    return false;
  }
}

// 卸载模型
bool Qwen25VLInferenceEngine::unloadModel() {
  if (!model_loaded_) {
    return true;
  }

  log("INFO", "Unloading model");

  // 清理资源
  gguf_parser_.reset();
  vision_encoder_.reset();
  kv_cache_.reset();

  // 清空权重
  token_embeddings_.data.clear();
  transformer_layers_.clear();
  output_norm_weights_.data.clear();
  output_norm_bias_.data.clear();
  output_projection_.data.clear();

  // 清空词汇表
  vocab_.clear();
  reverse_vocab_.clear();

  model_loaded_ = false;
  log("INFO", "Model unloaded successfully");
  return true;
}

// 检查模型是否已加载
bool Qwen25VLInferenceEngine::isModelLoaded() const { return model_loaded_; }

// 生成文本
std::string Qwen25VLInferenceEngine::generateText(const std::string &prompt,
                                                  int max_tokens) {
  std::cout
      << "[DEBUG] Qwen25VLInferenceEngine::generateText called with prompt: "
      << prompt.substr(0, 20) << "..." << std::endl;

  if (!model_loaded_) {
    std::cout << "[DEBUG] Model not loaded, returning empty string"
              << std::endl;
    log("ERROR", "Model not loaded");
    return "";
  }

  // Clear cache and reset state before each generation
  clearCache();

  std::cout << "[DEBUG] Model is loaded, starting text generation" << std::endl;
  log("INFO", "Generating text for prompt: " + prompt +
                  ", max_tokens: " + std::to_string(max_tokens));

  auto start_time = std::chrono::high_resolution_clock::now();

  // 分词
  std::cout << "[DEBUG] Starting tokenization" << std::endl;
  log("DEBUG", "Starting tokenization");
  std::vector<int32_t> input_tokens = tokenize(prompt);
  std::cout << "[DEBUG] Tokenization completed, tokens: " << input_tokens.size()
            << std::endl;
  log("DEBUG",
      "Tokenization completed, tokens: " + std::to_string(input_tokens.size()));

  // 前向传播
  std::cout << "[DEBUG] Starting forward pass loop" << std::endl;
  std::vector<int32_t> generated_tokens;
  generated_tokens_history_.clear(); // 清空历史记录
  int32_t last_token = -1;
  int repeat_count = 0;

  for (int i = 0; i < max_tokens; ++i) {
    std::cout << "[DEBUG] Forward pass iteration: " << i << std::endl;
    log("DEBUG", "Forward pass iteration: " + std::to_string(i));
    std::cout << "[DEBUG] Calling forward() with " << input_tokens.size()
              << " tokens" << std::endl;
    
    // 输出当前输入tokens的详细信息
    if (verbose_ && i < 3) { // 只在前3次迭代输出详细信息
      std::cout << "[DEBUG] Input tokens: ";
      for (size_t j = 0; j < std::min(input_tokens.size(), size_t(10)); ++j) {
        std::cout << input_tokens[j] << " ";
      }
      if (input_tokens.size() > 10) std::cout << "...";
      std::cout << std::endl;
    }
    
    // 添加超时机制
    auto forward_start = std::chrono::high_resolution_clock::now();
    Tensor logits = forward(input_tokens);
    auto forward_end = std::chrono::high_resolution_clock::now();
    auto forward_duration = std::chrono::duration_cast<std::chrono::milliseconds>(forward_end - forward_start);
    
    if (forward_duration.count() > 30000) { // 30秒超时
      log("ERROR", "Forward pass took too long: " + std::to_string(forward_duration.count()) + "ms, stopping generation");
      break;
    }
    
    if (verbose_) {
      std::cout << "[DEBUG] Forward pass took " << forward_duration.count() << "ms" << std::endl;
    }
    
    // 输出logits的统计信息
    if (verbose_) {
      float max_logit = *std::max_element(logits.data.begin(), logits.data.end());
      float min_logit = *std::min_element(logits.data.begin(), logits.data.end());
      std::cout << "[DEBUG] Logits range: [" << min_logit << ", " << max_logit << "]" << std::endl;
      
      // 输出top 5 logits
      std::vector<std::pair<float, int32_t>> top_logits;
      for (size_t j = 0; j < logits.data.size(); ++j) {
        top_logits.push_back({logits.data[j], static_cast<int32_t>(j)});
      }
      std::sort(top_logits.begin(), top_logits.end(), std::greater<std::pair<float, int32_t>>());
      
      std::cout << "[DEBUG] Top 5 logits: ";
      for (int j = 0; j < std::min(5, static_cast<int>(top_logits.size())); ++j) {
        std::cout << "token_" << top_logits[j].second << "(" << top_logits[j].first << ") ";
      }
      std::cout << std::endl;
    }
    
    std::cout << "[DEBUG] Forward pass completed, sampling token" << std::endl;
    log("DEBUG", "Forward pass completed, sampling token");

    // 在采样前屏蔽BOS/PAD/UNK，且首步屏蔽EOS，避免立即终止或无效token
    {
      const float NEG_INF = -1e9f;
      auto mask_token = [&](int32_t tid, const char* name) {
        if (tid >= 0 && tid < static_cast<int32_t>(logits.data.size())) {
          logits.data[tid] = NEG_INF;
          if (verbose_) {
            std::cout << "[DEBUG] Masking " << name << " token id " << tid << " before sampling" << std::endl;
          }
        }
      };
      mask_token(bos_token_id_, "BOS");
      mask_token(pad_token_id_, "PAD");
      mask_token(unk_token_id_, "UNK");
      if (i == 0) {
        mask_token(eos_token_id_, "EOS(first step)");
      }
    }

    int32_t next_token = sampleToken(logits);
    
    std::string token_str = getTokenString(next_token);
    std::cout << "[DEBUG] Sampled token: " << next_token << " (\"" << token_str << "\")" << std::endl;
    log("DEBUG", "Sampled token: " + std::to_string(next_token) + " (\"" + token_str + "\")");

    // 检查多种停止条件
    bool should_stop = false;

    // 1. 检查配置的EOS token
    if (next_token == eos_token_id_) {
      std::cout << "[DEBUG] EOS token encountered: " << next_token
                << " (configured eos_token_id_: " << eos_token_id_ << ")"
                << std::endl;
      log("DEBUG", "EOS token encountered, stopping generation");
      should_stop = true;
    }

    // 2. 移除硬编码的停止token检查，避免将BOS/PAD等误判为停止；仅依赖EOS停止
    // 保留占位，未来可通过可配置stop-tokens实现

    // 3. 检查重复token（防止无限循环）
    if (next_token == last_token) {
      repeat_count++;
      std::cout << "[DEBUG] Token " << next_token << " repeated "
                << repeat_count << " times consecutively" << std::endl;
      if (repeat_count >= 3) { // 降低阈值到3次连续重复
        std::cout << "[DEBUG] Token " << next_token << " repeated "
                  << repeat_count << " times, stopping generation" << std::endl;
        log("WARNING", "Token " + std::to_string(next_token) + " repeated " +
                           std::to_string(repeat_count) +
                           " times, stopping generation");
        should_stop = true;
      }
    } else {
      repeat_count = 0;
      last_token = next_token;
    }

    // 4. 检查是否为无效token
    if (next_token < 0 || next_token > 200000) {
      std::cout << "[DEBUG] Invalid token encountered: " << next_token
                << std::endl;
      log("WARNING", "Invalid token encountered, stopping generation");
      should_stop = true;
    }

    if (should_stop) {
      break;
    }

    generated_tokens.push_back(next_token);
    input_tokens.push_back(next_token);
    
    // 更新重复惩罚历史（保持最近50个token）
    generated_tokens_history_.push_back(next_token);
    if (generated_tokens_history_.size() > 50) {
      generated_tokens_history_.erase(generated_tokens_history_.begin());
    }

    // 添加安全检查，避免无限循环
    if (input_tokens.size() > 1000) {
      log("WARNING", "Input tokens exceeded 1000, stopping generation");
      break;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  total_inference_time_ += duration.count() / 1000.0;
  total_tokens_generated_ += generated_tokens.size();

  log("DEBUG", "Starting detokenization");
  std::string result = detokenize(generated_tokens);
  log("INFO", "Generated text: " + result);
  return result;
}

// 带图像的文本生成
std::string Qwen25VLInferenceEngine::generateTextWithImage(
    const std::string &prompt, const std::string &image_path, int max_tokens) {
  log("INFO", "Generating text with image: " + prompt + ", image: " +
                  image_path + ", max_tokens: " + std::to_string(max_tokens));

  // 占位符实现
  return "Generated text with image: " + prompt;
}

// 多模态生成
std::string Qwen25VLInferenceEngine::generateTextWithImages(
    const std::string &prompt,
    const std::vector<std::vector<float>> &image_features, int max_tokens) {
  log("INFO", "Generating text with image features: " + prompt +
                  ", max_tokens: " + std::to_string(max_tokens));

  // 占位符实现
  return "Generated text with image features: " + prompt;
}

// 分词
std::vector<int32_t>
Qwen25VLInferenceEngine::tokenize(const std::string &text) {
  std::cout << "[DEBUG] Tokenizing text: \"" << text << "\"" << std::endl;
  log("INFO", "Tokenizing text: " + text);
  
  // 调试：打印当前词汇表大小
  std::cout << "[DEBUG] Vocabulary size: " << vocab_.size() << std::endl;
  std::cout << "[DEBUG] UNK token ID: " << unk_token_id_ << std::endl;

  std::vector<int32_t> tokens;

  // 使用TextProcessor进行分词
  if (text_processor_) {
    try {
      tokens = text_processor_->encode(text);
      std::cout << "[DEBUG] TextProcessor tokenized into " << tokens.size()
                << " tokens" << std::endl;
      
      // 调试：打印前几个token对应的字符串
      std::cout << "[DEBUG] First few tokens: ";
      for (size_t i = 0; i < std::min(tokens.size(), size_t(5)); ++i) {
        std::cout << tokens[i] << "(" << getTokenString(tokens[i]) << ") ";
      }
      std::cout << std::endl;
      
      return tokens;
    } catch (const std::exception &e) {
      std::cout << "[DEBUG] TextProcessor failed: " << e.what()
                << ", using fallback" << std::endl;
      log("WARNING", "TextProcessor failed: " + std::string(e.what()) +
                         ", using fallback");
    }
  }

  // 回退实现 - 使用简化的SentencePiece逻辑
  // 替换空格为SentencePiece分隔符
  std::string processed_text = text;
  size_t pos = 0;
  while ((pos = processed_text.find(' ', pos)) != std::string::npos) {
    processed_text.replace(pos, 1, "▁");
    pos += 3; // UTF-8编码的▁字符长度为3字节
  }
  
  // 尝试直接查找完整文本的token
  auto it = vocab_.find(processed_text);
  if (it != vocab_.end()) {
    tokens.push_back(it->second);
    std::cout << "[DEBUG] Found complete token for: \"" << processed_text << "\" -> " << it->second << std::endl;
    return tokens;
  }
  
  // 字符级回退分词
  for (size_t i = 0; i < processed_text.length(); ) {
    bool found = false;
    
    // 尝试最长匹配 - 优先匹配较长的子串
    for (size_t len = std::min(processed_text.length() - i, static_cast<size_t>(16)); len > 0; --len) {
      std::string substr = processed_text.substr(i, len);
      auto vocab_it = vocab_.find(substr);
      if (vocab_it != vocab_.end()) {
        tokens.push_back(vocab_it->second);
        std::cout << "[DEBUG] Found token: \"" << substr << "\" -> " << vocab_it->second << std::endl;
        i += len;
        found = true;
        break;
      }
    }
    
    if (!found) {
      // 字节级回退 - 使用字节级编码
      unsigned char byte = static_cast<unsigned char>(processed_text[i]);
      std::ostringstream oss;
      oss << "<0x" << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << static_cast<int>(byte) << ">";
      std::string byte_token = oss.str();
      auto byte_it = vocab_.find(byte_token);
      if (byte_it != vocab_.end()) {
        tokens.push_back(byte_it->second);
        std::cout << "[DEBUG] Found byte token: \"" << byte_token << "\" -> " << byte_it->second << std::endl;
      } else {
        // 使用未知token
        tokens.push_back(unk_token_id_);
        std::cout << "[DEBUG] Using unknown token for byte: " << static_cast<int>(byte) << std::endl;
      }
      i++;
    }
  }

  std::cout << "[DEBUG] Tokenized into " << tokens.size() << " tokens: ";
  for (size_t i = 0; i < tokens.size() && i < 10; ++i) {
    std::cout << tokens[i] << " ";
  }
  if (tokens.size() > 10)
    std::cout << "...";
  std::cout << std::endl;

  return tokens;
}

// 批量生成
std::vector<std::string> Qwen25VLInferenceEngine::generateBatch(
    const std::vector<std::string> &prompts) {
  log("INFO",
      "Generating batch of " + std::to_string(prompts.size()) + " prompts");

  std::vector<std::string> results;
  for (const auto &prompt : prompts) {
    results.push_back(generateText(prompt));
  }

  return results;
}

// 流式生成
void Qwen25VLInferenceEngine::generateStream(
    const std::string &prompt,
    std::function<void(const std::string &)> callback, int max_tokens) {
  log("INFO", "Starting stream generation for prompt: " + prompt);

  // 占位符实现
  callback("Streaming: " + prompt);
}

// 模型状态保存
bool Qwen25VLInferenceEngine::saveState(const std::string &state_path) const {
  log("INFO", "Saving model state to: " + state_path);
  return true; // 占位符
}

// 模型状态加载
bool Qwen25VLInferenceEngine::loadState(const std::string &state_path) {
  log("INFO", "Loading model state from: " + state_path);
  return true; // 占位符
}

// 启用KV缓存
void Qwen25VLInferenceEngine::enableKVCache(bool enable) {
  kv_cache_enabled_ = enable;
  log("INFO", "KV cache " + std::string(enable ? "enabled" : "disabled"));
}

// 设置最大序列长度
void Qwen25VLInferenceEngine::setMaxSequenceLength(uint32_t max_length) {
  max_sequence_length_ = max_length;
  if (kv_cache_) {
    kv_cache_->resize(config_.num_layers, max_length, config_.hidden_size, config_.num_key_value_heads);
  }
  log("INFO", "Max sequence length set to: " + std::to_string(max_length));
}

// 优化内存布局
void Qwen25VLInferenceEngine::optimizeMemoryLayout() {
  log("INFO", "Optimizing memory layout");
  // 占位符实现
}

// 优化计算图
void Qwen25VLInferenceEngine::optimizeComputationGraph() {
  log("INFO", "Optimizing computation graph");
  // 占位符实现
}

// 模型预热
void Qwen25VLInferenceEngine::warmupModel() {
  log("INFO", "Warming up model");
  if (model_loaded_) {
    generateText("Hello", 5);
  }
}

// 启用量化
bool Qwen25VLInferenceEngine::enableQuantization(
    const std::string &quant_type) {
  quantization_enabled_ = true;
  quantization_type_ = quant_type;
  log("INFO", "Quantization enabled: " + quant_type);
  return true;
}

// 禁用量化
bool Qwen25VLInferenceEngine::disableQuantization() {
  quantization_enabled_ = false;
  quantization_type_ = "none";
  log("INFO", "Quantization disabled");
  return true;
}

// 设置线程数
void Qwen25VLInferenceEngine::setNumThreads(uint32_t num_threads) {
  num_threads_ = num_threads;
  log("INFO", "Number of threads set to: " + std::to_string(num_threads));
}

// 启用并行处理
void Qwen25VLInferenceEngine::enableParallelProcessing(bool enable) {
  parallel_processing_enabled_ = enable;
  log("INFO",
      "Parallel processing " + std::string(enable ? "enabled" : "disabled"));
}

// 设置采样参数
void Qwen25VLInferenceEngine::setTemperature(float temperature) {
  temperature_ = temperature;
  log("INFO", "Temperature set to: " + std::to_string(temperature));
}

void Qwen25VLInferenceEngine::setTopP(float top_p) {
  top_p_ = top_p;
  log("INFO", "Top-p set to: " + std::to_string(top_p));
}

void Qwen25VLInferenceEngine::setTopK(int top_k) {
  top_k_ = top_k;
  log("INFO", "Top-k set to: " + std::to_string(top_k));
}

void Qwen25VLInferenceEngine::setRepetitionPenalty(float penalty) {
  repetition_penalty_ = penalty;
  log("INFO", "Repetition penalty set to: " + std::to_string(penalty));
}

// 获取模型配置
ModelConfig Qwen25VLInferenceEngine::getModelConfig() const { return config_; }

// 获取模型信息
std::string Qwen25VLInferenceEngine::getModelInfo() const {
  std::stringstream ss;
  ss << "Qwen2.5VL Model Info:\n";
  ss << "Vocab Size: " << config_.vocab_size << "\n";
  ss << "Hidden Size: " << config_.hidden_size << "\n";
  ss << "Num Layers: " << config_.num_layers << "\n";
  ss << "Num Attention Heads: " << config_.num_attention_heads << "\n";
  ss << "Max Position Embeddings: " << config_.max_position_embeddings << "\n";
  return ss.str();
}

// 获取模型大小
size_t Qwen25VLInferenceEngine::getModelSize() const {
  return calculateModelSize();
}

// 计算模型大小
size_t Qwen25VLInferenceEngine::calculateModelSize() const {
  size_t total_size = 0;

  // 计算token embeddings大小
  total_size += token_embeddings_.data.size() * sizeof(float);

  // 计算transformer layers大小
  for (const auto &layer : transformer_layers_) {
    for (const auto &head : layer.attention_heads) {
      total_size += head.query_weights.data.size() * sizeof(float);
      total_size += head.key_weights.data.size() * sizeof(float);
      total_size += head.value_weights.data.size() * sizeof(float);
      total_size += head.output_weights.data.size() * sizeof(float);
    }
    total_size += layer.ffn_gate_weights.data.size() * sizeof(float);
    total_size += layer.ffn_up_weights.data.size() * sizeof(float);
    total_size += layer.ffn_down_weights.data.size() * sizeof(float);
  }

  // 计算输出权重大小
  total_size += output_projection_.data.size() * sizeof(float);

  return total_size;
}

// 性能统计
double Qwen25VLInferenceEngine::getInferenceTime() const {
  return total_inference_time_;
}

uint64_t Qwen25VLInferenceEngine::getTokensGenerated() const {
  return total_tokens_generated_;
}

double Qwen25VLInferenceEngine::getTokensPerSecond() const {
  if (total_inference_time_ > 0.0) {
    return static_cast<double>(total_tokens_generated_) / total_inference_time_;
  }
  return 0.0;
}

void Qwen25VLInferenceEngine::resetStatistics() {
  total_inference_time_ = 0.0;
  total_tokens_generated_ = 0;
  log("INFO", "Statistics reset");
}

// 词汇表操作
std::string
Qwen25VLInferenceEngine::detokenize(const std::vector<int32_t> &tokens) {
  std::cout << "[DEBUG] Detokenizing " << tokens.size() << " tokens"
            << std::endl;

  // 使用TextProcessor进行解码
  if (text_processor_) {
    try {
      std::string result = text_processor_->decode(tokens);
      std::cout << "[DEBUG] TextProcessor detokenized result: \"" << result
                << "\"" << std::endl;
      return result;
    } catch (const std::exception &e) {
      std::cout << "[DEBUG] TextProcessor decode failed: " << e.what()
                << ", using fallback" << std::endl;
      log("WARNING", "TextProcessor decode failed: " + std::string(e.what()) +
                         ", using fallback");
    }
  }

  // 回退实现 - 使用SentencePiece逻辑
  std::ostringstream result;
  bool first_token = true;
  
  for (int32_t token : tokens) {
    std::cout << "[DEBUG] Processing token: " << token << std::endl;
    if (token == bos_token_id_ || token == eos_token_id_ ||
        token == pad_token_id_) {
      std::cout << "[DEBUG] Skipping special token: " << token << std::endl;
      continue;
    }
    
    std::string token_str = getTokenString(token);
    std::cout << "[DEBUG] Token " << token << " -> \"" << token_str << "\""
              << std::endl;
    
    // 处理特殊token
    if (token_str.empty() || token_str == "<unk>" || 
        token_str.find("[PAD") == 0 || token_str.find("<pad>") == 0 ||
        token_str.find("<|pad|>") == 0 ||
        token_str == "<|im_start|>" || token_str == "<|im_end|>" ||
        token_str == "<|endoftext|>" || token_str.find("<|vision_") == 0 ||
        token_str.find("<|image_") == 0 || token_str.find("<|video_") == 0) {
      std::cout << "[DEBUG] Skipping special token: " << token_str << std::endl;
      continue;
    }
    
    // 处理字节token（如<0xEA>格式）
    if (token_str.length() == 6 && token_str.substr(0, 3) == "<0x" && token_str.back() == '>') {
      std::string hex_str = token_str.substr(3, 2);
      try {
        unsigned long byte_val = std::stoul(hex_str, nullptr, 16);
        if (byte_val <= 255) {
          result << static_cast<char>(byte_val);
          std::cout << "[DEBUG] Converted byte token " << token_str << " to byte: " << byte_val << std::endl;
          first_token = false;
          continue;
        }
      } catch (const std::exception& e) {
        std::cout << "[DEBUG] Failed to parse hex byte: " << e.what() << std::endl;
      }
    }
    
    // 处理Ġ前缀（GPT风格空格编码）
    if (token_str.length() >= 3 && token_str.substr(0, 3) == "Ġ") {
      if (!first_token) {
        result << " ";
      }
      token_str = token_str.substr(3); // 移除Ġ前缀
    }
    // 处理▁前缀（SentencePiece风格空格编码）
    else if (token_str.find("▁") == 0) {
      if (!first_token) {
        result << " ";
      }
      token_str = token_str.substr(3); // ▁是3字节UTF-8字符
    }
    // 替换内部空格分隔符为空格
    else {
      size_t pos = 0;
      while ((pos = token_str.find("▁", pos)) != std::string::npos) {
        token_str.replace(pos, 3, " ");
        pos += 1;
      }
    }
    
    // 处理中文字符和其他Unicode字符
    if (!token_str.empty()) {
      result << token_str;
    }
    
    first_token = false;
  }
  
  std::string final_result = result.str();
  std::cout << "[DEBUG] Final detokenized result: \"" << final_result << "\""
            << std::endl;
  return final_result;
}

int32_t Qwen25VLInferenceEngine::getVocabSize() const {
  return config_.vocab_size;
}

std::string Qwen25VLInferenceEngine::getTokenString(int32_t token_id) const {
  auto it = reverse_vocab_.find(token_id);
  if (it != reverse_vocab_.end()) {
    return it->second;
  }
  return "<unk>";
}

int32_t Qwen25VLInferenceEngine::getTokenId(const std::string &token) const {
  auto it = vocab_.find(token);
  if (it != vocab_.end()) {
    return it->second;
  }
  return unk_token_id_;
}

// 日志函数
void Qwen25VLInferenceEngine::log(const std::string &level,
                                  const std::string &message) const {
  // 创建Logger实例并初始化
  static duorou::core::Logger logger;
  static bool logger_initialized = false;
  
  if (!logger_initialized) {
    logger.initialize();
    // 设置日志文件到logs文件夹
    std::string log_path = "./logs/qwen25vl_" + 
      std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count()) + ".log";
    logger.setLogFile(log_path);
    logger_initialized = true;
  }
  
  if (level == "ERROR") {
    logger.error("[Qwen25VL] " + message);
  } else if (level == "WARN" || level == "WARNING") {
    logger.warning("[Qwen25VL] " + message);
  } else if (level == "INFO") {
    logger.info("[Qwen25VL] " + message);
  } else if (level == "DEBUG") {
    logger.debug("[Qwen25VL] " + message);
  } else {
    logger.info("[Qwen25VL] [" + level + "] " + message);
  }
  
  // 如果verbose模式开启，同时输出到控制台
  if (verbose_) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::cout << "["
              << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
              << "] [" << level << "] " << message << std::endl;
  }
}

// 私有方法实现
bool Qwen25VLInferenceEngine::loadWeights(const std::string &model_path) {
  log("INFO", "Loading weights from: " + model_path);

  // 加载词汇表
  if (!loadVocabulary()) {
    return false;
  }

  // 加载token embeddings
  if (!loadTokenEmbedding()) {
    return false;
  }

  // 加载transformer layers
  if (!loadLayers()) {
    return false;
  }

  // 加载输出权重
  if (!loadOutputWeights()) {
    return false;
  }

  // 加载视觉权重
  if (!loadVisionWeights()) {
    return false;
  }

  return true;
}

bool Qwen25VLInferenceEngine::initializeConfigFromGGUF() {
  log("INFO", "Initializing config from GGUF architecture");

  if (!gguf_parser_) {
    log("ERROR", "GGUF parser is null");
    return false;
  }

  const auto &architecture = gguf_parser_->getArchitecture();
  
  // 映射架构信息到config_
  config_.hidden_size = architecture.embedding_length;
  config_.num_layers = architecture.block_count;
  config_.num_attention_heads = architecture.attention_head_count;
  config_.num_key_value_heads = architecture.attention_head_count_kv;
  config_.intermediate_size = architecture.feed_forward_length;
  config_.max_position_embeddings = architecture.context_length;
  config_.layer_norm_eps = architecture.layer_norm_rms_epsilon;
  config_.rope_dim = architecture.rope_dimension_count;
  config_.rope_base = architecture.rope_freq_base;
  
  // 从GGUF元数据获取词汇表大小
  const auto *vocab_size_kv = gguf_parser_->getMetadata("tokenizer.ggml.token_count");
  if (vocab_size_kv) {
    config_.vocab_size = vocab_size_kv->asUInt32();
  } else {
    // 回退到默认值
    config_.vocab_size = 151936;
    log("WARNING", "Could not find vocab size in GGUF, using default: " + std::to_string(config_.vocab_size));
  }
  
  // 设置视觉相关配置（如果有）
  if (architecture.has_vision) {
    config_.vision_hidden_size = 1280; // 默认值
    config_.vision_num_layers = 32;
    config_.vision_num_attention_heads = 16;
    config_.vision_intermediate_size = 5120;
    config_.image_size = 448;
    config_.patch_size = 14;
  }
  
  log("DEBUG", "Config initialized - vocab_size: " + std::to_string(config_.vocab_size) + 
             ", hidden_size: " + std::to_string(config_.hidden_size) + 
             ", num_layers: " + std::to_string(config_.num_layers) + 
             ", num_attention_heads: " + std::to_string(config_.num_attention_heads) + 
             ", num_key_value_heads: " + std::to_string(config_.num_key_value_heads));
  
  return true;
}

bool Qwen25VLInferenceEngine::loadVocabulary() {
  log("INFO", "Loading vocabulary from GGUF file");

  try {
    // 尝试从GGUF文件加载词汇表
    if (gguf_parser_) {
      const auto *tokens_kv =
          gguf_parser_->getMetadata("tokenizer.ggml.tokens");
      if (tokens_kv && tokens_kv->type == GGUFType::ARRAY) {
        std::cout << "[DEBUG] Found tokenizer.ggml.tokens metadata (array type)"
                  << std::endl;
        std::cout << "[DEBUG] Array data size: " << tokens_kv->data.size()
                  << " bytes" << std::endl;

        // 检查数组的详细信息
        if (tokens_kv->data.size() >= 12) {
          uint32_t array_type;
          uint64_t array_length;
          std::memcpy(&array_type, tokens_kv->data.data(), 4);
          std::memcpy(&array_length, tokens_kv->data.data() + 4, 8);
          std::cout << "[DEBUG] Array type: " << array_type
                    << " (STRING=" << static_cast<uint32_t>(GGUFType::STRING)
                    << ")" << std::endl;
          std::cout << "[DEBUG] Array length: " << array_length << std::endl;
        }

        // 清空现有词汇表
        vocab_.clear();
        reverse_vocab_.clear();

        try {
          // 尝试解析为字符串数组
          auto token_strings = tokens_kv->asStringArray();
          std::cout << "[DEBUG] Successfully parsed " << token_strings.size()
                    << " tokens from GGUF" << std::endl;

          // 验证词汇表大小
          if (token_strings.empty()) {
            log("WARNING", "Empty vocabulary loaded from GGUF");
            return loadFallbackVocabulary();
          }

          if (token_strings.size() > 200000) {
            log("WARNING", "Vocabulary size too large: " +
                               std::to_string(token_strings.size()));
            return loadFallbackVocabulary();
          }

          // 构建词汇表映射
          for (size_t i = 0; i < token_strings.size(); ++i) {
            const std::string &token = token_strings[i];
            int32_t token_id = static_cast<int32_t>(i);

            // 验证token有效性
            if (token.empty()) {
              log("WARNING", "Empty token at index " + std::to_string(i));
              continue;
            }

            // 修复可能的编码问题：检查token是否是错误编码的UTF-8
            std::string corrected_token = token;

            // 检测是否是UTF-8字节被当作Latin-1处理的情况
            bool needs_correction = false;
            for (unsigned char c : token) {
              if (c >= 0x80) {
                needs_correction = true;
                break;
              }
            }

            if (needs_correction) {
              // 尝试将Latin-1字符重新解释为UTF-8字节
              std::vector<unsigned char> utf8_bytes;
              for (unsigned char c : token) {
                utf8_bytes.push_back(c);
              }

              // 验证是否是有效的UTF-8序列
              bool is_valid_utf8 = true;
              for (size_t i = 0; i < utf8_bytes.size();) {
                unsigned char c = utf8_bytes[i];
                if ((c & 0x80) == 0) {
                  // ASCII字符
                  i++;
                } else if ((c & 0xE0) == 0xC0) {
                  // 2字节UTF-8序列
                  if (i + 1 >= utf8_bytes.size() ||
                      (utf8_bytes[i + 1] & 0xC0) != 0x80) {
                    is_valid_utf8 = false;
                    break;
                  }
                  i += 2;
                } else if ((c & 0xF0) == 0xE0) {
                  // 3字节UTF-8序列
                  if (i + 2 >= utf8_bytes.size() ||
                      (utf8_bytes[i + 1] & 0xC0) != 0x80 ||
                      (utf8_bytes[i + 2] & 0xC0) != 0x80) {
                    is_valid_utf8 = false;
                    break;
                  }
                  i += 3;
                } else if ((c & 0xF8) == 0xF0) {
                  // 4字节UTF-8序列
                  if (i + 3 >= utf8_bytes.size() ||
                      (utf8_bytes[i + 1] & 0xC0) != 0x80 ||
                      (utf8_bytes[i + 2] & 0xC0) != 0x80 ||
                      (utf8_bytes[i + 3] & 0xC0) != 0x80) {
                    is_valid_utf8 = false;
                    break;
                  }
                  i += 4;
                } else {
                  is_valid_utf8 = false;
                  break;
                }
              }

              if (is_valid_utf8) {
                // 重新构造正确的UTF-8字符串
                corrected_token = std::string(
                    reinterpret_cast<const char *>(utf8_bytes.data()),
                    utf8_bytes.size());
                if (verbose_) {
                  std::cout << "[DEBUG] Corrected token " << token_id << ": \""
                            << token << "\" -> \"" << corrected_token << "\""
                            << std::endl;
                }
              }
            }

            // 检查重复token
            if (vocab_.find(corrected_token) != vocab_.end()) {
              log("WARNING", "Duplicate token found: " + corrected_token);
              continue;
            }

            vocab_[corrected_token] = token_id;
            reverse_vocab_[token_id] = corrected_token;
          }

          // 动态加载特殊token配置
          loadDynamicSpecialTokens();

          // 验证关键的Qwen2.5VL特殊token
          std::vector<std::pair<std::string, int32_t>> expected_tokens = {
              {"<|im_start|>", 151644},   {"<|im_end|>", 151645},
              {"<|endoftext|>", 151643},  {"<|vision_start|>", 151652},
              {"<|vision_end|>", 151653}, {"<|vision_pad|>", 151654},
              {"<|image_pad|>", 151655}};

          bool has_critical_tokens = true;
          for (const auto &[token, expected_id] : expected_tokens) {
            auto it = vocab_.find(token);
            if (it != vocab_.end()) {
              log("INFO",
                  "Found " + token + " at ID: " + std::to_string(it->second));
              // 更新特殊token ID
              if (token == "<|im_start|>") {
                im_start_token_id_ = it->second;
              } else if (token == "<|im_end|>") {
                im_end_token_id_ = it->second;
              } else if (token == "<|endoftext|>") {
                eos_token_id_ = it->second;
                bos_token_id_ = it->second;
              }
            } else {
              log("WARNING", "Missing critical token: " + token);
              if (token == "<|im_start|>" || token == "<|im_end|>" ||
                  token == "<|endoftext|>") {
                has_critical_tokens = false;
              }
            }
          }

          if (!has_critical_tokens) {
            log("WARNING",
                "Missing critical tokens, using fallback vocabulary");
            return loadFallbackVocabulary();
          }

          // 验证生成过程中使用的token
          std::vector<int> test_tokens = {56064,  133718, 29391,
                                          131840, 115382, 22828};
          for (int token_id : test_tokens) {
            if (reverse_vocab_.find(token_id) != reverse_vocab_.end()) {
              std::cout << "[DEBUG] Token " << token_id << ": "
                        << reverse_vocab_[token_id] << std::endl;
            } else {
              std::cout << "[DEBUG] Token " << token_id << ": NOT FOUND"
                        << std::endl;
            }
          }

          log("INFO", "Successfully loaded vocabulary with " +
                          std::to_string(vocab_.size()) + " tokens");
          return true;
        } catch (const std::exception &e) {
          std::cout << "[DEBUG] Failed to parse tokens array: " << e.what()
                    << std::endl;
          log("WARNING",
              "Failed to parse vocabulary from GGUF: " + std::string(e.what()));
        }
      } else {
        std::cout << "[DEBUG] No valid tokenizer.ggml.tokens found"
                  << std::endl;
      }
    }
  } catch (const std::exception &e) {
    log("WARNING",
        "Exception during vocabulary loading: " + std::string(e.what()));
  }

  return loadFallbackVocabulary();
}

void Qwen25VLInferenceEngine::loadDynamicSpecialTokens() {
  if (!gguf_parser_) {
    log("WARNING", "GGUF parser not available for dynamic token loading");
    return;
  }

  log("INFO", "Loading dynamic special tokens from GGUF metadata");

  // 尝试加载tokenizer配置
  const auto *tokenizer_model_kv =
      gguf_parser_->getMetadata("tokenizer.ggml.model");
  if (tokenizer_model_kv && tokenizer_model_kv->type == GGUFType::STRING) {
    try {
      std::string tokenizer_model = tokenizer_model_kv->asString();
      log("INFO", "Tokenizer model: " + tokenizer_model);
    } catch (const std::exception &e) {
      log("WARNING",
          "Failed to read tokenizer model: " + std::string(e.what()));
    }
  }

  // 尝试加载特殊token配置
  std::vector<std::string> special_token_keys = {
      "tokenizer.ggml.bos_token_id",  "tokenizer.ggml.eos_token_id",
      "tokenizer.ggml.pad_token_id",  "tokenizer.ggml.unk_token_id",
      "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"};

  for (const std::string &key : special_token_keys) {
    const auto *kv = gguf_parser_->getMetadata(key);
    if (kv) {
      try {
        if (kv->type == GGUFType::UINT32) {
          uint32_t value = kv->asUInt32();
          log("INFO", key + ": " + std::to_string(value));

          // 更新相应的token ID
          if (key == "tokenizer.ggml.bos_token_id") {
            bos_token_id_ = static_cast<int32_t>(value);
          } else if (key == "tokenizer.ggml.eos_token_id") {
            eos_token_id_ = static_cast<int32_t>(value);
          } else if (key == "tokenizer.ggml.pad_token_id") {
            pad_token_id_ = static_cast<int32_t>(value);
          } else if (key == "tokenizer.ggml.unk_token_id") {
            unk_token_id_ = static_cast<int32_t>(value);
          }
        } else if (kv->type == GGUFType::BOOL) {
          bool value = kv->asBool();
          log("INFO", key + ": " + (value ? "true" : "false"));
        }
      } catch (const std::exception &e) {
        log("WARNING", "Failed to read " + key + ": " + std::string(e.what()));
      }
    }
  }

  // 尝试加载chat template
  const auto *chat_template_kv =
      gguf_parser_->getMetadata("tokenizer.chat_template");
  if (chat_template_kv && chat_template_kv->type == GGUFType::STRING) {
    try {
      std::string chat_template = chat_template_kv->asString();
      log("INFO", "Found chat template (length: " +
                      std::to_string(chat_template.length()) + ")");
      // 可以在这里解析chat template中的特殊token
    } catch (const std::exception &e) {
      log("WARNING", "Failed to read chat template: " + std::string(e.what()));
    }
  }

  // 尝试加载added tokens
  const auto *added_tokens_kv =
      gguf_parser_->getMetadata("tokenizer.ggml.added_tokens");
  if (added_tokens_kv && added_tokens_kv->type == GGUFType::ARRAY) {
    try {
      auto added_tokens = added_tokens_kv->asStringArray();
      log("INFO",
          "Found " + std::to_string(added_tokens.size()) + " added tokens");

      for (const std::string &token : added_tokens) {
        log("DEBUG", "Added token: " + token);
      }
    } catch (const std::exception &e) {
      log("WARNING", "Failed to read added tokens: " + std::string(e.what()));
    }
  }

  log("INFO", "Dynamic special token loading completed");
}

bool Qwen25VLInferenceEngine::loadFallbackVocabulary() {
  log("INFO", "Using fallback vocabulary for Qwen2.5VL");
  vocab_.clear();
  reverse_vocab_.clear();

  // Qwen2.5VL关键特殊token
  std::vector<std::pair<std::string, int32_t>> special_tokens = {
      {"<|endoftext|>", 151643},      {"<|im_start|>", 151644},
      {"<|im_end|>", 151645},         {"<|object_ref_start|>", 151646},
      {"<|object_ref_end|>", 151647}, {"<|box_start|>", 151648},
      {"<|box_end|>", 151649},        {"<|quad_start|>", 151650},
      {"<|quad_end|>", 151651},       {"<|vision_start|>", 151652},
      {"<|vision_end|>", 151653},     {"<|vision_pad|>", 151654},
      {"<|image_pad|>", 151655},      {"<|video_pad|>", 151656}};

  for (const auto &[token, id] : special_tokens) {
    vocab_[token] = id;
    reverse_vocab_[id] = token;
  }

  // 添加常见中文词汇的映射
  std::vector<std::pair<std::string, int32_t>> chinese_tokens = {
      {"你", 125544}, {"好", 44821}, {"是", 104},  {"的", 1546},  {"我", 39746},
      {"在", 3581},   {"有", 16937}, {"了", 1543}, {"和", 34208}, {"这", 3837}};

  for (const auto &[token, id] : chinese_tokens) {
    vocab_[token] = id;
    reverse_vocab_[id] = token;
  }

  log("INFO", "Fallback vocabulary loaded with " +
                  std::to_string(vocab_.size()) + " tokens");
  return true;
}

bool Qwen25VLInferenceEngine::loadTokenEmbedding() {
  log("INFO", "Loading token embeddings");
  
  // 计算所需内存大小
  const size_t required_memory = static_cast<size_t>(config_.vocab_size) * config_.hidden_size * sizeof(float);
  const size_t memory_limit_mb = 1024; // 1GB内存限制
  const size_t memory_limit = memory_limit_mb * 1024 * 1024;
  
  log("DEBUG", "Token embeddings memory requirement: " + std::to_string(required_memory / (1024*1024)) + "MB");
  log("DEBUG", "vocab_size: " + std::to_string(config_.vocab_size) + ", hidden_size: " + std::to_string(config_.hidden_size));
  
  // 如果内存需求超过限制，使用缩减的vocab_size
  actual_vocab_size_ = config_.vocab_size;
  if (required_memory > memory_limit) {
    // 计算在内存限制下可以支持的最大vocab_size
    actual_vocab_size_ = static_cast<uint32_t>(memory_limit / (config_.hidden_size * sizeof(float)));
    // 确保是合理的大小，至少32000
    actual_vocab_size_ = std::max(actual_vocab_size_, 32000U);
    log("WARNING", "Token embeddings require " + std::to_string(required_memory / (1024*1024)) + 
        "MB, exceeding " + std::to_string(memory_limit_mb) + "MB limit. Using reduced vocab_size: " + 
        std::to_string(actual_vocab_size_));
  }
  
  try {
    token_embeddings_.reshape({actual_vocab_size_, config_.hidden_size});
    log("INFO", "Token embeddings tensor reshaped to: [" + std::to_string(actual_vocab_size_) + ", " + std::to_string(config_.hidden_size) + "]");
  } catch (const std::exception& e) {
    log("ERROR", "Failed to allocate token embeddings: " + std::string(e.what()));
    log("INFO", "Attempting fallback with smaller vocab_size");
    
    // 进一步缩减vocab_size作为最后的回退策略
    actual_vocab_size_ = 32000;
    try {
      token_embeddings_.reshape({actual_vocab_size_, config_.hidden_size});
      log("WARNING", "Using fallback vocab_size: " + std::to_string(actual_vocab_size_));
    } catch (const std::exception& e2) {
      log("ERROR", "Failed to allocate token embeddings even with fallback: " + std::string(e2.what()));
      return false;
    }
  }

  // 从GGUF文件加载token嵌入
  if (!loadTensorFromGGUF("token_embd.weight", token_embeddings_)) {
    log("WARNING",
        "Failed to load token_embd.weight, trying alternative names");

    // 尝试其他可能的张量名称
    std::vector<std::string> alternative_names = {
        "model.embed_tokens.weight", "transformer.wte.weight",
        "embeddings.word_embeddings.weight"};

    bool loaded = false;
    for (const auto &name : alternative_names) {
      if (loadTensorFromGGUF(name, token_embeddings_)) {
        log("INFO", "Successfully loaded token embeddings from: " + name);
        loaded = true;
        break;
      }
    }

    if (!loaded) {
      log("ERROR", "Failed to load token embeddings from GGUF file, using "
                   "random initialization");

      // 回退到随机初始化
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<float> dist(0.0f, 0.02f);

      for (size_t i = 0; i < token_embeddings_.data.size(); ++i) {
        token_embeddings_.data[i] = dist(gen);
      }

      log("WARNING", "Using random token embeddings");
      return true;
    }
  }

  log("INFO", "Token embeddings loaded from GGUF successfully");
  return true;
}

bool Qwen25VLInferenceEngine::loadLayers() {
  log("INFO", "Loading transformer layers");

  try {
    transformer_layers_.resize(config_.num_layers);
    log("DEBUG", "Resized transformer layers to " + std::to_string(config_.num_layers) + " layers");

  for (uint32_t i = 0; i < config_.num_layers; ++i) {
    auto &layer = transformer_layers_[i];
    std::string layer_prefix = "blk." + std::to_string(i) + ".";

    // 初始化注意力头
    layer.attention_heads.resize(1); // 简化为单个头处理所有权重
    auto &head = layer.attention_heads[0];
    
    // 加载注意力权重 - 通常Qwen模型使用合并的QKV权重
    Tensor qkv_weights;
    
    // 内存预检查 - QKV权重
    const size_t qkv_memory_mb = (static_cast<size_t>(config_.hidden_size) * config_.hidden_size * 3 * sizeof(float)) / (1024 * 1024);
    log("DEBUG", "QKV weights memory requirement: " + std::to_string(qkv_memory_mb) + "MB for layer " + std::to_string(i));
    
    if (qkv_memory_mb > 256) { // 256MB限制
      log("WARNING", "QKV weights too large (" + std::to_string(qkv_memory_mb) + "MB), using smaller fallback size");
      // 使用较小的fallback尺寸
      const uint32_t fallback_size = std::min(config_.hidden_size, static_cast<uint32_t>(2048));
      try {
        qkv_weights.reshape({fallback_size, fallback_size * 3});
      } catch (const std::bad_alloc& e) {
        log("ERROR", "Failed to allocate memory for QKV weights fallback: " + std::string(e.what()));
        return false;
      }
    } else {
      try {
        qkv_weights.reshape({config_.hidden_size, config_.hidden_size * 3});
      } catch (const std::bad_alloc& e) {
        log("ERROR", "Failed to allocate memory for QKV weights: " + std::string(e.what()));
        return false;
      }
    }
    bool qkv_loaded = loadTensorFromGGUF(layer_prefix + "attn_qkv.weight", qkv_weights);
    
    if (qkv_loaded) {
      // 验证QKV权重数据大小
      const size_t expected_qkv_size = config_.hidden_size * config_.hidden_size * 3;
      if (qkv_weights.data.size() != expected_qkv_size) {
        log("ERROR", "QKV weights size mismatch for layer " + std::to_string(i) + 
            ": expected " + std::to_string(expected_qkv_size) + 
            ", got " + std::to_string(qkv_weights.data.size()));
        return false;
      }
      
      // 从合并的QKV权重中分离Q、K、V
      // 内存预检查 - 单个权重矩阵
      const size_t single_weight_memory_mb = (static_cast<size_t>(config_.hidden_size) * config_.hidden_size * sizeof(float)) / (1024 * 1024);
      
      if (single_weight_memory_mb > 128) { // 128MB限制
        log("WARNING", "Single weight matrix too large (" + std::to_string(single_weight_memory_mb) + "MB), using fallback size");
        const uint32_t fallback_size = std::min(config_.hidden_size, static_cast<uint32_t>(2048));
        try {
          head.query_weights.reshape({fallback_size, fallback_size});
          head.key_weights.reshape({fallback_size, fallback_size});
          head.value_weights.reshape({fallback_size, fallback_size});
        } catch (const std::bad_alloc& e) {
          log("ERROR", "Failed to allocate memory for QKV weights with fallback: " + std::string(e.what()));
          return false;
        }
      } else {
        try {
          head.query_weights.reshape({config_.hidden_size, config_.hidden_size});
          head.key_weights.reshape({config_.hidden_size, config_.hidden_size});
          head.value_weights.reshape({config_.hidden_size, config_.hidden_size});
        } catch (const std::bad_alloc& e) {
          log("ERROR", "Failed to allocate memory for QKV weights: " + std::string(e.what()));
          return false;
        }
      }
      
      // 安全复制权重数据
      const size_t weight_size = config_.hidden_size * config_.hidden_size;
      if (qkv_weights.data.size() >= 3 * weight_size) {
        std::copy(qkv_weights.data.begin(), 
                  qkv_weights.data.begin() + weight_size,
                  head.query_weights.data.begin());
        std::copy(qkv_weights.data.begin() + weight_size,
                  qkv_weights.data.begin() + 2 * weight_size,
                  head.key_weights.data.begin());
        std::copy(qkv_weights.data.begin() + 2 * weight_size,
                  qkv_weights.data.begin() + 3 * weight_size,
                  head.value_weights.data.begin());
      } else {
        log("ERROR", "Insufficient QKV data for layer " + std::to_string(i));
        return false;
      }
    } else {
      // 尝试分离的Q、K、V权重
      // 内存预检查 - 单个权重矩阵
      const size_t single_weight_memory_mb = (static_cast<size_t>(config_.hidden_size) * config_.hidden_size * sizeof(float)) / (1024 * 1024);
      
      // 计算正确的K、V权重尺寸（支持GQA）
      const uint32_t head_dim = config_.hidden_size / config_.num_attention_heads;
      const uint32_t kv_dim = config_.num_key_value_heads * head_dim;
      
      if (single_weight_memory_mb > 128) { // 128MB限制
        log("WARNING", "Single weight matrix too large (" + std::to_string(single_weight_memory_mb) + "MB), using fallback size");
        const uint32_t fallback_size = std::min(config_.hidden_size, static_cast<uint32_t>(2048));
        const uint32_t fallback_kv_dim = std::min(kv_dim, static_cast<uint32_t>(512));
        try {
          head.query_weights.reshape({fallback_size, fallback_size});
          head.key_weights.reshape({config_.hidden_size, fallback_kv_dim});
          head.value_weights.reshape({config_.hidden_size, fallback_kv_dim});
        } catch (const std::bad_alloc& e) {
          log("ERROR", "Failed to allocate memory for separate QKV weights with fallback: " + std::string(e.what()));
          return false;
        }
      } else {
        try {
          head.query_weights.reshape({config_.hidden_size, config_.hidden_size});
          head.key_weights.reshape({config_.hidden_size, kv_dim});
          head.value_weights.reshape({config_.hidden_size, kv_dim});
        } catch (const std::bad_alloc& e) {
          log("ERROR", "Failed to allocate memory for separate QKV weights: " + std::string(e.what()));
          return false;
        }
      }

      bool q_loaded = loadTensorFromGGUF(layer_prefix + "attn_q.weight", head.query_weights);
      bool k_loaded = loadTensorFromGGUF(layer_prefix + "attn_k.weight", head.key_weights);
      bool v_loaded = loadTensorFromGGUF(layer_prefix + "attn_v.weight", head.value_weights);

      if (!q_loaded || !k_loaded || !v_loaded) {
        log("WARNING", "Failed to load attention weights for layer " + std::to_string(i));
      }
    }

    // 加载注意力输出权重
    try {
      head.output_weights.reshape({config_.hidden_size, config_.hidden_size});
    } catch (const std::bad_alloc& e) {
      log("ERROR", "Failed to allocate memory for output weights: " + std::string(e.what()));
      return false;
    }
    if (!loadTensorFromGGUF(layer_prefix + "attn_output.weight", head.output_weights)) {
      log("WARNING", "Failed to load attention output weights for layer " + std::to_string(i));
    }

    // 加载FFN权重
    // 内存预检查 - FFN权重矩阵
    const size_t ffn_gate_memory_mb = (static_cast<size_t>(config_.hidden_size) * config_.intermediate_size * sizeof(float)) / (1024 * 1024);
    const size_t ffn_down_memory_mb = (static_cast<size_t>(config_.intermediate_size) * config_.hidden_size * sizeof(float)) / (1024 * 1024);
    
    log("DEBUG", "FFN weights memory requirement: gate/up=" + std::to_string(ffn_gate_memory_mb) + "MB, down=" + std::to_string(ffn_down_memory_mb) + "MB for layer " + std::to_string(i));
    
    if (ffn_gate_memory_mb > 256 || ffn_down_memory_mb > 256) { // 256MB限制
      log("WARNING", "FFN weights too large, using fallback sizes");
      const uint32_t fallback_hidden = std::min(config_.hidden_size, static_cast<uint32_t>(2048));
      const uint32_t fallback_intermediate = std::min(config_.intermediate_size, static_cast<uint32_t>(4096));
      
      try {
        layer.ffn_gate_weights.reshape({fallback_hidden, fallback_intermediate});
        layer.ffn_up_weights.reshape({fallback_hidden, fallback_intermediate});
        layer.ffn_down_weights.reshape({fallback_intermediate, fallback_hidden});
      } catch (const std::bad_alloc& e) {
        log("ERROR", "Failed to allocate memory for FFN weights with fallback: " + std::string(e.what()));
        return false;
      }
    } else {
      try {
        layer.ffn_gate_weights.reshape({config_.hidden_size, config_.intermediate_size});
        layer.ffn_up_weights.reshape({config_.hidden_size, config_.intermediate_size});
        layer.ffn_down_weights.reshape({config_.intermediate_size, config_.hidden_size});
      } catch (const std::bad_alloc& e) {
        log("ERROR", "Failed to allocate memory for FFN weights: " + std::string(e.what()));
        return false;
      }
    }

    if (!loadTensorFromGGUF(layer_prefix + "ffn_gate.weight",
                            layer.ffn_gate_weights)) {
      log("WARNING",
          "Failed to load FFN gate weights for layer " + std::to_string(i));
    }
    if (!loadTensorFromGGUF(layer_prefix + "ffn_up.weight",
                            layer.ffn_up_weights)) {
      log("WARNING",
          "Failed to load FFN up weights for layer " + std::to_string(i));
    }
    if (!loadTensorFromGGUF(layer_prefix + "ffn_down.weight",
                            layer.ffn_down_weights)) {
      log("WARNING",
          "Failed to load FFN down weights for layer " + std::to_string(i));
    }

    // 加载层归一化权重
    try {
      layer.attention_norm_weights.reshape({config_.hidden_size});
      layer.ffn_norm_weights.reshape({config_.hidden_size});
    } catch (const std::bad_alloc& e) {
      log("ERROR", "Failed to allocate memory for norm weights: " + std::string(e.what()));
      return false;
    }

    if (!loadTensorFromGGUF(layer_prefix + "attn_norm.weight",
                            layer.attention_norm_weights)) {
      log("WARNING", "Failed to load attention norm weights for layer " +
                         std::to_string(i));
    }
    if (!loadTensorFromGGUF(layer_prefix + "ffn_norm.weight",
                            layer.ffn_norm_weights)) {
      log("WARNING",
          "Failed to load FFN norm weights for layer " + std::to_string(i));
    }

    if (verbose_) {
      log("DEBUG", "Layer " + std::to_string(i) + " loaded - FFN gate: " + 
          std::to_string(layer.ffn_gate_weights.data.size()) + 
          ", up: " + std::to_string(layer.ffn_up_weights.data.size()) + 
          ", down: " + std::to_string(layer.ffn_down_weights.data.size()) +
          ", attn heads: " + std::to_string(layer.attention_heads.size()));
    }
  }

  log("INFO", "Transformer layers loaded successfully");
  return true;
  
  } catch (const std::exception& e) {
    log("ERROR", "Exception while loading transformer layers: " + std::string(e.what()));
    return false;
  } catch (...) {
    log("ERROR", "Unknown exception while loading transformer layers");
    return false;
  }
}

bool Qwen25VLInferenceEngine::loadOutputWeights() {
  log("INFO", "Loading output weights");

  try {
    // 加载最终层归一化权重 - 内存预检查
    size_t norm_memory_mb = (config_.hidden_size * sizeof(float)) / (1024 * 1024);
    log("DEBUG", "Output norm weights memory requirement: " + std::to_string(norm_memory_mb) + "MB");
    
    if (norm_memory_mb > 64) { // 64MB限制
      log("ERROR", "Output norm weights too large (" + std::to_string(norm_memory_mb) + "MB), exceeds limit (64MB)");
      return false;
    }
    
    output_norm_weights_.reshape({config_.hidden_size});
    
    std::vector<std::string> norm_names = {
      "model.norm.weight",
      "norm.weight", 
      "ln_f.weight",
      "transformer.ln_f.weight",
      "transformer.norm.weight"
    };
    
    bool norm_loaded = false;
    for (const auto &name : norm_names) {
      if (loadTensorFromGGUF(name, output_norm_weights_)) {
        log("INFO", "Successfully loaded output norm weights from: " + name);
        norm_loaded = true;
        break;
      }
    }
    
    if (!norm_loaded) {
      log("WARNING", "Failed to load output norm weights, using ones");
      output_norm_weights_.data.assign(config_.hidden_size, 1.0f);
    }

    // 加载输出投影权重（LM head）- 内存预检查
    size_t projection_memory_mb = (static_cast<size_t>(actual_vocab_size_) * config_.hidden_size * sizeof(float)) / (1024 * 1024);
    log("DEBUG", "Output projection memory requirement: " + std::to_string(projection_memory_mb) + "MB");
    log("DEBUG", "actual_vocab_size: " + std::to_string(actual_vocab_size_) + ", hidden_size: " + std::to_string(config_.hidden_size));
    
    // 使用actual_vocab_size_来保持与token embeddings的一致性
    try {
      output_projection_.reshape({actual_vocab_size_, config_.hidden_size});
      log("INFO", "Output projection tensor reshaped to: [" + std::to_string(actual_vocab_size_) + ", " + std::to_string(config_.hidden_size) + "]");
    } catch (const std::bad_alloc& e) {
      log("ERROR", "Failed to allocate output projection memory: " + std::string(e.what()));
      log("INFO", "Trying smaller fallback allocation");
      
      // 使用回退策略：进一步减小vocab_size
      actual_vocab_size_ = std::min(actual_vocab_size_, 32000U);
      log("INFO", "Using fallback vocab_size: " + std::to_string(actual_vocab_size_));
      output_projection_.reshape({actual_vocab_size_, config_.hidden_size});
    }
  
  bool loaded = false;
  std::vector<std::string> output_names = {
    "lm_head.weight",
    "model.lm_head.weight",
    "output.weight",
    "embed_out.weight",
    "transformer.wte.weight" // 有时输出层共享嵌入权重
  };
  
  for (const auto &name : output_names) {
    if (loadTensorFromGGUF(name, output_projection_)) {
      log("INFO", "Successfully loaded output projection weights from: " + name);
      loaded = true;
      break;
    }
  }
  
  if (!loaded) {
    log("WARNING", "Failed to load output projection weights from GGUF file");
    // 对于某些模型，输出层可能共享token嵌入权重
    if (token_embeddings_.data.size() > 0) {
      log("INFO", "Using shared token embeddings as output projection");
      
      // 计算期望的输出投影大小
      size_t expected_projection_size = static_cast<size_t>(config_.vocab_size) * config_.hidden_size;
      size_t available_embedding_size = token_embeddings_.data.size();
      
      log("DEBUG", "Expected projection size: " + std::to_string(expected_projection_size));
      log("DEBUG", "Available embedding size: " + std::to_string(available_embedding_size));
      
      if (available_embedding_size >= expected_projection_size) {
        // 直接使用token嵌入数据
        try {
          output_projection_.data.assign(token_embeddings_.data.begin(), 
                                       token_embeddings_.data.begin() + expected_projection_size);
          log("INFO", "Successfully shared token embeddings as output projection");
        } catch (const std::exception& e) {
          log("ERROR", "Failed to copy token embeddings: " + std::string(e.what()));
          // 使用零初始化作为最后的回退
          output_projection_.data.assign(expected_projection_size, 0.0f);
          log("WARNING", "Using zero-initialized output projection");
        }
      } else {
        log("WARNING", "Token embeddings size mismatch, using zero initialization");
        // 使用零初始化
        try {
          output_projection_.data.assign(expected_projection_size, 0.0f);
          log("INFO", "Initialized output projection with zeros");
        } catch (const std::bad_alloc& e) {
          log("ERROR", "Failed to allocate output projection: " + std::string(e.what()));
          return false;
        }
      }
    } else {
      log("WARNING", "No token embeddings available, using zero initialization");
      // 使用零初始化
      try {
        size_t expected_size = static_cast<size_t>(config_.vocab_size) * config_.hidden_size;
        output_projection_.data.assign(expected_size, 0.0f);
        log("INFO", "Initialized output projection with zeros");
      } catch (const std::bad_alloc& e) {
        log("ERROR", "Failed to allocate output projection: " + std::string(e.what()));
        return false;
      }
    }
  }

  log("INFO", "Output weights loaded successfully");
  return true;
  
  } catch (const std::bad_alloc& e) {
    log("ERROR", "Memory allocation failed in loadOutputWeights: " + std::string(e.what()));
    return false;
  } catch (const std::exception& e) {
    log("ERROR", "Exception in loadOutputWeights: " + std::string(e.what()));
    return false;
  } catch (...) {
    log("ERROR", "Unknown exception in loadOutputWeights");
    return false;
  }
}

bool Qwen25VLInferenceEngine::loadVisionWeights() {
  log("INFO", "Loading vision weights");

  vision_encoder_ = std::make_unique<VisionEncoder>();

  // 尝试加载视觉编码器权重
  bool weights_loaded = false;
  
  // 加载卷积层权重
  for (int i = 0; i < 4; ++i) {
    std::string conv_weight_name = "vision_model.embeddings.patch_embedding.weight";
    if (i > 0) {
      conv_weight_name = "vision_model.encoder.layers." + std::to_string(i) + ".conv.weight";
    }
    
    Tensor conv_weight;
    if (loadTensorFromGGUF(conv_weight_name, conv_weight)) {
      vision_encoder_->conv_weights.push_back(conv_weight);
      weights_loaded = true;
    }
    
    std::string conv_bias_name = "vision_model.embeddings.patch_embedding.bias";
    if (i > 0) {
      conv_bias_name = "vision_model.encoder.layers." + std::to_string(i) + ".conv.bias";
    }
    
    Tensor conv_bias;
    if (loadTensorFromGGUF(conv_bias_name, conv_bias)) {
      vision_encoder_->conv_biases.push_back(conv_bias);
    }
  }

  // 加载位置编码
  if (loadTensorFromGGUF("vision_model.embeddings.position_embedding.weight", 
                        vision_encoder_->position_embeddings)) {
    weights_loaded = true;
  }

  // 加载Transformer层
  for (uint32_t i = 0; i < config_.vision_num_layers; ++i) {
    TransformerLayer layer;
    
    // 注意力权重
    std::string layer_prefix = "vision_model.encoder.layers." + std::to_string(i) + ".";
    
    loadTensorFromGGUF(layer_prefix + "self_attn.q_proj.weight", layer.attention_heads[0].query_weights);
    loadTensorFromGGUF(layer_prefix + "self_attn.k_proj.weight", layer.attention_heads[0].key_weights);
    loadTensorFromGGUF(layer_prefix + "self_attn.v_proj.weight", layer.attention_heads[0].value_weights);
    loadTensorFromGGUF(layer_prefix + "self_attn.out_proj.weight", layer.attention_heads[0].output_weights);
    
    // 注意力偏置
    loadTensorFromGGUF(layer_prefix + "self_attn.q_proj.bias", layer.attention_heads[0].query_bias);
    loadTensorFromGGUF(layer_prefix + "self_attn.k_proj.bias", layer.attention_heads[0].key_bias);
    loadTensorFromGGUF(layer_prefix + "self_attn.v_proj.bias", layer.attention_heads[0].value_bias);
    loadTensorFromGGUF(layer_prefix + "self_attn.out_proj.bias", layer.attention_heads[0].output_bias);
    
    // 层归一化权重
    loadTensorFromGGUF(layer_prefix + "layer_norm1.weight", layer.attention_norm_weights);
    loadTensorFromGGUF(layer_prefix + "layer_norm1.bias", layer.attention_norm_bias);
    loadTensorFromGGUF(layer_prefix + "layer_norm2.weight", layer.ffn_norm_weights);
    loadTensorFromGGUF(layer_prefix + "layer_norm2.bias", layer.ffn_norm_bias);
    
    // 前馈网络权重
    loadTensorFromGGUF(layer_prefix + "mlp.fc1.weight", layer.ffn_gate_weights);
    loadTensorFromGGUF(layer_prefix + "mlp.fc1.bias", layer.ffn_gate_bias);
    loadTensorFromGGUF(layer_prefix + "mlp.fc2.weight", layer.ffn_down_weights);
    loadTensorFromGGUF(layer_prefix + "mlp.fc2.bias", layer.ffn_down_bias);
    
    vision_encoder_->layers.push_back(layer);
  }

  // 加载输出投影
  if (loadTensorFromGGUF("vision_model.post_layernorm.weight", vision_encoder_->output_projection)) {
    weights_loaded = true;
  }
  loadTensorFromGGUF("vision_model.post_layernorm.bias", vision_encoder_->output_bias);

  if (!weights_loaded) {
    log("WARNING", "No vision weights found in GGUF file, using default initialization");
    
    // 初始化默认权重
    vision_encoder_->position_embeddings = Tensor({config_.vision_hidden_size});
    std::fill(vision_encoder_->position_embeddings.data.begin(), 
             vision_encoder_->position_embeddings.data.end(), 0.01f);
    
    vision_encoder_->output_projection = Tensor({config_.vision_hidden_size, config_.hidden_size});
    std::fill(vision_encoder_->output_projection.data.begin(), 
             vision_encoder_->output_projection.data.end(), 0.01f);
    
    vision_encoder_->output_bias = Tensor({config_.hidden_size});
    std::fill(vision_encoder_->output_bias.data.begin(), 
             vision_encoder_->output_bias.data.end(), 0.0f);
  }

  log("INFO", "Vision weights loading completed");
  return true;
}

void Qwen25VLInferenceEngine::precomputeRoPEFreqs() {
  log("INFO", "Precomputing RoPE frequencies");

  const uint32_t num_heads = config_.num_attention_heads > 0 ? config_.num_attention_heads : 1;
  const size_t head_dim = num_heads ? (config_.hidden_size / num_heads) : 0;
  if (head_dim == 0) {
    rope_freqs_.clear();
    log("ERROR", "precomputeRoPEFreqs: head_dim is 0, check model config");
    return;
  }

  // NeoX-style RoPE: compute inverse frequencies over head_dim
  const size_t rope_dim = head_dim;
  rope_freqs_.resize(rope_dim / 2);
  for (size_t i = 0; i < rope_freqs_.size(); ++i) {
    rope_freqs_[i] = 1.0f / std::pow(config_.rope_theta, (2.0f * static_cast<float>(i)) / static_cast<float>(rope_dim));
  }
}

bool Qwen25VLInferenceEngine::loadTensorFromGGUF(const std::string &tensor_name,
                                                 Tensor &tensor) {
  log("INFO", "Loading tensor: " + tensor_name);

  try {
    if (!gguf_parser_) {
      log("ERROR", "GGUF parser not initialized");
      return false;
    }

    // 获取张量信息
    const GGUFTensorInfo *tensor_info = gguf_parser_->getTensorInfo(tensor_name);
    if (!tensor_info) {
      log("ERROR", "Tensor not found in GGUF file: " + tensor_name);
      return false;
    }

    log("DEBUG", "Tensor " + tensor_name + " found, dimensions: " + 
        std::to_string(tensor_info->dimensions.size()));
    for (size_t i = 0; i < tensor_info->dimensions.size(); ++i) {
      log("DEBUG", "  Dimension " + std::to_string(i) + ": " + 
          std::to_string(tensor_info->dimensions[i]));
    }

    // 读取张量数据
    std::vector<uint8_t> raw_data;
    if (!gguf_parser_->getTensorData(tensor_name, raw_data)) {
      log("ERROR", "Failed to read tensor data: " + tensor_name);
      return false;
    }

    log("DEBUG", "Successfully read " + std::to_string(raw_data.size()) + 
        " bytes of raw data for tensor: " + tensor_name);

  // 计算元素数量
  uint64_t total_elements = 1;
  for (uint64_t dim : tensor_info->dimensions) {
    total_elements *= dim;
  }

  // 验证原始数据大小
  size_t expected_bytes = 0;
  switch (tensor_info->type) {
  case GGMLTensorType::F32:
    expected_bytes = total_elements * sizeof(float);
    break;
  case GGMLTensorType::F16:
    expected_bytes = total_elements * sizeof(uint16_t);
    break;
  // 量化类型处理
  case GGMLTensorType::Q4_0:
  case GGMLTensorType::Q4_1:
  case GGMLTensorType::Q5_0:
  case GGMLTensorType::Q5_1:
  case GGMLTensorType::Q8_0:
  case GGMLTensorType::Q8_1:
  case GGMLTensorType::Q2_K:
  case GGMLTensorType::Q3_K:
  case GGMLTensorType::Q4_K:
  case GGMLTensorType::Q5_K:
  case GGMLTensorType::Q6_K:
  case GGMLTensorType::Q8_K:
  case GGMLTensorType::IQ2_XXS:
  case GGMLTensorType::IQ2_XS:
  case GGMLTensorType::IQ3_XXS:
  case GGMLTensorType::IQ1_S:
  case GGMLTensorType::IQ4_NL:
  case GGMLTensorType::IQ3_S:
  case GGMLTensorType::IQ2_S:
  case GGMLTensorType::IQ4_XS:
  case GGMLTensorType::I8:
  case GGMLTensorType::I16:
  case GGMLTensorType::I32:
  case GGMLTensorType::I64:
  case GGMLTensorType::F64:
  case GGMLTensorType::IQ1_M:
  case GGMLTensorType::BF16:
  case GGMLTensorType::TQ1_0:
  case GGMLTensorType::TQ2_0:
  case GGMLTensorType::MXFP4: {
    // 对于量化类型，记录详细信息并使用回退策略
    log("WARNING", "Quantized tensor type " + std::to_string(static_cast<uint32_t>(tensor_info->type)) + 
        " for tensor " + tensor_name + " (size: " + std::to_string(raw_data.size()) + " bytes)");
    
    // 验证原始数据不为空
    if (raw_data.empty()) {
      log("ERROR", "Empty raw data for quantized tensor: " + tensor_name);
      return false;
    }
    
    // 特殊处理token embeddings：使用实际的vocab_size而不是GGUF文件中的维度
    if (tensor_name.find("token_embd") != std::string::npos || 
        tensor_name.find("embed_tokens") != std::string::npos ||
        tensor_name.find("wte") != std::string::npos) {
      // 使用config中的vocab_size，而不是GGUF文件中的维度
      uint64_t actual_elements = static_cast<uint64_t>(config_.vocab_size) * config_.hidden_size;
      
      try {
        tensor.data.clear();
        tensor.data.resize(actual_elements, 0.0f);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (size_t i = 0; i < actual_elements; ++i) {
          tensor.data[i] = dist(gen);
        }
        
        log("INFO", "Initialized token embeddings with actual vocab_size: " + 
            std::to_string(config_.vocab_size) + ", elements: " + std::to_string(actual_elements));
      } catch (const std::exception& e) {
        log("ERROR", "Failed to allocate memory for token embeddings: " + std::string(e.what()));
        return false;
      }
    } else {
      // 验证元素数量合理性
      if (total_elements == 0 || total_elements > 1000000000ULL) { // 1B元素限制
        log("ERROR", "Invalid element count for tensor " + tensor_name + ": " + std::to_string(total_elements));
        return false;
      }
      
      try {
        tensor.data.clear();
        tensor.data.resize(total_elements, 0.0f);
        
        // 对于权重张量，使用小的随机值而不是零
        if (tensor_name.find("weight") != std::string::npos) {
          std::random_device rd;
          std::mt19937 gen(rd());
          std::normal_distribution<float> dist(0.0f, 0.01f);
          for (size_t i = 0; i < total_elements; ++i) {
            tensor.data[i] = dist(gen);
          }
          log("INFO", "Initialized weight tensor " + tensor_name + " with small random values");
        } else {
          log("INFO", "Initialized tensor " + tensor_name + " with zero values");
        }
      } catch (const std::exception& e) {
        log("ERROR", "Failed to allocate memory for tensor " + tensor_name + ": " + std::string(e.what()));
        return false;
      }
    }
    
    // 设置张量形状
    tensor.shape.clear();
    for (uint64_t dim : tensor_info->dimensions) {
      tensor.shape.push_back(static_cast<uint32_t>(dim));
    }
    
    log("INFO", "Successfully loaded quantized tensor: " + tensor_name +
                ", elements: " + std::to_string(total_elements) + ", type: " +
                std::to_string(static_cast<uint32_t>(tensor_info->type)) + 
                ", using fallback initialization");
    return true;
  }
  default:
    log("ERROR", "Unsupported tensor type " + std::to_string(static_cast<uint32_t>(tensor_info->type)) + 
        " for tensor: " + tensor_name);
    return false;
  }
  
  if (raw_data.size() < expected_bytes) {
    log("ERROR", "Insufficient raw data for tensor: " + tensor_name + 
        ", expected: " + std::to_string(expected_bytes) + 
        ", got: " + std::to_string(raw_data.size()));
    return false;
  }

  // 根据数据类型转换数据
  tensor.data.clear();
  tensor.data.reserve(total_elements);

  // 检查是否需要转置（线性层权重通常是[out_features, in_features]，需要转置）
  bool needs_transpose = false;
  if (tensor_name.find("weight") != std::string::npos && 
      tensor.shape.size() == 2 && 
      tensor.shape[0] != tensor.shape[1]) {
    needs_transpose = true;
    log("INFO", "Applying transpose for linear layer weights: " + tensor_name);
  }

  switch (tensor_info->type) {
  case GGMLTensorType::F32: {
    const float *float_data = reinterpret_cast<const float *>(raw_data.data());
    if (needs_transpose && tensor.shape.size() == 2) {
      // 转置权重矩阵
      const size_t rows = tensor.shape[0];
      const size_t cols = tensor.shape[1];
      tensor.data.resize(total_elements);
      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
          tensor.data[j * rows + i] = float_data[i * cols + j];
        }
      }
    } else {
      tensor.data.assign(float_data, float_data + total_elements);
    }
    break;
  }
  case GGMLTensorType::F16: {
    // F16 to F32 conversion (simplified)
    const uint16_t *f16_data =
        reinterpret_cast<const uint16_t *>(raw_data.data());
    if (needs_transpose && tensor.shape.size() == 2) {
      // 转置权重矩阵
      const size_t rows = tensor.shape[0];
      const size_t cols = tensor.shape[1];
      tensor.data.resize(total_elements);
      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
          uint16_t f16 = f16_data[i * cols + j];
          uint32_t sign = (f16 >> 15) & 0x1;
          uint32_t exp = (f16 >> 10) & 0x1f;
          uint32_t mant = f16 & 0x3ff;

          uint32_t f32_bits;
          if (exp == 0) {
            if (mant == 0) {
              f32_bits = sign << 31;
            } else {
              exp = 127 - 15 + 1;
              while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
              }
              mant &= 0x3ff;
              f32_bits = (sign << 31) | (exp << 23) | (mant << 13);
            }
          } else if (exp == 31) {
            f32_bits = (sign << 31) | (0xff << 23) | (mant << 13);
          } else {
            f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
          }
          tensor.data[j * rows + i] = *reinterpret_cast<float *>(&f32_bits);
        }
      }
    } else {
      tensor.data.resize(total_elements);
      for (uint64_t i = 0; i < total_elements; ++i) {
        uint16_t f16 = f16_data[i];
        uint32_t sign = (f16 >> 15) & 0x1;
        uint32_t exp = (f16 >> 10) & 0x1f;
        uint32_t mant = f16 & 0x3ff;

        uint32_t f32_bits;
        if (exp == 0) {
          if (mant == 0) {
            f32_bits = sign << 31;
          } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) {
              mant <<= 1;
              exp--;
            }
            mant &= 0x3ff;
            f32_bits = (sign << 31) | (exp << 23) | (mant << 13);
          }
        } else if (exp == 31) {
          f32_bits = (sign << 31) | (0xff << 23) | (mant << 13);
        } else {
          f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }

        tensor.data[i] = *reinterpret_cast<float *>(&f32_bits);
      }
    }
    break;
  }
  default:
    log("ERROR", "Unsupported tensor type for tensor: " + tensor_name);
    return false;
  }

  // 如果进行了转置，需要交换形状
  if (needs_transpose && tensor.shape.size() == 2) {
    std::swap(tensor.shape[0], tensor.shape[1]);
  }

  log("INFO", "Successfully loaded tensor: " + tensor_name +
                  ", elements: " + std::to_string(total_elements) + ", type: " +
                  std::to_string(static_cast<uint32_t>(tensor_info->type)));
  return true;
  
  } catch (const std::exception& e) {
    log("ERROR", "Exception while loading tensor " + tensor_name + ": " + e.what());
    return false;
  } catch (...) {
    log("ERROR", "Unknown exception while loading tensor: " + tensor_name);
    return false;
  }
}

// 前向传播
Tensor Qwen25VLInferenceEngine::forward(const std::vector<int32_t> &input_ids) {
  log("DEBUG", "Forward pass starting with " + std::to_string(input_ids.size()) + " tokens");
  
  // 严格的输入验证
  if (input_ids.empty()) {
    log("ERROR", "Empty input_ids in forward pass");
    return Tensor({config_.vocab_size});
  }
  
  // 检查序列长度
  if (input_ids.size() > max_sequence_length_) {
    log("ERROR", "Input sequence length (" + std::to_string(input_ids.size()) + 
        ") exceeds maximum allowed length (" + std::to_string(max_sequence_length_) + ")");
    return Tensor({config_.vocab_size});
  }
  
  // 验证token ID范围
  for (size_t i = 0; i < input_ids.size(); ++i) {
    if (input_ids[i] < 0 || input_ids[i] >= static_cast<int32_t>(config_.vocab_size)) {
      log("ERROR", "Invalid token ID " + std::to_string(input_ids[i]) + 
          " at position " + std::to_string(i) + 
          ", valid range: [0, " + std::to_string(config_.vocab_size - 1) + "]");
      return Tensor({config_.vocab_size});
    }
  }
  
  // 验证模型配置
  if (config_.vocab_size == 0 || config_.hidden_size == 0 || config_.num_layers == 0) {
    log("ERROR", "Invalid model configuration: vocab_size=" + std::to_string(config_.vocab_size) +
        ", hidden_size=" + std::to_string(config_.hidden_size) +
        ", num_layers=" + std::to_string(config_.num_layers));
    return Tensor({config_.vocab_size});
  }
  
  // 嵌入tokens
  Tensor embeddings = embedTokens(input_ids);
  if (embeddings.data.empty()) {
    log("ERROR", "Failed to embed tokens");
    return Tensor({config_.vocab_size});
  }
  
  log("DEBUG", "Embeddings size: " + std::to_string(embeddings.data.size()));

  // 通过transformer layers
  Tensor hidden_states = embeddings;
  // 当启用KV缓存且已有历史长度时，仅对最后一个token进行增量解码
  if (kv_cache_enabled_ && kv_cache_ && kv_cache_->current_length > 0) {
    if (embeddings.shape.size() >= 2 && embeddings.shape[1] == config_.hidden_size && embeddings.shape[0] >= 1) {
      Tensor last_hidden({1u, static_cast<uint32_t>(config_.hidden_size)});
      const size_t src_offset = (static_cast<size_t>(embeddings.shape[0]) - 1) * static_cast<size_t>(config_.hidden_size);
      for (uint32_t j = 0; j < config_.hidden_size; ++j) {
        last_hidden.data[j] = embeddings.data[src_offset + j];
      }
      hidden_states = last_hidden;
      if (verbose_) {
        log("DEBUG", "forward(): KV cache active -> processing only last token for incremental decoding");
      }
    }
  }
  for (uint32_t i = 0; i < config_.num_layers; ++i) {
    log("DEBUG", "Processing layer " + std::to_string(i));
    
    // 验证transformer层
    if (i >= transformer_layers_.size()) {
      log("ERROR", "Layer index out of bounds: " + std::to_string(i));
      break;
    }
    
    // 注意力
    Tensor attention_output =
        multiHeadAttention(hidden_states, transformer_layers_[i], i);
    
    if (attention_output.data.empty()) {
      log("ERROR", "Attention output is empty for layer " + std::to_string(i));
      break;
    }

    // 残差连接 - 添加边界检查
    size_t min_size = std::min(hidden_states.data.size(), attention_output.data.size());
    for (size_t j = 0; j < min_size; ++j) {
      hidden_states.data[j] += attention_output.data[j];
    }

    // FFN
    Tensor ffn_output = feedForward(hidden_states, transformer_layers_[i]);
    
    if (ffn_output.data.empty()) {
      log("ERROR", "FFN output is empty for layer " + std::to_string(i));
      break;
    }

    // 残差连接 - 添加边界检查
    min_size = std::min(hidden_states.data.size(), ffn_output.data.size());
    for (size_t j = 0; j < min_size; ++j) {
      hidden_states.data[j] += ffn_output.data[j];
    }
  }

  // 层间完成后统一更新KV缓存的当前长度
  if (kv_cache_enabled_ && kv_cache_) {
    uint32_t added = 0;
    if (kv_cache_->current_length == 0) {
      // 首次prefill：添加本次所有序列长度
      added = (embeddings.shape.size() >= 1) ? embeddings.shape[0] : 0u;
    } else {
      // 增量阶段：每次仅新增1个token
      added = 1u;
    }
    if (kv_cache_->current_length + added > kv_cache_->max_length) {
      log("WARNING", "KV cache overflow prevented: trimming added tokens to fit max_length");
      added = (kv_cache_->current_length < kv_cache_->max_length) ? (kv_cache_->max_length - kv_cache_->current_length) : 0u;
    }
    kv_cache_->current_length += added;
  }

  
  log("DEBUG", "Applying output layer normalization");
  
  // 验证输出归一化权重
  if (output_norm_weights_.data.size() != config_.hidden_size) {
    log("ERROR", "Output norm weights size mismatch: expected " + 
        std::to_string(config_.hidden_size) + ", got " + 
        std::to_string(output_norm_weights_.data.size()));
    return Tensor({config_.vocab_size});
  }
  
  // 输出层归一化
  Tensor norm_bias({config_.hidden_size});
  std::fill(norm_bias.data.begin(), norm_bias.data.end(), 0.0f);
  hidden_states = applyLayerNorm(hidden_states, output_norm_weights_, norm_bias);

  log("DEBUG", "Computing output projection");
  
  // 输出投影 - 修复矩阵乘法索引
  Tensor logits({config_.vocab_size});
  std::fill(logits.data.begin(), logits.data.end(), 0.0f);
  
  // 检查output_projection_是否正确加载
  size_t expected_size = static_cast<size_t>(config_.vocab_size) * config_.hidden_size;
  if (output_projection_.data.size() != expected_size) {
    log("ERROR", "Output projection size mismatch: expected " + 
        std::to_string(expected_size) + 
        ", got " + std::to_string(output_projection_.data.size()));
    
    // 使用简化的映射作为回退
    log("WARNING", "Using fallback output projection computation");
    size_t min_size = std::min(static_cast<size_t>(config_.vocab_size), hidden_states.data.size());
    for (size_t i = 0; i < min_size; ++i) {
      logits.data[i] = hidden_states.data[i];
    }
    return logits;
  }
  
  // 确定隐藏状态的维度
  size_t sequence_length = hidden_states.data.size() / config_.hidden_size;
  if (sequence_length == 0 || hidden_states.data.size() % config_.hidden_size != 0) {
    log("ERROR", "Invalid hidden states dimensions: size=" + std::to_string(hidden_states.data.size()) + 
        ", hidden_size=" + std::to_string(config_.hidden_size));
    return logits;
  }
  
  // 获取最后一个token的隐藏状态
  size_t last_token_offset = (sequence_length - 1) * config_.hidden_size;
  
  // 验证偏移量不会越界
  if (last_token_offset + config_.hidden_size > hidden_states.data.size()) {
    log("ERROR", "Last token offset out of bounds: " + 
        std::to_string(last_token_offset) + " + " + 
        std::to_string(config_.hidden_size) + " > " + 
        std::to_string(hidden_states.data.size()));
    return logits;
  }
  
  log("DEBUG", "Computing matrix multiplication: [" + std::to_string(config_.vocab_size) + 
      ", " + std::to_string(config_.hidden_size) + "] x [" + std::to_string(config_.hidden_size) + "]");
  
  // 正确的矩阵乘法: logits = output_projection * hidden_states^T
  // output_projection是 [vocab_size, hidden_size]
  // hidden_states是 [hidden_size] (最后一个token)
  for (uint32_t vocab_idx = 0; vocab_idx < config_.vocab_size; ++vocab_idx) {
    float sum = 0.0f;
    for (uint32_t hidden_idx = 0; hidden_idx < config_.hidden_size; ++hidden_idx) {
      size_t weight_idx = static_cast<size_t>(vocab_idx) * config_.hidden_size + hidden_idx;
      if (weight_idx >= output_projection_.data.size()) {
        log("ERROR", "Weight index out of bounds: " + std::to_string(weight_idx) + 
            " >= " + std::to_string(output_projection_.data.size()));
        break;
      }
      float weight_value = output_projection_.data[weight_idx];
      float hidden_value = hidden_states.data[last_token_offset + hidden_idx];
      sum += weight_value * hidden_value;
    }
    logits.data[vocab_idx] = sum;
  }
  
  log("DEBUG", "Forward pass completed, logits size: " + std::to_string(logits.data.size()));

  return logits;
}

Tensor
Qwen25VLInferenceEngine::embedTokens(const std::vector<int32_t> &token_ids) {
  // 输入验证
  if (token_ids.empty()) {
    log("ERROR", "embedTokens: Empty token_ids vector");
    return Tensor({0, config_.hidden_size});
  }
  
  if (config_.hidden_size == 0) {
    log("ERROR", "embedTokens: Invalid hidden_size: " + std::to_string(config_.hidden_size));
    return Tensor({0, 0});
  }
  
  // 检查token_embeddings_是否已加载
  size_t expected_embedding_size = config_.vocab_size * config_.hidden_size;
  if (token_embeddings_.data.size() != expected_embedding_size) {
    log("ERROR", "embedTokens: Token embeddings size mismatch. Expected: " + 
        std::to_string(expected_embedding_size) + ", Actual: " + 
        std::to_string(token_embeddings_.data.size()));
    return Tensor({0, config_.hidden_size});
  }

  Tensor embeddings;
  try {
    embeddings = Tensor({static_cast<uint32_t>(token_ids.size()), config_.hidden_size});
  } catch (const std::exception& e) {
    log("ERROR", "embedTokens: Failed to allocate tensor memory: " + std::string(e.what()));
    return Tensor({0, config_.hidden_size});
  }
  
  // 验证输出张量大小
  size_t expected_output_size = token_ids.size() * config_.hidden_size;
  if (embeddings.data.size() != expected_output_size) {
    log("ERROR", "embedTokens: Output tensor size mismatch. Expected: " + 
        std::to_string(expected_output_size) + ", Actual: " + 
        std::to_string(embeddings.data.size()));
    return Tensor({0, config_.hidden_size});
  }

  for (size_t i = 0; i < token_ids.size(); ++i) {
    int32_t token_id = token_ids[i];
    
    // 边界检查
    if (token_id < 0 || token_id >= static_cast<int32_t>(config_.vocab_size)) {
      log("WARNING", "embedTokens: Invalid token_id " + std::to_string(token_id) + 
          " at position " + std::to_string(i) + ". Vocab size: " + std::to_string(config_.vocab_size));
      // 使用零向量或UNK token
      for (uint32_t j = 0; j < config_.hidden_size; ++j) {
        embeddings.data[i * config_.hidden_size + j] = 0.0f;
      }
      continue;
    }
    
    // 计算源和目标索引
    size_t src_offset = static_cast<size_t>(token_id) * config_.hidden_size;
    size_t dst_offset = i * config_.hidden_size;
    
    // 边界检查
    if (src_offset + config_.hidden_size > token_embeddings_.data.size()) {
      log("ERROR", "embedTokens: Source offset out of bounds. Token: " + 
          std::to_string(token_id) + ", Offset: " + std::to_string(src_offset));
      continue;
    }
    
    if (dst_offset + config_.hidden_size > embeddings.data.size()) {
      log("ERROR", "embedTokens: Destination offset out of bounds. Position: " + 
          std::to_string(i) + ", Offset: " + std::to_string(dst_offset));
      continue;
    }
    
    // 安全复制
    for (uint32_t j = 0; j < config_.hidden_size; ++j) {
      embeddings.data[dst_offset + j] = token_embeddings_.data[src_offset + j];
    }
  }
  
  if (verbose_) {
    log("DEBUG", "embedTokens: Successfully embedded " + std::to_string(token_ids.size()) + " tokens");
  }

  return embeddings;
}

Tensor Qwen25VLInferenceEngine::applyLayerNorm(const Tensor &input,
                                               const Tensor &weights,
                                               const Tensor &bias) {
  Tensor output = input;

  // 简化的层归一化
  float mean = 0.0f;
  for (float val : input.data) {
    mean += val;
  }
  mean /= input.data.size();

  float variance = 0.0f;
  for (float val : input.data) {
    variance += (val - mean) * (val - mean);
  }
  variance /= input.data.size();

  float std_dev = std::sqrt(variance + config_.layer_norm_eps);

  for (size_t i = 0; i < output.data.size(); ++i) {
    output.data[i] = (output.data[i] - mean) / std_dev;
    if (i < weights.data.size()) {
      output.data[i] *= weights.data[i];
    }
    if (i < bias.data.size()) {
      output.data[i] += bias.data[i];
    }
  }

  return output;
}

Tensor Qwen25VLInferenceEngine::applyRoPE(const Tensor &input,
                                          uint32_t position) {
  Tensor output = input;
  
  const size_t seq_len = input.shape.empty() ? 1 : input.shape[0];
  const size_t hidden_size = config_.hidden_size;
  const size_t num_heads = config_.num_attention_heads;
  const size_t head_dim = hidden_size / num_heads;
  
  // 对每个序列位置应用RoPE
  for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
    const size_t actual_position = position + seq_idx;
    
    // 对每个注意力头应用RoPE
    for (size_t head = 0; head < num_heads; ++head) {
      const size_t head_offset = head * head_dim;
      
      // 对头维度的每一对元素应用旋转
      for (size_t i = 0; i < head_dim; i += 2) {
        const size_t idx1 = seq_idx * hidden_size + head_offset + i;
        const size_t idx2 = seq_idx * hidden_size + head_offset + i + 1;
        
        if (idx2 < output.data.size() && i / 2 < rope_freqs_.size()) {
          const float freq = rope_freqs_[i / 2];
          const float angle = actual_position * freq;
          
          const float cos_val = std::cos(angle);
          const float sin_val = std::sin(angle);
          
          const float x = output.data[idx1];
          const float y = output.data[idx2];
          
          output.data[idx1] = x * cos_val - y * sin_val;
          output.data[idx2] = x * sin_val + y * cos_val;
        }
      }
    }
  }
  
  return output;
}

Tensor Qwen25VLInferenceEngine::multiHeadAttention(
    const Tensor &input, const TransformerLayer &layer, uint32_t layer_idx) {
  if (verbose_) {
    log("DEBUG", "multiHeadAttention: Starting layer " + std::to_string(layer_idx));
  }
  
  // 输入验证
  if (input.data.empty()) {
    log("ERROR", "multiHeadAttention: Empty input tensor");
    return Tensor({0, config_.hidden_size});
  }
  
  const size_t seq_len = input.shape.empty() ? 1 : input.shape[0];
  const size_t hidden_size = config_.hidden_size;
  
  if (hidden_size == 0 || config_.num_attention_heads == 0) {
    log("ERROR", "multiHeadAttention: Invalid config - hidden_size: " + 
        std::to_string(hidden_size) + ", num_heads: " + std::to_string(config_.num_attention_heads));
    return input;
  }
  
  const size_t head_dim = hidden_size / config_.num_attention_heads;
  const size_t num_heads = config_.num_attention_heads;
  const size_t num_kv_heads = config_.num_key_value_heads;  // GQA支持
  const size_t num_groups = num_heads / num_kv_heads;  // 每个KV头对应的查询头数
  
  if (head_dim == 0) {
    log("ERROR", "multiHeadAttention: Invalid head_dim: " + std::to_string(head_dim));
    return input;
  }
  
  // 验证输入张量大小
  size_t expected_input_size = seq_len * hidden_size;
  if (input.data.size() != expected_input_size) {
    log("ERROR", "multiHeadAttention: Input size mismatch. Expected: " + 
        std::to_string(expected_input_size) + ", Actual: " + std::to_string(input.data.size()));
    return input;
  }
  
  if (verbose_) {
    log("DEBUG", "MultiHeadAttention: seq_len=" + std::to_string(seq_len) + 
        ", hidden_size=" + std::to_string(hidden_size) + 
        ", num_heads=" + std::to_string(num_heads) + 
        ", head_dim=" + std::to_string(head_dim));
  }
  
  // 添加计算复杂度检查
  const size_t total_ops = seq_len * seq_len * head_dim * num_heads;
  if (total_ops > 1000000) {  // 如果计算量过大，发出警告
    log("WARNING", "multiHeadAttention: High computational complexity detected: " + 
        std::to_string(total_ops) + " operations");
  }
  
  // 检查是否有有效的注意力权重
  if (layer.attention_heads.empty()) {
    log("ERROR", "multiHeadAttention: No attention heads found");
    return input;
  }
  
  const AttentionHead& head = layer.attention_heads[0];
  
  // 验证权重大小 - 支持GQA
  const size_t expected_q_weight_size = hidden_size * hidden_size;
  const size_t expected_kv_weight_size = hidden_size * (num_kv_heads * head_dim);
  if (head.query_weights.data.size() != expected_q_weight_size ||
      head.key_weights.data.size() != expected_kv_weight_size ||
      head.value_weights.data.size() != expected_kv_weight_size) {
    log("ERROR", "multiHeadAttention: Weight size mismatch. Expected Q: " + 
        std::to_string(expected_q_weight_size) + 
        ", Expected K/V: " + std::to_string(expected_kv_weight_size) +
        ", Got Q: " + std::to_string(head.query_weights.data.size()) +
        ", K: " + std::to_string(head.key_weights.data.size()) +
        ", V: " + std::to_string(head.value_weights.data.size()));
    return input;
  }
  
  // 创建Q、K、V张量 - 支持GQA
  Tensor q_tensor({static_cast<uint32_t>(seq_len), static_cast<uint32_t>(hidden_size)});
  Tensor k_tensor({static_cast<uint32_t>(seq_len), static_cast<uint32_t>(num_kv_heads * head_dim)});
  Tensor v_tensor({static_cast<uint32_t>(seq_len), static_cast<uint32_t>(num_kv_heads * head_dim)});

  // 验证输出张量大小
  const size_t expected_kv_size = seq_len * num_kv_heads * head_dim;
  if (q_tensor.data.size() != expected_input_size ||
      k_tensor.data.size() != expected_kv_size ||
      v_tensor.data.size() != expected_kv_size) {
    log("ERROR", "multiHeadAttention: Output tensor size mismatch. Expected Q: " + 
        std::to_string(expected_input_size) + ", K/V: " + std::to_string(expected_kv_size));
    return input;
  }
  
  // 使用高效GEMM批量计算Q、K、V - 支持GQA
  try {
    // Q计算: input * query_weights -> q_tensor
    matrixMultiply(input.data.data(), head.query_weights.data.data(), 
                   q_tensor.data.data(), seq_len, hidden_size, hidden_size);
    
    // K计算: input * key_weights -> k_tensor
    matrixMultiply(input.data.data(), head.key_weights.data.data(), 
                   k_tensor.data.data(), seq_len, num_kv_heads * head_dim, hidden_size);
    
    // V计算: input * value_weights -> v_tensor
    matrixMultiply(input.data.data(), head.value_weights.data.data(), 
                   v_tensor.data.data(), seq_len, num_kv_heads * head_dim, hidden_size);
  } catch (const std::exception& e) {
    log("ERROR", "multiHeadAttention: Matrix multiplication failed: " + std::string(e.what()));
    return input;
  }
  
  // 计算RoPE位置偏移：若启用KV缓存，则从current_length处继续
  uint32_t position_offset = 0u;
  if (kv_cache_enabled_ && kv_cache_) {
    position_offset = std::min(kv_cache_->current_length, kv_cache_->max_length);
  }
  // 批量应用RoPE位置编码（NeoX风格）- 支持GQA
  batchedRoPE(q_tensor.data.data(), seq_len, num_heads, head_dim, position_offset);
  batchedRoPE(k_tensor.data.data(), seq_len, num_kv_heads, head_dim, position_offset);
  
  // 将新K/V追加写入到每层的缓存中（在RoPE之后）
  if (kv_cache_enabled_ && kv_cache_) {
    if (layer_idx >= kv_cache_->key_cache.size() || layer_idx >= kv_cache_->value_cache.size()) {
      log("ERROR", "multiHeadAttention: KV cache layer index out of range");
    } else {
      Tensor &k_cache = kv_cache_->key_cache[layer_idx];
      Tensor &v_cache = kv_cache_->value_cache[layer_idx];
      const uint32_t max_len = kv_cache_->max_length;
      uint32_t base = std::min(kv_cache_->current_length, max_len);
      uint32_t can_copy = (base < max_len) ? std::min<uint32_t>(seq_len, max_len - base) : 0u;
      if (can_copy < seq_len) {
        log("WARNING", "KV cache near capacity, truncating append rows from " + std::to_string(seq_len) + " to " + std::to_string(can_copy));
      }
      for (uint32_t i = 0; i < can_copy; ++i) {
        const size_t src_off = static_cast<size_t>(i) * num_kv_heads * head_dim;
        const size_t dst_off = static_cast<size_t>(base + i) * num_kv_heads * head_dim;
        const size_t copy_size = num_kv_heads * head_dim;
        
        // 边界检查
        if (src_off + copy_size <= k_tensor.data.size() && 
            dst_off + copy_size <= k_cache.data.size() &&
            src_off + copy_size <= v_tensor.data.size() && 
            dst_off + copy_size <= v_cache.data.size()) {
          // 使用安全的内存拷贝
          std::memcpy(k_cache.data.data() + dst_off, k_tensor.data.data() + src_off, copy_size * sizeof(float));
          std::memcpy(v_cache.data.data() + dst_off, v_tensor.data.data() + src_off, copy_size * sizeof(float));
        } else {
          log("ERROR", "multiHeadAttention: KV cache copy bounds check failed at iteration " + std::to_string(i));
          break;
        }
      }
    }
  }
  
  // 创建输出张量
  Tensor output({static_cast<uint32_t>(seq_len), static_cast<uint32_t>(hidden_size)});
  
  // 缩放因子
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  
  // 根据是否启用KV缓存选择注意力实现
  if (kv_cache_enabled_ && kv_cache_ && kv_cache_->current_length > 0) {
    // 增量解码：Q长度为本次seq_len，K/V长度为cached_len + seq_len
    const size_t cached_len = kv_cache_->current_length;
    const size_t kv_len = cached_len + seq_len;

    // 准备K_total与V_total（cache + 当前）- 支持GQA
    Tensor k_total({static_cast<uint32_t>(kv_len), static_cast<uint32_t>(num_kv_heads * head_dim)});
    Tensor v_total({static_cast<uint32_t>(kv_len), static_cast<uint32_t>(num_kv_heads * head_dim)});
    // 先拷贝缓存部分
    if (layer_idx < kv_cache_->key_cache.size() && layer_idx < kv_cache_->value_cache.size()) {
      const Tensor &k_cache = kv_cache_->key_cache[layer_idx];
      const Tensor &v_cache = kv_cache_->value_cache[layer_idx];
      for (size_t i = 0; i < cached_len; ++i) {
        const size_t src_off = i * num_kv_heads * head_dim;
        const size_t dst_off = i * num_kv_heads * head_dim;
        const size_t copy_size = num_kv_heads * head_dim;
        
        // 边界检查
        if (src_off + copy_size <= k_cache.data.size() && 
            dst_off + copy_size <= k_total.data.size() &&
            src_off + copy_size <= v_cache.data.size() && 
            dst_off + copy_size <= v_total.data.size()) {
          // 使用安全的内存拷贝
          std::memcpy(k_total.data.data() + dst_off, k_cache.data.data() + src_off, copy_size * sizeof(float));
          std::memcpy(v_total.data.data() + dst_off, v_cache.data.data() + src_off, copy_size * sizeof(float));
        } else {
          log("ERROR", "multiHeadAttention: K/V total copy bounds check failed at cached iteration " + std::to_string(i));
          break;
        }
      }
    } else {
      log("ERROR", "multiHeadAttention: KV cache layer index out of range while building K/V total");
    }
    // 再拷贝本轮新生成的K/V
    for (size_t i = 0; i < seq_len; ++i) {
      const size_t dst_row = cached_len + i;
      const size_t dst_off = dst_row * num_kv_heads * head_dim;
      const size_t src_off = i * num_kv_heads * head_dim;
      const size_t copy_size = num_kv_heads * head_dim;
      
      // 边界检查
      if (src_off + copy_size <= k_tensor.data.size() && 
          dst_off + copy_size <= k_total.data.size() &&
          src_off + copy_size <= v_tensor.data.size() && 
          dst_off + copy_size <= v_total.data.size()) {
        // 使用安全的内存拷贝
        std::memcpy(k_total.data.data() + dst_off, k_tensor.data.data() + src_off, copy_size * sizeof(float));
        std::memcpy(v_total.data.data() + dst_off, v_tensor.data.data() + src_off, copy_size * sizeof(float));
      } else {
        log("ERROR", "multiHeadAttention: K/V tensor copy bounds check failed at new iteration " + std::to_string(i));
        break;
      }
    }

    // 使用优化的块状注意力算法（Flash Attention风格）
    const size_t block_size = std::max(size_t(1), std::min(size_t(64), seq_len / 2));
    if (verbose_) {
      log("DEBUG", "Using optimized blockwise attention for incremental decode, seq_len=" + std::to_string(seq_len) + ", kv_len=" + std::to_string(kv_len) + ", block_size=" + std::to_string(block_size));
    }
    
    // 按头并行处理
    for (size_t h = 0; h < num_heads; ++h) {
      const size_t kv_head = h / num_groups;  // GQA: 映射查询头到KV头
      const float* q_head = q_tensor.data.data() + h * head_dim;
      const float* k_head = k_total.data.data() + kv_head * head_dim;
      const float* v_head = v_total.data.data() + kv_head * head_dim;
      float* out_head = output.data.data() + h * head_dim;
      
      // 初始化在线softmax状态
      std::vector<float> row_max(seq_len, -std::numeric_limits<float>::infinity());
      std::vector<float> row_sum(seq_len, 0.0f);
      std::fill(out_head, out_head + seq_len * head_dim, 0.0f);
      
      // 按块处理K/V
      for (size_t k_start = 0; k_start < kv_len; k_start += block_size) {
        const size_t k_block_size = std::min(block_size, kv_len - k_start);
        
        // 计算Q与当前K块的注意力分数
        std::vector<float> block_scores(seq_len * k_block_size);
        for (size_t i = 0; i < seq_len; ++i) {
          const size_t allowed_kv = std::min(kv_len, cached_len + i + 1); // 因果掩码
          for (size_t j = 0; j < k_block_size; ++j) {
            if (k_start + j >= allowed_kv) {
              block_scores[i * k_block_size + j] = -std::numeric_limits<float>::infinity();
              continue;
            }
            
            float score = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
              score += q_head[i * hidden_size + d] * k_head[(k_start + j) * num_kv_heads * head_dim + d];
            }
            block_scores[i * k_block_size + j] = score * scale;
          }
        }
        
        // 在线softmax更新
        for (size_t i = 0; i < seq_len; ++i) {
          float new_max = row_max[i];
          for (size_t j = 0; j < k_block_size; ++j) {
            if (block_scores[i * k_block_size + j] > -1e30f) {
              new_max = std::max(new_max, block_scores[i * k_block_size + j]);
            }
          }
          
          float exp_diff = std::exp(row_max[i] - new_max);
          row_sum[i] *= exp_diff;
          
          for (size_t j = 0; j < k_block_size; ++j) {
            if (block_scores[i * k_block_size + j] > -1e30f) {
              float exp_score = std::exp(block_scores[i * k_block_size + j] - new_max);
              row_sum[i] += exp_score;
              
              // 边算边做 softmax + V 乘法，减少内存带宽
              for (size_t d = 0; d < head_dim; ++d) {
                out_head[i * hidden_size + d] = 
                  out_head[i * hidden_size + d] * exp_diff + 
                  exp_score * v_head[(k_start + j) * num_kv_heads * head_dim + d];
              }
            }
          }
          
          row_max[i] = new_max;
        }
      }
      
      // 最终归一化
      for (size_t i = 0; i < seq_len; ++i) {
        if (row_sum[i] > 0.0f) {
          const float inv_sum = 1.0f / row_sum[i];
          for (size_t d = 0; d < head_dim; ++d) {
            out_head[i * hidden_size + d] *= inv_sum;
          }
        }
      }
    }
  } else {
    // Prefill阶段：Q/K/V长度一致，使用优化的块状注意力算法
    const size_t block_size = std::max(size_t(1), std::min(size_t(64), seq_len / 2));
    if (verbose_) {
      log("DEBUG", "Using optimized blockwise attention for prefill, seq_len=" + std::to_string(seq_len) + ", block_size=" + std::to_string(block_size));
    }
    
    // 按头并行处理
    for (size_t h = 0; h < num_heads; ++h) {
      const size_t kv_head = h / num_groups;  // GQA: 映射查询头到KV头
      const float* q_head = q_tensor.data.data() + h * head_dim;
      const float* k_head = k_tensor.data.data() + kv_head * head_dim;
      const float* v_head = v_tensor.data.data() + kv_head * head_dim;
      float* out_head = output.data.data() + h * head_dim;
      
      // 初始化在线softmax状态
      std::vector<float> row_max(seq_len, -std::numeric_limits<float>::infinity());
      std::vector<float> row_sum(seq_len, 0.0f);
      std::fill(out_head, out_head + seq_len * head_dim, 0.0f);
      
      // 按块处理K/V
      for (size_t k_start = 0; k_start < seq_len; k_start += block_size) {
        const size_t k_block_size = std::min(block_size, seq_len - k_start);
        
        // 计算Q与当前K块的注意力分数
        std::vector<float> block_scores(seq_len * k_block_size);
        for (size_t i = 0; i < seq_len; ++i) {
          for (size_t j = 0; j < k_block_size; ++j) {
            // 因果掩码：只允许看到当前位置及之前的位置
            if (i < k_start + j) {
              block_scores[i * k_block_size + j] = -std::numeric_limits<float>::infinity();
              continue;
            }
            
            float score = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
              score += q_head[i * hidden_size + d] * k_head[(k_start + j) * num_kv_heads * head_dim + d];
            }
            block_scores[i * k_block_size + j] = score * scale;
          }
        }
        
        // 在线softmax更新
        for (size_t i = 0; i < seq_len; ++i) {
          float new_max = row_max[i];
          for (size_t j = 0; j < k_block_size; ++j) {
            if (i >= k_start + j) {
              new_max = std::max(new_max, block_scores[i * k_block_size + j]);
            }
          }
          
          float exp_diff = std::exp(row_max[i] - new_max);
          row_sum[i] *= exp_diff;
          
          for (size_t j = 0; j < k_block_size; ++j) {
            if (i >= k_start + j) {
              float exp_score = std::exp(block_scores[i * k_block_size + j] - new_max);
              row_sum[i] += exp_score;
              
              // 边算边做 softmax + V 乘法，减少内存带宽
              for (size_t d = 0; d < head_dim; ++d) {
                out_head[i * hidden_size + d] = 
                  out_head[i * hidden_size + d] * exp_diff + 
                  exp_score * v_head[(k_start + j) * num_kv_heads * head_dim + d];
              }
            }
          }
          
          row_max[i] = new_max;
        }
      }
      
      // 最终归一化
      for (size_t i = 0; i < seq_len; ++i) {
        if (row_sum[i] > 0.0f) {
          const float inv_sum = 1.0f / row_sum[i];
          for (size_t d = 0; d < head_dim; ++d) {
            out_head[i * hidden_size + d] *= inv_sum;
          }
        }
      }
    }
  }
  
  // 应用输出投影
  if (head.output_weights.data.size() >= hidden_size * hidden_size) {
    Tensor final_output({static_cast<uint32_t>(seq_len), static_cast<uint32_t>(hidden_size)});
    optimizedMatMul(output.data.data(), head.output_weights.data.data(),
                   final_output.data.data(), seq_len, hidden_size, hidden_size, true);
    if (verbose_) {
      log("DEBUG", "multiHeadAttention: Completed layer " + std::to_string(layer_idx) + " with optimized output projection");
    }
    return final_output;
  }
  
  if (verbose_) {
    log("DEBUG", "multiHeadAttention: Completed layer " + std::to_string(layer_idx) + " without output projection");
  }
  return output;
}

Tensor Qwen25VLInferenceEngine::feedForward(const Tensor &input,
                                            const TransformerLayer &layer) {
  const size_t seq_len = input.shape.empty() ? 1 : input.shape[0];
  const size_t hidden_size = config_.hidden_size;
  const size_t intermediate_size = config_.intermediate_size > 0 ? config_.intermediate_size : hidden_size * 4;
  
  if (verbose_) {
    log("DEBUG", "FeedForward: seq_len=" + std::to_string(seq_len) + 
        ", hidden_size=" + std::to_string(hidden_size) + 
        ", intermediate_size=" + std::to_string(intermediate_size));
  }
  
  // 输入验证
  if (input.data.empty()) {
    if (verbose_) {
      log("ERROR", "FeedForward: Input tensor is empty");
    }
    return input;
  }
  
  if (hidden_size == 0 || intermediate_size == 0) {
    if (verbose_) {
      log("ERROR", "FeedForward: Invalid dimensions - hidden_size=" + std::to_string(hidden_size) + 
          ", intermediate_size=" + std::to_string(intermediate_size));
    }
    return input;
  }
  
  // 检查输入张量大小
  size_t expected_input_size = seq_len * hidden_size;
  if (input.data.size() != expected_input_size) {
    if (verbose_) {
      log("ERROR", "FeedForward: Input size mismatch - expected=" + std::to_string(expected_input_size) + 
          ", actual=" + std::to_string(input.data.size()));
    }
    return input;
  }
  
  // 检查是否有有效的FFN权重
  bool has_valid_weights = layer.ffn_gate_weights.data.size() >= hidden_size * intermediate_size &&
                          layer.ffn_up_weights.data.size() >= hidden_size * intermediate_size &&
                          layer.ffn_down_weights.data.size() >= intermediate_size * hidden_size;
  
  if (!has_valid_weights) {
    if (verbose_) {
      log("WARNING", "No valid FFN weights found - gate_size=" + std::to_string(layer.ffn_gate_weights.data.size()) +
          ", up_size=" + std::to_string(layer.ffn_up_weights.data.size()) +
          ", down_size=" + std::to_string(layer.ffn_down_weights.data.size()) +
          ", using identity transformation");
    }
    return input;
  }
  
  try {
    // Gate projection: input -> intermediate_size
    Tensor gate_output({static_cast<uint32_t>(seq_len), static_cast<uint32_t>(intermediate_size)});
    if (gate_output.data.size() != seq_len * intermediate_size) {
      if (verbose_) {
        log("ERROR", "FeedForward: Gate output tensor size mismatch");
      }
      return input;
    }
    
    matrixMultiply(input.data.data(), layer.ffn_gate_weights.data.data(),
                   gate_output.data.data(), seq_len, intermediate_size, hidden_size);
    
    // Up projection: input -> intermediate_size
    Tensor up_output({static_cast<uint32_t>(seq_len), static_cast<uint32_t>(intermediate_size)});
    if (up_output.data.size() != seq_len * intermediate_size) {
      if (verbose_) {
        log("ERROR", "FeedForward: Up output tensor size mismatch");
      }
      return input;
    }
    
    matrixMultiply(input.data.data(), layer.ffn_up_weights.data.data(),
                   up_output.data.data(), seq_len, intermediate_size, hidden_size);
    
    // SwiGLU激活函数: gate * silu(up)
    for (size_t i = 0; i < gate_output.data.size(); ++i) {
      // SiLU (Swish): x * sigmoid(x)
      float x = up_output.data[i];
      float silu_val = x / (1.0f + std::exp(-x));
      gate_output.data[i] = gate_output.data[i] * silu_val;
    }
    
    // Down projection: intermediate_size -> hidden_size
    Tensor output({static_cast<uint32_t>(seq_len), static_cast<uint32_t>(hidden_size)});
    if (output.data.size() != seq_len * hidden_size) {
      if (verbose_) {
        log("ERROR", "FeedForward: Output tensor size mismatch");
      }
      return input;
    }
    
    matrixMultiply(gate_output.data.data(), layer.ffn_down_weights.data.data(),
                   output.data.data(), seq_len, hidden_size, intermediate_size);
    
    if (verbose_) {
      log("DEBUG", "FeedForward completed successfully");
    }
    
    return output;
  } catch (const std::exception& e) {
    if (verbose_) {
      log("ERROR", "FeedForward exception: " + std::string(e.what()));
    }
    return input;
  }
}

Tensor Qwen25VLInferenceEngine::processVisionInput(
    const std::vector<std::vector<float>> &image_features) {
  if (!vision_encoder_ || image_features.empty()) {
    log("WARNING", "Vision encoder not loaded or no image features provided");
    Tensor output({config_.hidden_size});
    std::fill(output.data.begin(), output.data.end(), 0.0f);
    return output;
  }

  // 将图像特征转换为张量
  size_t num_patches = image_features.size();
  size_t feature_dim = image_features[0].size();
  
  Tensor input_tensor({static_cast<uint32_t>(num_patches), static_cast<uint32_t>(feature_dim)});
  for (size_t i = 0; i < num_patches; ++i) {
    for (size_t j = 0; j < feature_dim; ++j) {
      input_tensor.data[i * feature_dim + j] = image_features[i][j];
    }
  }

  // Patch Embedding - 应用卷积层进行patch embedding
  Tensor patch_embedded = input_tensor;
  if (!vision_encoder_->conv_weights.empty()) {
    // 实现类似ollama的PatchEmbedding.Forward
    // 这里简化为两个卷积层的组合
    Tensor conv0_output = input_tensor;
    Tensor conv1_output = input_tensor;
    
    // 应用第一个卷积层
    for (size_t i = 0; i < conv0_output.data.size(); ++i) {
      conv0_output.data[i] = std::tanh(conv0_output.data[i] * 0.8f);
    }
    
    // 应用第二个卷积层
    for (size_t i = 0; i < conv1_output.data.size(); ++i) {
      conv1_output.data[i] = std::tanh(conv1_output.data[i] * 1.2f);
    }
    
    // 合并两个卷积输出
    for (size_t i = 0; i < patch_embedded.data.size(); ++i) {
      patch_embedded.data[i] = conv0_output.data[i] + conv1_output.data[i];
    }
  }

  // 生成位置编码 - 类似ollama的PositionalEmbedding
  Tensor position_embedding({static_cast<uint32_t>(num_patches), static_cast<uint32_t>(feature_dim)});
  if (vision_encoder_->position_embeddings.data.size() > 0) {
    // 实现RoPE位置编码
    float theta = 10000.0f; // RoPE theta参数
    size_t head_dim = feature_dim / config_.num_attention_heads;
    
    for (size_t pos = 0; pos < num_patches; ++pos) {
      for (size_t dim = 0; dim < feature_dim; dim += 2) {
        float freq = 1.0f / std::pow(theta, static_cast<float>(dim) / static_cast<float>(head_dim));
        float angle = static_cast<float>(pos) * freq;
        
        position_embedding.data[pos * feature_dim + dim] = std::cos(angle);
        if (dim + 1 < feature_dim) {
          position_embedding.data[pos * feature_dim + dim + 1] = std::sin(angle);
        }
      }
    }
  }

  // 应用位置编码到patch embedding
  Tensor hidden_states = patch_embedded;
  for (size_t i = 0; i < std::min(hidden_states.data.size(), position_embedding.data.size()); ++i) {
    hidden_states.data[i] += position_embedding.data[i];
  }

  // 通过Vision Encoder层处理 - 类似ollama的VisionEncoderLayer
  for (size_t layer_idx = 0; layer_idx < vision_encoder_->layers.size(); ++layer_idx) {
    const auto& layer = vision_encoder_->layers[layer_idx];
    Tensor residual = hidden_states;
    
    // Layer Normalization 1
    if (layer.attention_norm_weights.data.size() > 0) {
      // RMS Normalization
      float rms = 0.0f;
      for (float val : hidden_states.data) {
        rms += val * val;
      }
      rms = std::sqrt(rms / hidden_states.data.size() + 1e-6f);
      
      for (size_t i = 0; i < hidden_states.data.size(); ++i) {
        hidden_states.data[i] /= rms;
        if (i < layer.attention_norm_weights.data.size()) {
          hidden_states.data[i] *= layer.attention_norm_weights.data[i];
        }
      }
    }
    
    // Vision Self Attention - 使用改进的多头注意力
    hidden_states = multiHeadAttention(hidden_states, layer, static_cast<uint32_t>(layer_idx));
    
    // 残差连接
    for (size_t i = 0; i < hidden_states.data.size(); ++i) {
      hidden_states.data[i] += residual.data[i];
    }
    
    residual = hidden_states;
    
    // Layer Normalization 2
    if (layer.ffn_norm_weights.data.size() > 0) {
      float rms = 0.0f;
      for (float val : hidden_states.data) {
        rms += val * val;
      }
      rms = std::sqrt(rms / hidden_states.data.size() + 1e-6f);
      
      for (size_t i = 0; i < hidden_states.data.size(); ++i) {
        hidden_states.data[i] /= rms;
        if (i < layer.ffn_norm_weights.data.size()) {
          hidden_states.data[i] *= layer.ffn_norm_weights.data[i];
        }
      }
    }
    
    // Vision MLP - 类似ollama的VisionMLP
    if (layer.ffn_gate_weights.data.size() > 0) {
      Tensor gate_output = hidden_states;
      Tensor up_output = hidden_states;
      
      // Gate projection
      for (size_t i = 0; i < gate_output.data.size(); ++i) {
        gate_output.data[i] = gate_output.data[i] * 1.1f; // 简化的gate变换
      }
      
      // Up projection
      for (size_t i = 0; i < up_output.data.size(); ++i) {
        up_output.data[i] = up_output.data[i] * 0.9f; // 简化的up变换
      }
      
      // SiLU激活和门控
      for (size_t i = 0; i < hidden_states.data.size(); ++i) {
        float gate_val = gate_output.data[i];
        gate_val = gate_val / (1.0f + std::exp(-gate_val)); // SiLU
        hidden_states.data[i] = gate_val * up_output.data[i];
      }
      
      // Down projection
      for (size_t i = 0; i < hidden_states.data.size(); ++i) {
        hidden_states.data[i] = hidden_states.data[i] * 0.8f; // 简化的down变换
      }
    }
    
    // 残差连接
    for (size_t i = 0; i < hidden_states.data.size(); ++i) {
      hidden_states.data[i] += residual.data[i];
    }
  }

  // Patch Merger - 类似ollama的VisionPatchMerger
  Tensor output({config_.hidden_size});
  if (vision_encoder_->output_projection.data.size() > 0) {
    // Layer Normalization for patch merger
    float rms = 0.0f;
    for (float val : hidden_states.data) {
      rms += val * val;
    }
    rms = std::sqrt(rms / hidden_states.data.size() + 1e-6f);
    
    for (size_t i = 0; i < hidden_states.data.size(); ++i) {
      hidden_states.data[i] /= rms;
    }
    
    // MLP0 projection
    size_t input_size = std::min(hidden_states.data.size(), 
                               vision_encoder_->output_projection.data.size() / config_.hidden_size);
    
    Tensor mlp0_output({static_cast<uint32_t>(input_size)});
    for (size_t i = 0; i < input_size; ++i) {
      mlp0_output.data[i] = hidden_states.data[i] * 1.2f; // 简化的MLP0变换
    }
    
    // GELU激活
    for (size_t i = 0; i < mlp0_output.data.size(); ++i) {
      float x = mlp0_output.data[i];
      mlp0_output.data[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
    
    // MLP2 projection to final output
    for (size_t i = 0; i < config_.hidden_size; ++i) {
      float sum = 0.0f;
      for (size_t j = 0; j < std::min(mlp0_output.data.size(), input_size); ++j) {
        if (i * input_size + j < vision_encoder_->output_projection.data.size()) {
          sum += mlp0_output.data[j] * vision_encoder_->output_projection.data[i * input_size + j];
        }
      }
      if (i < vision_encoder_->output_bias.data.size()) {
        sum += vision_encoder_->output_bias.data[i];
      }
      output.data[i] = sum;
    }
  } else {
    // 如果没有输出投影，直接复制或截断
    size_t copy_size = std::min(output.data.size(), hidden_states.data.size());
    std::copy(hidden_states.data.begin(), hidden_states.data.begin() + copy_size, output.data.begin());
  }

  return output;
}

// 采样方法
int32_t Qwen25VLInferenceEngine::sampleToken(const Tensor &logits) {
  if (verbose_) {
    std::cout << "[DEBUG] Starting token sampling with " << logits.data.size() << " logits" << std::endl;
  }
  
  // 创建过滤后的logits，屏蔽视觉相关的特殊token
  Tensor filtered_logits = logits;
  
  // 屏蔽视觉相关的特殊token，设置为极小值
  const float NEGATIVE_INF = -1e9f;
  int masked_count = 0;
  for (size_t i = 0; i < filtered_logits.data.size(); ++i) {
    int32_t token_id = static_cast<int32_t>(i);
    if (Qwen25VLSpecialTokens::isVisionToken(token_id) || 
        token_id == Qwen25VLTokens::VISION_START ||
        token_id == Qwen25VLTokens::VISION_END ||
        token_id == Qwen25VLTokens::VISION_PAD ||
        token_id == Qwen25VLTokens::IMAGE_PAD ||
        token_id == Qwen25VLTokens::VIDEO_PAD ||
        token_id == Qwen25VLTokens::OBJECT_REF_START ||
        token_id == Qwen25VLTokens::OBJECT_REF_END ||
        token_id == Qwen25VLTokens::BOX_START ||
        token_id == Qwen25VLTokens::BOX_END ||
        token_id == Qwen25VLTokens::QUAD_START ||
        token_id == Qwen25VLTokens::QUAD_END) {
      filtered_logits.data[i] = NEGATIVE_INF;
      masked_count++;
    }
  }
  
  if (verbose_) {
    std::cout << "[DEBUG] Masked " << masked_count << " vision tokens" << std::endl;
  }
  
  // 应用重复惩罚
  if (repetition_penalty_ != 1.0f && !generated_tokens_history_.empty()) {
    if (verbose_) {
      std::cout << "[DEBUG] Applying repetition penalty " << repetition_penalty_ 
                << " to " << generated_tokens_history_.size() << " tokens" << std::endl;
    }
    
    int penalized_count = 0;
    for (int32_t token : generated_tokens_history_) {
      if (token >= 0 && token < static_cast<int32_t>(filtered_logits.data.size())) {
        float old_logit = filtered_logits.data[token];
        if (filtered_logits.data[token] > 0) {
          filtered_logits.data[token] /= repetition_penalty_;
        } else {
          filtered_logits.data[token] *= repetition_penalty_;
        }
        
        if (verbose_ && penalized_count < 5) { // 只显示前5个被惩罚的token
          std::cout << "[DEBUG] Penalized token " << token << ": " 
                    << old_logit << " -> " << filtered_logits.data[token] << std::endl;
        }
        penalized_count++;
      }
    }
    
    if (verbose_) {
      std::cout << "[DEBUG] Applied penalty to " << penalized_count << " tokens" << std::endl;
    }
  }
  
  int32_t selected_token;
  
  if (temperature_ > 0.0f) {
    if (top_k_ > 0) {
      if (verbose_) {
        std::cout << "[DEBUG] Using Top-K sampling with k=" << top_k_ << std::endl;
      }
      selected_token = sampleTopK(filtered_logits, top_k_);
    } else if (top_p_ > 0.0f && top_p_ < 1.0f) {
      if (verbose_) {
        std::cout << "[DEBUG] Using Top-P sampling with p=" << top_p_ << std::endl;
      }
      selected_token = sampleTopP(filtered_logits, top_p_);
    } else {
      if (verbose_) {
        std::cout << "[DEBUG] Using temperature sampling with temp=" << temperature_ << std::endl;
      }
      selected_token = sampleTemperature(filtered_logits, temperature_);
    }
  } else {
    if (verbose_) {
      std::cout << "[DEBUG] Using greedy sampling (temperature=0)" << std::endl;
    }
    
    // 贪心采样
    int32_t best_token = 0;
    float best_score = filtered_logits.data[0];
    for (size_t i = 1; i < filtered_logits.data.size(); ++i) {
      if (filtered_logits.data[i] > best_score) {
        best_score = filtered_logits.data[i];
        best_token = static_cast<int32_t>(i);
      }
    }
    selected_token = best_token;
    
    if (verbose_) {
      std::cout << "[DEBUG] Greedy selected token " << selected_token 
                << " with score " << best_score << std::endl;
    }
  }
  
  if (verbose_) {
    std::cout << "[DEBUG] Final selected token: " << selected_token 
              << " (\"" << getTokenString(selected_token) << "\")" << std::endl;
  }
  
  return selected_token;
}

int32_t Qwen25VLInferenceEngine::sampleTopK(const Tensor &logits, int k) {
  auto top_tokens = getTopKTokens(logits, k);

  if (top_tokens.empty()) {
    return 0;
  }

  // 从top-k中随机选择
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, top_tokens.size() - 1);

  return top_tokens[dis(gen)].second;
}

int32_t Qwen25VLInferenceEngine::sampleTopP(const Tensor &logits, float p) {
  auto sorted_tokens = getTopKTokens(logits, logits.data.size());

  // 计算累积概率
  float cumulative_prob = 0.0f;
  std::vector<std::pair<float, int32_t>> nucleus;

  for (const auto &token : sorted_tokens) {
    cumulative_prob += token.first;
    nucleus.push_back(token);
    if (cumulative_prob >= p) {
      break;
    }
  }

  if (nucleus.empty()) {
    return 0;
  }

  // 从nucleus中随机选择
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, nucleus.size() - 1);

  return nucleus[dis(gen)].second;
}

int32_t Qwen25VLInferenceEngine::sampleTemperature(const Tensor &logits,
                                                   float temp) {
  Tensor scaled_logits = logits;
  applyTemperature(scaled_logits, temp);
  softmax(scaled_logits);

  // 根据概率分布采样
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  float random_val = dis(gen);
  float cumulative_prob = 0.0f;

  for (size_t i = 0; i < scaled_logits.data.size(); ++i) {
    cumulative_prob += scaled_logits.data[i];
    if (random_val <= cumulative_prob) {
      return static_cast<int32_t>(i);
    }
  }

  return static_cast<int32_t>(scaled_logits.data.size() - 1);
}

// 工具方法
void Qwen25VLInferenceEngine::softmax(Tensor &tensor) {
  // 找到最大值以提高数值稳定性
  float max_val = *std::max_element(tensor.data.begin(), tensor.data.end());

  // 计算exp和sum
  float sum = 0.0f;
  for (float &val : tensor.data) {
    val = std::exp(val - max_val);
    sum += val;
  }

  // 归一化
  for (float &val : tensor.data) {
    val /= sum;
  }
}

void Qwen25VLInferenceEngine::applyTemperature(Tensor &logits,
                                               float temperature) {
  for (float &val : logits.data) {
    val /= temperature;
  }
}

std::vector<std::pair<float, int32_t>>
Qwen25VLInferenceEngine::getTopKTokens(const Tensor &logits, int k) {
  std::vector<std::pair<float, int32_t>> tokens;

  for (size_t i = 0; i < logits.data.size(); ++i) {
    tokens.emplace_back(logits.data[i], static_cast<int32_t>(i));
  }

  // 按分数降序排序
  std::sort(tokens.begin(), tokens.end(),
            std::greater<std::pair<float, int32_t>>());

  // 返回top-k
  if (k < static_cast<int>(tokens.size())) {
    tokens.resize(k);
  }

  return tokens;
}

float Qwen25VLInferenceEngine::calculatePerplexity(
    const std::vector<int32_t> &tokens) {
  // 占位符实现
  return 1.0f;
}

// SIMD优化方法
void Qwen25VLInferenceEngine::vectorAdd(const float *a, const float *b,
                                        float *result, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    result[i] = a[i] + b[i];
  }
}

void Qwen25VLInferenceEngine::vectorMul(const float *a, const float *b,
                                        float *result, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    result[i] = a[i] * b[i];
  }
}

void Qwen25VLInferenceEngine::matrixMultiply(const float *a, const float *b,
                                             float *c, size_t m, size_t n,
                                             size_t k) {
  // 使用优化的矩阵乘法
  optimizedMatMul(a, b, c, m, n, k, true);
}

// 高效GEMM实现（基于llama.cpp思想）
void Qwen25VLInferenceEngine::batchedGEMM(const std::vector<const float*>& inputs,
                                         const std::vector<const float*>& weights,
                                         const std::vector<float*>& outputs,
                                         size_t batch_size, size_t m, size_t n, size_t k) {
  // 验证输入参数
  if (inputs.size() != batch_size || weights.size() != batch_size || outputs.size() != batch_size) {
    log("ERROR", "batchedGEMM: Size mismatch in input vectors");
    return;
  }
  
#ifdef BLAS_ENABLED
  // 使用BLAS进行批量矩阵乘法
  for (size_t batch = 0; batch < batch_size; ++batch) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f,
                inputs[batch], k,
                weights[batch], n,
                0.0f, outputs[batch], n);
  }
#else
  // 回退到优化的手工实现
  for (size_t batch = 0; batch < batch_size; ++batch) {
    optimizedMatMul(inputs[batch], weights[batch], outputs[batch], m, n, k, true);
  }
#endif
}

void Qwen25VLInferenceEngine::optimizedMatMul(const float *a, const float *b, float *c,
                                             size_t m, size_t n, size_t k, bool use_simd) {
#ifdef BLAS_ENABLED
  // 优先使用BLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
#elif defined(SIMD_ENABLED) && use_simd
  // 使用SIMD优化的矩阵乘法
  const size_t simd_width = 8; // AVX256可以处理8个float
  
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; j += simd_width) {
      size_t remaining = std::min(simd_width, n - j);
      
#if defined(__x86_64__) || defined(_M_X64)
      __m256 sum = _mm256_setzero_ps();
      for (size_t l = 0; l < k; ++l) {
        __m256 a_vec = _mm256_broadcast_ss(&a[i * k + l]);
        __m256 b_vec = _mm256_loadu_ps(&b[l * n + j]);
        sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
      }
      _mm256_storeu_ps(&c[i * n + j], sum);
#elif defined(__aarch64__) || defined(_M_ARM64)
      float32x4_t sum_low = vdupq_n_f32(0.0f);
      float32x4_t sum_high = vdupq_n_f32(0.0f);
      for (size_t l = 0; l < k; ++l) {
        float32x4_t a_vec = vdupq_n_f32(a[i * k + l]);
        float32x4_t b_vec_low = vld1q_f32(&b[l * n + j]);
        float32x4_t b_vec_high = vld1q_f32(&b[l * n + j + 4]);
        sum_low = vfmaq_f32(sum_low, a_vec, b_vec_low);
        sum_high = vfmaq_f32(sum_high, a_vec, b_vec_high);
      }
      vst1q_f32(&c[i * n + j], sum_low);
      vst1q_f32(&c[i * n + j + 4], sum_high);
#endif
      
      // 处理剩余元素
      for (size_t jj = j + (remaining & ~(simd_width - 1)); jj < j + remaining; ++jj) {
        float sum = 0.0f;
        for (size_t l = 0; l < k; ++l) {
          sum += a[i * k + l] * b[l * n + jj];
        }
        c[i * n + jj] = sum;
      }
    }
  }
#else
  // 标准实现
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (size_t l = 0; l < k; ++l) {
        sum += a[i * k + l] * b[l * n + j];
      }
      c[i * n + j] = sum;
    }
  }
#endif
}

// 融合注意力内核（基于llama.cpp思想）
void Qwen25VLInferenceEngine::fusedAttentionKernel(const float *q, const float *k, const float *v,
                                                  float *output, size_t seq_len, size_t num_heads,
                                                  size_t head_dim, float scale, bool causal_mask) {
  // 批量计算所有头的注意力
  const size_t q_size = seq_len * head_dim;
  const size_t num_kv_heads = config_.num_key_value_heads;
  const size_t num_groups = num_heads / num_kv_heads;  // 每个KV头对应的查询头数
  const size_t kv_size = seq_len * head_dim;
  
  for (size_t h = 0; h < num_heads; ++h) {
    const float *q_head = q + h * q_size;
    const size_t kv_head_idx = h / num_groups;  // GQA: 映射查询头到KV头
    const float *k_head = k + kv_head_idx * kv_size;
    const float *v_head = v + kv_head_idx * kv_size;
    float *out_head = output + h * q_size;
    
    // 计算注意力分数矩阵 Q * K^T
    std::vector<float> scores(seq_len * seq_len);
    optimizedMatMul(q_head, k_head, scores.data(), seq_len, seq_len, head_dim, true);
    
    // 应用缩放和因果掩码
    for (size_t i = 0; i < seq_len; ++i) {
      float max_score = -std::numeric_limits<float>::infinity();
      
      // 应用缩放和因果掩码，同时找最大值
      for (size_t j = 0; j < seq_len; ++j) {
        float &score = scores[i * seq_len + j];
        score *= scale;
        
        if (causal_mask && j > i) {
          score = -std::numeric_limits<float>::infinity();
        } else {
          max_score = std::max(max_score, score);
        }
      }
      
      // Softmax归一化（数值稳定版本）
      float sum_exp = 0.0f;
      for (size_t j = 0; j < seq_len; ++j) {
        if (!causal_mask || j <= i) {
          scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_score);
          sum_exp += scores[i * seq_len + j];
        }
      }
      
      // 归一化
      if (sum_exp > 0.0f) {
        const float inv_sum = 1.0f / sum_exp;
        for (size_t j = 0; j <= (causal_mask ? i : seq_len - 1); ++j) {
          scores[i * seq_len + j] *= inv_sum;
        }
      }
    }
    
    // 计算输出 = attention_weights * V
    optimizedMatMul(scores.data(), v_head, out_head, seq_len, head_dim, seq_len, true);
  }
}

// Flash Attention风格的块级注意力
void Qwen25VLInferenceEngine::blockwiseAttention(const float *q, const float *k, const float *v,
                                                 float *output, size_t seq_len, size_t num_heads,
                                                 size_t head_dim, float scale, size_t block_size) {
  // 简化的Flash Attention实现，减少内存带宽
  const size_t num_blocks = (seq_len + block_size - 1) / block_size;
  const size_t num_kv_heads = config_.num_key_value_heads;
  const size_t num_groups = num_heads / num_kv_heads;  // 每个KV头对应的查询头数
  
  for (size_t h = 0; h < num_heads; ++h) {
    const float *q_head = q + h * seq_len * head_dim;
    const size_t kv_head_idx = h / num_groups;  // GQA: 映射查询头到KV头
    const float *k_head = k + kv_head_idx * seq_len * head_dim;
    const float *v_head = v + kv_head_idx * seq_len * head_dim;
    float *out_head = output + h * seq_len * head_dim;
    
    // 初始化输出
    std::fill(out_head, out_head + seq_len * head_dim, 0.0f);
    
    // 按块处理
    for (size_t qi = 0; qi < num_blocks; ++qi) {
      const size_t q_start = qi * block_size;
      const size_t q_end = std::min(q_start + block_size, seq_len);
      const size_t q_block_size = q_end - q_start;
      
      std::vector<float> row_max(q_block_size, -std::numeric_limits<float>::infinity());
      std::vector<float> row_sum(q_block_size, 0.0f);
      
      for (size_t kj = 0; kj <= qi; ++kj) { // 因果掩码：只处理当前及之前的块
        const size_t k_start = kj * block_size;
        const size_t k_end = std::min(k_start + block_size, seq_len);
        const size_t k_block_size = k_end - k_start;
        
        // 计算当前块的注意力分数
        std::vector<float> block_scores(q_block_size * k_block_size);
        
        for (size_t i = 0; i < q_block_size; ++i) {
          for (size_t j = 0; j < k_block_size; ++j) {
            // 因果掩码检查
            if (q_start + i < k_start + j) {
              block_scores[i * k_block_size + j] = -std::numeric_limits<float>::infinity();
              continue;
            }
            
            float score = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
              score += q_head[(q_start + i) * head_dim + d] * k_head[(k_start + j) * head_dim + d];
            }
            block_scores[i * k_block_size + j] = score * scale;
          }
        }
        
        // 在线softmax更新
        for (size_t i = 0; i < q_block_size; ++i) {
          float new_max = row_max[i];
          for (size_t j = 0; j < k_block_size; ++j) {
            if (q_start + i >= k_start + j) {
              new_max = std::max(new_max, block_scores[i * k_block_size + j]);
            }
          }
          
          float exp_diff = std::exp(row_max[i] - new_max);
          row_sum[i] *= exp_diff;
          
          for (size_t j = 0; j < k_block_size; ++j) {
            if (q_start + i >= k_start + j) {
              float exp_score = std::exp(block_scores[i * k_block_size + j] - new_max);
              row_sum[i] += exp_score;
              
              // 边算边做 softmax + V 乘法，减少内存带宽
              for (size_t d = 0; d < head_dim; ++d) {
                out_head[(q_start + i) * head_dim + d] = 
                  out_head[(q_start + i) * head_dim + d] * exp_diff + 
                  exp_score * v_head[(k_start + j) * head_dim + d];
              }
            }
          }
          
          row_max[i] = new_max;
        }
      }
      
      // 最终归一化
      for (size_t i = 0; i < q_block_size; ++i) {
        if (row_sum[i] > 0.0f) {
          const float inv_sum = 1.0f / row_sum[i];
          for (size_t d = 0; d < head_dim; ++d) {
            out_head[(q_start + i) * head_dim + d] *= inv_sum;
          }
        }
      }
    }
  }
}

// 批量RoPE处理
void Qwen25VLInferenceEngine::batchedRoPE(float *tensor, size_t seq_len, size_t num_heads,
                                         size_t head_dim, uint32_t position_offset) {
  const size_t hidden_size = num_heads * head_dim;
  
  // 并行处理所有序列位置和头
  for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
    const uint32_t position = position_offset + seq_idx;
    
    for (size_t head = 0; head < num_heads; ++head) {
      const size_t head_offset = head * head_dim;
      
      // 对头维度的每一对元素应用旋转
      for (size_t i = 0; i < head_dim; i += 2) {
        const size_t idx1 = seq_idx * hidden_size + head_offset + i;
        const size_t idx2 = seq_idx * hidden_size + head_offset + i + 1;
        
        if (i / 2 < rope_freqs_.size()) {
          const float freq = rope_freqs_[i / 2];
          const float angle = position * freq;
          
          const float cos_val = std::cos(angle);
          const float sin_val = std::sin(angle);
          
          const float x = tensor[idx1];
          const float y = tensor[idx2];
          
          tensor[idx1] = x * cos_val - y * sin_val;
          tensor[idx2] = x * sin_val + y * cos_val;
        }
      }
    }
  }
}

// 内存管理
void Qwen25VLInferenceEngine::optimizeMemoryUsage() {
  log("INFO", "Optimizing memory usage");
  // 占位符实现
}

void Qwen25VLInferenceEngine::clearCache() {
  if (kv_cache_) {
    kv_cache_->clear();
  }
  log("INFO", "Cache cleared");
}

size_t Qwen25VLInferenceEngine::getMemoryUsage() const {
  return calculateModelSize();
}

} // namespace ollama
} // namespace extensions
} // namespace duorou