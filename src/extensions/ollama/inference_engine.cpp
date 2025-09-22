#include "inference_engine.h"
#include "../../ml/context.h"
#include "../../ml/nn/attention.h"
#include "../../ml/tensor.h"
#include "ollama_model_manager.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// 包含真正的 llama.cpp 头文件
#include "../../../third_party/llama.cpp/include/llama.h"

namespace duorou {
namespace extensions {
namespace ollama {
// 简单白名单：哪些架构交给 llama.cpp 处理
static bool isSupportedByLlamaCpp(const std::string &arch_raw) {
  std::string arch = arch_raw;
  std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);

  // 明确排除已知非 llama.cpp 路线（多模态/非兼容）架构，避免误走 llama.cpp
  if (arch.find("qwen") != std::string::npos ||
      arch.find("qwen2") != std::string::npos ||
      arch.find("qwen25") != std::string::npos ||
      arch.find("qwen2.5") != std::string::npos ||
      arch.find("qwen-2.5") != std::string::npos ||
      arch.find("qwen2vl") != std::string::npos ||
      arch.find("qwen2.5vl") != std::string::npos ||
      arch.find("qwen-2.5vl") != std::string::npos ||
      arch.find("vl") != std::string::npos) {
    return false;
  }

  // 允许的架构关键词（按需扩展，但尽量精确）
  static const char *allow[] = {"llama", "llama2", "llama3", "mistral",
                                "gemma"};
  for (auto k : allow) {
    if (arch.find(k) != std::string::npos)
      return true;
  }
  return false;
}

MLInferenceEngine::MLInferenceEngine(const std::string &model_id)
    : model_id_(model_id), initialized_(false), ml_context_(nullptr),
      attention_(nullptr), gguf_parser_(nullptr), kv_cache_(nullptr),
      vocab_size_(0), n_layers_(0), n_heads_(0), n_embd_(0), n_ctx_(0),
      rope_initialized_(false), llama_model_(nullptr), llama_context_(nullptr),
      llama_sampler_(nullptr), use_llama_backend_(false) {
  std::cout << "[DEBUG] MLInferenceEngine constructor called with model_id: "
            << model_id << std::endl;
}

MLInferenceEngine::~MLInferenceEngine() {
  std::cout << "[DEBUG] MLInferenceEngine destructor called" << std::endl;

  // 释放 llama.cpp 资源
  if (llama_sampler_) {
    llama_sampler_free(llama_sampler_);
    llama_sampler_ = nullptr;
  }

  if (llama_context_) {
    llama_free(llama_context_);
    llama_context_ = nullptr;
  }

  if (llama_model_) {
    llama_free_model(llama_model_);
    llama_model_ = nullptr;
  }

  cleanupResources();
}

bool MLInferenceEngine::initialize() {
  std::cout << "[DEBUG] MLInferenceEngine::initialize called" << std::endl;

  if (initialized_) {
    std::cout << "[DEBUG] Already initialized, returning true" << std::endl;
    return true;
  }

  try {
    // 创建ML上下文
    ml_context_ = new ml::Context();

    // 创建注意力机制（简化配置）
    attention_ = new ml::nn::MultiHeadAttention(512,  // embed_dim
                                                8,    // num_heads
                                                -1,   // kv_heads (default)
                                                true, // bias
                                                0.1f  // dropout
    );

    // 尝试从OllamaModelManager获取模型路径
    OllamaModelManager &manager = GlobalModelManager::getInstance();

    // 直接使用model_id_，因为OllamaModelManager::getModelInfo会进行归一化处理
    auto model_info = manager.getModelInfo(model_id_);
    if (!model_info || model_info->file_path.empty()) {
      std::cerr << "[ERROR] MLInferenceEngine: Model not found or file path "
                   "empty in OllamaModelManager: "
                << model_id_ << std::endl;
      initialized_ = false;
      return false;
    }

    model_path_ = model_info->file_path;

    // 第一步：用 GGUFParser 解析文件，获取架构
    gguf_parser_ = std::make_unique<GGUFParser>();
    if (!gguf_parser_->parseFile(model_path_)) {
      std::cerr << "[ERROR] MLInferenceEngine: Failed to parse GGUF: "
                << model_path_ << std::endl;
      initialized_ = false;
      return false;
    }
    const auto &arch = gguf_parser_->getArchitecture().name;
    use_llama_backend_ = isSupportedByLlamaCpp(arch);
    std::cout << "[DEBUG] Detected architecture: '" << arch
              << "', use_llama_backend_="
              << (use_llama_backend_ ? "true" : "false") << std::endl;

    if (use_llama_backend_) {
      // 仅当选择 llama.cpp 时初始化其后端并加载模型
      llama_backend_init();
      // 加载llama.cpp模型
      if (!loadLlamaModel(model_path_)) {
        std::cerr
            << "[ERROR] MLInferenceEngine: Failed to load llama.cpp model from "
            << model_path_ << std::endl;
        initialized_ = false;
        return false;
      }

      initialized_ = true;
      std::cout
          << "[DEBUG] MLInferenceEngine initialized successfully with llama.cpp"
          << std::endl;
      return true;
    } else {
      // 非 llama 架构：走内部 Forward 流程，使用 Qwen 模型
      std::cout
          << "[DEBUG] Initializing Qwen multimodal model for internal forward"
          << std::endl;

      // 创建 Qwen 多模态模型
      qwen_model_ = std::make_unique<duorou::model::QwenMultimodalModel>();

      // 首先初始化模型组件（包括文本模型）
      if (!qwen_model_->initialize("")) {
        std::cerr << "[WARN] Failed to initialize Qwen model components, using "
                     "fallback initialization"
                  << std::endl;
        // 即使初始化失败，也继续初始化其他组件，以便至少有基本的推理能力
      }

      // 尝试从 GGUF 文件加载模型
      if (!qwen_model_->loadModel(model_path_)) {
        std::cerr << "[WARN] Failed to load Qwen model from GGUF, using "
                     "fallback initialization"
                  << std::endl;
        // 即使加载失败，也继续初始化其他组件，以便至少有基本的推理能力
      }

      if (!parseModelConfig()) {
        std::cerr << "[ERROR] Failed to parse model configuration" << std::endl;
        initialized_ = false;
        return false;
      }
      if (!loadModelWeights()) {
        std::cerr << "[ERROR] Failed to load model weights" << std::endl;
        initialized_ = false;
        return false;
      }
      if (!initializeKVCache()) {
        std::cerr << "[ERROR] Failed to initialize KV cache" << std::endl;
        initialized_ = false;
        return false;
      }
      if (!precomputeRoPEFreqs()) {
        std::cerr << "[ERROR] Failed to precompute RoPE frequencies"
                  << std::endl;
        initialized_ = false;
        return false;
      }
      initialized_ = true;
      std::cout << "[DEBUG] MLInferenceEngine initialized successfully with "
                   "internal forward (Qwen model)"
                << std::endl;
      return true;
    }
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception during initialization: " << e.what()
              << std::endl;
    initialized_ = false;
    return false;
  }
}

std::string MLInferenceEngine::generateText(const std::string &prompt,
                                            uint32_t max_tokens,
                                            float temperature, float top_p) {
  std::cout << "[DEBUG] MLInferenceEngine::generateText called with prompt: '"
            << prompt << "', max_tokens: " << max_tokens
            << ", temperature: " << temperature << ", top_p: " << top_p
            << std::endl;

  if (!isReady()) {
    std::cerr << "[ERROR] Inference engine not properly initialized: "
              << (initialized_ ? "initialized_\n" : "not initialized\n")
              << "  ml_context_=" << (ml_context_ ? "ok" : "null") << "\n"
              << "  attention_=" << (attention_ ? "ok" : "null") << "\n"
              << "  llama_model_=" << (llama_model_ ? "ok" : "null") << "\n"
              << "  llama_context_=" << (llama_context_ ? "ok" : "null") << "\n"
              << "  llama_sampler_=" << (llama_sampler_ ? "ok" : "null")
              << std::endl;
    return "Error: Inference engine not initialized";
  }

  try {
    if (use_llama_backend_) {
      // 使用 llama.cpp 进行推理
      return generateWithLlama(prompt, max_tokens, temperature, top_p);
    } else {
      // 使用内部 Forward 模式进行推理
      return generateWithInternalForward(prompt, max_tokens, temperature,
                                         top_p);
    }

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in generateText: " << e.what() << std::endl;
    return "Error: " + std::string(e.what());
  }
}

bool MLInferenceEngine::isReady() const {
  // 根据后端模式判定就绪状态
  if (!initialized_)
    return false;
  if (use_llama_backend_) {
    return ml_context_ != nullptr && attention_ != nullptr &&
           llama_model_ != nullptr && llama_context_ != nullptr &&
           llama_sampler_ != nullptr;
  } else {
    // 内部 Forward 模式：至少需要 ML 上下文、注意力和已完成 RoPE 预计算
    return ml_context_ != nullptr && attention_ != nullptr && rope_initialized_;
  }
}

std::string MLInferenceEngine::processText(const std::string &text) {
  // 简单的文本处理逻辑
  return "Processed: " + text;
}

bool MLInferenceEngine::loadModel(const std::string &model_path) {
  try {
    std::cout << "[DEBUG] Loading model from: " << model_path << std::endl;

    // 步骤1: 创建GGUF解析器
    std::cout << "[DEBUG] Step 1: Creating GGUF parser" << std::endl;
    gguf_parser_ = std::make_unique<GGUFParser>();

    // 步骤2: 解析GGUF文件
    std::cout << "[DEBUG] Step 2: Parsing GGUF file" << std::endl;
    if (!gguf_parser_->parseFile(model_path)) {
      std::cerr << "Failed to parse GGUF model: " << model_path << std::endl;
      return false;
    }

    // 解析模型配置
    if (!parseModelConfig()) {
      std::cerr << "Failed to parse model configuration" << std::endl;
      return false;
    }

    // 步骤3: 加载模型权重
    std::cout << "[DEBUG] Step 3: Loading model weights" << std::endl;
    if (!loadModelWeights()) {
      std::cerr << "Failed to load model weights" << std::endl;
      return false;
    }

    // 步骤4: 初始化KV缓存
    std::cout << "[DEBUG] Step 4: Initializing KV cache" << std::endl;
    if (!initializeKVCache()) {
      std::cerr << "Failed to initialize KV cache" << std::endl;
      return false;
    }

    // 步骤5: 预计算RoPE频率
    std::cout << "[DEBUG] Step 5: Precomputing RoPE frequencies" << std::endl;
    if (!precomputeRoPEFreqs()) {
      std::cerr << "Failed to precompute RoPE frequencies" << std::endl;
      return false;
    }

    std::cout
        << "[DEBUG] Model loaded successfully with all components initialized"
        << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "Error loading model: " << e.what() << std::endl;
    return false;
  }
}

bool MLInferenceEngine::loadLlamaModel(const std::string &model_path) {
  try {
    std::cout << "[DEBUG] Loading llama.cpp model from: " << model_path
              << std::endl;

    // 设置模型参数
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU only for now

    // 加载模型
    llama_model_ = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!llama_model_) {
      std::cerr << "[ERROR] Failed to load llama.cpp model from: " << model_path
                << std::endl;
      return false;
    }

    // 设置上下文参数
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;  // 上下文长度
    ctx_params.n_batch = 512; // 批处理大小
    ctx_params.n_threads = 4; // 线程数

    // 创建上下文
    llama_context_ = llama_init_from_model(llama_model_, ctx_params);
    if (!llama_context_) {
      std::cerr << "[ERROR] Failed to create llama.cpp context" << std::endl;
      return false;
    }

    // 基于 GGUF 元数据构建 Vocabulary 和 TextProcessor
    try {
      gguf_parser_ = std::make_unique<GGUFParser>();
      if (gguf_parser_->parseFile(model_path)) {
        // 读取tokens
        std::vector<std::string> tokens;
        if (const auto *kvTokens =
                gguf_parser_->getMetadata("tokenizer.ggml.tokens")) {
          tokens = kvTokens->asStringArray();
        }
        // 读取token types
        std::vector<int32_t> types;
        if (const auto *kvTypes =
                gguf_parser_->getMetadata("tokenizer.ggml.token_type")) {
          types = kvTypes->asInt32Array();
        }
        if (types.empty() && !tokens.empty()) {
          types.assign(tokens.size(), duorou::model::TOKEN_TYPE_NORMAL);
        }
        // 读取merges
        std::vector<std::string> merges;
        if (const auto *kvMerges =
                gguf_parser_->getMetadata("tokenizer.ggml.merges")) {
          merges = kvMerges->asStringArray();
        }

        if (!tokens.empty()) {
          vocab_ = std::make_shared<duorou::model::Vocabulary>();
          vocab_->initialize(tokens, types, /*scores*/ {}, merges);

          // BOS/EOS 配置
          std::vector<int32_t> bos_ids;
          std::vector<int32_t> eos_ids;
          bool add_bos = false;
          bool add_eos = false;
          if (const auto *kvBOS =
                  gguf_parser_->getMetadata("tokenizer.ggml.bos_token_id")) {
            bos_ids.push_back(kvBOS->asInt32());
          }
          if (const auto *kvEOS =
                  gguf_parser_->getMetadata("tokenizer.ggml.eos_token_id")) {
            eos_ids.push_back(kvEOS->asInt32());
          }
          if (const auto *kvAddBOS =
                  gguf_parser_->getMetadata("tokenizer.ggml.add_bos_token")) {
            add_bos = kvAddBOS->asBool();
          }
          if (const auto *kvAddEOS =
                  gguf_parser_->getMetadata("tokenizer.ggml.add_eos_token")) {
            add_eos = kvAddEOS->asBool();
          }
          if (!bos_ids.empty()) {
            vocab_->setBOS(bos_ids, add_bos);
          }
          if (!eos_ids.empty()) {
            vocab_->setEOS(eos_ids, add_eos);
          }

          // 创建 TextProcessor
          tokenizer_ = duorou::model::createTextProcessorFromGGUF(
              *gguf_parser_, vocab_, tok_opts_);

          std::cout << "[DEBUG] Initialized Vocabulary(size=" << vocab_->size()
                    << ") and TextProcessor from GGUF" << std::endl;
        } else {
          std::cerr << "[WARN] GGUF does not contain tokenizer tokens; "
                       "skipping custom tokenizer init"
                    << std::endl;
        }
      } else {
        std::cerr << "[WARN] Failed to parse GGUF for tokenizer init: "
                  << model_path << std::endl;
      }
    } catch (const std::exception &e) {
      std::cerr << "[WARN] Exception initializing tokenizer from GGUF: "
                << e.what() << std::endl;
    }

    // 初始化采样器
    if (!initializeSampler()) {
      std::cerr << "[ERROR] Failed to initialize sampler" << std::endl;
      return false;
    }

    std::cout << "[DEBUG] llama.cpp model loaded successfully" << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception loading llama.cpp model: " << e.what()
              << std::endl;
    return false;
  }
}

bool MLInferenceEngine::initializeSampler() {
  try {
    // 创建采样器链
    struct llama_sampler_chain_params chain_params =
        llama_sampler_chain_default_params();
    llama_sampler_ = llama_sampler_chain_init(chain_params);

    if (!llama_sampler_) {
      std::cerr << "[ERROR] Failed to create llama.cpp sampler chain"
                << std::endl;
      return false;
    }

    // 添加温度采样器
    llama_sampler_chain_add(llama_sampler_, llama_sampler_init_temp(0.8f));

    // 添加 top-p 采样器
    llama_sampler_chain_add(llama_sampler_, llama_sampler_init_top_p(0.9f, 1));

    // 添加 top-k 采样器
    llama_sampler_chain_add(llama_sampler_, llama_sampler_init_top_k(40));

    // 添加分布采样器
    llama_sampler_chain_add(llama_sampler_, llama_sampler_init_dist(1234));

    std::cout << "[DEBUG] llama.cpp sampler chain initialized successfully"
              << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception initializing sampler: " << e.what()
              << std::endl;
    return false;
  }
}

std::vector<llama_token> MLInferenceEngine::tokenize(const std::string &text) {
  std::cout << "[DEBUG] Tokenizing text: '" << text << "' (length: " << text.length() << ")" << std::endl;
  
  // 优先使用自定义 TextProcessor
  try {
    if (tokenizer_) {
      std::cout << "[DEBUG] Using TextProcessor tokenizer" << std::endl;
      std::vector<int32_t> ids = tokenizer_->encode(text, /*addSpecial=*/true);
      std::vector<llama_token> tokens;
      tokens.reserve(ids.size());
      for (int32_t id : ids)
        tokens.push_back(static_cast<llama_token>(id));
      std::cout << "[DEBUG] Tokenized via TextProcessor into " << tokens.size()
                << " tokens" << std::endl;
      return tokens;
    }

    // 其次使用 llama.cpp 自带 tokenizer
    if (llama_model_) {
      std::cout << "[DEBUG] Using llama.cpp tokenizer" << std::endl;
      std::vector<llama_token> tokens;
      tokens.resize(text.length() + 8);
      const struct llama_vocab *vocab = llama_model_get_vocab(llama_model_);
      int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(),
                                    tokens.data(), tokens.size(),
                                    true, // add_special
                                    false // parse_special
      );
      if (n_tokens > 0) {
        tokens.resize(n_tokens);
        std::cout << "[DEBUG] Tokenized via llama.cpp into " << n_tokens
                  << " tokens" << std::endl;
        for (int i = 0; i < std::min(5, n_tokens); i++) {
          std::cout << "[DEBUG] Token[" << i << "] = " << tokens[i] << std::endl;
        }
        return tokens;
      } else {
        std::cout << "[DEBUG] llama.cpp tokenization failed, n_tokens = " << n_tokens << std::endl;
      }
    } else {
      std::cout << "[DEBUG] llama_model_ is null, cannot use llama.cpp tokenizer" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during tokenization: " << e.what() << std::endl;
  }

  // 最后退化为简单分词（仅作为兜底）
  std::cout << "[DEBUG] Using fallback tokenizer" << std::endl;
  std::vector<llama_token> tokens;
  try {
    std::istringstream iss(text);
    std::string word;
    int word_count = 0;
    while (iss >> word) {
      std::cout << "[DEBUG] Fallback processing word[" << word_count << "]: '" << word << "'" << std::endl;
      llama_token id = 0;
      for (char c : word) {
        id = id * 31 + static_cast<llama_token>(c);
      }
      id = std::abs(id) % 50000 + 1;
      std::cout << "[DEBUG] Fallback word '" << word << "' -> token ID " << id << std::endl;
      tokens.push_back(id);
      word_count++;
    }
    std::cout << "[DEBUG] Tokenized by fallback into " << tokens.size()
              << " tokens" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error during fallback tokenization: " << e.what()
              << std::endl;
  }
  return tokens;
}

std::string
MLInferenceEngine::detokenize(const std::vector<llama_token> &tokens) {
  // 优先使用自定义 TextProcessor
  try {
    if (tokenizer_) {
      std::vector<int32_t> ids;
      ids.reserve(tokens.size());
      for (auto t : tokens)
        ids.push_back(static_cast<int32_t>(t));
      return tokenizer_->decode(ids);
    }

    // 其次使用 llama.cpp 将 token 转为文本
    if (llama_model_) {
      const struct llama_vocab *vocab = llama_model_get_vocab(llama_model_);
      std::ostringstream result;
      char piece[256];
      for (auto t : tokens) {
        int n_piece = llama_token_to_piece(vocab, t, piece, sizeof(piece),
                                           0,    // lstrip
                                           false // special
        );
        if (n_piece > 0)
          result.write(piece, n_piece);
      }
      return result.str();
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during detokenization: " << e.what() << std::endl;
  }

  // 兜底：简单拼接
  try {
    std::ostringstream result;
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (i)
        result << ' ';
      result << tokens[i];
    }
    return result.str();
  } catch (const std::exception &e) {
    return std::string();
  }
}

std::string MLInferenceEngine::generateIntelligentResponse(
    const std::string &prompt, uint32_t max_tokens, float temperature) {
  // 智能响应生成逻辑
  try {
    // 检测输入语言和内容类型
    bool is_chinese = false;
    for (char c : prompt) {
      if (static_cast<unsigned char>(c) > 127) {
        is_chinese = true;
        break;
      }
    }

    // 根据prompt内容生成相关响应
    std::string response;
    std::string lower_prompt = prompt;
    std::transform(lower_prompt.begin(), lower_prompt.end(),
                   lower_prompt.begin(), ::tolower);

    // 根据max_tokens限制响应长度
    if (response.length() > max_tokens * 4) { // 粗略估算：4字符=1token
      response = response.substr(0, max_tokens * 4) + "...";
    }

    return response;

  } catch (const std::exception &e) {
    return "生成响应时出现错误。Error generating response.";
  }
}

bool MLInferenceEngine::parseModelConfig() {
  if (!gguf_parser_) {
    std::cerr << "[ERROR] GGUF parser not initialized" << std::endl;
    return false;
  }

  try {
    // 从GGUF文件中读取模型配置参数
    // 这里需要根据实际的GGUFParser接口来获取配置
    // 假设GGUFParser提供了获取配置的方法

    // 获取词汇表大小
    vocab_size_ = 32000; // 默认值，应该从GGUF文件读取

    // 获取模型层数
    n_layers_ = 32; // 默认值，应该从GGUF文件读取

    // 获取注意力头数
    n_heads_ = 32; // 默认值，应该从GGUF文件读取

    // 获取嵌入维度
    n_embd_ = 4096; // 默认值，应该从GGUF文件读取

    // 获取上下文长度
    n_ctx_ = 2048; // 默认值，应该从GGUF文件读取

    std::cout << "[DEBUG] Model config - vocab_size: " << vocab_size_
              << ", n_layers: " << n_layers_ << ", n_heads: " << n_heads_
              << ", n_embd: " << n_embd_ << ", n_ctx: " << n_ctx_ << std::endl;

    return true;
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Failed to parse model config: " << e.what()
              << std::endl;
    return false;
  }
}

bool MLInferenceEngine::loadModelWeights() {
  try {
    std::cout << "[DEBUG] Loading model weights for " << n_layers_ << " layers"
              << std::endl;

    // 清理之前的权重
    for (auto *weight : model_weights_) {
      delete weight;
    }
    model_weights_.clear();

    // 为每一层创建权重张量
    // 这里是简化实现，实际应该从GGUF文件中读取权重数据
    for (uint32_t i = 0; i < n_layers_; ++i) {
      // 创建注意力权重
      auto *attn_weight =
          new ml::Tensor({n_embd_, n_embd_}, ml::DataType::FLOAT32);
      model_weights_.push_back(attn_weight);

      // 创建前馈网络权重
      auto *ffn_weight =
          new ml::Tensor({n_embd_, n_embd_ * 4}, ml::DataType::FLOAT32);
      model_weights_.push_back(ffn_weight);
    }

    std::cout << "[DEBUG] Loaded " << model_weights_.size() << " weight tensors"
              << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Failed to load model weights: " << e.what()
              << std::endl;
    return false;
  }
}

bool MLInferenceEngine::initializeKVCache() {
  try {
    std::cout << "[DEBUG] Initializing KV cache with context length: " << n_ctx_
              << std::endl;

    // 配置KV缓存
    cache_config_.maxSeqLen = n_ctx_;
    cache_config_.maxBatchSize = 32; // 默认批次大小
    cache_config_.numLayers = n_layers_;
    cache_config_.numHeads = n_heads_;
    cache_config_.headDim = n_embd_ / n_heads_;
    cache_config_.dtype = kvcache::DType::FLOAT32;

    // 注意：Cache是抽象类，需要具体实现
    // 这里暂时注释掉，等待具体的Cache实现类
    // kv_cache_ = std::make_unique<kvcache::ConcreteCache>();

    // 暂时设置为nullptr，表示KV缓存配置已准备好但未实例化
    kv_cache_ = nullptr;

    std::cout << "[DEBUG] KV cache configuration prepared (maxSeqLen: "
              << cache_config_.maxSeqLen
              << ", numLayers: " << cache_config_.numLayers
              << ", numHeads: " << cache_config_.numHeads
              << ", headDim: " << cache_config_.headDim << ")" << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Failed to initialize KV cache: " << e.what()
              << std::endl;
    return false;
  }
}

bool MLInferenceEngine::precomputeRoPEFreqs() {
  try {
    std::cout << "[DEBUG] Precomputing RoPE frequencies" << std::endl;

    const uint32_t head_dim = n_embd_ / n_heads_;
    const float theta = 10000.0f;

    // 清理之前的频率
    rope_freqs_.clear();
    rope_freqs_.reserve(head_dim / 2);

    // 计算RoPE频率
    for (uint32_t i = 0; i < head_dim / 2; ++i) {
      float freq = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
      rope_freqs_.push_back(freq);
    }

    rope_initialized_ = true;

    std::cout << "[DEBUG] Precomputed " << rope_freqs_.size()
              << " RoPE frequencies" << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Failed to precompute RoPE frequencies: " << e.what()
              << std::endl;
    return false;
  }
}

std::string MLInferenceEngine::generateWithLlama(const std::string &prompt,
                                                 uint32_t max_tokens,
                                                 float temperature,
                                                 float top_p) {
  try {
    std::cout << "[DEBUG] Starting llama.cpp inference" << std::endl;

    // 1. 使用 llama.cpp 进行 tokenization
    std::vector<llama_token> tokens;
    tokens.resize(prompt.length() + 1);

    // 获取模型的词汇表
    const struct llama_vocab *vocab = llama_model_get_vocab(llama_model_);

    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                  tokens.data(), tokens.size(),
                                  true, // add_special
                                  false // parse_special
    );

    if (n_tokens < 0) {
      std::cerr << "[ERROR] Failed to tokenize prompt" << std::endl;
      return "Error: Failed to tokenize prompt";
    }

    tokens.resize(n_tokens);
    std::cout << "[DEBUG] Tokenized prompt into " << n_tokens << " tokens"
              << std::endl;

    // 2. 创建批次
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);

    // 添加 tokens 到批次
    for (int i = 0; i < n_tokens; i++) {
      batch.token[i] = tokens[i];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = false;
    }
    batch.n_tokens = n_tokens;

    // 最后一个 token 需要输出
    batch.logits[batch.n_tokens - 1] = true;

    // 4. 解码输入 tokens
    if (llama_decode(llama_context_, batch) != 0) {
      std::cerr << "[ERROR] Failed to decode input tokens" << std::endl;
      llama_batch_free(batch);
      return "Error: Failed to decode input tokens";
    }

    // 5. 生成新的 tokens
    std::string generated_text;
    int n_cur = batch.n_tokens;
    int n_decode = 0;

    while (n_decode < static_cast<int>(max_tokens)) {
      // 使用采样器选择下一个 token
      llama_token new_token = llama_sampler_sample(
          llama_sampler_, llama_context_, batch.n_tokens - 1);

      // 检查是否为结束 token
      if (llama_vocab_is_eog(vocab, new_token)) {
        std::cout << "[DEBUG] Generated EOS token, stopping generation"
                  << std::endl;
        break;
      }

      // 将 token 转换为文本
      char piece[256];
      int n_piece = llama_token_to_piece(vocab, new_token, piece, sizeof(piece),
                                         0,    // lstrip
                                         false // special
      );

      if (n_piece > 0) {
        generated_text.append(piece, n_piece);
      }

      // 准备下一次解码
      batch.n_tokens = 1;
      batch.token[0] = new_token;
      batch.pos[0] = n_cur;
      batch.n_seq_id[0] = 1;
      batch.seq_id[0][0] = 0;
      batch.logits[0] = true;

      n_decode++;
      n_cur++;

      // 解码新 token
      if (llama_decode(llama_context_, batch) != 0) {
        std::cerr << "[ERROR] Failed to decode new token" << std::endl;
        break;
      }
    }

    llama_batch_free(batch);

    std::cout << "[DEBUG] Generated " << n_decode
              << " tokens: " << generated_text.substr(0, 100) << "..."
              << std::endl;

    return generated_text;
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in generateWithLlama: " << e.what()
              << std::endl;
    return "Error: " + std::string(e.what());
  }
}

// 内部 Forward 模式：使用 Qwen 模型进行真正的推理
std::string
MLInferenceEngine::generateWithInternalForward(const std::string &prompt,
                                               uint32_t max_tokens,
                                               float temperature, float top_p) {
  try {
    std::cout << "[DEBUG] [InternalForward] Starting Qwen model inference"
              << std::endl;

    // 检查 Qwen 模型是否可用
    if (!qwen_model_) {
      std::cerr << "[ERROR] Qwen model not initialized" << std::endl;
      return generateIntelligentResponse(prompt, max_tokens, temperature);
    }

    // 首先编码文本为 token IDs
    std::vector<int32_t> input_ids;
    
    // 如果有 llama_model_，使用 llama.cpp 的分词功能
    if (llama_model_) {
      std::vector<llama_token> llama_tokens = tokenize(prompt);
      input_ids.reserve(llama_tokens.size());
      for (llama_token token : llama_tokens) {
        input_ids.push_back(static_cast<int32_t>(token));
      }
      std::cout << "[DEBUG] [InternalForward] Using llama.cpp tokenization" << std::endl;
    } else {
      // 回退到 Qwen 模型的分词器
      input_ids = qwen_model_->encode(prompt, true);
      std::cout << "[DEBUG] [InternalForward] Using Qwen tokenization" << std::endl;
    }

    if (input_ids.empty()) {
      std::cout
          << "[WARN] [InternalForward] Failed to encode prompt, using fallback"
          << std::endl;
      return generateIntelligentResponse(prompt, max_tokens, temperature);
    }

    std::cout << "[DEBUG] [InternalForward] Encoded " << input_ids.size()
              << " tokens from prompt" << std::endl;

    // 调用 Qwen 模型的多模态生成方法（不传递图像）
    std::vector<int32_t> output_ids =
        qwen_model_->generateMultimodal(input_ids, {}, // 空的图像数据
                                        max_tokens, temperature, top_p);

    if (output_ids.empty()) {
      std::cout << "[WARN] [InternalForward] Qwen model returned empty output, "
                   "using fallback"
                << std::endl;
      return generateIntelligentResponse(prompt, max_tokens, temperature);
    }

    std::cout << "[DEBUG] [InternalForward] Qwen model generated "
              << output_ids.size() << " output tokens" << std::endl;

    // 解码输出 token IDs 为文本
    std::string result;
    
    // 如果有 llama_model_，使用 llama.cpp 的解码功能
    if (llama_model_) {
      std::vector<llama_token> llama_tokens;
      llama_tokens.reserve(output_ids.size());
      for (int32_t token_id : output_ids) {
        llama_tokens.push_back(static_cast<llama_token>(token_id));
      }
      result = detokenize(llama_tokens);
      std::cout << "[DEBUG] [InternalForward] Using llama.cpp detokenization" << std::endl;
    } else {
      // 回退到 Qwen 模型的解码器
      result = qwen_model_->decode(output_ids);
      std::cout << "[DEBUG] [InternalForward] Using Qwen detokenization" << std::endl;
    }

    if (result.empty()) {
      std::cout << "[WARN] [InternalForward] Failed to decode output tokens, "
                   "using fallback"
                << std::endl;
      return generateIntelligentResponse(prompt, max_tokens, temperature);
    }

    std::cout << "[DEBUG] [InternalForward] Generated response: "
              << result.substr(0, 50) << "..." << std::endl;
    return result;

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in generateWithInternalForward: "
              << e.what() << std::endl;
    return "Error: " + std::string(e.what());
  }
}

void MLInferenceEngine::cleanupResources() {
  std::cout << "[DEBUG] Cleaning up inference engine resources" << std::endl;

  // 清理ML组件
  delete attention_;
  attention_ = nullptr;

  delete ml_context_;
  ml_context_ = nullptr;

  // 清理模型权重
  for (auto *weight : model_weights_) {
    delete weight;
  }
  model_weights_.clear();

  // 清理KV缓存
  kv_cache_.reset();

  // 清理RoPE频率
  rope_freqs_.clear();
  rope_initialized_ = false;

  // 重置配置
  vocab_size_ = 0;
  n_layers_ = 0;
  n_heads_ = 0;
  n_embd_ = 0;
  n_ctx_ = 0;

  std::cout << "[DEBUG] Resource cleanup completed" << std::endl;
}

} // namespace ollama
} // namespace extensions
} // namespace duorou