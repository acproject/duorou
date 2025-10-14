// Include standard library headers first
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// Include the actual llama.cpp header files
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include "../../../third_party/llama.cpp/include/llama.h"
#include "ggml-cpu.h"
#include "ggml.h"

// Include project headers after third-party headers
#include "../../kvcache/causal.h"
#include "../../ml/backend/backend.h"
#include "../../ml/context.h"
#include "../../ml/nn/attention.h"
#include "../../ml/tensor.h"
#include "inference_engine.h"
#include "ollama_model_manager.h"

#include "llama-model.h"

namespace duorou {
namespace extensions {
namespace ollama {
// Simple whitelist: which architectures to handle with llama.cpp
static bool isSupportedByLlamaCpp(const std::string &arch_raw) {
  std::string arch = arch_raw;
  std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);

  // Special-case: allow Qwen2.5-VL family only when explicitly enabled
  // Recognized aliases for Qwen2.5-VL
  static const std::unordered_set<std::string> qwen25vl_aliases = {
      "qwen25vl", "qwen2.5vl", "qwen-2.5vl", "qwen2_5vl", "qwen-2_5vl"};

  // If the architecture name mentions Qwen or VL, only allow if it's Qwen2.5-VL
  if (arch.find("qwen") != std::string::npos || arch.find("vl") != std::string::npos) {
    for (const auto &alias : qwen25vl_aliases) {
      if (arch.find(alias) != std::string::npos) {
        const char *env = std::getenv("DUOROU_ENABLE_LLAMA_QWEN25VL");
        bool enabled = env && std::string(env) == "1";
        return enabled; // require opt-in to use llama.cpp for Qwen2.5-VL
      }
    }
    return false; // other Qwen/VL variants are not supported by vanilla llama.cpp
  }

  // Allowed architecture keywords (expand as needed, but keep precise)
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
      head_dim_q_(0), head_dim_k_(0),
      rope_freqs_(), rope_initialized_(false), llama_model_(nullptr),
      llama_context_(nullptr), llama_sampler_(nullptr),
      use_llama_backend_(false) {
  std::cout << "[DEBUG] MLInferenceEngine constructor called with model_id: "
            << model_id << std::endl;
}

MLInferenceEngine::~MLInferenceEngine() {
  std::cout << "[DEBUG] MLInferenceEngine destructor called" << std::endl;

  // Release llama.cpp resources
  if (llama_sampler_) {
    llama_sampler_free(llama_sampler_);
    llama_sampler_ = nullptr;
  }

  if (llama_context_) {
    llama_free(llama_context_);
    llama_context_ = nullptr;
  }

  if (llama_model_) {
    llama_model_free(llama_model_);
    llama_model_ = nullptr;
  }

  cleanupResources();

  // Important: do NOT aggressively cleanup ML backends here.
  // Some tensors may still be alive and hold a Backend* for deallocation.
  // Premature backend cleanup can cause dangling pointers and crash in
  // duorou::ml::Tensor::deallocate. Let process exit handle final cleanup,
  // or allow explicit cleanup via environment flag if needed.
  const char *cleanup_flag = std::getenv("DUOROU_CLEANUP_BACKENDS");
  if (cleanup_flag && std::string(cleanup_flag) == std::string("1")) {
    std::cout << "[DEBUG] Cleaning up ML backends (DUOROU_CLEANUP_BACKENDS=1)"
              << std::endl;
    duorou::ml::BackendManager::getInstance().cleanup();
  } else {
    std::cout << "[DEBUG] Skipping ML backend cleanup to avoid dangling "
                 "Backend* in tensors"
              << std::endl;
  }
}

bool MLInferenceEngine::initialize() {
  std::cout << "[DEBUG] MLInferenceEngine::initialize called" << std::endl;

  if (initialized_) {
    std::cout << "[DEBUG] Already initialized, returning true" << std::endl;
    return true;
  }

  try {
    // 获取模型路径
    OllamaModelManager &manager = GlobalModelManager::getInstance();
    auto model_info = manager.getModelInfo(model_id_);
    if (!model_info || model_info->file_path.empty()) {
      std::cerr << "[ERROR] MLInferenceEngine: Model not found or file path empty in OllamaModelManager: "
                << model_id_ << std::endl;
      initialized_ = false;
      return false;
    }

    model_path_ = model_info->file_path;

    // 首先解析 GGUF 以确定架构与内部前向支持
    gguf_parser_ = std::make_unique<GGUFParser>();
    if (!gguf_parser_->parseFile(model_path_)) {
      std::cerr << "[ERROR] Failed to parse GGUF for model: " << model_path_ << std::endl;
      initialized_ = false;
      return false;
    }

    std::string arch_for_check = gguf_parser_->getArchitecture().name;
    std::cout << "[DEBUG] Parsed GGUF architecture: " << arch_for_check << std::endl;

    // 如果 llama.cpp 不支持该架构，且内部前向支持，则走内部前向路径
    bool llama_supported = true;
    if (!arch_for_check.empty() && !isSupportedByLlamaCpp(arch_for_check)) {
      llama_supported = false;
    }

    bool internal_supported = checkInternalForwardSupport();

    if (!llama_supported && internal_supported) {
      std::cout << "[INFO] Llama.cpp unsupported for arch, enabling InternalForward mode" << std::endl;
      use_llama_backend_ = false;

      // 先解析模型配置，提供注意力层与 KV 缓存的维度参数
      if (!parseModelConfig()) {
        std::cerr << "[ERROR] Failed to parse model config from GGUF metadata" << std::endl;
        initialized_ = false;
        return false;
      }

      // 初始化 ML 上下文与注意力模块（使用解析到的维度参数）
      ml_context_ = new duorou::ml::Context();
      attention_ = new duorou::ml::nn::MultiHeadAttention(
          static_cast<int64_t>(n_embd_),
          static_cast<int64_t>(n_heads_),
          static_cast<int64_t>(n_kv_heads_ > 0 ? n_kv_heads_ : -1),
          true,
          0.0f);

      // 配置 RoPE（基于 GGUF 元数据）
      if (attention_) {
        duorou::ml::nn::RoPEConfig rc{};
        rc.dimension = static_cast<int64_t>(rope_dim_ > 0 ? rope_dim_ : head_dim_q_);
        rc.theta = rope_freq_base_ > 0.0f ? rope_freq_base_ : 10000.0f;
        const auto arch = gguf_parser_->getArchitecture();
        rc.sections.clear();
        for (auto v : arch.rope_dimension_sections) rc.sections.push_back(static_cast<int64_t>(v));
        attention_->setRoPEConfig(rc);
      }

      // 初始化 KV 缓存（根据解析到的 n_layers/n_heads/head_dim 等）
      if (!initializeKVCache()) {
        std::cerr << "[WARN] Failed to initialize KV cache for InternalForward" << std::endl;
      }

      // 创建并初始化 Qwen 模型（优先使用 GGUF 中的词汇表与分词器）
      try {
        if (!vocab_) {
          vocab_ = duorou::model::createVocabularyFromGGUF(*gguf_parser_);
        }
        if (!tokenizer_) {
          tokenizer_ = duorou::model::createTextProcessorFromGGUF(*gguf_parser_, vocab_, tok_opts_);
        }
      } catch (const std::exception &e) {
        std::cerr << "[WARN] Failed to initialize vocab/tokenizer from GGUF: " << e.what() << std::endl;
      }

      try {
        // 使用可选外部词汇表创建多模态模型
        if (vocab_) {
          qwen_model_.reset(static_cast<duorou::model::QwenMultimodalModel*>(
              duorou::model::createQwenMultimodalModel(model_path_, vocab_).release()));
        } else {
          qwen_model_.reset(static_cast<duorou::model::QwenMultimodalModel*>(
              duorou::model::createQwenMultimodalModel(model_path_).release()));
        }
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] Failed to create QwenMultimodalModel: " << e.what() << std::endl;
        initialized_ = false;
        return false;
      }

      if (!qwen_model_) {
        std::cerr << "[ERROR] Qwen model creation returned null" << std::endl;
        initialized_ = false;
        return false;
      }

      // 加载模型权重或配置
      if (!qwen_model_->loadModel(model_path_)) {
        std::cerr << "[ERROR] Failed to load Qwen model from GGUF: " << model_path_ << std::endl;
        initialized_ = false;
        return false;
      }

      initialized_ = true;
      std::cout << "[DEBUG] MLInferenceEngine initialized successfully (InternalForward)" << std::endl;
      return true;
    }

    // 默认或支持时：尝试使用 llama.cpp 后端
    use_llama_backend_ = true;
    llama_backend_init();

    if (!loadLlamaModel(model_path_)) {
      std::cerr << "[ERROR] MLInferenceEngine: Failed to load llama.cpp model from "
                << model_path_ << std::endl;
      // 直接回退到内部前向（如果支持）
      use_llama_backend_ = false;
      if (internal_supported) {
        std::cout << "[INFO] Falling back to InternalForward mode after llama load failure" << std::endl;
        
        // 解析模型配置，确保维度就绪
        if (!parseModelConfig()) {
          std::cerr << "[ERROR] Failed to parse model config from GGUF metadata" << std::endl;
          initialized_ = false;
          return false;
        }

        // 初始化 ML 组件与注意力模块
        ml_context_ = new duorou::ml::Context();
        attention_ = new duorou::ml::nn::MultiHeadAttention(
            static_cast<int64_t>(n_embd_),
            static_cast<int64_t>(n_heads_),
            static_cast<int64_t>(n_kv_heads_ > 0 ? n_kv_heads_ : -1),
            true,
            0.0f);
        // 配置 RoPE（基于 GGUF 元数据）
        if (attention_) {
          duorou::ml::nn::RoPEConfig rc{};
          rc.dimension = static_cast<int64_t>(rope_dim_ > 0 ? rope_dim_ : head_dim_q_);
          rc.theta = rope_freq_base_ > 0.0f ? rope_freq_base_ : 10000.0f;
          const auto arch = gguf_parser_->getArchitecture();
          rc.sections.clear();
          for (auto v : arch.rope_dimension_sections) rc.sections.push_back(static_cast<int64_t>(v));
          attention_->setRoPEConfig(rc);
        }
        if (!initializeKVCache()) {
          std::cerr << "[WARN] Failed to initialize KV cache for InternalForward" << std::endl;
        }
        // 词汇表和分词器
        try {
          if (!vocab_) {
            vocab_ = duorou::model::createVocabularyFromGGUF(*gguf_parser_);
          }
          if (!tokenizer_) {
            tokenizer_ = duorou::model::createTextProcessorFromGGUF(*gguf_parser_, vocab_, tok_opts_);
          }
        } catch (const std::exception &e) {
          std::cerr << "[WARN] Failed to initialize vocab/tokenizer from GGUF: " << e.what() << std::endl;
        }
        // 创建 Qwen 模型
        try {
          if (vocab_) {
            qwen_model_.reset(static_cast<duorou::model::QwenMultimodalModel*>(
                duorou::model::createQwenMultimodalModel(model_path_, vocab_).release()));
          } else {
            qwen_model_.reset(static_cast<duorou::model::QwenMultimodalModel*>(
                duorou::model::createQwenMultimodalModel(model_path_).release()));
          }
        } catch (const std::exception &e) {
          std::cerr << "[ERROR] Failed to create QwenMultimodalModel: " << e.what() << std::endl;
          initialized_ = false;
          return false;
        }
        if (!qwen_model_ || !qwen_model_->loadModel(model_path_)) {
          std::cerr << "[ERROR] Failed to initialize Qwen model during fallback" << std::endl;
          initialized_ = false;
          return false;
        }
        if (!parseModelConfig()) {
          std::cerr << "[WARN] Failed to parse model config from GGUF metadata" << std::endl;
        }
        initialized_ = true;
        std::cout << "[DEBUG] InternalForward initialization succeeded after llama load failure" << std::endl;
        return true;
      }
      initialized_ = false;
      return false;
    }

    // 初始化采样器
    if (!initializeSampler()) {
      std::cerr << "[ERROR] Failed to initialize llama sampler" << std::endl;
      initialized_ = false;
      return false;
    }

    initialized_ = true;
    std::cout << "[DEBUG] MLInferenceEngine initialized successfully (llama.cpp)" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception during initialize: " << e.what() << std::endl;
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
    // 根据后端选择生成
    if (use_llama_backend_) {
      return generateWithLlama(prompt, max_tokens, temperature, top_p);
    }
    // // 内部前向优先用于 Qwen 架构
    // if (checkInternalForwardSupport()) {
    //   return generateWithInternalForward(prompt, max_tokens, temperature, top_p);
    // }
    // 兜底：尝试 GGLM 简化路径
    return generateWithGGLM(prompt, max_tokens, temperature, top_p);

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in generateText: " << e.what() << std::endl;
    return "Error: " + std::string(e.what());
  }
}

bool MLInferenceEngine::isReady() const {
  if (!initialized_) return false;
  if (use_llama_backend_) {
    return llama_model_ != nullptr && llama_context_ != nullptr && llama_sampler_ != nullptr;
  }
  // 内部前向模式的就绪检查：需要 qwen_model_ 与 ml_context_ 存在
  return qwen_model_ != nullptr && ml_context_ != nullptr;
}

std::string MLInferenceEngine::processText(const std::string &text) {
  // Simple text processing logic
  return "Processed: " + text;
}

bool MLInferenceEngine::loadModel(const std::string &model_path) {
  try {
    std::cout << "[DEBUG] Loading model from: " << model_path << std::endl;

    // Step 1: Create GGUF parser
    std::cout << "[DEBUG] Step 1: Creating GGUF parser" << std::endl;
    gguf_parser_ = std::make_unique<GGUFParser>();

    // Step 2: Parse GGUF file
    std::cout << "[DEBUG] Step 2: Parsing GGUF file" << std::endl;
    if (!gguf_parser_->parseFile(model_path)) {
      std::cerr << "Failed to parse GGUF model: " << model_path << std::endl;
      return false;
    }

    // Parse model configuration
    if (!parseModelConfig()) {
      std::cerr << "Failed to parse model configuration" << std::endl;
      return false;
    }

    // Step 3: Load model weights
    std::cout << "[DEBUG] Step 3: Loading model weights" << std::endl;
    if (!loadModelWeights()) {
      std::cerr << "Failed to load model weights" << std::endl;
      return false;
    }

    // Step 4: Initialize KV cache
    std::cout << "[DEBUG] Step 4: Initializing KV cache" << std::endl;
    if (!initializeKVCache()) {
      std::cerr << "Failed to initialize KV cache" << std::endl;
      return false;
    }

    // Step 5: Precompute RoPE frequencies
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

    // Check architecture support before attempting to load with llama.cpp
    std::string arch_for_check;
    try {
      if (gguf_parser_ && !gguf_parser_->getArchitecture().name.empty()) {
        arch_for_check = gguf_parser_->getArchitecture().name;
      } else {
        auto tmp = std::make_unique<GGUFParser>();
        if (tmp->parseFile(model_path)) {
          arch_for_check = tmp->getArchitecture().name;
        }
      }
    } catch (...) {
    }
    if (!arch_for_check.empty() && !isSupportedByLlamaCpp(arch_for_check)) {
      std::cerr << "[WARN] llama.cpp does not support architecture '"
                << arch_for_check << "'; skipping llama.cpp load" << std::endl;
      return false;
    }

    // Set model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU only for now

    // Load model
    llama_model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!llama_model_) {
      std::cerr << "[ERROR] Failed to load llama.cpp model from: " << model_path
                << std::endl;
      return false;
    }

    // Set context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;  // Context length
    ctx_params.n_batch = 512; // Batch size
    ctx_params.n_threads = 4; // Number of threads

    // Create context
    llama_context_ = llama_init_from_model(llama_model_, ctx_params);
    if (!llama_context_) {
      std::cerr << "[ERROR] Failed to create llama.cpp context" << std::endl;
      return false;
    }

    // Build Vocabulary and TextProcessor based on GGUF metadata
    try {
      gguf_parser_ = std::make_unique<GGUFParser>();
      if (gguf_parser_->parseFile(model_path)) {
        // Read tokens
        std::vector<std::string> tokens;
        if (const auto *kvTokens =
                gguf_parser_->getMetadata("tokenizer.ggml.tokens")) {
          tokens = kvTokens->asStringArray();
        }
        // Read token types
        std::vector<int32_t> types;
        if (const auto *kvTypes =
                gguf_parser_->getMetadata("tokenizer.ggml.token_type")) {
          types = kvTypes->asInt32Array();
        }
        if (types.empty() && !tokens.empty()) {
          types.assign(tokens.size(), duorou::model::TOKEN_TYPE_NORMAL);
        }
        // Read merges
        std::vector<std::string> merges;
        if (const auto *kvMerges =
                gguf_parser_->getMetadata("tokenizer.ggml.merges")) {
          merges = kvMerges->asStringArray();
        }

        if (!tokens.empty()) {
          vocab_ = std::make_shared<duorou::model::Vocabulary>();
          vocab_->initialize(tokens, types, /*scores*/ {}, merges);

          // BOS/EOS configuration
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

          // Create TextProcessor
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

    // Initialize sampler
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
    // Create sampler chain
    struct llama_sampler_chain_params chain_params =
        llama_sampler_chain_default_params();
    llama_sampler_ = llama_sampler_chain_init(chain_params);

    if (!llama_sampler_) {
      std::cerr << "[ERROR] Failed to create llama.cpp sampler chain"
                << std::endl;
      return false;
    }

    // Add temperature sampler
    llama_sampler_chain_add(llama_sampler_, llama_sampler_init_temp(0.8f));

    // Add top-p sampler
    llama_sampler_chain_add(llama_sampler_, llama_sampler_init_top_p(0.9f, 1));

    // Add top-k sampler
    llama_sampler_chain_add(llama_sampler_, llama_sampler_init_top_k(40));

    // Add distribution sampler
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
  std::cout << "[DEBUG] Tokenizing text: '" << text
            << "' (length: " << text.length() << ")" << std::endl;

  // Prefer using custom TextProcessor
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

    // Next, use llama.cpp's built-in tokenizer
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
          std::cout << "[DEBUG] Token[" << i << "] = " << tokens[i]
                    << std::endl;
        }
        return tokens;
      } else {
        std::cout << "[DEBUG] llama.cpp tokenization failed, n_tokens = "
                  << n_tokens << std::endl;
      }
    } else {
      std::cout
          << "[DEBUG] llama_model_ is null, cannot use llama.cpp tokenizer"
          << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during tokenization: " << e.what() << std::endl;
  }

  // Finally, fall back to simple tokenization (as a last resort)
  std::cout << "[DEBUG] Using fallback tokenizer" << std::endl;
  std::vector<llama_token> tokens;
  try {
    std::istringstream iss(text);
    std::string word;
    int word_count = 0;
    while (iss >> word) {
      std::cout << "[DEBUG] Fallback processing word[" << word_count << "]: '"
                << word << "'" << std::endl;
      llama_token id = 0;
      for (char c : word) {
        id = id * 31 + static_cast<llama_token>(c);
      }
      id = std::abs(id) % 50000 + 1;
      std::cout << "[DEBUG] Fallback word '" << word << "' -> token ID " << id
                << std::endl;
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
  // Prefer using custom TextProcessor
  try {
    if (tokenizer_) {
      std::vector<int32_t> ids;
      ids.reserve(tokens.size());
      for (auto t : tokens)
        ids.push_back(static_cast<int32_t>(t));
      return tokenizer_->decode(ids);
    }

    // Next, use llama.cpp to convert tokens to text
    if (llama_model_) {
      const struct llama_vocab *vocab = llama_model_get_vocab(llama_model_);
      std::ostringstream result;
      char piece[256];
      for (auto t : tokens) {
        int32_t n_piece = llama_token_to_piece(vocab, t, piece, sizeof(piece),
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

  // Fallback: simple concatenation
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
  // Intelligent response generation logic
  try {
    // Detect input language and content type
    bool is_chinese = false;
    for (char c : prompt) {
      if (static_cast<unsigned char>(c) > 127) {
        is_chinese = true;
        break;
      }
    }

    // Generate relevant response based on prompt content
    std::string response;
    std::string lower_prompt = prompt;
    std::transform(lower_prompt.begin(), lower_prompt.end(),
                   lower_prompt.begin(), ::tolower);

    // Limit response length based on max_tokens
    if (response.length() >
        max_tokens * 4) { // Rough estimate: 4 characters = 1 token
      response = response.substr(0, max_tokens * 4) + "...";
    }

    return response;

  } catch (const std::exception &e) {
    return "Error generating response.";
  }
}

bool MLInferenceEngine::parseModelConfig() {
  if (!gguf_parser_) {
    std::cerr << "[ERROR] GGUF parser not initialized" << std::endl;
    return false;
  }

  try {
    // 1) 架构字段对齐
    const auto &arch = gguf_parser_->getArchitecture();

    // 直接映射基础参数
    n_layers_ = arch.block_count ? arch.block_count : n_layers_;
    n_heads_ = arch.attention_head_count ? arch.attention_head_count : n_heads_;
    n_kv_heads_ = arch.attention_head_count_kv
                      ? arch.attention_head_count_kv
                      : (n_kv_heads_ ? n_kv_heads_ : n_heads_);
    n_embd_ = arch.embedding_length ? arch.embedding_length : n_embd_;
    n_ctx_ = arch.context_length ? arch.context_length : n_ctx_;

    // 注意力每头维度（Q 与 K/V），若缺失则从 n_embd_/n_heads_ 回退
    head_dim_q_ = arch.attention_head_dim ? arch.attention_head_dim
                                          : (n_heads_ ? (n_embd_ / n_heads_) : 0);
    head_dim_k_ = arch.attention_head_dim_k ? arch.attention_head_dim_k
                                            : head_dim_q_;

    // RoPE 参数
    rope_dim_ = arch.rope_dimension_count
                    ? arch.rope_dimension_count
                    : head_dim_q_;
    rope_freq_base_ =
        arch.rope_freq_base > 0.0f ? arch.rope_freq_base : 10000.0f;

    // 2) 词表大小优先从 {arch}.vocab_size 读取，退化到 tokenizer.ggml.tokens
    // 长度
    uint32_t vocab_from_meta = 0;
    if (!arch.name.empty()) {
      std::string key_vs = arch.name + ".vocab_size";
      if (const auto *kvVS = gguf_parser_->getMetadata(key_vs)) {
        // GGUF 中 {arch}.vocab_size 通常为 uint32
        vocab_from_meta = kvVS->asUInt32();
      }
    }

    if (vocab_from_meta > 0) {
      vocab_size_ = vocab_from_meta;
    } else {
      // 回退：使用 tokenizer.ggml.tokens 的数量
      if (const auto *kvTokens =
              gguf_parser_->getMetadata("tokenizer.ggml.tokens")) {
        const auto tokens = kvTokens->asStringArray();
        if (!tokens.empty()) {
          vocab_size_ = static_cast<uint32_t>(tokens.size());
        }
      }
    }

    // 保底默认值（避免为 0 导致后续崩溃）
    if (vocab_size_ == 0)
      vocab_size_ = 32000;
    if (n_layers_ == 0)
      n_layers_ = 32;
    if (n_heads_ == 0)
      n_heads_ = 32;
    if (n_kv_heads_ == 0)
      n_kv_heads_ = n_heads_;
    if (n_embd_ == 0)
      n_embd_ = 4096;
    if (n_ctx_ == 0)
      n_ctx_ = 2048;
    if (rope_dim_ == 0)
      rope_dim_ = head_dim_q_;

    // 调试输出（含 RoPE mrope sections）
    std::cout << "[DEBUG] Parsed GGUF architecture: '" << arch.name << "'"
              << std::endl;
    if (!arch.rope_dimension_sections.empty()) {
      std::cout << "[DEBUG] RoPE mrope sections: ";
      for (size_t i = 0; i < arch.rope_dimension_sections.size(); ++i) {
        std::cout << arch.rope_dimension_sections[i]
                  << (i + 1 < arch.rope_dimension_sections.size() ? "," : "");
      }
      std::cout << std::endl;
    }

    std::cout << "[DEBUG] Model config - vocab_size: " << vocab_size_
              << ", n_layers: " << n_layers_ << ", n_heads: " << n_heads_
              << ", n_kv_heads: " << n_kv_heads_ << ", n_embd: " << n_embd_
              << ", n_ctx: " << n_ctx_ << ", head_dim_q: " << head_dim_q_
              << ", head_dim_k: " << head_dim_k_ << ", rope_dim: "
              << rope_dim_ << ", rope_freq_base: " << rope_freq_base_
              << std::endl;

    // 基本合法性检查
    if (n_layers_ == 0 || n_heads_ == 0 || n_embd_ == 0 || n_ctx_ == 0 ||
        vocab_size_ == 0) {
      std::cerr << "[ERROR] Invalid model configuration parsed from GGUF"
                << std::endl;
      return false;
    }

    return true;
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Failed to parse model config: " << e.what()
              << std::endl;
    return false;
  }
}

bool MLInferenceEngine::loadModelWeights() {
  try {
    std::cout << "[DEBUG] Loading model weights from GGUF" << std::endl;

    // Clean up previous weights (will be handled by cleanupResources uniqueness
    // too)
    for (auto *weight : model_weights_) {
      delete weight;
    }
    model_weights_.clear();

    if (!gguf_parser_) {
      std::cerr << "[ERROR] GGUF parser not initialized" << std::endl;
      return false;
    }

    // Map all tensors from GGUF into weight_map_
    if (!mapTensorWeights()) {
      std::cerr << "[ERROR] Failed to map tensor weights from GGUF"
                << std::endl;
      return false;
    }

    // Populate model_weights_ vector for convenience/legacy paths
    model_weights_.reserve(weight_map_.size());
    for (const auto &kv : weight_map_) {
      model_weights_.push_back(kv.second);
    }

    std::cout << "[DEBUG] Loaded " << model_weights_.size()
              << " tensors via GGUF mapping" << std::endl;

    // Validate weight ranges to detect potential loading issues
    std::cout << "[DEBUG] Validating weight ranges..." << std::endl;
    size_t checked_tensors = 0;
    size_t problematic_tensors = 0;

    for (const auto &kv : weight_map_) {
      const std::string &name = kv.first;
      duorou::ml::Tensor *tensor = kv.second;

      if (!tensor || tensor->numel() == 0)
        continue;

      // Sample a few values to check for obvious issues
      size_t sample_size = std::min(static_cast<size_t>(100),
                                    static_cast<size_t>(tensor->numel()));
      std::vector<float> sample_data(sample_size);
      tensor->copyToHost(sample_data.data(),
                         sample_data.size() * sizeof(float));

      auto minmax = std::minmax_element(sample_data.begin(), sample_data.end());
      float min_val = *minmax.first;
      float max_val = *minmax.second;

      bool has_nan = false, has_inf = false, all_zeros = true;
      for (float val : sample_data) {
        if (std::isnan(val))
          has_nan = true;
        if (std::isinf(val))
          has_inf = true;
        if (val != 0.0f)
          all_zeros = false;
      }

      if (has_nan || has_inf || all_zeros || max_val > 1000.0f ||
          min_val < -1000.0f) {
        std::cerr << "[WARN] Tensor '" << name << "' has suspicious values: "
                  << "range=[" << min_val << ", " << max_val << "], "
                  << "NaN=" << has_nan << ", Inf=" << has_inf
                  << ", AllZeros=" << all_zeros << std::endl;
        problematic_tensors++;
      }

      checked_tensors++;
      if (checked_tensors >= 10)
        break; // Limit checking to avoid excessive output
    }

    if (problematic_tensors > 0) {
      std::cerr << "[WARN] Found " << problematic_tensors
                << " tensors with suspicious values out of " << checked_tensors
                << " checked" << std::endl;
    } else {
      std::cout << "[DEBUG] Weight validation passed for " << checked_tensors
                << " tensors" << std::endl;
    }

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

    // Configure KV cache
    cache_config_.maxSeqLen = n_ctx_;
    cache_config_.maxBatchSize = 32; // Default batch size
    cache_config_.numLayers = n_layers_;
    // 对齐 KV 头数（GQA）：优先使用 n_kv_heads_
    cache_config_.numHeads = (n_kv_heads_ > 0 ? n_kv_heads_ : n_heads_);
    // KV 缓存的每头维度优先使用 K/V 维度，其次 Q 维度，最后回退到 n_embd_/n_heads_
    cache_config_.headDim = head_dim_k_ > 0
                                ? head_dim_k_
                                : (head_dim_q_ > 0 ? head_dim_q_
                                                   : (n_heads_ ? (n_embd_ / n_heads_) : 0));
    cache_config_.dtype = kvcache::DType::FLOAT32;

    // Adapter bridging ml::Backend to kvcache::Backend for backend-aware KV
    // cache allocations
    struct MLKVBackendAdapter : public duorou::kvcache::Backend {
      explicit MLKVBackendAdapter(duorou::ml::Backend *backend)
          : mlBackend(backend) {}
      void *allocate(size_t bytes) override {
        if (mlBackend)
          return mlBackend->allocate(bytes);
        return std::malloc(bytes);
      }
      void deallocate(void *ptr) override {
        if (!ptr)
          return;
        if (mlBackend)
          mlBackend->deallocate(ptr);
        else
          std::free(ptr);
      }
      void copy(void *dst, const void *src, size_t bytes) override {
        if (!dst || !src || bytes == 0)
          return;
        if (mlBackend)
          mlBackend->copyDeviceToDevice(dst, src, bytes);
        else
          std::memcpy(dst, src, bytes);
      }
      duorou::ml::Backend *mlBackend;
    };

    MLKVBackendAdapter kvAdapter(ml_context_ ? ml_context_->getBackend()
                                             : nullptr);
    kvcache::Context kvCtx(&kvAdapter);

    // Instantiate causal KV cache and initialize with backend-aware context
    kv_cache_ = std::make_unique<kvcache::CausalCache>();
    kv_cache_->init(kvCtx, cache_config_);

    std::cout << "[DEBUG] KV cache instantiated and initialized (type=Causal)"
              << ", maxSeqLen: " << cache_config_.maxSeqLen
              << ", numLayers: " << cache_config_.numLayers
              << ", numHeads(GQA): " << cache_config_.numHeads
              << ", head_dim_q: " << head_dim_q_
              << ", head_dim_k: " << head_dim_k_
              << ", headDim(cache): " << cache_config_.headDim << std::endl;
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

    if (n_heads_ == 0) {
      std::cerr << "[ERROR] n_heads_ is zero, cannot compute head_dim"
                << std::endl;
      return false;
    }

    const uint32_t head_dim = head_dim_q_ > 0 ? head_dim_q_ : (n_heads_ ? (n_embd_ / n_heads_) : 0);
    uint32_t rope_dim = rope_dim_ > 0 ? rope_dim_ : 0;
    bool rope_fallback = false;
    if (rope_dim == 0) {
      rope_dim = head_dim;
      rope_fallback = true;
      std::cerr << "[WARN] rope_dim is zero or missing; falling back to head_dim_q="
                << head_dim << std::endl;
    }
    const float theta = rope_freq_base_ > 0.0f ? rope_freq_base_ : 10000.0f;

    // Clean up previous frequencies
    rope_freqs_.clear();
    rope_freqs_.reserve(rope_dim / 2);

    // Calculate RoPE frequencies（与 GGUF 的 rope.dimension_count &
    // rope.freq_base 对齐）
    for (uint32_t i = 0; i < rope_dim / 2; ++i) {
      float freq =
          1.0f / std::pow(theta, (2.0f * i) / static_cast<float>(rope_dim));
      rope_freqs_.push_back(freq);
    }

    rope_initialized_ = true;

    std::cout << "[DEBUG] Precomputed " << rope_freqs_.size()
              << " RoPE frequencies (head_dim_q=" << head_dim
              << ", rope_dim=" << rope_dim << ", theta=" << theta << ")"
              << (rope_fallback ? " [fallback applied]" : "") << std::endl;
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

    // 1. Use llama.cpp for tokenization
    std::vector<llama_token> tokens;
    tokens.resize(prompt.length() + 1);

    // Get model vocabulary
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

    // 2. Create batch
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);

    // Add tokens to batch
    for (int i = 0; i < n_tokens; i++) {
      batch.token[i] = tokens[i];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = false;
    }
    batch.n_tokens = n_tokens;

    // Last token needs output
    batch.logits[batch.n_tokens - 1] = true;

    // 4. Decode input tokens
    if (llama_decode(llama_context_, batch) != 0) {
      std::cerr << "[ERROR] Failed to decode input tokens" << std::endl;
      llama_batch_free(batch);
      return "Error: Failed to decode input tokens";
    }

    // 5. Generate new tokens
    std::string generated_text;
    int n_cur = batch.n_tokens;
    int n_decode = 0;

    while (n_decode < static_cast<int>(max_tokens)) {
      // Use sampler to select next token
      llama_token new_token = llama_sampler_sample(
          llama_sampler_, llama_context_, batch.n_tokens - 1);

      // Check if it's an end token
      if (llama_vocab_is_eog(vocab, new_token)) {
        std::cout << "[DEBUG] Generated EOS token, stopping generation"
                  << std::endl;
        break;
      }

      // Convert token to text
      char piece[256];
      int n_piece = llama_token_to_piece(vocab, new_token, piece, sizeof(piece),
                                         0,    // lstrip
                                         false // special
      );

      if (n_piece > 0) {
        generated_text.append(piece, n_piece);
      }

      // Prepare for next decoding
      batch.n_tokens = 1;
      batch.token[0] = new_token;
      batch.pos[0] = n_cur;
      batch.n_seq_id[0] = 1;
      batch.seq_id[0][0] = 0;
      batch.logits[0] = true;

      n_decode++;
      n_cur++;

      // Decode new token
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

// std::string MLInferenceEngine::generateWithGGLM(const std::string &prompt,
//                                                 uint32_t max_tokens,
//                                                 float temperature,
//                                                 float top_p) {
//   try {
//     if (!initialized_) {
//       return "Error: Inference engine not initialized";
//     }
//     if (!qwen_model_ || !ml_context_) {
//       return "Error: Qwen model or ML context not initialized";
//     }

//     // Encode with Qwen tokenizer (apply minimal ChatML formatting for Qwen architectures)
//     std::string effective_prompt = prompt;
//     bool used_chatml = false;
//     if (gguf_parser_) {
//       auto archName = gguf_parser_->getArchitecture().name;
//       bool is_qwen = (archName.find("qwen") != std::string::npos) ||
//                      (archName.find("Qwen") != std::string::npos);
//       if (is_qwen) {
//         // Minimal ChatML commonly used by Qwen instruct models
//         effective_prompt = std::string("<|im_start|>system\n你是一个有帮助的助手。<|im_end|>\n") +
//                            "<|im_start|>user\n" + prompt + "\n<|im_end|>\n" +
//                            "<|im_start|>assistant\n";
//         used_chatml = true;
//       }
//     }
//     std::vector<int32_t> input_ids = qwen_model_->encode(effective_prompt, /*addSpecial=*/!used_chatml);
//     if (input_ids.empty()) {
//       return "Error: Failed to encode prompt";
//     }
//     if (used_chatml) {
//       std::cout << "[DEBUG] [GGLM] Using Qwen ChatML-formatted prompt" << std::endl;
//     }

//     // Get EOS from GGUF or vocabulary
//     int32_t eos_id = -1;
//     if (gguf_parser_) {
//       if (const auto *kvEOS =
//               gguf_parser_->getMetadata("tokenizer.ggml.eos_token_id")) {
//         eos_id = kvEOS->asInt32();
//       }
//     }
//     if (eos_id < 0) {
//       const duorou::model::Vocabulary *v = qwen_model_->getVocabulary();
//       eos_id = v ? v->getSpecialId(duorou::model::Special::EOS) : -1;
//     }
//     if (!input_ids.empty() && eos_id >= 0 && input_ids.back() == eos_id) {
//       input_ids.pop_back();
//     }

//     std::vector<int32_t> sequence_ids = input_ids;
//     std::vector<int32_t> generated_ids;

//     static thread_local std::mt19937 rng(std::random_device{}());

//     // Precompute PAD-like token ids from vocabulary strings (e.g., "[PAD]", "[PAD270]")
//     // This helps mask tokens that are not marked as PAD in GGUF but exist in the vocabulary.
//     std::vector<int32_t> pad_like_ids;
//     const duorou::model::Vocabulary *vocab_ptr = qwen_model_->getVocabulary();
//     if (vocab_ptr) {
//       const auto &vals = vocab_ptr->getValues();
//       pad_like_ids.reserve(vals.size() / 128); // heuristic reserve
//       for (size_t i = 0; i < vals.size(); ++i) {
//         const std::string &tok = vals[i];
//         // Match common PAD patterns, e.g., "[PAD]", "[PAD270]", "<pad>", "<PAD>"
//         bool is_pad_like = false;
//         if (!tok.empty()) {
//           if (tok.size() >= 5 && tok.rfind("[PAD", 0) == 0) {
//             // Starts with "[PAD" (e.g., [PAD], [PAD270])
//             is_pad_like = true;
//           } else if (tok == "<pad>" || tok == "<PAD>" || tok == "[PAD]") {
//             is_pad_like = true;
//           }
//         }
//         if (is_pad_like) {
//           pad_like_ids.push_back(static_cast<int32_t>(i));
//         }
//       }
//     }

//     size_t generated = 0;
//     while (generated < max_tokens) {
//       auto t_loop_start = std::chrono::high_resolution_clock::now();
//       // Build input tensor: full sequence for first step, last token afterwards
//       duorou::ml::Tensor input_tensor;
//       if (generated == 0) {
//         input_tensor =
//             duorou::ml::Tensor({static_cast<int64_t>(sequence_ids.size())},
//                                duorou::ml::DataType::INT32);
//         if (ml_context_->getBackend())
//           input_tensor.setBackend(ml_context_->getBackend());
//         input_tensor.allocate(input_tensor.backend());
//         input_tensor.copyFromHost(sequence_ids.data(),
//                                   sequence_ids.size() * sizeof(int32_t));
//       } else {
//         int32_t last_id = sequence_ids.back();
//         input_tensor = duorou::ml::Tensor({1}, duorou::ml::DataType::INT32);
//         if (ml_context_->getBackend())
//           input_tensor.setBackend(ml_context_->getBackend());
//         input_tensor.allocate(input_tensor.backend());
//         input_tensor.copyFromHost(&last_id, sizeof(int32_t));
//       }

//       // Use multimodal forward for consistency (no images here)
//       auto t_fwd_start = std::chrono::high_resolution_clock::now();
//       duorou::ml::Tensor logits_tensor =
//           qwen_model_->forward(*ml_context_, input_tensor, {}, kv_cache_.get());
//       auto t_fwd_end = std::chrono::high_resolution_clock::now();
//       if (logits_tensor.numel() <= 0) {
//         break;
//       }

//       // Diagnostics on first step: vocab vs logits size
//       if (generated == 0) {
//         size_t vocab_sz = qwen_model_->getVocabSize();
//         const auto &sh = logits_tensor.shape();
//         std::cout << "[DEBUG] [GGLM] logits size=" << logits_tensor.numel()
//                   << ", model vocab size=" << vocab_sz << ", shape=[";
//         for (size_t i = 0; i < sh.size(); ++i) {
//           std::cout << sh[i] << (i + 1 < sh.size() ? "," : "");
//         }
//         std::cout << "]" << std::endl;
//         auto dur_fwd = std::chrono::duration_cast<std::chrono::milliseconds>(
//             t_fwd_end - t_fwd_start);
//         std::cout << "[DEBUG] [GGLM] forward took " << dur_fwd.count()
//                   << " ms" << std::endl;
//       }

//       // Apply temperature scaling directly on tensor
//       if (temperature > 0.0f) {
//         // Create temperature tensor and divide
//         duorou::ml::Tensor temp_tensor({1}, duorou::ml::DataType::FLOAT32);
//         if (ml_context_->getBackend())
//           temp_tensor.setBackend(ml_context_->getBackend());
//         temp_tensor.allocate(temp_tensor.backend());
//         float temp_val = temperature;
//         temp_tensor.copyFromHost(&temp_val, sizeof(float));

//         // Temperature scaling: logits = logits / temperature
//         logits_tensor = logits_tensor.div(*ml_context_, temp_tensor);
//       }

//       // Copy logits for the last step row only when needed
//       std::vector<float> logits;
//       {
//         const auto &sh = logits_tensor.shape();
//         const size_t model_vocab_sz = qwen_model_->getVocabSize();
//         if (sh.size() == 1) {
//           const size_t V = static_cast<size_t>(sh[0]);
//           if (generated == 0 && V != model_vocab_sz) {
//             std::cout << "[DEBUG] [GGLM] logits 1D size " << V
//                       << " differs from model vocab " << model_vocab_sz
//                       << "; proceeding with V=" << V << std::endl;
//           }
//           logits.resize(V);
//           logits_tensor.copyToHost(logits.data(), logits.size() * sizeof(float));
//         } else if (sh.size() == 2) {
//           // Layout: [T, V]. Copy entire then slice the last row.
//           const size_t T = static_cast<size_t>(sh[0]);
//           const size_t V = static_cast<size_t>(sh[1]);
//           if (generated == 0 && V != model_vocab_sz) {
//             std::cout << "[DEBUG] [GGLM] logits 2D shape [" << T << "," << V
//                       << "] differs from model vocab " << model_vocab_sz
//                       << "; slicing last row with V=" << V << std::endl;
//           }
//           std::vector<float> tmp(static_cast<size_t>(logits_tensor.numel()));
//           logits_tensor.copyToHost(tmp.data(), tmp.size() * sizeof(float));
//           const size_t offset = (T > 0 ? (T - 1) * V : 0);
//           logits.assign(tmp.begin() + static_cast<std::ptrdiff_t>(offset),
//                         tmp.begin() + static_cast<std::ptrdiff_t>(offset + V));
//         } else {
//           // Fallback: copy all and use as-is (rare). Avoid noisy warnings.
//           if (generated == 0) {
//             std::cout << "[DEBUG] [GGLM] unexpected logits shape ndim="
//                       << sh.size() << "; copying all" << std::endl;
//           }
//           logits.resize(static_cast<size_t>(logits_tensor.numel()));
//           logits_tensor.copyToHost(logits.data(),
//                                    logits.size() * sizeof(float));
//         }
//       }

//       // Mask special tokens to avoid degenerate outputs like [PADxxx]
//       // Retrieve PAD/UNK/BOS ids from GGUF or vocabulary and set logits to a large negative
//       auto get_special_id = [&](duorou::model::Special sp) -> int32_t {
//         int32_t id = -1;
//         if (gguf_parser_) {
//           const char *key = nullptr;
//           switch (sp) {
//           case duorou::model::Special::PAD:
//             key = "tokenizer.ggml.pad_token_id";
//             break;
//           case duorou::model::Special::UNK:
//             key = "tokenizer.ggml.unk_token_id";
//             break;
//           case duorou::model::Special::BOS:
//             key = "tokenizer.ggml.bos_token_id";
//             break;
//           default:
//             key = nullptr;
//             break;
//           }
//           if (key) {
//             if (const auto *kv = gguf_parser_->getMetadata(key)) {
//               id = kv->asInt32();
//             }
//           }
//         }
//         if (id < 0) {
//           const duorou::model::Vocabulary *v = qwen_model_->getVocabulary();
//           if (v)
//             id = v->getSpecialId(sp);
//         }
//         return id;
//       };

//       const int32_t pad_id = get_special_id(duorou::model::Special::PAD);
//       const int32_t unk_id = get_special_id(duorou::model::Special::UNK);
//       const int32_t bos_id = get_special_id(duorou::model::Special::BOS);

//       if (generated == 0) {
//         std::cout << "[DEBUG] [GGLM] special ids - PAD=" << pad_id
//                   << ", UNK=" << unk_id << ", BOS=" << bos_id << std::endl;
//       }

//       auto mask_logit = [&](int32_t id) {
//         if (id >= 0 && static_cast<size_t>(id) < logits.size()) {
//           logits[static_cast<size_t>(id)] = -1e30f; // effectively -inf
//         }
//       };
//       mask_logit(pad_id);
//       mask_logit(unk_id);
//       // Avoid re-emitting BOS inside the sequence
//       mask_logit(bos_id);

//       // Mask PAD-like tokens inferred from vocabulary strings
//       if (!pad_like_ids.empty()) {
//         for (int32_t pid : pad_like_ids) {
//           if (pid >= 0 && static_cast<size_t>(pid) < logits.size()) {
//             logits[static_cast<size_t>(pid)] = -1e30f;
//           }
//         }
//         if (generated == 0) {
//           // Log summary of masked PAD-like ids for diagnostics
//           std::cout << "[DEBUG] [GGLM] masked " << pad_like_ids.size()
//                     << " PAD-like tokens by string pattern" << std::endl;
//           size_t show = std::min<size_t>(5, pad_like_ids.size());
//           for (size_t k = 0; k < show; ++k) {
//             int32_t pid = pad_like_ids[k];
//             const auto &vals = vocab_ptr ? vocab_ptr->getValues() : std::vector<std::string>();
//             std::string label = (vocab_ptr && static_cast<size_t>(pid) < vals.size()) ? vals[pid] : std::string("(unknown)");
//             std::cout << "[DEBUG] [GGLM] PAD-like id=" << pid << " token='" << label << "'" << std::endl;
//           }
//         }
//       }

//       // Argmax shortcut when temperature is effectively 0
//       int32_t next_token = -1;
//       if (temperature <= 0.0f) {
//         next_token = static_cast<int32_t>(
//             std::max_element(logits.begin(), logits.end()) - logits.begin());
//       } else {
//         // CPU softmax (stable): exp(logit - max) / sum(exp(...))
//         auto t_softmax_start = std::chrono::high_resolution_clock::now();
//         std::vector<float> probs(logits.size());
//         float max_logit = -std::numeric_limits<float>::infinity();
//         for (float v : logits) max_logit = std::max(max_logit, v);
//         double sum = 0.0;
//         for (size_t i = 0; i < logits.size(); ++i) {
//           float x = logits[i] - max_logit;
//           float ex = std::exp(x);
//           probs[i] = ex;
//           sum += ex;
//         }
//         if (sum <= 0.0) sum = 1.0;
//         for (auto &p : probs) p = static_cast<float>(p / sum);
//         auto t_softmax_end = std::chrono::high_resolution_clock::now();

//         // top-p sampling
//         const size_t vocab = probs.size();
//         if (top_p >= 1.0f) {
//           std::discrete_distribution<int> dist(probs.begin(), probs.end());
//           next_token = dist(rng);
//         } else {
//           auto t_sort_start = std::chrono::high_resolution_clock::now();
//           std::vector<int> idx(vocab);
//           std::iota(idx.begin(), idx.end(), 0);
//           std::sort(idx.begin(), idx.end(),
//                     [&](int a, int b) { return probs[a] > probs[b]; });
//           float acc = 0.0f;
//           size_t cut = 0;
//           for (; cut < idx.size(); ++cut) {
//             acc += probs[idx[cut]];
//             if (acc >= top_p)
//               break;
//           }
//           if (cut >= idx.size())
//             cut = idx.size() - 1;
//           std::vector<float> kept_probs(cut + 1);
//           for (size_t i = 0; i <= cut; ++i)
//             kept_probs[i] = probs[idx[i]];
//           for (auto &p : kept_probs)
//             p /= (acc > 0.0f ? acc : 1.0f);
//           std::discrete_distribution<int> dist(kept_probs.begin(),
//                                                kept_probs.end());
//           int pick = dist(rng);
//           next_token = idx[pick];
//           auto t_sort_end = std::chrono::high_resolution_clock::now();
//           auto dur_softmax = std::chrono::duration_cast<std::chrono::milliseconds>(
//               t_softmax_end - t_softmax_start);
//           auto dur_sort = std::chrono::duration_cast<std::chrono::milliseconds>(
//               t_sort_end - t_sort_start);
//           std::cout << "[DEBUG] [GGLM] step " << generated
//                     << " softmax=" << dur_softmax.count()
//                     << " ms, sort/sample=" << dur_sort.count() << " ms"
//                     << std::endl;
//         }
//       }

//       if (next_token < 0)
//         break;
//       if (eos_id >= 0 && next_token == eos_id)
//         break;

//       sequence_ids.push_back(next_token);
//       generated_ids.push_back(next_token);
//       generated += 1;

//       auto t_loop_end = std::chrono::high_resolution_clock::now();
//       auto dur_loop = std::chrono::duration_cast<std::chrono::milliseconds>(
//           t_loop_end - t_loop_start);
//       std::cout << "[DEBUG] [GGLM] step " << generated
//                 << " done, seq_len=" << sequence_ids.size()
//                 << ", took " << dur_loop.count() << " ms, next_token="
//                 << next_token << std::endl;
//     }

//     // Decode only generated part to text using Qwen detokenizer
//     std::string out = generated_ids.empty()
//                           ? std::string()
//                           : qwen_model_->decode(generated_ids);
//     return out.empty() ? std::string("(empty)") : out;
//   } catch (const std::exception &e) {
//     std::cerr << "[ERROR] Exception in generateWithGGLM: " << e.what()
//               << std::endl;
//     return "Error: " + std::string(e.what());
//   }
// }

std::string MLInferenceEngine::generateWithGGLM(const std::string &prompt,
                                                  uint32_t max_tokens,
                                                  float temperature,
                                                  float top_p) {
  try {
    std::cout << "[DEBUG] [GGML] Starting Qwen2VL GGML-based inference" << std::endl;
    
    // 验证初始化状态
    if (!gguf_parser_ || !ml_context_) {
      throw std::runtime_error("GGML context or GGUF parser not initialized");
    }
    
    // 获取模型配置
    const auto arch = gguf_parser_->getArchitecture();
    const int64_t n_embd = static_cast<int64_t>(n_embd_);
    const int64_t n_heads = static_cast<int64_t>(n_heads_);
    const int64_t n_kv_heads = static_cast<int64_t>(n_kv_heads_ > 0 ? n_kv_heads_ : n_heads_);
    const int64_t n_layers = static_cast<int64_t>(n_layers_);
    const int64_t head_dim = n_embd / n_heads;
    const int64_t vocab_size = static_cast<int64_t>(vocab_size_);
    
    std::cout << "[DEBUG] [GGML] Model config - n_embd: " << n_embd 
              << ", n_heads: " << n_heads << ", n_kv_heads: " << n_kv_heads
              << ", n_layers: " << n_layers << ", head_dim: " << head_dim << std::endl;
    
    // 分词
    std::vector<llama_token> input_tokens;
    if (tokenizer_) {
      input_tokens = tokenizer_->encode(prompt);
    } else {
      throw std::runtime_error("Tokenizer not available");
    }
    
    if (input_tokens.empty()) {
      return "Error: Failed to tokenize input";
    }
    
    const int64_t n_tokens = static_cast<int64_t>(input_tokens.size());
    std::cout << "[DEBUG] [GGML] Input tokens: " << n_tokens << std::endl;
    
    // 创建 GGML 计算图上下文
    struct ggml_init_params params = {
      .mem_size = 1024 * 1024 * 1024, // 1GB
      .mem_buffer = nullptr,
      .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
      throw std::runtime_error("Failed to initialize GGML context");
    }
    
    // 创建输入张量
    struct ggml_tensor* input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    memcpy(input_ids->data, input_tokens.data(), n_tokens * sizeof(llama_token));
    
    // 创建位置编码张量
    struct ggml_tensor* pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    for (int64_t i = 0; i < n_tokens; ++i) {
      ((int32_t*)pos_ids->data)[i] = static_cast<int32_t>(i);
    }
    
    // 获取嵌入权重
    auto emb_it = weight_map_.find("model.embed_tokens.weight");
    if (emb_it == weight_map_.end() || !emb_it->second) {
      throw std::runtime_error("Embedding weights not found");
    }
    
    // 词嵌入查找
    struct ggml_tensor* emb_weight = emb_it->second->to_ggml(ctx);
    struct ggml_tensor* embeddings = ggml_get_rows(ctx, emb_weight, input_ids);
    struct ggml_tensor* cur = embeddings; // [n_tokens, n_embd]
    
    std::cout << "[DEBUG] [GGML] Embeddings shape: [" << cur->ne[0] << ", " << cur->ne[1] << "]" << std::endl;
    
    // RoPE 配置
    const float rope_freq_base = rope_freq_base_ > 0.0f ? rope_freq_base_ : 10000.0f;
    const float rope_freq_scale = 1.0f;
    const int64_t rope_dim = static_cast<int64_t>(rope_dim_ > 0 ? rope_dim_ : head_dim);
    
    // 多段 RoPE 配置
    std::vector<int> rope_sections;
    for (auto v : arch.rope_dimension_sections) {
      rope_sections.push_back(static_cast<int>(v));
    }
    if (rope_sections.empty()) {
      rope_sections = {static_cast<int>(rope_dim)};
    }
    
    // Transformer 层循环
    for (int64_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
      std::cout << "[DEBUG] [GGML] Processing layer " << layer_idx << std::endl;
      
      // 保存残差连接的输入
      struct ggml_tensor* residual = cur;
      
      // 1. 输入层归一化 (RMSNorm)
      std::string attn_norm_name = "model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight";
      auto attn_norm_it = weight_map_.find(attn_norm_name);
      if (attn_norm_it != weight_map_.end() && attn_norm_it->second) {
        cur = ggml_rms_norm(ctx, cur, 1e-6f);
        struct ggml_tensor* norm_weight = attn_norm_it->second->to_ggml(ctx);
        cur = ggml_mul(ctx, cur, norm_weight);
      }
      
      // 2. 自注意力机制
      {
        // Q, K, V 投影
        std::string q_proj_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight";
        std::string k_proj_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight";
        std::string v_proj_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight";
        std::string o_proj_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight";
        
        auto q_weight_it = weight_map_.find(q_proj_name);
        auto k_weight_it = weight_map_.find(k_proj_name);
        auto v_weight_it = weight_map_.find(v_proj_name);
        auto o_weight_it = weight_map_.find(o_proj_name);
        
        if (q_weight_it != weight_map_.end() && k_weight_it != weight_map_.end() && 
            v_weight_it != weight_map_.end() && o_weight_it != weight_map_.end()) {
          
          // 计算 Q, K, V
          struct ggml_tensor* q_weight = q_weight_it->second->to_ggml(ctx);
          struct ggml_tensor* k_weight = k_weight_it->second->to_ggml(ctx);
          struct ggml_tensor* v_weight = v_weight_it->second->to_ggml(ctx);
          struct ggml_tensor* Q = ggml_mul_mat(ctx, q_weight, cur);
          struct ggml_tensor* K = ggml_mul_mat(ctx, k_weight, cur);
          struct ggml_tensor* V = ggml_mul_mat(ctx, v_weight, cur);
          
          // 添加偏置（如果存在）
          std::string q_bias_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.bias";
          std::string k_bias_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.bias";
          std::string v_bias_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.bias";
          
          auto q_bias_it = weight_map_.find(q_bias_name);
          auto k_bias_it = weight_map_.find(k_bias_name);
          auto v_bias_it = weight_map_.find(v_bias_name);
          
          if (q_bias_it != weight_map_.end() && q_bias_it->second) {
            struct ggml_tensor* q_bias = q_bias_it->second->to_ggml(ctx);
            Q = ggml_add(ctx, Q, q_bias);
          }
          if (k_bias_it != weight_map_.end() && k_bias_it->second) {
            struct ggml_tensor* k_bias = k_bias_it->second->to_ggml(ctx);
            K = ggml_add(ctx, K, k_bias);
          }
          if (v_bias_it != weight_map_.end() && v_bias_it->second) {
            struct ggml_tensor* v_bias = v_bias_it->second->to_ggml(ctx);
            V = ggml_add(ctx, V, v_bias);
          }
          
          // 重塑为多头格式
          Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, n_tokens);
          K = ggml_reshape_3d(ctx, K, head_dim, n_kv_heads, n_tokens);
          V = ggml_reshape_3d(ctx, V, head_dim, n_kv_heads, n_tokens);
          
          // 应用 RoPE（多段）
          if (!rope_sections.empty()) {
            // 简化的 RoPE 实现，使用第一个段的维度
            int rope_n_dims = rope_sections[0];
            Q = ggml_rope_ext(ctx, Q, pos_ids, nullptr, rope_n_dims, 0, 0, 
                             rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(ctx, K, pos_ids, nullptr, rope_n_dims, 0, 0,
                             rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
          }
          
          // 注意力计算
          // Q: [head_dim, n_heads, n_tokens] -> [n_heads, n_tokens, head_dim]
          Q = ggml_cont(ctx, ggml_permute(ctx, Q, 1, 2, 0, 3));
          K = ggml_cont(ctx, ggml_permute(ctx, K, 1, 2, 0, 3));
          V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));
          
          // 缩放点积注意力
          const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
          
          // Q @ K^T
          struct ggml_tensor* scores = ggml_mul_mat(ctx, K, Q); // [n_heads, n_tokens, n_tokens]
          scores = ggml_scale(ctx, scores, scale);
          
          // Softmax
          scores = ggml_soft_max(ctx, scores);
          
          // scores @ V
          struct ggml_tensor* attn_out = ggml_mul_mat(ctx, V, scores); // [n_heads, n_tokens, head_dim]
          
          // 重塑回原始形状
          attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 2, 0, 1, 3));
          attn_out = ggml_reshape_2d(ctx, attn_out, n_embd, n_tokens);
          
          // 输出投影
          struct ggml_tensor* o_weight = o_weight_it->second->to_ggml(ctx);
          cur = ggml_mul_mat(ctx, o_weight, attn_out);
        }
      }
      
      // 3. 残差连接
      cur = ggml_add(ctx, cur, residual);
      residual = cur;
      
      // 4. FFN 前的层归一化
      std::string ffn_norm_name = "model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight";
      auto ffn_norm_it = weight_map_.find(ffn_norm_name);
      if (ffn_norm_it != weight_map_.end() && ffn_norm_it->second) {
        cur = ggml_rms_norm(ctx, cur, 1e-6f);
        struct ggml_tensor* ffn_norm_weight = ffn_norm_it->second->to_ggml(ctx);
        cur = ggml_mul(ctx, cur, ffn_norm_weight);
      }
      
      // 5. FFN (SwiGLU)
      {
        std::string gate_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight";
        std::string up_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight";
        std::string down_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight";
        
        auto gate_weight_it = weight_map_.find(gate_proj_name);
        auto up_weight_it = weight_map_.find(up_proj_name);
        auto down_weight_it = weight_map_.find(down_proj_name);
        
        if (gate_weight_it != weight_map_.end() && up_weight_it != weight_map_.end() && 
            down_weight_it != weight_map_.end()) {
          
          struct ggml_tensor* gate_weight = gate_weight_it->second->to_ggml(ctx);
          struct ggml_tensor* up_weight = up_weight_it->second->to_ggml(ctx);
          struct ggml_tensor* down_weight = down_weight_it->second->to_ggml(ctx);
          
          struct ggml_tensor* gate = ggml_mul_mat(ctx, gate_weight, cur);
          struct ggml_tensor* up = ggml_mul_mat(ctx, up_weight, cur);
          
          // SiLU 激活函数
          gate = ggml_silu(ctx, gate);
          
          // 门控机制
          struct ggml_tensor* ffn_out = ggml_mul(ctx, gate, up);
          
          // 下投影
          cur = ggml_mul_mat(ctx, down_weight, ffn_out);
        }
      }
      
      // 6. 最终残差连接
      cur = ggml_add(ctx, cur, residual);
    }
    
    // 最终层归一化
    auto final_norm_it = weight_map_.find("model.norm.weight");
    if (final_norm_it != weight_map_.end() && final_norm_it->second) {
      cur = ggml_rms_norm(ctx, cur, 1e-6f);
      struct ggml_tensor* final_norm_weight = final_norm_it->second->to_ggml(ctx);
      cur = ggml_mul(ctx, cur, final_norm_weight);
    }
    
    // LM Head 投影到词汇表
    auto lm_head_it = weight_map_.find("lm_head.weight");
    if (lm_head_it != weight_map_.end() && lm_head_it->second) {
      struct ggml_tensor* lm_head_weight = lm_head_it->second->to_ggml(ctx);
      cur = ggml_mul_mat(ctx, lm_head_weight, cur);
    }
    
    // 构建计算图
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);
    
    // 执行计算
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    
    // 获取最后一个 token 的 logits
    const int64_t last_token_idx = n_tokens - 1;
    float* logits_data = (float*)cur->data;
    const int64_t logits_offset = last_token_idx * vocab_size;
    
    // 生成循环
    std::vector<llama_token> generated_tokens;
    std::vector<llama_token> sequence = input_tokens;
    
    for (uint32_t step = 0; step < max_tokens; ++step) {
      // 应用温度
      std::vector<float> probs(vocab_size);
      for (int64_t i = 0; i < vocab_size; ++i) {
        probs[i] = logits_data[logits_offset + i] / temperature;
      }
      
      // Softmax
      float max_logit = *std::max_element(probs.begin(), probs.end());
      float sum_exp = 0.0f;
      for (auto& p : probs) {
        p = expf(p - max_logit);
        sum_exp += p;
      }
      for (auto& p : probs) {
        p /= sum_exp;
      }
      
      // Top-p 采样
      std::vector<std::pair<float, int>> prob_idx;
      for (int i = 0; i < vocab_size; ++i) {
        prob_idx.emplace_back(probs[i], i);
      }
      std::sort(prob_idx.begin(), prob_idx.end(), std::greater<>());
      
      float cumsum = 0.0f;
      int cutoff = vocab_size;
      for (int i = 0; i < vocab_size; ++i) {
        cumsum += prob_idx[i].first;
        if (cumsum >= top_p) {
          cutoff = i + 1;
          break;
        }
      }
      
      // 随机采样
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0.0, 1.0);
      
      float r = dis(gen);
      cumsum = 0.0f;
      llama_token next_token = prob_idx[0].second;
      
      for (int i = 0; i < cutoff; ++i) {
        cumsum += prob_idx[i].first;
        if (r <= cumsum) {
          next_token = prob_idx[i].second;
          break;
        }
      }
      
      // 检查停止条件
      if (next_token == 151645 || next_token == 151643) { // EOS tokens for Qwen
        break;
      }
      
      generated_tokens.push_back(next_token);
      sequence.push_back(next_token);
      
      // 对于后续 token，需要重新计算（这里简化处理）
      break; // 暂时只生成一个 token
    }
    
    // 解码生成的文本
    std::string result;
    if (tokenizer_ && !generated_tokens.empty()) {
      result = tokenizer_->decode(generated_tokens);
    }
    
    // 清理
    ggml_free(ctx);
    
    std::cout << "[DEBUG] [GGML] Generated text: " << result << std::endl;
    return result.empty() ? "(empty)" : result;
    
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] Exception in generateWithGGLM: " << e.what() << std::endl;
    return "Error: " + std::string(e.what());
  }
}

// Internal Forward mode: Use Qwen model for actual inference
// std::string
// MLInferenceEngine::generateWithInternalForward(const std::string &prompt,
//                                                uint32_t max_tokens,
//                                                float temperature, float top_p) {
//   try {
//     std::cout << "[DEBUG] [InternalForward] Starting Qwen model tensor forward "
//                  "with KV cache"
//               << std::endl;
//     // Diagnostics: model configuration and components
//     std::cout << "[DEBUG] [InternalForward] Model config - vocab_size: "
//               << vocab_size_ << ", n_layers: " << n_layers_
//               << ", n_heads: " << n_heads_ << ", n_kv_heads: " << n_kv_heads_
//               << ", n_embd: " << n_embd_ << ", n_ctx: " << n_ctx_
//               << ", rope_dim: " << rope_dim_
//               << ", rope_freq_base: " << rope_freq_base_ << std::endl;
//     std::cout << "[DEBUG] [InternalForward] Weight map tensors: "
//               << weight_map_.size() << std::endl;
//     // Log key tensor shapes to aid diagnosis
//     {
//       auto logTensorShape = [&](const char *name) {
//         auto it = weight_map_.find(name);
//         if (it != weight_map_.end() && it->second) {
//           const auto &sh = it->second->shape();
//           std::cout << "[DEBUG] [InternalForward] tensor '" << name
//                     << "' shape: [";
//           for (size_t i = 0; i < sh.size(); ++i) {
//             std::cout << sh[i] << (i + 1 < sh.size() ? "," : "");
//           }
//           std::cout << "], dtype="
//                     << duorou::ml::dataTypeToString(it->second->dtype())
//                     << std::endl;
//         } else {
//           std::cout << "[DEBUG] [InternalForward] tensor '" << name
//                     << "' not found" << std::endl;
//         }
//       };
//       logTensorShape("token_embd.weight");
//       logTensorShape("output.weight");
//       logTensorShape("output_norm.weight");
//       // Scan a few critical weights for dtype and basic stats
//       auto scanTensor = [&](const char *name) {
//         auto it = weight_map_.find(name);
//         if (it != weight_map_.end() && it->second) {
//           const auto &t = it->second;
//           std::cout << "[DEBUG] [InternalForward] scan '" << name
//                     << "' dtype=" << duorou::ml::dataTypeToString(t->dtype())
//                     << ", numel=" << t->numel() << std::endl;
//           size_t n = static_cast<size_t>(std::min<int64_t>(t->numel(), 1024));
//           if (n > 0) {
//             std::vector<float> buf(n);
//             // Best-effort read as float
//             t->copyToHost(buf.data(), n * sizeof(float));
//             bool has_nan = false, has_inf = false;
//             bool all_zero = true;
//             float mn = buf[0], mx = buf[0];
//             for (size_t i = 0; i < n; ++i) {
//               float v = buf[i];
//               if (std::isnan(v))
//                 has_nan = true;
//               if (std::isinf(v))
//                 has_inf = true;
//               if (v != 0.0f)
//                 all_zero = false;
//               mn = std::min(mn, v);
//               mx = std::max(mx, v);
//             }
//             std::cout << "[DEBUG] [InternalForward] '" << name
//                       << "' sample stats min=" << mn << ", max=" << mx
//                       << ", NaN=" << has_nan << ", Inf=" << has_inf
//                       << ", all_zero=" << all_zero << std::endl;
//           }
//         }
//       };
//       scanTensor("token_embd.weight");
//       scanTensor("output.weight");
//       scanTensor("output_norm.weight");
//     }
//     if (kv_cache_) {
//       std::cout << "[DEBUG] [InternalForward] KV cache type: Causal; cache "
//                    "initialized"
//                 << std::endl;
//     } else {
//       std::cout << "[DEBUG] [InternalForward] KV cache not present"
//                 << std::endl;
//     }

//     // Check if Qwen model and ML context are available
//     if (!qwen_model_ || !ml_context_) {
//       std::cerr << "[ERROR] Qwen model or ML context not initialized"
//                 << std::endl;
//       return generateIntelligentResponse(prompt, max_tokens, temperature);
//     }

//     // First encode text to token IDs using Qwen tokenizer (llama.cpp tokenizer
//     // disabled)
//     std::vector<int32_t> input_ids = qwen_model_->encode(prompt, true);
//     std::cout << "[DEBUG] [InternalForward] Using Qwen tokenization (llama.cpp "
//                  "tokenizer disabled)"
//               << std::endl;

//     if (input_ids.empty()) {
//       std::cout
//           << "[WARN] [InternalForward] Failed to encode prompt, using fallback"
//           << std::endl;
//       return generateIntelligentResponse(prompt, max_tokens, temperature);
//     }

//     std::cout << "[DEBUG] [InternalForward] Encoded " << input_ids.size()
//               << " tokens from prompt" << std::endl;

//     // Remove trailing EOS from input to allow generation to proceed,
//     // and remember prompt length so we can strip it from the final output
//     size_t prompt_len = input_ids.size();

//     // Get EOS token ID from GGUF metadata (Qwen2.5VL uses 151645)
//     int32_t eos_id = -1;
//     if (gguf_parser_) {
//       if (const auto *kvEOS =
//               gguf_parser_->getMetadata("tokenizer.ggml.eos_token_id")) {
//         eos_id = kvEOS->asInt32();
//         std::cout << "[DEBUG] [InternalForward] Using EOS token ID from GGUF: "
//                   << eos_id << std::endl;
//       }
//     }

//     // Fallback to vocabulary if GGUF parsing fails
//     if (eos_id < 0) {
//       const duorou::model::Vocabulary *v = qwen_model_->getVocabulary();
//       eos_id = v ? v->getSpecialId(duorou::model::Special::EOS) : -1;
//       std::cout
//           << "[DEBUG] [InternalForward] Using EOS token ID from vocabulary: "
//           << eos_id << std::endl;
//     }

//     if (!input_ids.empty() && eos_id >= 0 && input_ids.back() == eos_id) {
//       input_ids.pop_back();
//       prompt_len = input_ids.size();
//       std::cout
//           << "[DEBUG] [InternalForward] Removed trailing EOS from prompt tokens"
//           << std::endl;
//     }

//     // Prepare generation loop
//     std::vector<int32_t> sequence_ids = input_ids; // prompt + generated

//     // Ensure KV cache is initialized
//     if (!kv_cache_) {
//       std::cout << "[WARN] [InternalForward] KV cache not initialized; "
//                    "proceeding without cache"
//                 << std::endl;
//     } else {
//       std::cout << "[DEBUG] [InternalForward] KV cache available; will be "
//                    "passed into forward()"
//                 << std::endl;
//     }

//     // Random engine for sampling
//     static thread_local std::mt19937 rng(std::random_device{}());

//     auto apply_temperature = [&](std::vector<float> &logits, float temp) {
//       if (temp <= 0.0f)
//         return; // handled by argmax path later
//       for (auto &x : logits) {
//         x /= temp;
//       }
//     };

//     auto softmax = [&](const std::vector<float> &logits) {
//       std::vector<float> probs(logits.size());
//       if (logits.empty())
//         return probs;
//       float max_logit = *std::max_element(logits.begin(), logits.end());
//       double sum = 0.0;
//       for (size_t i = 0; i < logits.size(); ++i) {
//         double e = std::exp(static_cast<double>(logits[i] - max_logit));
//         probs[i] = static_cast<float>(e);
//         sum += e;
//       }
//       if (sum > 0.0) {
//         for (auto &p : probs)
//           p = static_cast<float>(p / sum);
//       }
//       return probs;
//     };

//     auto sample_top_p = [&](const std::vector<float> &probs, float tp) {
//       if (probs.empty())
//         return int32_t(-1);
//       if (tp >= 1.0f) {
//         // Full distribution sampling
//         std::discrete_distribution<int> dist(probs.begin(), probs.end());
//         return static_cast<int32_t>(dist(rng));
//       }
//       // Sort indices by prob desc
//       std::vector<int> idx(probs.size());
//       std::iota(idx.begin(), idx.end(), 0);
//       std::sort(idx.begin(), idx.end(),
//                 [&](int a, int b) { return probs[a] > probs[b]; });
//       // Accumulate until reaching top_p
//       std::vector<int> kept;
//       std::vector<float> kept_probs;
//       float acc = 0.0f;
//       for (int id : idx) {
//         kept.push_back(id);
//         kept_probs.push_back(probs[id]);
//         acc += probs[id];
//         if (acc >= tp)
//           break;
//       }
//       // Normalize kept_probs
//       if (acc > 0.0f) {
//         for (auto &p : kept_probs)
//           p = p / acc;
//       }
//       // Sample among kept
//       std::discrete_distribution<int> dist(kept_probs.begin(),
//                                            kept_probs.end());
//       int pick = dist(rng);
//       return static_cast<int32_t>(kept[pick]);
//     };

//     // Main generation loop performing tensor forward per step
//     size_t generated = 0;
//     while (generated < max_tokens) {
//       duorou::ml::Tensor logits_tensor;

//       // Unified inference: always use multimodal model's forward method for
//       // consistency
//       duorou::ml::Tensor input_tensor;

//       if (generated == 0) {
//         // Prime KV cache with full prompt on first pass
//         input_tensor =
//             duorou::ml::Tensor({static_cast<int64_t>(sequence_ids.size())},
//                                duorou::ml::DataType::INT32);
//         if (ml_context_->getBackend()) {
//           input_tensor.setBackend(ml_context_->getBackend());
//         }
//         if (!sequence_ids.empty()) {
//           input_tensor.copyFromHost(sequence_ids.data(),
//                                     sequence_ids.size() * sizeof(int32_t));
//         }
//         std::cout << "[DEBUG] [InternalForward] First pass: processing "
//                   << sequence_ids.size() << " tokens" << std::endl;
//         {
//           const auto &sh = input_tensor.shape();
//           std::cout << "[DEBUG] [InternalForward] input_tensor shape: [";
//           for (size_t i = 0; i < sh.size(); ++i) {
//             std::cout << sh[i] << (i + 1 < sh.size() ? "," : "");
//           }
//           std::cout << "], dtype="
//                     << duorou::ml::dataTypeToString(input_tensor.dtype())
//                     << std::endl;
//         }
//       } else {
//         // Step-by-step decode using only the last generated token
//         int32_t last_id = sequence_ids.back();
//         input_tensor = duorou::ml::Tensor({1}, duorou::ml::DataType::INT32);
//         if (ml_context_->getBackend()) {
//           input_tensor.setBackend(ml_context_->getBackend());
//         }
//         input_tensor.copyFromHost(&last_id, sizeof(int32_t));
//         std::cout << "[DEBUG] [InternalForward] Step " << generated
//                   << ": processing token " << last_id << std::endl;
//         {
//           const auto &sh = input_tensor.shape();
//           std::cout << "[DEBUG] [InternalForward] input_tensor shape: [";
//           for (size_t i = 0; i < sh.size(); ++i) {
//             std::cout << sh[i] << (i + 1 < sh.size() ? "," : "");
//           }
//           std::cout << "], dtype="
//                     << duorou::ml::dataTypeToString(input_tensor.dtype())
//                     << std::endl;
//         }
//       }

//       // Always use multimodal model's forward method for consistency
//       logits_tensor =
//           qwen_model_->forward(*ml_context_, input_tensor, {}, kv_cache_.get());

//       std::cout
//           << "[DEBUG] [InternalForward] Forward pass completed, logits numel="
//           << logits_tensor.numel()
//           << ", dtype=" << duorou::ml::dataTypeToString(logits_tensor.dtype())
//           << std::endl;

//       // Convert logits to host vector
//       std::vector<float> logits;
//       if (logits_tensor.numel() > 0) {
//         logits.resize(static_cast<size_t>(logits_tensor.numel()));
//         logits_tensor.copyToHost(logits.data(), logits.size() * sizeof(float));
//       }
//       if (logits.empty()) {
//         std::cout
//             << "[WARN] [InternalForward] Empty logits from forward(); breaking"
//             << std::endl;
//         break;
//       }

//       // Validate logits size equals vocabulary size; otherwise assert and
//       // fallback to llama.cpp
//       {
//         size_t vocab_size = qwen_model_->getVocabSize();
//         if (vocab_size == 0 || logits.size() != vocab_size) {
//           std::cerr << "[ERROR] [InternalForward] Logits size mismatch: got "
//                     << logits.size() << ", expected vocab size " << vocab_size
//                     << "; falling back to llama.cpp" << std::endl;
//           // Attempt auto fallback and delegate to llama backend
//           if (tryAutoFallback("Logits size mismatch in internal forward")) {
//             // Prefer intelligent response to reduce llama.cpp dependency
//             return generateIntelligentResponse(prompt, max_tokens, temperature);
//           } else {
//             return generateIntelligentResponse(prompt, max_tokens, temperature);
//           }
//         }

//         // Debug: Check logits validity and range
//         auto minmax = std::minmax_element(logits.begin(), logits.end());
//         float min_logit = *minmax.first;
//         float max_logit = *minmax.second;

//         std::cout << "[DEBUG] [InternalForward] Logits range: [" << min_logit
//                   << ", " << max_logit << "]" << std::endl;

//         // Check for invalid values
//         bool has_nan = false, has_inf = false;
//         for (float logit : logits) {
//           if (std::isnan(logit))
//             has_nan = true;
//           if (std::isinf(logit))
//             has_inf = true;
//         }

//         if (has_nan || has_inf) {
//           std::cerr
//               << "[ERROR] [InternalForward] Invalid logits detected - NaN: "
//               << has_nan << ", Inf: " << has_inf
//               << "; falling back to llama.cpp" << std::endl;
//           if (tryAutoFallback("Invalid logits (NaN/Inf) in internal forward")) {
//             return generateIntelligentResponse(prompt, max_tokens, temperature);
//           } else {
//             return generateIntelligentResponse(prompt, max_tokens, temperature);
//           }
//         }

//         // Check if logits are all zeros (indicating potential weight loading
//         // issues)
//         bool all_zeros = std::all_of(logits.begin(), logits.end(),
//                                      [](float x) { return x == 0.0f; });
//         if (all_zeros) {
//           std::cerr
//               << "[ERROR] [InternalForward] All logits are zero - potential "
//                  "weight loading issue; falling back to llama.cpp"
//               << std::endl;
//           if (tryAutoFallback("All-zero logits in internal forward")) {
//             return generateIntelligentResponse(prompt, max_tokens, temperature);
//           } else {
//             return generateIntelligentResponse(prompt, max_tokens, temperature);
//           }
//         }

//         // Check for extremely large values that might indicate numerical
//         // instability
//         if (max_logit > 1000.0f || min_logit < -1000.0f) {
//           std::cerr << "[WARN] [InternalForward] Extreme logit values detected "
//                        "- may indicate numerical instability"
//                     << std::endl;
//         }
//       }

//       int32_t next_token = -1;
//       if (temperature <= 0.0f) {
//         // Greedy: pick argmax
//         next_token = static_cast<int32_t>(std::distance(
//             logits.begin(), std::max_element(logits.begin(), logits.end())));
//       } else {
//         // Temperature scaling then top-p sampling
//         apply_temperature(logits, temperature);
//         std::vector<float> probs = softmax(logits);
//         next_token = sample_top_p(probs, std::clamp(top_p, 0.0f, 1.0f));
//       }

//       if (next_token < 0) {
//         std::cout << "[WARN] [InternalForward] Sampling returned invalid "
//                      "token; stopping"
//                   << std::endl;
//         break;
//       }

//       sequence_ids.push_back(next_token);
//       generated += 1;

//       // Stop at EOS if available
//       if (eos_id >= 0 && next_token == eos_id) {
//         std::cout << "[DEBUG] [InternalForward] Reached EOS token; stopping "
//                      "generation"
//                   << std::endl;
//         break;
//       }
//     }

//     // Decode output token IDs to text
//     std::string result;

//     // Only decode newly generated tokens (exclude the prompt)
//     std::vector<int32_t> gen_ids;
//     if (sequence_ids.size() > prompt_len) {
//       gen_ids.assign(sequence_ids.begin() +
//                          static_cast<std::ptrdiff_t>(prompt_len),
//                      sequence_ids.end());
//     }

//     // Unified: always use Qwen decoder for internal forward output
//     result = gen_ids.empty() ? std::string() : qwen_model_->decode(gen_ids);
//     std::cout << "[DEBUG] [InternalForward] Using Qwen detokenization "
//                  "(excluded prompt tokens)"
//               << std::endl;

//     if (result.empty()) {
//       std::cout << "[WARN] [InternalForward] Failed to decode output tokens, "
//                    "using fallback"
//                 << std::endl;
//       return generateIntelligentResponse(prompt, max_tokens, temperature);
//     }

//     std::cout << "[DEBUG] [InternalForward] Generated response: "
//               << result.substr(0, 50) << "..." << std::endl;
//     return result;

//   } catch (const std::exception &e) {
//     std::cerr << "[ERROR] Exception in generateWithInternalForward: "
//               << e.what() << std::endl;
//     return "Error: " + std::string(e.what());
//   }
// }

void MLInferenceEngine::cleanupResources() {
  std::cout << "[DEBUG] Cleaning up inference engine resources" << std::endl;

  // Clean up ML components
  delete attention_;
  attention_ = nullptr;

  delete ml_context_;
  ml_context_ = nullptr;

  // Delete unique tensors referenced by model_weights_ and weight_map_
  {
    std::unordered_set<duorou::ml::Tensor *> unique;
    for (auto *w : model_weights_) {
      if (w)
        unique.insert(w);
    }
    for (auto &kv : weight_map_) {
      if (kv.second)
        unique.insert(kv.second);
    }
    for (auto *t : unique) {
      delete t;
    }
  }
  model_weights_.clear();
  weight_map_.clear();

  // Clean up KV cache
  kv_cache_.reset();

  // Clean up RoPE frequencies
  rope_freqs_.clear();
  rope_initialized_ = false;

  // Reset configuration
  vocab_size_ = 0;
  n_layers_ = 0;
  n_heads_ = 0;
  n_kv_heads_ = 0;
  n_embd_ = 0;
  n_ctx_ = 0;
  rope_dim_ = 0;
  rope_freq_base_ = 10000.0f;

  std::cout << "[DEBUG] Resource cleanup completed" << std::endl;
}

} // namespace ollama
} // namespace extensions
} // namespace duorou

duorou::ml::DataType
duorou::extensions::ollama::MLInferenceEngine::convertGGMLDataType(
    GGMLTensorType ggmlType) {
  switch (ggmlType) {
  case GGMLTensorType::F32:
    return duorou::ml::DataType::FLOAT32;
  case GGMLTensorType::F16:
    return duorou::ml::DataType::FLOAT16;
  case GGMLTensorType::BF16:
    return duorou::ml::DataType::BF16;
  default:
    // For quantized types, store as INT8 by default (block-quantized layout)
    return duorou::ml::DataType::INT8;
  }
}

bool duorou::extensions::ollama::MLInferenceEngine::loadTensorData(
    const std::string &tensorName, duorou::ml::Tensor *tensor) {
  if (!gguf_parser_) {
    std::cerr << "[ERROR] GGUF parser not initialized" << std::endl;
    return false;
  }
  const auto *info = gguf_parser_->getTensorInfo(tensorName);
  if (!info) {
    std::cerr << "[WARN] Tensor info not found in GGUF: " << tensorName
              << std::endl;
    return false;
  }

  // Ensure tensor has memory allocated
  if (!tensor->backend()) {
    // Attach current backend
    if (ml_context_ && ml_context_->getBackend()) {
      tensor->setBackend(ml_context_->getBackend());
    }
  }
  if (!tensor->isValid()) {
    tensor->allocate(tensor->backend());
  }

  const size_t bytes = gguf_parser_->getTensorSize(tensorName);
  if (bytes == 0) {
    std::cerr << "[WARN] Tensor size is 0 in GGUF: " << tensorName << std::endl;
    return false;
  }

  // Read tensor data into tensor->data()
  if (!gguf_parser_->readTensorData(*info, tensor->data(), bytes)) {
    std::cerr << "[ERROR] Failed to read tensor data: " << tensorName
              << std::endl;
    return false;
  }
  return true;
}

bool duorou::extensions::ollama::MLInferenceEngine::mapTensorWeights() {
  if (!gguf_parser_) {
    std::cerr << "[ERROR] GGUF parser not initialized for mapping" << std::endl;
    return false;
  }

  const auto &infos = gguf_parser_->getAllTensorInfos();
  if (infos.empty()) {
    std::cerr << "[ERROR] No tensor infos found in GGUF" << std::endl;
    return false;
  }

  size_t mapped = 0;
  for (const auto &info : infos) {
    // Build shape from dimensions
    std::vector<int64_t> shape;
    shape.reserve(info.dimensions.size());
    for (auto d : info.dimensions) {
      shape.push_back(static_cast<int64_t>(d));
    }
    duorou::ml::DataType dtype = convertGGMLDataType(info.type);
    auto *t = new duorou::ml::Tensor(shape, dtype);
    // Attach backend and allocate
    if (ml_context_ && ml_context_->getBackend()) {
      t->setBackend(ml_context_->getBackend());
    }
    t->allocate(t->backend());

    if (!loadTensorData(info.name, t)) {
      std::cerr << "[WARN] Skipping tensor due to read failure: " << info.name
                << std::endl;
      delete t;
      continue;
    }

    weight_map_[info.name] = t;
    mapped++;
  }
  std::cout << "[DEBUG] mapTensorWeights: mapped " << mapped << " tensors"
            << std::endl;
  return mapped > 0;
}

bool duorou::extensions::ollama::MLInferenceEngine::
    checkInternalForwardSupport() {
  if (!gguf_parser_)
    return false;
  std::string arch = gguf_parser_->getArchitecture().name;
  std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);
  // Prefer internal forward for Qwen-family architectures, esp. multimodal
  // variants (vl)
  bool is_qwen = arch.find("qwen") != std::string::npos;
  if (!is_qwen) {
    return false;
  }
  // Basic config presence check
  return true;
}

bool duorou::extensions::ollama::MLInferenceEngine::tryAutoFallback(
    const std::string &reason) {
  // Allow disabling automatic fallback via environment variable
  const char *disable_env = std::getenv("DUOROU_DISABLE_LLAMA_FALLBACK");
  bool disable_fallback = disable_env && std::string(disable_env) != "0" &&
                          std::string(disable_env).size() > 0;
  if (disable_fallback) {
    std::cerr << "[INFO] Auto fallback to llama.cpp disabled by "
                 "DUOROU_DISABLE_LLAMA_FALLBACK. Reason: "
              << reason << std::endl;
    use_llama_backend_ = false;
    return false;
  }

  // Check if llama.cpp supports the current architecture before attempting
  // fallback
  std::string arch =
      gguf_parser_ ? gguf_parser_->getArchitecture().name : std::string();
  std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);
  bool llama_supported = isSupportedByLlamaCpp(arch);
  if (!llama_supported) {
    std::cerr << "[WARN] Auto fallback skipped: architecture '" << arch
              << "' is unsupported by llama.cpp; staying on internal forward"
              << std::endl;
    use_llama_backend_ = false;
    // Do not attempt llama.cpp; let caller handle initialization failure
    // gracefully
    return false;
  }

  std::cerr << "[INFO] Attempting auto fallback to llama.cpp due to: " << reason
            << std::endl;
  use_llama_backend_ = true;
  try {
    llama_backend_init();
    if (!loadLlamaModel(model_path_)) {
      std::cerr
          << "[ERROR] Auto fallback failed: could not load llama.cpp model"
          << std::endl;
      use_llama_backend_ = false;
      return false;
    }
    initialized_ = true;
    std::cout << "[DEBUG] Auto fallback to llama.cpp succeeded" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception during auto fallback: " << e.what()
              << std::endl;
    use_llama_backend_ = false;
    return false;
  }
}