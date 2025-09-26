// Include standard library headers first
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
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <unordered_set>

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

// Include project headers after third-party headers
#include "inference_engine.h"
#include "../../ml/context.h"
#include "../../ml/nn/attention.h"
#include "../../ml/tensor.h"
#include "../../ml/backend/backend.h"
#include "../../kvcache/causal.h"
#include "ollama_model_manager.h"

namespace duorou {
namespace extensions {
namespace ollama {
// Simple whitelist: which architectures to handle with llama.cpp
static bool isSupportedByLlamaCpp(const std::string &arch_raw) {
  std::string arch = arch_raw;
  std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);

  // Explicitly exclude known non-llama.cpp (multimodal/incompatible) architectures to avoid misrouting to llama.cpp
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
       rope_freqs_(), rope_initialized_(false), llama_model_(nullptr), llama_context_(nullptr),
      llama_sampler_(nullptr), use_llama_backend_(false) {
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
    llama_free_model(llama_model_);
    llama_model_ = nullptr;
  }

  cleanupResources();

  // Cleanup ML backends
  duorou::ml::BackendManager::getInstance().cleanup();
}

bool MLInferenceEngine::initialize() {
  std::cout << "[DEBUG] MLInferenceEngine::initialize called" << std::endl;

  if (initialized_) {
    std::cout << "[DEBUG] Already initialized, returning true" << std::endl;
    return true;
  }

  try {
    // Initialize and select ML backend (prefer GGML for internal forward path)
    auto &backendMgr = duorou::ml::BackendManager::getInstance();
    if (!backendMgr.getCurrentBackend()) {
      backendMgr.initializeBackends();
      // Prefer GGML if available; fallback to existing current backend
      bool switched = backendMgr.setCurrentBackend(duorou::ml::DeviceType::GGML);
      std::cout << "[DEBUG] Backend initialized, selected="
                << duorou::ml::deviceTypeToString(
                       switched ? duorou::ml::DeviceType::GGML
                                : backendMgr.getCurrentBackend()
                                      ? backendMgr.getCurrentBackend()->getType()
                                      : duorou::ml::DeviceType::CPU)
                << std::endl;
    }

    // Create ML context using the selected backend
    ml_context_ = new duorou::ml::Context(backendMgr.getCurrentBackend());

    // Create attention mechanism (simplified configuration)
    attention_ = new duorou::ml::nn::MultiHeadAttention(512,  // embed_dim
                                                8,    // num_heads
                                                -1,   // kv_heads (default)
                                                true, // bias
                                                0.1f  // dropout
    );

    // Try to get model path from OllamaModelManager
    OllamaModelManager &manager = GlobalModelManager::getInstance();

    // Use model_id_ directly, as OllamaModelManager::getModelInfo will perform normalization
    auto model_info = manager.getModelInfo(model_id_);
    if (!model_info || model_info->file_path.empty()) {
      std::cerr << "[ERROR] MLInferenceEngine: Model not found or file path "
                   "empty in OllamaModelManager: "
                << model_id_ << std::endl;
      initialized_ = false;
      return false;
    }

    model_path_ = model_info->file_path;

    // Step 1: Use GGUFParser to parse file and get architecture
    gguf_parser_ = std::make_unique<GGUFParser>();
    if (!gguf_parser_->parseFile(model_path_)) {
      std::cerr << "[ERROR] MLInferenceEngine: Failed to parse GGUF: "
                << model_path_ << std::endl;
      initialized_ = false;
      return false;
    }
    const auto &arch = gguf_parser_->getArchitecture().name;
    // Allow forcing llama.cpp via environment variable DUOROU_FORCE_LLAMA
    const char* force_llama_env = std::getenv("DUOROU_FORCE_LLAMA");
    bool force_llama = force_llama_env && std::string(force_llama_env) != "0" && std::string(force_llama_env).size() > 0;
    bool llama_supported = isSupportedByLlamaCpp(arch);
    if (force_llama && !llama_supported) {
      std::cerr << "[WARN] DUOROU_FORCE_LLAMA requested but architecture '" << arch << "' is unsupported by llama.cpp; using internal forward instead" << std::endl;
    }

    use_llama_backend_ = force_llama && llama_supported;
    std::cout << "[DEBUG] Detected architecture: '" << arch
              << "', use_llama_backend_="
              << (use_llama_backend_ ? "true" : "false")
              << ((force_llama && llama_supported) ? " (forced by DUOROU_FORCE_LLAMA)" : "")
              << std::endl;

    if (use_llama_backend_) {
      // Only initialize llama.cpp backend and load model when llama.cpp is selected
      llama_backend_init();
      // Load llama.cpp model
      if (!loadLlamaModel(model_path_)) {
        std::cerr
            << "[ERROR] MLInferenceEngine: Failed to load llama.cpp model from "
            << model_path_ << std::endl;
        // Gracefully fall back to internal forward initialization
        std::cout << "[INFO] Falling back to internal forward initialization" << std::endl;
        use_llama_backend_ = false;

        // Non-llama architecture: use internal Forward flow with Qwen model
        std::cout
            << "[DEBUG] Initializing Qwen multimodal model for internal forward"
            << std::endl;

        // Create Qwen multimodal model
        qwen_model_ = std::make_unique<duorou::model::QwenMultimodalModel>();

        // First initialize model components (including text model)
        if (!qwen_model_->initialize("")) {
          std::cerr << "[WARN] Failed to initialize Qwen model components, using "
                       "fallback initialization"
                    << std::endl;
          // Even if initialization fails, continue initializing other components for basic inference capability
        }

        // Try to load model from GGUF file
        if (!qwen_model_->loadModel(model_path_)) {
          std::cerr << "[WARN] Failed to load Qwen model from GGUF, using "
                       "fallback initialization"
                    << std::endl;
          // Even if loading fails, continue initializing other components for basic inference capability
        }

        // Check internal forward support based on GGUF architecture and parsed config
        if (!checkInternalForwardSupport()) {
          std::cerr << "[ERROR] Internal forward not supported; aborting initialization" << std::endl;
          initialized_ = false;
          return false;
        }

        // Parse model configuration
        if (!parseModelConfig()) {
          std::cerr << "[ERROR] Failed to parse model configuration from GGUF" << std::endl;
          initialized_ = false;
          return false;
        }

        // Load model weights
        if (!loadModelWeights()) {
          std::cerr << "[ERROR] Failed to load model weights from GGUF" << std::endl;
          initialized_ = false;
          return false;
        }

        // Initialize KV cache
        if (!initializeKVCache()) {
          std::cerr << "[ERROR] Failed to initialize KV cache" << std::endl;
          initialized_ = false;
          return false;
        }

        // Precompute RoPE frequencies
        if (!precomputeRoPEFreqs()) {
          std::cerr << "[ERROR] Failed to precompute RoPE frequencies" << std::endl;
          initialized_ = false;
          return false;
        }

        initialized_ = true;
        std::cout
            << "[DEBUG] MLInferenceEngine initialized successfully with internal forward (fallback from llama.cpp)"
            << std::endl;
        return true;
      }

      initialized_ = true;
      std::cout
          << "[DEBUG] MLInferenceEngine initialized successfully with llama.cpp"
          << std::endl;
      return true;
    }
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
    if (use_llama_backend_) {
      // Use llama.cpp for inference
      return generateWithLlama(prompt, max_tokens, temperature, top_p);
    } else {
      // Use internal Forward mode for inference
      return generateWithInternalForward(prompt, max_tokens, temperature,
                                         top_p);
    }

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in generateText: " << e.what() << std::endl;
    return "Error: " + std::string(e.what());
  }
}

bool MLInferenceEngine::isReady() const {
  // Determine ready status based on backend mode
  if (!initialized_)
    return false;
  if (use_llama_backend_) {
    return ml_context_ != nullptr && attention_ != nullptr &&
           llama_model_ != nullptr && llama_context_ != nullptr &&
           llama_sampler_ != nullptr;
  } else {
    // Internal Forward mode: requires at least ML context, attention, and completed RoPE precomputation
    return ml_context_ != nullptr && attention_ != nullptr && rope_initialized_;
  }
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
    } catch (...) {}
    if (!arch_for_check.empty() && !isSupportedByLlamaCpp(arch_for_check)) {
      std::cerr << "[WARN] llama.cpp does not support architecture '" << arch_for_check
                << "'; skipping llama.cpp load" << std::endl;
      return false;
    }

    // Set model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU only for now

    // Load model
    llama_model_ = llama_load_model_from_file(model_path.c_str(), model_params);
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
  std::cout << "[DEBUG] Tokenizing text: '" << text << "' (length: " << text.length() << ")" << std::endl;
  
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

  // Finally, fall back to simple tokenization (as a last resort)
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
    if (response.length() > max_tokens * 4) { // Rough estimate: 4 characters = 1 token
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
    n_layers_   = arch.block_count ? arch.block_count : n_layers_;
    n_heads_    = arch.attention_head_count ? arch.attention_head_count : n_heads_;
    n_kv_heads_ = arch.attention_head_count_kv ? arch.attention_head_count_kv : (n_kv_heads_ ? n_kv_heads_ : n_heads_);
    n_embd_     = arch.embedding_length ? arch.embedding_length : n_embd_;
    n_ctx_      = arch.context_length ? arch.context_length : n_ctx_;

    // RoPE 参数
    rope_dim_       = arch.rope_dimension_count ? arch.rope_dimension_count : (n_heads_ ? (n_embd_ / n_heads_) : 0);
    rope_freq_base_ = arch.rope_freq_base > 0.0f ? arch.rope_freq_base : 10000.0f;

    // 2) 词表大小优先从 {arch}.vocab_size 读取，退化到 tokenizer.ggml.tokens 长度
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
      if (const auto *kvTokens = gguf_parser_->getMetadata("tokenizer.ggml.tokens")) {
        const auto tokens = kvTokens->asStringArray();
        if (!tokens.empty()) {
          vocab_size_ = static_cast<uint32_t>(tokens.size());
        }
      }
    }

    // 保底默认值（避免为 0 导致后续崩溃）
    if (vocab_size_ == 0) vocab_size_ = 32000;
    if (n_layers_   == 0) n_layers_   = 32;
    if (n_heads_    == 0) n_heads_    = 32;
    if (n_kv_heads_ == 0) n_kv_heads_ = n_heads_;
    if (n_embd_     == 0) n_embd_     = 4096;
    if (n_ctx_      == 0) n_ctx_      = 2048;
    if (rope_dim_   == 0) rope_dim_   = (n_heads_ ? (n_embd_ / n_heads_) : 0);

    // 调试输出（含 RoPE mrope sections）
    std::cout << "[DEBUG] Parsed GGUF architecture: '" << arch.name << "'" << std::endl;
    if (!arch.rope_dimension_sections.empty()) {
      std::cout << "[DEBUG] RoPE mrope sections: ";
      for (size_t i = 0; i < arch.rope_dimension_sections.size(); ++i) {
        std::cout << arch.rope_dimension_sections[i] << (i + 1 < arch.rope_dimension_sections.size() ? "," : "");
      }
      std::cout << std::endl;
    }

    std::cout << "[DEBUG] Model config - vocab_size: " << vocab_size_
              << ", n_layers: " << n_layers_ << ", n_heads: " << n_heads_
              << ", n_kv_heads: " << n_kv_heads_ << ", n_embd: " << n_embd_
              << ", n_ctx: " << n_ctx_ << ", rope_dim: " << rope_dim_
              << ", rope_freq_base: " << rope_freq_base_ << std::endl;

    // 基本合法性检查
    if (n_layers_ == 0 || n_heads_ == 0 || n_embd_ == 0 || n_ctx_ == 0 || vocab_size_ == 0) {
      std::cerr << "[ERROR] Invalid model configuration parsed from GGUF" << std::endl;
      return false;
    }

    return true;
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Failed to parse model config: " << e.what() << std::endl;
    return false;
  }
}

bool MLInferenceEngine::loadModelWeights() {
  try {
    std::cout << "[DEBUG] Loading model weights from GGUF" << std::endl;

    // Clean up previous weights (will be handled by cleanupResources uniqueness too)
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
      std::cerr << "[ERROR] Failed to map tensor weights from GGUF" << std::endl;
      return false;
    }

    // Populate model_weights_ vector for convenience/legacy paths
    model_weights_.reserve(weight_map_.size());
    for (const auto &kv : weight_map_) {
      model_weights_.push_back(kv.second);
    }

    std::cout << "[DEBUG] Loaded " << model_weights_.size() << " tensors via GGUF mapping" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Failed to load model weights: " << e.what() << std::endl;
    return false;
  }
}

bool MLInferenceEngine::initializeKVCache() {
  try {
    std::cout << "[DEBUG] Initializing KV cache with context length: " << n_ctx_ << std::endl;

    // Configure KV cache
    cache_config_.maxSeqLen = n_ctx_;
    cache_config_.maxBatchSize = 32; // Default batch size
    cache_config_.numLayers = n_layers_;
    // 对齐 KV 头数（GQA）：优先使用 n_kv_heads_
    cache_config_.numHeads = (n_kv_heads_ > 0 ? n_kv_heads_ : n_heads_);
    cache_config_.headDim = n_heads_ ? (n_embd_ / n_heads_) : 0;
    cache_config_.dtype = kvcache::DType::FLOAT32;

    // Adapter bridging ml::Backend to kvcache::Backend for backend-aware KV cache allocations
    struct MLKVBackendAdapter : public duorou::kvcache::Backend {
      explicit MLKVBackendAdapter(duorou::ml::Backend* backend) : mlBackend(backend) {}
      void* allocate(size_t bytes) override {
        if (mlBackend) return mlBackend->allocate(bytes);
        return std::malloc(bytes);
      }
      void deallocate(void* ptr) override {
        if (!ptr) return;
        if (mlBackend) mlBackend->deallocate(ptr);
        else std::free(ptr);
      }
      void copy(void* dst, const void* src, size_t bytes) override {
        if (!dst || !src || bytes == 0) return;
        if (mlBackend) mlBackend->copyDeviceToDevice(dst, src, bytes);
        else std::memcpy(dst, src, bytes);
      }
      duorou::ml::Backend* mlBackend;
    };

    MLKVBackendAdapter kvAdapter(ml_context_ ? ml_context_->getBackend() : nullptr);
    kvcache::Context kvCtx(&kvAdapter);

    // Instantiate causal KV cache and initialize with backend-aware context
    kv_cache_ = std::make_unique<kvcache::CausalCache>();
    kv_cache_->init(kvCtx, cache_config_);

    std::cout << "[DEBUG] KV cache instantiated and initialized (type=Causal, maxSeqLen: "
              << cache_config_.maxSeqLen
              << ", numLayers: " << cache_config_.numLayers
              << ", numHeads: " << cache_config_.numHeads
              << ", headDim: " << cache_config_.headDim << ")" << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Failed to initialize KV cache: " << e.what() << std::endl;
    return false;
  }
}

bool MLInferenceEngine::precomputeRoPEFreqs() {
  try {
    std::cout << "[DEBUG] Precomputing RoPE frequencies" << std::endl;

    if (n_heads_ == 0) {
      std::cerr << "[ERROR] n_heads_ is zero, cannot compute head_dim" << std::endl;
      return false;
    }

    const uint32_t head_dim = n_embd_ / n_heads_;
    const uint32_t rope_dim = rope_dim_ > 0 ? rope_dim_ : head_dim;
    const float theta = rope_freq_base_ > 0.0f ? rope_freq_base_ : 10000.0f;

    // Clean up previous frequencies
    rope_freqs_.clear();
    rope_freqs_.reserve(rope_dim / 2);

    // Calculate RoPE frequencies（与 GGUF 的 rope.dimension_count & rope.freq_base 对齐）
    for (uint32_t i = 0; i < rope_dim / 2; ++i) {
      float freq = 1.0f / std::pow(theta, (2.0f * i) / static_cast<float>(rope_dim));
      rope_freqs_.push_back(freq);
    }

    rope_initialized_ = true;

    std::cout << "[DEBUG] Precomputed " << rope_freqs_.size()
              << " RoPE frequencies (rope_dim=" << rope_dim
              << ", theta=" << theta << ")" << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Failed to precompute RoPE frequencies: " << e.what() << std::endl;
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

// Internal Forward mode: Use Qwen model for actual inference
std::string
MLInferenceEngine::generateWithInternalForward(const std::string &prompt,
                                               uint32_t max_tokens,
                                               float temperature, float top_p) {
  try {
    std::cout << "[DEBUG] [InternalForward] Starting Qwen model tensor forward with KV cache" << std::endl;

    // Check if Qwen model and ML context are available
    if (!qwen_model_ || !ml_context_) {
      std::cerr << "[ERROR] Qwen model or ML context not initialized" << std::endl;
      return generateIntelligentResponse(prompt, max_tokens, temperature);
    }

    // First encode text to token IDs
    std::vector<int32_t> input_ids;

    // If llama_model_ exists, use llama.cpp tokenization
    if (llama_model_) {
      std::vector<llama_token> llama_tokens = tokenize(prompt);
      input_ids.reserve(llama_tokens.size());
      for (llama_token token : llama_tokens) {
        input_ids.push_back(static_cast<int32_t>(token));
      }
      std::cout << "[DEBUG] [InternalForward] Using llama.cpp tokenization" << std::endl;
    } else {
      // Fall back to Qwen model tokenizer
      input_ids = qwen_model_->encode(prompt, true);
      std::cout << "[DEBUG] [InternalForward] Using Qwen tokenization" << std::endl;
    }

    if (input_ids.empty()) {
      std::cout << "[WARN] [InternalForward] Failed to encode prompt, using fallback" << std::endl;
      return generateIntelligentResponse(prompt, max_tokens, temperature);
    }

    std::cout << "[DEBUG] [InternalForward] Encoded " << input_ids.size()
              << " tokens from prompt" << std::endl;

    // Remove trailing EOS from input to allow generation to proceed,
    // and remember prompt length so we can strip it from the final output
    size_t prompt_len = input_ids.size();
    const duorou::model::Vocabulary* v = qwen_model_->getVocabulary();
    int32_t eos_id = v ? v->getSpecialId(duorou::model::Special::EOS) : -1;
    if (!input_ids.empty() && eos_id >= 0 && input_ids.back() == eos_id) {
      input_ids.pop_back();
      prompt_len = input_ids.size();
      std::cout << "[DEBUG] [InternalForward] Removed trailing EOS from prompt tokens" << std::endl;
    }

    // Prepare generation loop
    std::vector<int32_t> sequence_ids = input_ids; // prompt + generated

    // Ensure KV cache is initialized
    if (!kv_cache_) {
      std::cout << "[WARN] [InternalForward] KV cache not initialized; proceeding without cache" << std::endl;
    } else {
      std::cout << "[DEBUG] [InternalForward] KV cache available; will be passed into forward()" << std::endl;
    }

    // Random engine for sampling
    static thread_local std::mt19937 rng(std::random_device{}());

    auto apply_temperature = [&](std::vector<float>& logits, float temp){
      if (temp <= 0.0f) return; // handled by argmax path later
      for (auto &x : logits) {
        x /= temp;
      }
    };

    auto softmax = [&](const std::vector<float>& logits){
      std::vector<float> probs(logits.size());
      if (logits.empty()) return probs;
      float max_logit = *std::max_element(logits.begin(), logits.end());
      double sum = 0.0;
      for (size_t i = 0; i < logits.size(); ++i) {
        double e = std::exp(static_cast<double>(logits[i] - max_logit));
        probs[i] = static_cast<float>(e);
        sum += e;
      }
      if (sum > 0.0) {
        for (auto &p : probs) p = static_cast<float>(p / sum);
      }
      return probs;
    };

    auto sample_top_p = [&](const std::vector<float>& probs, float tp){
      if (probs.empty()) return int32_t(-1);
      if (tp >= 1.0f) {
        // Full distribution sampling
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return static_cast<int32_t>(dist(rng));
      }
      // Sort indices by prob desc
      std::vector<int> idx(probs.size());
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });
      // Accumulate until reaching top_p
      std::vector<int> kept;
      std::vector<float> kept_probs;
      float acc = 0.0f;
      for (int id : idx) {
        kept.push_back(id);
        kept_probs.push_back(probs[id]);
        acc += probs[id];
        if (acc >= tp) break;
      }
      // Normalize kept_probs
      if (acc > 0.0f) {
        for (auto &p : kept_probs) p = p / acc;
      }
      // Sample among kept
      std::discrete_distribution<int> dist(kept_probs.begin(), kept_probs.end());
      int pick = dist(rng);
      return static_cast<int32_t>(kept[pick]);
    };

    // Main generation loop performing tensor forward per step
    size_t generated = 0;
    while (generated < max_tokens) {
      duorou::ml::Tensor logits_tensor;

      if (generated == 0) {
        // Prime KV cache with full prompt on first pass
        duorou::ml::Tensor input_tensor({static_cast<int64_t>(sequence_ids.size())}, duorou::ml::DataType::INT32);
        if (ml_context_->getBackend()) {
          input_tensor.setBackend(ml_context_->getBackend());
        }
        if (!sequence_ids.empty()) {
          input_tensor.copyFromHost(sequence_ids.data(), sequence_ids.size() * sizeof(int32_t));
        }
        // Forward pass using tensors and (optionally) KV cache via multimodal model
        logits_tensor = qwen_model_->forward(*ml_context_, input_tensor, {}, kv_cache_.get());
      } else {
        // Step-by-step decode using only the last generated token via text model
        int32_t last_id = sequence_ids.back();
        duorou::ml::Tensor last_token_tensor({1}, duorou::ml::DataType::INT32);
        if (ml_context_->getBackend()) {
          last_token_tensor.setBackend(ml_context_->getBackend());
        }
        last_token_tensor.copyFromHost(&last_id, sizeof(int32_t));
        auto *textModel = qwen_model_->getTextModel();
        if (!textModel) {
          std::cerr << "[ERROR] [InternalForward] textModel is null; falling back" << std::endl;
          return tryAutoFallback("textModel null in stepDecode") ? generateWithLlama(prompt, max_tokens, temperature, top_p)
                                                                  : generateIntelligentResponse(prompt, max_tokens, temperature);
        }
        logits_tensor = textModel->stepDecode(*ml_context_, last_token_tensor, kv_cache_.get());
      }

      // Convert logits to host vector
      std::vector<float> logits;
      if (logits_tensor.numel() > 0) {
        logits.resize(static_cast<size_t>(logits_tensor.numel()));
        logits_tensor.copyToHost(logits.data(), logits.size() * sizeof(float));
      }
      if (logits.empty()) {
        std::cout << "[WARN] [InternalForward] Empty logits from forward(); breaking" << std::endl;
        break;
      }

      // Validate logits size equals vocabulary size; otherwise assert and fallback to llama.cpp
      {
        size_t vocab_size = qwen_model_->getVocabSize();
        if (vocab_size == 0 || logits.size() != vocab_size) {
          std::cerr << "[ERROR] [InternalForward] Logits size mismatch: got " << logits.size()
                    << ", expected vocab size " << vocab_size << "; falling back to llama.cpp" << std::endl;
          // Attempt auto fallback and delegate to llama backend
          if (tryAutoFallback("Logits size mismatch in internal forward")) {
            return generateWithLlama(prompt, max_tokens, temperature, top_p);
          } else {
            // If fallback fails, return intelligent response
            return generateIntelligentResponse(prompt, max_tokens, temperature);
          }
        }
      }

      int32_t next_token = -1;
      if (temperature <= 0.0f) {
        // Greedy: pick argmax
        next_token = static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
      } else {
        // Temperature scaling then top-p sampling
        apply_temperature(logits, temperature);
        std::vector<float> probs = softmax(logits);
        next_token = sample_top_p(probs, std::clamp(top_p, 0.0f, 1.0f));
      }

      if (next_token < 0) {
        std::cout << "[WARN] [InternalForward] Sampling returned invalid token; stopping" << std::endl;
        break;
      }

      sequence_ids.push_back(next_token);
      generated += 1;

      // Stop at EOS if available
      if (eos_id >= 0 && next_token == eos_id) {
        std::cout << "[DEBUG] [InternalForward] Reached EOS token; stopping generation" << std::endl;
        break;
      }
    }

    // Decode output token IDs to text
    std::string result;

    // Only decode newly generated tokens (exclude the prompt)
    std::vector<int32_t> gen_ids;
    if (sequence_ids.size() > prompt_len) {
      gen_ids.assign(sequence_ids.begin() + static_cast<std::ptrdiff_t>(prompt_len), sequence_ids.end());
    }

    if (llama_model_) {
      // llama.cpp detokenization
      std::vector<llama_token> llama_tokens;
      llama_tokens.reserve(gen_ids.size());
      for (int32_t token_id : gen_ids) {
        llama_tokens.push_back(static_cast<llama_token>(token_id));
      }
      result = gen_ids.empty() ? std::string() : detokenize(llama_tokens);
      std::cout << "[DEBUG] [InternalForward] Using llama.cpp detokenization (excluded prompt tokens)" << std::endl;
    } else {
      // Qwen decoder
      result = gen_ids.empty() ? std::string() : qwen_model_->decode(gen_ids);
      std::cout << "[DEBUG] [InternalForward] Using Qwen detokenization (excluded prompt tokens)" << std::endl;
    }

    if (result.empty()) {
      std::cout << "[WARN] [InternalForward] Failed to decode output tokens, using fallback" << std::endl;
      return generateIntelligentResponse(prompt, max_tokens, temperature);
    }

    std::cout << "[DEBUG] [InternalForward] Generated response: "
              << result.substr(0, 50) << "..." << std::endl;
    return result;

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in generateWithInternalForward: " << e.what() << std::endl;
    return "Error: " + std::string(e.what());
  }
}

void MLInferenceEngine::cleanupResources() {
  std::cout << "[DEBUG] Cleaning up inference engine resources" << std::endl;

  // Clean up ML components
  delete attention_;
  attention_ = nullptr;

  delete ml_context_;
  ml_context_ = nullptr;

  // Delete unique tensors referenced by model_weights_ and weight_map_
  {
    std::unordered_set<duorou::ml::Tensor*> unique;
    for (auto *w : model_weights_) {
      if (w) unique.insert(w);
    }
    for (auto &kv : weight_map_) {
      if (kv.second) unique.insert(kv.second);
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

duorou::ml::DataType duorou::extensions::ollama::MLInferenceEngine::convertGGMLDataType(GGMLTensorType ggmlType) {
  switch (ggmlType) {
  case GGMLTensorType::F32:
    return duorou::ml::DataType::FLOAT32;
  case GGMLTensorType::F16:
    return duorou::ml::DataType::FLOAT16;
  default:
    // For quantized types, store as INT8 by default (block-quantized layout)
    return duorou::ml::DataType::INT8;
  }
}

bool duorou::extensions::ollama::MLInferenceEngine::loadTensorData(const std::string &tensorName, duorou::ml::Tensor *tensor) {
  if (!gguf_parser_) {
    std::cerr << "[ERROR] GGUF parser not initialized" << std::endl;
    return false;
  }
  const auto *info = gguf_parser_->getTensorInfo(tensorName);
  if (!info) {
    std::cerr << "[WARN] Tensor info not found in GGUF: " << tensorName << std::endl;
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
    std::cerr << "[ERROR] Failed to read tensor data: " << tensorName << std::endl;
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
      std::cerr << "[WARN] Skipping tensor due to read failure: " << info.name << std::endl;
      delete t;
      continue;
    }

    weight_map_[info.name] = t;
    mapped++;
  }
  std::cout << "[DEBUG] mapTensorWeights: mapped " << mapped << " tensors" << std::endl;
  return mapped > 0;
}

bool duorou::extensions::ollama::MLInferenceEngine::checkInternalForwardSupport() {
  if (!gguf_parser_) return false;
  std::string arch = gguf_parser_->getArchitecture().name;
  std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);
  // Prefer internal forward for Qwen-family architectures, esp. multimodal variants (vl)
  bool is_qwen = arch.find("qwen") != std::string::npos;
  if (!is_qwen) {
    return false;
  }
  // Basic config presence check
  return true;
}

bool duorou::extensions::ollama::MLInferenceEngine::tryAutoFallback(const std::string &reason) {
  // Check if llama.cpp supports the current architecture before attempting fallback
  std::string arch = gguf_parser_ ? gguf_parser_->getArchitecture().name : std::string();
  std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);
  bool llama_supported = isSupportedByLlamaCpp(arch);
  if (!llama_supported) {
    std::cerr << "[WARN] Auto fallback skipped: architecture '" << arch
              << "' is unsupported by llama.cpp; staying on internal forward" << std::endl;
    use_llama_backend_ = false;
    // Do not attempt llama.cpp; let caller handle initialization failure gracefully
    return false;
  }

  std::cerr << "[INFO] Attempting auto fallback to llama.cpp due to: " << reason << std::endl;
  use_llama_backend_ = true;
  try {
    llama_backend_init();
    if (!loadLlamaModel(model_path_)) {
      std::cerr << "[ERROR] Auto fallback failed: could not load llama.cpp model" << std::endl;
      use_llama_backend_ = false;
      return false;
    }
    initialized_ = true;
    std::cout << "[DEBUG] Auto fallback to llama.cpp succeeded" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception during auto fallback: " << e.what() << std::endl;
    use_llama_backend_ = false;
    return false;
  }
}