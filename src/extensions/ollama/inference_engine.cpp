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
    ml_context_ = new ml::Context(backendMgr.getCurrentBackend());

    // Create attention mechanism (simplified configuration)
    attention_ = new ml::nn::MultiHeadAttention(512,  // embed_dim
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

    use_llama_backend_ = force_llama ? true : isSupportedByLlamaCpp(arch);
    std::cout << "[DEBUG] Detected architecture: '" << arch
              << "', use_llama_backend_="
              << (use_llama_backend_ ? "true" : "false")
              << (force_llama ? " (forced by DUOROU_FORCE_LLAMA)" : "")
              << std::endl;

    if (use_llama_backend_) {
      // Only initialize llama.cpp backend and load model when llama.cpp is selected
      llama_backend_init();
      // Load llama.cpp model
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
    // Read model configuration parameters from GGUF file
    // Need to get configuration based on actual GGUFParser interface
    // Assume GGUFParser provides methods to get configuration

    // Get vocabulary size
    vocab_size_ = 32000; // Default value, should be read from GGUF file

    // Get number of model layers
    n_layers_ = 32; // Default value, should be read from GGUF file

    // Get number of attention heads
    n_heads_ = 32; // Default value, should be read from GGUF file

    // Get embedding dimension
    n_embd_ = 4096; // Default value, should be read from GGUF file

    // Get context length
    n_ctx_ = 2048; // Default value, should be read from GGUF file

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

    // Clean up previous weights
    for (auto *weight : model_weights_) {
      delete weight;
    }
    model_weights_.clear();

    // Create weight tensors for each layer
    // This is a simplified implementation, should actually read weight data from GGUF file
    for (uint32_t i = 0; i < n_layers_; ++i) {
      // Create attention weights
      auto *attn_weight =
          new ml::Tensor({n_embd_, n_embd_}, ml::DataType::FLOAT32);
      model_weights_.push_back(attn_weight);

      // Create feedforward network weights
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

    // Configure KV cache
    cache_config_.maxSeqLen = n_ctx_;
    cache_config_.maxBatchSize = 32; // Default batch size
    cache_config_.numLayers = n_layers_;
    cache_config_.numHeads = n_heads_;
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
    std::cerr << "[ERROR] Failed to initialize KV cache: " << e.what()
              << std::endl;
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
    const float theta = 10000.0f;

    // Clean up previous frequencies
    rope_freqs_.clear();
    rope_freqs_.reserve(head_dim / 2);

    // Calculate RoPE frequencies
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
    std::cout << "[DEBUG] [InternalForward] Starting Qwen model inference"
              << std::endl;

    // Check if Qwen model is available
    if (!qwen_model_) {
      std::cerr << "[ERROR] Qwen model not initialized" << std::endl;
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
      std::cout
          << "[WARN] [InternalForward] Failed to encode prompt, using fallback"
          << std::endl;
      return generateIntelligentResponse(prompt, max_tokens, temperature);
    }

    std::cout << "[DEBUG] [InternalForward] Encoded " << input_ids.size()
              << " tokens from prompt" << std::endl;

    // Remove trailing EOS from input to allow generation to proceed,
    // and remember prompt length so we can strip it from the final output
    size_t prompt_len = input_ids.size();
    if (!llama_model_) {
      const duorou::model::Vocabulary* v = qwen_model_->getVocabulary();
      int32_t eos_id = v ? v->getSpecialId(duorou::model::Special::EOS) : -1;
      if (!input_ids.empty() && eos_id >= 0 && input_ids.back() == eos_id) {
        input_ids.pop_back();
        prompt_len = input_ids.size();
        std::cout << "[DEBUG] [InternalForward] Removed trailing EOS from prompt tokens" << std::endl;
      }
    }

    // Call Qwen model's multimodal generation method (without passing images)
    std::vector<int32_t> output_ids =
        qwen_model_->generateMultimodal(input_ids, {}, // Empty image data
                                        max_tokens, temperature, top_p);

    if (output_ids.empty()) {
      std::cout << "[WARN] [InternalForward] Qwen model returned empty output, "
                   "using fallback"
                << std::endl;
      return generateIntelligentResponse(prompt, max_tokens, temperature);
    }

    std::cout << "[DEBUG] [InternalForward] Qwen model generated "
              << output_ids.size() << " output tokens" << std::endl;

    // Decode output token IDs to text
    std::string result;
    
    // If llama_model_ exists, use llama.cpp decoding
    if (llama_model_) {
      // Only decode newly generated tokens (exclude the prompt)
      std::vector<int32_t> gen_ids;
      if (output_ids.size() > prompt_len) {
        gen_ids.assign(output_ids.begin() + static_cast<std::ptrdiff_t>(prompt_len), output_ids.end());
      }

      // Always decode only newly generated tokens; if none, return empty string
      std::vector<llama_token> llama_tokens;
      llama_tokens.reserve(gen_ids.size());
      for (int32_t token_id : gen_ids) {
        llama_tokens.push_back(static_cast<llama_token>(token_id));
      }
      result = gen_ids.empty() ? std::string() : detokenize(llama_tokens);
      std::cout << "[DEBUG] [InternalForward] Using llama.cpp detokenization (excluded prompt tokens)" << std::endl;
    } else {
      // Fall back to Qwen model decoder
      // Only decode newly generated tokens (exclude the prompt)
      std::vector<int32_t> gen_ids;
      if (output_ids.size() > prompt_len) {
        gen_ids.assign(output_ids.begin() + static_cast<std::ptrdiff_t>(prompt_len), output_ids.end());
      }
      // Always decode only newly generated tokens; if none, result will be empty
      result = gen_ids.empty() ? std::string() : qwen_model_->decode(gen_ids);
      std::cout << "[DEBUG] [InternalForward] Using Qwen detokenization (excluded prompt tokens)" << std::endl;
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

  // Clean up ML components
  delete attention_;
  attention_ = nullptr;

  delete ml_context_;
  ml_context_ = nullptr;

  // Clean up model weights
  for (auto *weight : model_weights_) {
    delete weight;
  }
  model_weights_.clear();

  // Clean up KV cache
  kv_cache_.reset();

  // Clean up RoPE frequencies
  rope_freqs_.clear();
  rope_initialized_ = false;

  // Reset configuration
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