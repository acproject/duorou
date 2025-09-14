#include "llama_vocab_adapter.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace ollama {

LlamaVocabAdapter::LlamaVocabAdapter()
    : vocab_(nullptr), model_(nullptr), bos_token_(LLAMA_TOKEN_NULL),
      eos_token_(LLAMA_TOKEN_NULL), unk_token_(LLAMA_TOKEN_NULL) {}

LlamaVocabAdapter::~LlamaVocabAdapter() {
  if (model_) {
    ::llama_free_model(model_);
  }
}

bool LlamaVocabAdapter::initializeFromFile(const std::string &modelPath) {
  try {
    // Create model parameters with vocab_only = true
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;

    // Load model (vocab only)
    model_ = ::llama_load_model_from_file(modelPath.c_str(), model_params);
    if (!model_) {
      std::cerr << "Failed to load model from: " << modelPath << std::endl;
      return false;
    }

    // Get vocab from model
    vocab_ = ::llama_model_get_vocab(model_);
    if (!vocab_) {
      std::cerr << "Failed to get vocab from model" << std::endl;
      ::llama_free_model(model_);
      model_ = nullptr;
      return false;
    }

    updateSpecialTokens();
    return true;

  } catch (const std::exception &e) {
    std::cerr << "Exception in initializeFromFile: " << e.what() << std::endl;
    return false;
  }
}

bool LlamaVocabAdapter::initializeWithVocab(const struct ::llama_vocab *vocab) {
  if (!vocab) {
    return false;
  }

  vocab_ = vocab;
  updateSpecialTokens();
  return true;
}

bool LlamaVocabAdapter::initialize(const std::string &vocabPath) {
  return initializeFromFile(vocabPath);
}

std::vector<int> LlamaVocabAdapter::encode(const std::string &text) {
  if (!vocab_) {
    throw std::runtime_error("Vocabulary not initialized");
  }

  // Get max possible tokens (conservative estimate)
  int32_t max_tokens = text.length() + 16;
  std::vector<llama_token> tokens(max_tokens);

  // Use llama.cpp tokenization
  int32_t n_tokens = ::llama_tokenize(vocab_, text.c_str(), text.length(),
                                      tokens.data(), max_tokens, true, true);

  if (n_tokens < 0) {
    throw std::runtime_error("Tokenization failed");
  }

  // Convert to int vector
  std::vector<int> result;
  result.reserve(n_tokens);
  for (int32_t i = 0; i < n_tokens; ++i) {
    result.push_back(static_cast<int>(tokens[i]));
  }

  return result;
}

std::string LlamaVocabAdapter::decode(const std::vector<int> &tokens) {
  if (!vocab_) {
    throw std::runtime_error("Vocabulary not initialized");
  }

  // Convert int vector to llama_token vector
  std::vector<llama_token> llama_tokens;
  llama_tokens.reserve(tokens.size());
  for (int token : tokens) {
    llama_tokens.push_back(static_cast<llama_token>(token));
  }

  // Estimate buffer size
  int32_t buffer_size = tokens.size() * 8; // Conservative estimate
  std::vector<char> buffer(buffer_size);

  // Use llama.cpp detokenization
  int32_t result_len = ::llama_detokenize(
      vocab_, llama_tokens.data(), static_cast<int32_t>(llama_tokens.size()),
      buffer.data(), buffer_size, false, true);

  if (result_len < 0) {
    throw std::runtime_error("Detokenization failed");
  }

  return std::string(buffer.data(), result_len);
}

bool LlamaVocabAdapter::is(int token,
                           duorou::extensions::ollama::TokenType type) {
  if (!vocab_) {
    return false;
  }

  llama_token llama_tok = static_cast<llama_token>(token);
  llama_token_attr attr = ::llama_vocab_get_attr(vocab_, llama_tok);

  using TokenType = duorou::extensions::ollama::TokenType;
  switch (type) {
  case duorou::extensions::ollama::TOKEN_TYPE_NORMAL:
    return (attr & LLAMA_TOKEN_ATTR_NORMAL) != 0;
  case duorou::extensions::ollama::TOKEN_TYPE_UNKNOWN:
    return (attr & LLAMA_TOKEN_ATTR_UNKNOWN) != 0;
  case duorou::extensions::ollama::TOKEN_TYPE_CONTROL:
    return (attr & LLAMA_TOKEN_ATTR_CONTROL) != 0;
  case duorou::extensions::ollama::TOKEN_TYPE_USER_DEFINED:
    return (attr & LLAMA_TOKEN_ATTR_USER_DEFINED) != 0;
  case duorou::extensions::ollama::TOKEN_TYPE_UNUSED:
    return (attr & LLAMA_TOKEN_ATTR_UNUSED) != 0;
  case duorou::extensions::ollama::TOKEN_TYPE_BYTE:
    return (attr & LLAMA_TOKEN_ATTR_BYTE) != 0;
  default:
    return false;
  }
}

void LlamaVocabAdapter::setBOS(int token) {
  bos_token_ = static_cast<llama_token>(token);
}

void LlamaVocabAdapter::setEOS(int token) {
  eos_token_ = static_cast<llama_token>(token);
}

void LlamaVocabAdapter::merge(const Vocabulary &other) {
  // For llama.cpp vocab, merging is not typically supported
  // This would require creating a new combined vocabulary
  throw std::runtime_error(
      "Vocabulary merging not supported for llama.cpp vocab");
}

std::unordered_map<std::string, int> LlamaVocabAdapter::getSpecialVocabulary() {
  std::unordered_map<std::string, int> specials;

  if (!vocab_) {
    return specials;
  }

  // Add known special tokens
  llama_token bos = ::llama_vocab_bos(vocab_);
  if (bos != LLAMA_TOKEN_NULL) {
    specials["<bos>"] = static_cast<int>(bos);
  }

  llama_token eos = ::llama_vocab_eos(vocab_);
  if (eos != LLAMA_TOKEN_NULL) {
    specials["<eos>"] = static_cast<int>(eos);
  }

  llama_token unk = ::llama_vocab_pad(vocab_); // Use pad as unk fallback
  if (unk != LLAMA_TOKEN_NULL) {
    specials["<unk>"] = static_cast<int>(unk);
  }

  llama_token pad = ::llama_vocab_pad(vocab_);
  if (pad != LLAMA_TOKEN_NULL) {
    specials["<pad>"] = static_cast<int>(pad);
  }

  return specials;
}

void LlamaVocabAdapter::addSpecials(
    const std::unordered_map<std::string, int> &specials) {
  // For llama.cpp vocab, special tokens are typically fixed
  // This operation is not supported
  throw std::runtime_error(
      "Adding special tokens not supported for llama.cpp vocab");
}

llama_token LlamaVocabAdapter::getBOSToken() const {
  if (vocab_) {
    return ::llama_vocab_bos(vocab_);
  }
  return bos_token_;
}

llama_token LlamaVocabAdapter::getEOSToken() const {
  if (vocab_) {
    return ::llama_vocab_eos(vocab_);
  }
  return eos_token_;
}

llama_token LlamaVocabAdapter::getUNKToken() const {
  if (vocab_) {
    return ::llama_vocab_pad(vocab_); // Use pad as unk fallback
  }
  return unk_token_;
}

uint32_t LlamaVocabAdapter::getVocabSize() const {
  if (vocab_) {
    return static_cast<uint32_t>(::llama_vocab_n_tokens(vocab_));
  }
  return 0;
}

std::string LlamaVocabAdapter::getTokenizerModel() const {
  if (vocab_) {
    enum llama_vocab_type type = ::llama_vocab_type(vocab_);
    switch (type) {
    case LLAMA_VOCAB_TYPE_SPM:
      return "llama";
    case LLAMA_VOCAB_TYPE_BPE:
      return "gpt2";
    case LLAMA_VOCAB_TYPE_WPM:
      return "bert";
    case LLAMA_VOCAB_TYPE_UGM:
      return "t5";
    case LLAMA_VOCAB_TYPE_RWKV:
      return "rwkv";
    case LLAMA_VOCAB_TYPE_PLAMO2:
      return "plamo2";
    default:
      return "unknown";
    }
  }
  return "unknown";
}

void LlamaVocabAdapter::updateSpecialTokens() {
  if (vocab_) {
    bos_token_ = ::llama_vocab_bos(vocab_);
    eos_token_ = ::llama_vocab_eos(vocab_);
    unk_token_ = ::llama_vocab_pad(vocab_); // Use pad as unk fallback
  }
}

duorou::extensions::ollama::TokenType
LlamaVocabAdapter::llamaAttrToTokenType(llama_token_attr attr) {
  using TokenType = duorou::extensions::ollama::TokenType;
  if (attr & LLAMA_TOKEN_ATTR_NORMAL)
    return duorou::extensions::ollama::TOKEN_TYPE_NORMAL;
  if (attr & LLAMA_TOKEN_ATTR_UNKNOWN)
    return duorou::extensions::ollama::TOKEN_TYPE_UNKNOWN;
  if (attr & LLAMA_TOKEN_ATTR_CONTROL)
    return duorou::extensions::ollama::TOKEN_TYPE_CONTROL;
  if (attr & LLAMA_TOKEN_ATTR_USER_DEFINED)
    return duorou::extensions::ollama::TOKEN_TYPE_USER_DEFINED;
  if (attr & LLAMA_TOKEN_ATTR_UNUSED)
    return duorou::extensions::ollama::TOKEN_TYPE_UNUSED;
  if (attr & LLAMA_TOKEN_ATTR_BYTE)
    return duorou::extensions::ollama::TOKEN_TYPE_BYTE;
  return duorou::extensions::ollama::TOKEN_TYPE_NORMAL; // default
}

// Implementation of base Vocabulary interface methods
void LlamaVocabAdapter::initialize(const std::vector<std::string> &values,
                                   const std::vector<int32_t> &types,
                                   const std::vector<float> &scores,
                                   const std::vector<std::string> &merges) {
  // This adapter uses llama.cpp vocabulary, so we don't use these parameters
  // They are kept for interface compatibility
}

int32_t LlamaVocabAdapter::encode(const std::string &token) const {
  // Use llama.cpp tokenization for single token
  if (!vocab_)
    return -1;

  // Call the non-const version through const_cast for compatibility
  std::vector<int> tokens =
      const_cast<LlamaVocabAdapter *>(this)->encode(token);
  return tokens.empty() ? -1 : tokens[0];
}

std::string LlamaVocabAdapter::decode(int32_t id) const {
  if (!vocab_)
    return "";
  // Call the non-const version through const_cast for compatibility
  return const_cast<LlamaVocabAdapter *>(this)->decode(
      std::vector<int>{static_cast<int>(id)});
}

bool LlamaVocabAdapter::is(int32_t id,
                           duorou::extensions::ollama::Special special) const {
  switch (special) {
  case duorou::extensions::ollama::SPECIAL_BOS:
    return id == getBOSToken();
  case duorou::extensions::ollama::SPECIAL_EOS:
    return id == getEOSToken();
  default:
    return false;
  }
}

void LlamaVocabAdapter::setBOS(const std::vector<int32_t> &bos_tokens,
                               bool add_bos) {
  // llama.cpp manages BOS tokens internally
}

void LlamaVocabAdapter::setEOS(const std::vector<int32_t> &eos_tokens,
                               bool add_eos) {
  // llama.cpp manages EOS tokens internally
}

int LlamaVocabAdapter::merge(const std::string &left,
                             const std::string &right) const {
  // llama.cpp handles merges internally
  return -1;
}

std::vector<std::string> LlamaVocabAdapter::getSpecialVocabulary() const {
  std::vector<std::string> specials;
  if (vocab_) {
    // Add known special tokens
    specials.push_back("<bos>");
    specials.push_back("<eos>");
    specials.push_back("<unk>");
    specials.push_back("<pad>");
  }
  return specials;
}

std::vector<int32_t>
LlamaVocabAdapter::addSpecials(const std::vector<int32_t> &ids) const {
  // llama.cpp handles special token addition internally
  return ids;
}

} // namespace ollama