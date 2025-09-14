#pragma once

#include "../../../third_party/llama.cpp/include/llama.h"
#include "text_processor.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ollama {

// Forward declarations for llama.cpp types
struct llama_vocab;
struct llama_model;
using llama_token = int32_t;
using llama_token_attr = uint32_t;

/**
 * Adapter class that bridges llama.cpp's vocabulary implementation
 * with the existing Vocabulary interface
 */
class LlamaVocabAdapter : public duorou::extensions::ollama::Vocabulary {
public:
  LlamaVocabAdapter();
  ~LlamaVocabAdapter();

  // Initialize from model file (vocab only)
  bool initializeFromFile(const std::string &modelPath);

  // Initialize with existing vocab
  bool initializeWithVocab(const struct ::llama_vocab *vocab);

  // Vocabulary interface implementation
  void initialize(const std::vector<std::string> &values,
                  const std::vector<int32_t> &types,
                  const std::vector<float> &scores,
                  const std::vector<std::string> &merges) override;
  int32_t encode(const std::string &token) const override;
  std::string decode(int32_t id) const override;
  bool is(int32_t id,
          duorou::extensions::ollama::Special special) const override;
  void setBOS(const std::vector<int32_t> &bos_tokens, bool add_bos) override;
  void setEOS(const std::vector<int32_t> &eos_tokens, bool add_eos) override;
  int merge(const std::string &left, const std::string &right) const override;
  std::vector<std::string> getSpecialVocabulary() const override;
  std::vector<int32_t>
  addSpecials(const std::vector<int32_t> &ids) const override;

  // Additional methods for llama.cpp compatibility
  bool initialize(const std::string &vocabPath);
  std::vector<int> encode(const std::string &text);
  std::string decode(const std::vector<int> &tokens);
  bool is(int token, duorou::extensions::ollama::TokenType type);
  void setBOS(int token);
  void setEOS(int token);
  void merge(const Vocabulary &other);
  std::unordered_map<std::string, int> getSpecialVocabulary();
  void addSpecials(const std::unordered_map<std::string, int> &specials);

  // Additional llama.cpp specific methods
  llama_token getBOSToken() const;
  llama_token getEOSToken() const;
  llama_token getUNKToken() const;
  uint32_t getVocabSize() const;
  std::string getTokenizerModel() const;

private:
  const struct ::llama_vocab *vocab_;
  struct ::llama_model *model_; // For keeping model alive

  llama_token bos_token_;
  llama_token eos_token_;
  llama_token unk_token_;

  void updateSpecialTokens();
  duorou::extensions::ollama::TokenType
  llamaAttrToTokenType(llama_token_attr attr);
};

} // namespace ollama