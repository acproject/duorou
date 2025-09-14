#ifndef LLAMA_TEXT_PROCESSOR_H
#define LLAMA_TEXT_PROCESSOR_H

#include "llama_vocab_adapter.h"
#include "text_processor.h"
#include <memory>

namespace duorou {
namespace extensions {
namespace ollama {

// Text processor implementation using llama.cpp vocabulary
class LlamaTextProcessor : public TextProcessor {
public:
  explicit LlamaTextProcessor(std::shared_ptr<LlamaVocabAdapter> vocab);
  ~LlamaTextProcessor() override = default;

  // Encode text to token IDs
  std::vector<int32_t> encode(const std::string &text,
                              bool add_special = true) override;

  // Decode token IDs to text
  std::string decode(const std::vector<int32_t> &tokens) override;

  // Check if token is a special token
  bool is(int32_t token_id, Special special) const override;

  // Get vocabulary (returns the adapter as base Vocabulary)
  const Vocabulary *getVocabulary() const override;

  // Get vocabulary size
  size_t getVocabSize() const override;

private:
  std::shared_ptr<LlamaVocabAdapter> vocab_adapter_;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // LLAMA_TEXT_PROCESSOR_H