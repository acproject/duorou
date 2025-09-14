#include "llama_text_processor.h"

namespace duorou {
namespace extensions {
namespace ollama {

LlamaTextProcessor::LlamaTextProcessor(std::shared_ptr<LlamaVocabAdapter> vocab)
    : vocab_adapter_(vocab) {}

std::vector<int32_t> LlamaTextProcessor::encode(const std::string &text,
                                                bool add_special) {
  if (!vocab_adapter_) {
    return {};
  }
  return vocab_adapter_->encode(text);
}

std::string LlamaTextProcessor::decode(const std::vector<int32_t> &tokens) {
  if (!vocab_adapter_) {
    return "";
  }
  return vocab_adapter_->decode(tokens);
}

bool LlamaTextProcessor::is(int32_t token_id, Special special) const {
  if (!vocab_adapter_) {
    return false;
  }

  switch (special) {
  case SPECIAL_BOS:
    return token_id == vocab_adapter_->getBOSToken();
  case SPECIAL_EOS:
    return token_id == vocab_adapter_->getEOSToken();
  default:
    return false;
  }
}

const Vocabulary *LlamaTextProcessor::getVocabulary() const {
  return vocab_adapter_.get();
}

size_t LlamaTextProcessor::getVocabSize() const {
  if (!vocab_adapter_) {
    return 0;
  }
  return vocab_adapter_->getVocabSize();
}

} // namespace ollama
} // namespace extensions
} // namespace duorou