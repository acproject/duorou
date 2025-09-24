#pragma once

#include <cctype>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "text_processor.h"
#include "vocabulary.h"

// GGUF parser (from extensions)
#include "../extensions/ollama/gguf_parser.h"

namespace duorou {
namespace model {

using KVMap = std::unordered_map<std::string, std::string>;

struct TokenizerFactoryOptions {
  // Override tokenizer type: "bpe" or "spm" ("sentencepiece")
  std::string override_type;
  // Override BPE pre-tokenization regex pattern
  std::string override_bpe_pattern;
};

// TextProcessor registry creator signature
using TextProcessorCreator = std::function<std::unique_ptr<TextProcessor>(
    const KVMap &kv, std::shared_ptr<Vocabulary> vocab,
    const TokenizerFactoryOptions &opts)>;

// Register a TextProcessor creator by tokenizer model key (e.g., "llama",
// "gpt2")
void registerTextProcessor(const std::string &key,
                           TextProcessorCreator creator);

// Create TextProcessor by reading tokenizer-related kv (e.g.,
// tokenizer.ggml.model / .pre)
std::unique_ptr<TextProcessor>
getTextProcessor(const KVMap &kv, std::shared_ptr<Vocabulary> vocab,
                 const TokenizerFactoryOptions &opts = {});

// Create Vocabulary from GGUF metadata
std::shared_ptr<Vocabulary>
createVocabularyFromGGUF(const duorou::extensions::ollama::GGUFParser &parser);

// Create TextProcessor from GGUF metadata and a prepared Vocabulary
std::unique_ptr<TextProcessor> createTextProcessorFromGGUF(
    const duorou::extensions::ollama::GGUFParser &parser,
    std::shared_ptr<Vocabulary> vocab,
    const TokenizerFactoryOptions &opts = {});

// Create TextProcessor from GGUF metadata (creates Vocabulary internally)
std::unique_ptr<TextProcessor> createTextProcessorFromGGUF(
    const duorou::extensions::ollama::GGUFParser &parser,
    const TokenizerFactoryOptions &opts = {});

// Create TextProcessor from architecture name (without GGUF), useful for tests
// or non-GGUF loaders
std::unique_ptr<TextProcessor>
createTextProcessorForArchitecture(const std::string &architecture,
                                   std::shared_ptr<Vocabulary> vocab,
                                   const TokenizerFactoryOptions &opts = {});

} // namespace model
} // namespace duorou