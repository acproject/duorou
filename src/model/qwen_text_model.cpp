#include "qwen_text_model.h"
#include "../extensions/ollama/gguf_parser.h"
#include "tokenizer_factory.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <cstring>

namespace duorou {
namespace model {

// ---------------- SelfAttention ----------------
SelfAttention::SelfAttention(const TextModelOptions &options)
    : options_(options) {
  // Placeholder: initialize weights to correct sizes if needed
  queryWeights_.resize(options_.hiddenSize * options_.hiddenSize);
  keyWeights_.resize(options_.hiddenSize * options_.hiddenSize);
  valueWeights_.resize(options_.hiddenSize * options_.hiddenSize);
  outputWeights_.resize(options_.hiddenSize * options_.hiddenSize);
}

std::vector<float>
SelfAttention::forward(duorou::ml::Context &ctx,
                       const std::vector<float> &input,
                       const std::vector<float> & /*attentionMask*/,
                       duorou::kvcache::Cache* /*cache*/) {
  (void)ctx;
  // Placeholder forward: pass-through
  return input;
}

bool SelfAttention::loadWeights(const std::string & /*weightsPath*/) {
  weightsLoaded_ = true;
  return true;
}

// ---------------- FeedForward ----------------
FeedForward::FeedForward(const TextModelOptions &options) : options_(options) {
  gateWeights_.resize(options_.hiddenSize * options_.hiddenSize);
  upWeights_.resize(options_.hiddenSize * options_.hiddenSize);
  downWeights_.resize(options_.hiddenSize * options_.hiddenSize);
}

std::vector<float> FeedForward::forward(const std::vector<float> &input) {
  // Placeholder forward: pass-through
  return input;
}

bool FeedForward::loadWeights(const std::string & /*weightsPath*/) {
  weightsLoaded_ = true;
  return true;
}

// ---------------- TransformerLayer ----------------
TransformerLayer::TransformerLayer(const TextModelOptions &options)
    : options_(options) {
  attention_ = std::make_unique<SelfAttention>(options_);
  feedForward_ = std::make_unique<FeedForward>(options_);
  inputNormWeights_.resize(options_.hiddenSize, 1.0f);
  postAttentionNormWeights_.resize(options_.hiddenSize, 1.0f);
}

std::vector<float>
TransformerLayer::forward(duorou::ml::Context &ctx,
                          const std::vector<float> &input,
                          const std::vector<float> &attentionMask,
                          duorou::kvcache::Cache* cache) {
  // Placeholder: input norm -> attention -> post-attention norm -> FFN
  auto hidden = input;
  // input norm (simplified)
  (void)attentionMask; // unused placeholder
  (void)cache;
  // attention
  hidden = attention_->forward(ctx, hidden, attentionMask, cache);
  // post-attention norm (simplified)
  // FFN
  hidden = feedForward_->forward(hidden);
  return hidden;
}

bool TransformerLayer::loadWeights(const std::string & /*weightsPath*/,
                                   size_t /*layerIndex*/) {
  return attention_->loadWeights("") && feedForward_->loadWeights("");
}

// ---------------- QwenTextModel ----------------
QwenTextModel::QwenTextModel() : QwenTextModel(TextModelOptions{}) {}

QwenTextModel::QwenTextModel(const TextModelOptions &options)
    : options_(options) {
  modelType_ = "qwen-text";

  // Initialize transformer layers
  layers_.reserve(options_.blockCount);
  for (size_t i = 0; i < options_.blockCount; ++i) {
    layers_.push_back(std::make_unique<TransformerLayer>(options_));
  }

  // Initialize embedding and output weights with default vocab size
  size_t vocabSize = 151936; // Qwen default vocab size
  tokenEmbeddings_.resize(vocabSize * options_.hiddenSize);
  outputWeights_.resize(options_.hiddenSize * vocabSize);
  outputNormWeights_.resize(options_.hiddenSize, 1.0f);
}

std::vector<int32_t> QwenTextModel::encode(const std::string &text,
                                           bool addSpecial) {
  if (!tokenizer_) {
    std::cerr << "Error: Tokenizer not initialized. Cannot encode text."
              << std::endl;
    throw std::runtime_error("Tokenizer not initialized");
  }

  auto tokens = tokenizer_->encode(text, addSpecial);
  if (tokens.empty() && !text.empty()) {
    std::cerr << "Warning: Tokenizer returned empty tokens for non-empty text: "
              << text << std::endl;
    // Return a fallback token (prefer UNK special id if available)
    const Vocabulary *v = tokenizer_->getVocabulary();
    int32_t unk_id = v ? v->getSpecialId(Special::UNK) : -1;
    return {unk_id >= 0 ? unk_id : 0};
  }

  return tokens;
}

std::string QwenTextModel::decode(const std::vector<int32_t> &ids) {
  if (!tokenizer_) {
    std::cerr << "Error: Tokenizer not initialized. Cannot decode tokens."
              << std::endl;
    throw std::runtime_error("Tokenizer not initialized");
  }
  return tokenizer_->decode(ids);
}

size_t QwenTextModel::getVocabSize() const {
  if (tokenizer_) {
    return tokenizer_->getVocabSize();
  }
  if (vocabulary_) {
    return vocabulary_->size();
  }
  return 151936; // Default Qwen vocab size
}

const Vocabulary *QwenTextModel::getVocabulary() const {
  if (tokenizer_) {
    return tokenizer_->getVocabulary();
  }
  return vocabulary_.get();
}

bool QwenTextModel::initialize(const std::string &configPath) {
  try {
    // Load configuration
    if (!loadConfig(configPath)) {
      std::cerr << "Failed to load config from: " << configPath << std::endl;
      return false;
    }

    // 1) Prefer loading tokenizer from GGUF (vocab + merges)
    namespace fs = std::filesystem;
    auto findGGUF = [](const std::string &path) -> std::string {
      try {
        fs::path p(path);
        if (fs::exists(p) && fs::is_regular_file(p) &&
            p.extension() == ".gguf") {
          return p.string();
        }
        if (fs::exists(p) && fs::is_directory(p)) {
          for (const auto &entry : fs::directory_iterator(p)) {
            if (entry.is_regular_file() &&
                entry.path().extension() == ".gguf") {
              return entry.path().string();
            }
          }
        }
      } catch (...) {
        // ignore
      }
      return std::string();
    };

    std::string ggufFile = findGGUF(configPath);
    if (ggufFile.empty()) {
      try {
        fs::path p(configPath);
        if (fs::exists(p) && fs::is_regular_file(p)) {
          ggufFile = findGGUF(p.parent_path().string());
        }
      } catch (...) {
        // ignore
      }
    }

    if (!ggufFile.empty()) {
      std::cout << "[DEBUG] QwenTextModel: Found GGUF file: " << ggufFile
                << std::endl;
      duorou::extensions::ollama::GGUFParser parser(/*verbose=*/true);
      if (parser.parseFile(ggufFile)) {
        // Use the new unified GGUF vocabulary creation function
        TokenizerFactoryOptions opts; // defaults; env may override
        tokenizer_ = createTextProcessorFromGGUF(parser, opts);

        if (tokenizer_) {
          std::cout << "[DEBUG] QwenTextModel: Tokenizer created from GGUF successfully"
                    << std::endl;
          // Ensure embedding/output sizes match tokenizer vocab size
          size_t newVocab = tokenizer_->getVocabSize();
          if (newVocab > 0) {
            size_t hidden = options_.hiddenSize;
            tokenEmbeddings_.assign(newVocab * hidden, 0.0f);
            outputWeights_.assign(hidden * newVocab, 0.0f);
            if (outputNormWeights_.size() != hidden) {
              outputNormWeights_.assign(hidden, 1.0f);
            }
          }
          // Try to load weights from GGUF
          if (!loadWeights(ggufFile)) {
            std::cout << "[WARN] QwenTextModel: Failed to load weights from GGUF, continuing with zeros" << std::endl;
          }
          initialized_ = true;
          return true;
        } else {
          std::cerr
              << "[ERROR] QwenTextModel: Failed to create tokenizer from GGUF"
              << std::endl;
        }
      } else {
        std::cerr << "[WARN] QwenTextModel: Failed to parse GGUF file; falling back to default vocab"
                  << std::endl;
      }
    }

    // 2) Fallback: Initialize built-in placeholder vocabulary and BPE tokenizer
    vocabulary_ = std::make_unique<Vocabulary>();

    // Initialize vocabulary with default Qwen vocabulary
    std::vector<std::string> defaultVocab;
    std::vector<int32_t> defaultTypes;
    std::vector<float> defaultScores;
    std::vector<std::string> defaultMerges;

    size_t vocabSize = 151936; // Qwen default vocab size
    if (vocabulary_ && vocabulary_->size() > 0) {
      vocabSize = vocabulary_->size();
    }
    defaultVocab.reserve(vocabSize);
    defaultTypes.reserve(vocabSize);
    defaultScores.reserve(vocabSize);

    // Set special tokens first
    defaultVocab.push_back("<unk>"); // 0: UNK
    defaultTypes.push_back(duorou::model::TOKEN_TYPE_CONTROL);
    defaultScores.push_back(0.0f);

    defaultVocab.push_back("<bos>"); // 1: BOS
    defaultTypes.push_back(duorou::model::TOKEN_TYPE_CONTROL);
    defaultScores.push_back(0.0f);

    defaultVocab.push_back("<eos>"); // 2: EOS
    defaultTypes.push_back(duorou::model::TOKEN_TYPE_CONTROL);
    defaultScores.push_back(0.0f);

    // Fill remaining with placeholder tokens
    while (defaultVocab.size() < vocabSize) {
      defaultVocab.push_back("<token_" + std::to_string(defaultVocab.size()) + ">");
      defaultTypes.push_back(duorou::model::TOKEN_TYPE_NORMAL);
      defaultScores.push_back(0.0f);
    }

    // Initialize vocabulary and specials
    vocabulary_->initialize(defaultVocab, defaultTypes, defaultScores, defaultMerges);
    vocabulary_->setUNK({0});
    vocabulary_->setBOS({1}, true);
    vocabulary_->setEOS({2}, true);

    auto vocabPtr = std::shared_ptr<Vocabulary>(vocabulary_.get(), [](Vocabulary*){});

    // Create tokenizer via factory for Qwen architecture
    std::cout << "[DEBUG] Creating Qwen tokenizer via factory..." << std::endl;
    TokenizerFactoryOptions opts2;
    opts2.override_type = "bpe";
    tokenizer_ = createTextProcessorForArchitecture("qwen", vocabPtr, opts2);

    if (!tokenizer_) {
      std::cerr << "[ERROR] Failed to create tokenizer for Qwen architecture" << std::endl;
      return false;
    }

    // Ensure embedding/output sizes match tokenizer vocab size
    {
      size_t newVocab = tokenizer_->getVocabSize();
      if (newVocab == 0 && vocabPtr)
        newVocab = vocabPtr->size();
      if (newVocab > 0) {
        size_t hidden = options_.hiddenSize;
        tokenEmbeddings_.assign(newVocab * hidden, 0.0f);
        outputWeights_.assign(hidden * newVocab, 0.0f);
        if (outputNormWeights_.size() != hidden) {
          outputNormWeights_.assign(hidden, 1.0f);
        }
      }
    }

    // Initialize layers if not already done
    if (layers_.empty()) {
      layers_.reserve(options_.blockCount);
      for (size_t i = 0; i < options_.blockCount; ++i) {
        layers_.push_back(std::make_unique<TransformerLayer>(options_));
      }
    }

    initialized_ = true;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error initializing QwenTextModel: " << e.what() << std::endl;
    return false;
  }
}

// ---- QwenTextModel helper and forward/generate implementations ----
std::vector<float> QwenTextModel::layerNorm(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    float eps) {
  // Simplified layer norm: pass-through for now
  (void)weights;
  (void)eps;
  return input;
}

std::vector<float> QwenTextModel::softmax(const std::vector<float>& logits) {
  // Stable softmax over the whole vector (placeholder)
  if (logits.empty()) return {};
  float maxLogit = *std::max_element(logits.begin(), logits.end());
  std::vector<float> exps(logits.size());
  float sumExp = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i) {
    exps[i] = std::exp(logits[i] - maxLogit);
    sumExp += exps[i];
  }
  if (sumExp <= 0.0f) sumExp = 1.0f;
  for (auto& v : exps) v /= sumExp;
  return exps;
}

int32_t QwenTextModel::sampleToken(const std::vector<float>& probabilities, float temperature, float topP) {
  (void)temperature; (void)topP;
  if (probabilities.empty()) return 0;
  return static_cast<int32_t>(std::distance(
      probabilities.begin(),
      std::max_element(probabilities.begin(), probabilities.end())));
}

bool QwenTextModel::loadConfig(const std::string& /*configPath*/) {
  return true;
}

bool QwenTextModel::loadWeights(const std::string& /*weightsPath*/) {
  return true;
}

std::vector<float> QwenTextModel::embedTokens(const std::vector<int32_t>& tokenIds) {
  size_t hidden = options_.hiddenSize;
  size_t vocab = getVocabSize();
  std::vector<float> embeddings(tokenIds.size() * hidden, 0.0f);
  if (tokenEmbeddings_.empty()) return embeddings;
  for (size_t t = 0; t < tokenIds.size(); ++t) {
    int32_t id = tokenIds[t];
    if (id < 0) id = 0;
    if (static_cast<size_t>(id) >= vocab) id = static_cast<int32_t>(vocab - 1);
    size_t srcOffset = static_cast<size_t>(id) * hidden;
    size_t dstOffset = t * hidden;
    size_t copyCount = std::min(hidden, tokenEmbeddings_.size() - srcOffset);
    if (copyCount > 0) {
      std::copy_n(tokenEmbeddings_.begin() + srcOffset, copyCount,
                  embeddings.begin() + dstOffset);
    }
  }
  return embeddings;
}

std::vector<float> QwenTextModel::applyPositionalEncoding(
    const std::vector<float>& embeddings, size_t /*sequenceLength*/) {
  return embeddings;
}

std::vector<float> QwenTextModel::forward(const std::vector<int32_t>& inputIds) {
  if (inputIds.empty()) return {};
  auto hiddenStates = embedTokens(inputIds);
  hiddenStates = applyPositionalEncoding(hiddenStates, inputIds.size());

  std::vector<float> attentionMask; // unused placeholder
  duorou::ml::Context dummyCtx;
  for (auto& layer : layers_) {
    hiddenStates = layer->forward(dummyCtx, hiddenStates, attentionMask, nullptr);
  }
  hiddenStates = layerNorm(hiddenStates, outputNormWeights_, options_.eps);
  return hiddenStates;
}

std::vector<int32_t> QwenTextModel::generate(
    const std::vector<int32_t>& inputIds,
    size_t /*maxLength*/, float /*temperature*/, float /*topP*/) {
  std::vector<int32_t> result = inputIds;
  const Vocabulary* v = getVocabulary();
  int32_t eos_id = v ? v->getSpecialId(Special::EOS) : -1;
  if (eos_id >= 0) result.push_back(eos_id);
  return result;
}

bool QwenTextModel::loadModel(const std::string& modelPath) {
  return initialize(modelPath);
}

void QwenTextModel::setOptions(const TextModelOptions& options) {
  options_ = options;
}

bool QwenTextModel::initialize(const std::string& configPath, bool /*skipVocabInit*/) {
  return initialize(configPath);
}

::duorou::ml::Tensor QwenTextModel::forward(
    ::duorou::ml::Context& /*ctx*/,
    const ::duorou::ml::Tensor& inputIds,
    ::duorou::kvcache::Cache* /*cache*/) {
  size_t seqLen = static_cast<size_t>(inputIds.numel());
  const int32_t* idsPtr = inputIds.data<int32_t>();
  std::vector<int32_t> ids;
  ids.reserve(seqLen);
  for (size_t i = 0; i < seqLen; ++i) ids.push_back(idsPtr ? idsPtr[i] : 0);

  std::vector<float> hiddenStates = forward(ids);

  size_t hidden = options_.hiddenSize;
  ::duorou::ml::Tensor out = ::duorou::ml::Tensor::zeros({(int64_t)seqLen, (int64_t)hidden}, ::duorou::ml::DataType::FLOAT32);
  float* outData = out.data<float>();
  if (outData && !hiddenStates.empty()) {
    std::copy(hiddenStates.begin(), hiddenStates.end(), outData);
  }
  return out;
}

std::unique_ptr<BaseModel> createQwenTextModel(const std::string& configPath) {
  auto model = std::make_unique<QwenTextModel>();
  if (!model->initialize(configPath)) {
    std::cerr << "[ERROR] QwenTextModel factory: initialization failed for " << configPath << std::endl;
    return nullptr;
  }
  return model;
}

} // namespace model
} // namespace duorou