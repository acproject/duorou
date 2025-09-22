#include "qwen_text_model.h"
#include "../extensions/ollama/gguf_parser.h"
#include "tokenizer_factory.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

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
SelfAttention::forward(const std::vector<float> &input,
                       const std::vector<float> & /*attentionMask*/) {
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
TransformerLayer::forward(const std::vector<float> &input,
                          const std::vector<float> &attentionMask) {
  // Placeholder: input norm -> attention -> post-attention norm -> FFN
  auto hidden = input;
  // input norm (simplified)
  // attention
  hidden = attention_->forward(hidden, attentionMask);
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
          std::cout << "[DEBUG] QwenTextModel: Tokenizer created from GGUF "
                       "successfully"
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
          initialized_ = true;
          return true;
        } else {
          std::cerr
              << "[ERROR] QwenTextModel: Failed to create tokenizer from GGUF"
              << std::endl;
        }
      } else {
        std::cerr << "[WARN] QwenTextModel: Failed to parse GGUF file; falling "
                     "back to default vocab"
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

    // IMPORTANT: do NOT call getVocabSize() here because we have just created
    // an empty `vocabulary_`, which would incorrectly return 0 and lead to
    // initializing a tiny fallback vocab (e.g., 259 tokens). Instead, default
    // to Qwen's known vocab size unless an existing non-empty vocabulary is
    // set.
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

    // Add some ASCII bytes for basic coverage (optional)
    for (int i = 0; i < 256 && defaultVocab.size() < vocabSize; ++i) {
      std::string byteToken;
      if (i < 32 || i >= 127) {
        byteToken = std::string("<0x") + std::to_string(i) + ">";
      } else {
        byteToken = std::string(1, static_cast<char>(i));
      }
      defaultVocab.push_back(byteToken);
      defaultTypes.push_back(duorou::model::TOKEN_TYPE_NORMAL);
      defaultScores.push_back(0.0f);
    }

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

    std::cout << "[DEBUG] Vocabulary initialized with " << vocabulary_->size()
              << " tokens" << std::endl;

    auto vocabPtr = std::shared_ptr<Vocabulary>(vocabulary_.get(), [](Vocabulary*){});

    // Create tokenizer via factory for Qwen architecture
    std::cout << "[DEBUG] Creating Qwen tokenizer via factory..." << std::endl;
    TokenizerFactoryOptions opts;
    opts.override_type = "bpe";
    tokenizer_ = createTextProcessorForArchitecture("qwen", vocabPtr, opts);

    if (!tokenizer_) {
      std::cerr << "[ERROR] Failed to create tokenizer for Qwen architecture" << std::endl;
      return false;
    }

    // Ensure embedding/output sizes match tokenizer vocab size
    {
      size_t newVocab = tokenizer_->getVocabSize();
      if (newVocab == 0 && vocabPtr) newVocab = vocabPtr->size();
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

bool QwenTextModel::initialize(const std::string &configPath, bool skipVocabInit) {
  if (skipVocabInit) {
    try {
      if (!loadConfig(configPath)) {
        std::cerr << "Failed to load config from: " << configPath << std::endl;
        return false;
      }
      if (layers_.empty()) {
        layers_.reserve(options_.blockCount);
        for (size_t i = 0; i < options_.blockCount; ++i) {
          layers_.push_back(std::make_unique<TransformerLayer>(options_));
        }
      }
      initialized_ = true;
      std::cout << "[DEBUG] QwenTextModel initialized with external vocabulary (skipped vocab init)" << std::endl;
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Error initializing QwenTextModel with skipVocabInit: " << e.what() << std::endl;
      return false;
    }
  }
  return initialize(configPath);
}

std::vector<int32_t>
QwenTextModel::generate(const std::vector<int32_t> &inputIds, size_t maxLength,
                        float temperature, float topP) {
  if (!initialized_) {
    return {};
  }
  std::vector<int32_t> result = inputIds;
  int32_t eos_id = 2;
  if (tokenizer_) {
    if (const Vocabulary *v = tokenizer_->getVocabulary()) {
      int32_t vid = v->getSpecialId(Special::EOS);
      if (vid >= 0) eos_id = vid;
    }
  }
  if (!inputIds.empty() && inputIds.back() == eos_id) {
    return result;
  }
  for (size_t i = inputIds.size(); i < maxLength; ++i) {
    auto logits = forward(result);
    auto probabilities = softmax(logits);
    int32_t nextToken = sampleToken(probabilities, temperature, topP);
    result.push_back(nextToken);
    if (nextToken == eos_id) break;
  }
  return result;
}

std::vector<float>
QwenTextModel::forward(const std::vector<int32_t> &inputIds) {
  if (!initialized_ || inputIds.empty()) {
    return {};
  }
  auto embeddings = embedTokens(inputIds);
  embeddings = applyPositionalEncoding(embeddings, inputIds.size());
  auto hidden = embeddings;
  for (auto &layer : layers_) {
    hidden = layer->forward(hidden);
  }
  hidden = layerNorm(hidden, outputNormWeights_);
  size_t hiddenSize = options_.hiddenSize;
  size_t lastTokenStart = (inputIds.size() - 1) * hiddenSize;
  std::vector<float> logits(getVocabSize(), 0.0f);
  for (size_t i = 0; i < logits.size() && i < hiddenSize; ++i) {
    logits[i] = hidden[lastTokenStart + i];
  }
  return logits;
}

std::vector<float>
QwenTextModel::applyPositionalEncoding(const std::vector<float> &embeddings,
                                       size_t sequenceLength) {
  std::vector<float> result = embeddings;
  size_t hiddenSize = options_.hiddenSize;
  for (size_t pos = 0; pos < sequenceLength; ++pos) {
    for (size_t i = 0; i < hiddenSize; ++i) {
      size_t idx = pos * hiddenSize + i;
      result[idx] += std::sin(pos / 10000.0f * (i + 1));
    }
  }
  return result;
}

bool QwenTextModel::loadConfig(const std::string &configPath) {
  std::ifstream file(configPath);
  if (!file.is_open()) {
    return true;
  }
  // TODO: parse and set options_
  return true;
}

bool QwenTextModel::loadWeights(const std::string & /*weightsPath*/) {
  return true;
}

std::vector<float> QwenTextModel::layerNorm(const std::vector<float> &input,
                                            const std::vector<float> &weights,
                                            float eps) {
  std::vector<float> output = input;
  size_t hiddenSize = weights.size();
  if (hiddenSize == 0) return output;
  size_t sequenceLength = input.size() / hiddenSize;
  for (size_t seq = 0; seq < sequenceLength; ++seq) {
    size_t start = seq * hiddenSize;
    float mean = 0.0f;
    for (size_t i = 0; i < hiddenSize; ++i) mean += input[start + i];
    mean /= static_cast<float>(hiddenSize);
    float variance = 0.0f;
    for (size_t i = 0; i < hiddenSize; ++i) {
      float diff = input[start + i] - mean;
      variance += diff * diff;
    }
    variance /= static_cast<float>(hiddenSize);
    float invStd = 1.0f / std::sqrt(variance + eps);
    for (size_t i = 0; i < hiddenSize; ++i) {
      output[start + i] = (input[start + i] - mean) * invStd * weights[i];
    }
  }
  return output;
}

std::vector<float> QwenTextModel::softmax(const std::vector<float> &logits) {
  std::vector<float> probabilities(logits.size());
  if (logits.empty()) return probabilities;
  float maxLogit = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i) {
    probabilities[i] = std::exp(logits[i] - maxLogit);
    sum += probabilities[i];
  }
  if (sum <= 0.0f) return probabilities;
  for (size_t i = 0; i < probabilities.size(); ++i) probabilities[i] /= sum;
  return probabilities;
}

int32_t QwenTextModel::sampleToken(const std::vector<float> &probabilities,
                                   float temperature, float /*topP*/) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  if (probabilities.empty()) return 0;
  std::vector<float> scaled(probabilities.size());
  float invTemp = 1.0f / std::max(temperature, 1e-6f);
  float sum = 0.0f;
  for (size_t i = 0; i < probabilities.size(); ++i) {
    scaled[i] = std::pow(probabilities[i], invTemp);
    sum += scaled[i];
  }
  if (sum <= 0.0f) return 0;
  for (float &p : scaled) p /= sum;
  float r = dis(gen);
  float c = 0.0f;
  for (size_t i = 0; i < scaled.size(); ++i) {
    c += scaled[i];
    if (r <= c) return static_cast<int32_t>(i);
  }
  return static_cast<int32_t>(scaled.size() - 1);
}

std::vector<float> QwenTextModel::embedTokens(const std::vector<int32_t> &tokenIds) {
  size_t hidden = options_.hiddenSize;
  if (hidden == 0 || tokenIds.empty()) return {};
  size_t vocab = getVocabSize();
  std::vector<float> output(tokenIds.size() * hidden, 0.0f);
  for (size_t t = 0; t < tokenIds.size(); ++t) {
    int32_t id = tokenIds[t];
    if (id < 0 || static_cast<size_t>(id) >= vocab) continue;
    size_t embOffset = static_cast<size_t>(id) * hidden;
    size_t outOffset = t * hidden;
    if (embOffset + hidden <= tokenEmbeddings_.size()) {
      std::copy_n(tokenEmbeddings_.begin() + embOffset, hidden,
                  output.begin() + outOffset);
    }
  }
  return output;
}

void QwenTextModel::setOptions(const TextModelOptions &options) {
  options_ = options;
  layers_.clear();
  layers_.reserve(options_.blockCount);
  for (size_t i = 0; i < options_.blockCount; ++i) {
    layers_.push_back(std::make_unique<TransformerLayer>(options_));
  }
  size_t hidden = options_.hiddenSize;
  size_t vocab = getVocabSize();
  tokenEmbeddings_.assign(vocab * hidden, 0.0f);
  outputWeights_.assign(hidden * vocab, 0.0f);
  outputNormWeights_.assign(hidden, 1.0f);
}

bool QwenTextModel::loadModel(const std::string &modelPath) {
  bool ok = loadConfig(modelPath);
  ok = loadWeights(modelPath) && ok;
  return ok;
}

// Factory function
std::unique_ptr<BaseModel> createQwenTextModel(const std::string &configPath) {
  auto model = std::make_unique<QwenTextModel>();
  if (model->initialize(configPath)) {
    return std::move(model);
  }
  return nullptr;
}

} // namespace model
} // namespace duorou