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
  (void)attentionMask; // unused placeholder
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
      if (vid >= 0)
        eos_id = vid;
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
    if (nextToken == eos_id)
      break;
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

  size_t vocab = getVocabSize();
  std::vector<float> logits(vocab, 0.0f);
  if (!outputWeights_.empty() && outputWeights_.size() == hiddenSize * vocab) {
    const float *h = hidden.data() + lastTokenStart;
    const float *W = outputWeights_.data();
    for (size_t i = 0; i < vocab; ++i) {
      float sum = 0.0f;
      for (size_t j = 0; j < hiddenSize; ++j) {
        sum += h[j] * W[j * vocab + i];
      }
      logits[i] = sum;
    }
  } else {
    // Fallback: copy a slice
    for (size_t i = 0; i < logits.size() && i < hiddenSize; ++i) {
      logits[i] = hidden[lastTokenStart + i];
    }
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
      result[idx] += std::sin(static_cast<float>(pos) / 10000.0f * (i + 1));
    }
  }
  return result;
}

bool QwenTextModel::loadConfig(const std::string &configPath) {
  std::ifstream file(configPath);
  if (!file.is_open()) {
    // Optional config; proceed
    return true;
  }
  // TODO: parse if needed
  return true;
}

bool QwenTextModel::loadWeights(const std::string &weightsPath) {
  namespace fs = std::filesystem;
  auto pickGGUF = [](const std::string &path) -> std::string {
    try {
      fs::path p(path);
      if (fs::exists(p) && fs::is_regular_file(p) && p.extension() == ".gguf") {
        return p.string();
      }
      if (fs::exists(p) && fs::is_directory(p)) {
        for (const auto &entry : fs::directory_iterator(p)) {
          if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            return entry.path().string();
          }
        }
      }
      if (fs::exists(p) && fs::is_regular_file(p)) {
        auto parent = p.parent_path();
        if (!parent.empty() && fs::exists(parent) && fs::is_directory(parent)) {
          for (const auto &entry : fs::directory_iterator(parent)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
              return entry.path().string();
            }
          }
        }
      }
    } catch (...) {
    }
    return std::string();
  };

  std::string ggufFile = pickGGUF(weightsPath);
  if (ggufFile.empty()) {
    std::cout << "[WARN] QwenTextModel::loadWeights: No GGUF file found near " << weightsPath << std::endl;
    return true; // not fatal for now
  }

  duorou::extensions::ollama::GGUFParser parser(/*verbose=*/false);
  if (!parser.parseFile(ggufFile)) {
    std::cerr << "[ERROR] QwenTextModel::loadWeights: Failed to parse GGUF: " << ggufFile << std::endl;
    return false;
  }

  auto arch = parser.getArchitecture();
  if (arch.embedding_length > 0 && arch.embedding_length != options_.hiddenSize) {
    options_.hiddenSize = arch.embedding_length;
    std::cout << "[DEBUG] Updated hiddenSize from GGUF to " << options_.hiddenSize << std::endl;
  }

  auto halfToFloat = [](uint16_t h) -> float {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);
    uint32_t f;
    if (exp == 0) {
      if (mant == 0) {
        f = sign;
      } else {
        float m = mant / 1024.0f;
        float val = std::ldexp(m, -14);
        std::memcpy(&f, &val, sizeof(float));
      }
    } else if (exp == 0x1F) {
      f = sign | 0x7F800000 | (mant << 13);
    } else {
      int32_t exp32 = int32_t(exp) - 15 + 127;
      f = sign | (uint32_t(exp32) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
  };

  auto readTensorToFloat = [&](const std::string &name, std::vector<float> &dst, size_t expectedCount) -> bool {
    const auto *info = parser.getTensorInfo(name);
    if (!info) {
      std::cout << "[WARN] Tensor not found in GGUF: " << name << std::endl;
      return false;
    }
    size_t bytes = parser.getTensorSize(name);
    if (bytes == 0) {
      std::cout << "[WARN] Tensor size is 0: " << name << std::endl;
      return false;
    }
    dst.resize(0);
    if (info->type == duorou::extensions::ollama::GGMLTensorType::F32) {
      size_t count = bytes / sizeof(float);
      std::vector<float> buf(count);
      if (!parser.readTensorData(name, buf.data(), bytes)) return false;
      dst.swap(buf);
    } else if (info->type == duorou::extensions::ollama::GGMLTensorType::F16) {
      size_t count = bytes / sizeof(uint16_t);
      std::vector<uint16_t> hbuf(count);
      if (!parser.readTensorData(name, hbuf.data(), bytes)) return false;
      dst.resize(count);
      for (size_t i = 0; i < count; ++i) dst[i] = halfToFloat(hbuf[i]);
    } else {
      std::cout << "[WARN] Unsupported tensor type for " << name << ", skipping" << std::endl;
      return false;
    }
    if (expectedCount > 0 && dst.size() != expectedCount) {
      std::cout << "[WARN] Unexpected element count for " << name << ": got " << dst.size() << ", expected " << expectedCount << std::endl;
    }
    return true;
  };

  size_t vocab = getVocabSize();
  size_t hidden = options_.hiddenSize;

  {
    std::vector<float> emb;
    if (readTensorToFloat("token_embd.weight", emb, vocab * hidden)) {
      tokenEmbeddings_.swap(emb);
      std::cout << "[DEBUG] Loaded token_embd.weight (" << tokenEmbeddings_.size() << " floats)" << std::endl;
    } else {
      std::cout << "[WARN] Using zero-initialized token embeddings" << std::endl;
    }
  }

  {
    std::vector<float> onorm;
    if (readTensorToFloat("output_norm.weight", onorm, hidden)) {
      outputNormWeights_ = onorm;
      std::cout << "[DEBUG] Loaded output_norm.weight (" << outputNormWeights_.size() << " floats)" << std::endl;
    } else if (outputNormWeights_.size() != hidden) {
      outputNormWeights_.assign(hidden, 1.0f);
    }
  }

  {
    std::vector<float> lm;
    if (readTensorToFloat("output.weight", lm, hidden * vocab)) {
      outputWeights_.swap(lm);
      std::cout << "[DEBUG] Loaded output.weight (" << outputWeights_.size() << " floats)" << std::endl;
    } else {
      std::cout << "[WARN] Using zero-initialized output weights" << std::endl;
    }
  }

  return true;
}

std::vector<float> QwenTextModel::layerNorm(const std::vector<float> &input,
                                            const std::vector<float> &weights,
                                            float eps) {
  std::vector<float> output = input;
  size_t hiddenSize = weights.size();
  if (hiddenSize == 0)
    return output;
  size_t sequenceLength = input.size() / hiddenSize;
  for (size_t seq = 0; seq < sequenceLength; ++seq) {
    size_t start = seq * hiddenSize;
    float mean = 0.0f;
    for (size_t i = 0; i < hiddenSize; ++i)
      mean += input[start + i];
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
  if (logits.empty())
    return probabilities;
  float maxLogit = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i) {
    probabilities[i] = std::exp(logits[i] - maxLogit);
    sum += probabilities[i];
  }
  if (sum <= 0.0f)
    return probabilities;
  for (size_t i = 0; i < probabilities.size(); ++i)
    probabilities[i] /= sum;
  return probabilities;
}

int32_t QwenTextModel::sampleToken(const std::vector<float> &probabilities,
                                   float temperature, float topP) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  if (probabilities.empty()) return 0;

  // 1) Temperature scaling
  float invTemp = 1.0f / std::max(temperature, 1e-6f);
  std::vector<float> scaled(probabilities.size());
  float sum = 0.0f;
  for (size_t i = 0; i < probabilities.size(); ++i) {
    float p = probabilities[i];
    p = std::max(p, 0.0f);
    scaled[i] = std::pow(p, invTemp);
    sum += scaled[i];
  }
  if (sum <= 0.0f) return 0;
  for (float &p : scaled) p /= sum;

  // 2) Nucleus (top-p) sampling
  topP = std::clamp(topP, 0.0f, 1.0f);
  if (topP > 0.0f && topP < 1.0f) {
    // sort indices by prob desc
    std::vector<size_t> idx(scaled.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return scaled[a] > scaled[b]; });
    // accumulate until reaching topP
    float csum = 0.0f;
    size_t cutoff = 0;
    for (; cutoff < idx.size(); ++cutoff) {
      csum += scaled[idx[cutoff]];
      if (csum >= topP) { ++cutoff; break; }
    }
    if (cutoff == 0) cutoff = 1; // ensure at least one token
    // renormalize selected subset
    float subsetSum = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) subsetSum += scaled[idx[i]];
    if (subsetSum <= 0.0f) return static_cast<int32_t>(idx[0]);
    // sample within subset
    float r = dis(gen);
    float acc = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) {
      acc += scaled[idx[i]] / subsetSum;
      if (r <= acc) return static_cast<int32_t>(idx[i]);
    }
    return static_cast<int32_t>(idx[cutoff - 1]);
  }

  // Fallback: sample from full distribution
  float r = dis(gen);
  float c = 0.0f;
  for (size_t i = 0; i < scaled.size(); ++i) {
    c += scaled[i];
    if (r <= c) return static_cast<int32_t>(i);
  }
  return static_cast<int32_t>(scaled.size() - 1);
}

std::vector<float>
QwenTextModel::embedTokens(const std::vector<int32_t> &tokenIds) {
  size_t hidden = options_.hiddenSize;
  if (hidden == 0 || tokenIds.empty())
    return {};
  size_t vocab = getVocabSize();
  std::vector<float> output(tokenIds.size() * hidden, 0.0f);
  for (size_t t = 0; t < tokenIds.size(); ++t) {
    int32_t id = tokenIds[t];
    if (id < 0 || static_cast<size_t>(id) >= vocab)
      continue;
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

std::unique_ptr<BaseModel> createQwenTextModel(const std::string &configPath) {
  auto model = std::make_unique<QwenTextModel>();
  if (model->initialize(configPath)) {
    return std::move(model);
  }
  return nullptr;
}

} // namespace model
} // namespace duorou