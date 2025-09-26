#include "qwen_text_model.h"
#include "../extensions/ollama/gguf_parser.h"
#include "tokenizer_factory.h"
#include "../ml/backend/backend.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <cstring>
#include <cstdlib>

// KV Cache backend adapter bridging ML backend to KV cache backend
namespace {
struct MLKVBackendAdapter : public duorou::kvcache::Backend {
  explicit MLKVBackendAdapter(duorou::ml::Backend* backend) : mlBackend(backend) {}
  void* allocate(size_t bytes) override { if (mlBackend) return mlBackend->allocate(bytes); return std::malloc(bytes); }
  void deallocate(void* ptr) override { if (!ptr) return; if (mlBackend) mlBackend->deallocate(ptr); else std::free(ptr); }
  void copy(void* dst, const void* src, size_t bytes) override { if (!dst || !src || bytes == 0) return; if (mlBackend) mlBackend->copyDeviceToDevice(dst, src, bytes); else std::memcpy(dst, src, bytes); }
  duorou::ml::Backend* mlBackend;
};
} // anonymous namespace

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
  // Initialize MultiHeadAttention (optional, prepared for future tensor-based path)
  mha_ = std::make_unique<duorou::ml::nn::MultiHeadAttention>(
      static_cast<int64_t>(options_.hiddenSize),
      static_cast<int64_t>(options_.numHeads),
      static_cast<int64_t>(options_.numKVHeads),
      /*bias=*/true,
      /*dropout=*/0.0f);
}

std::vector<float>
SelfAttention::forward(duorou::ml::Context &ctx,
                       const std::vector<float> &input,
                       const std::vector<float> &attentionMask,
                       duorou::kvcache::Cache* cache) {
  (void)attentionMask;
  // Keep functional behavior: pass-through hidden, but wire KV cache for future acceleration
  const size_t hiddenSize = options_.hiddenSize;
  if (!cache || hiddenSize == 0 || input.empty() || (input.size() % hiddenSize) != 0) {
    return input;
  }

  // Prepare KV tensors: shape [seq_len, num_kv_heads, head_dim]
  const size_t seqLen = input.size() / hiddenSize;
  const int kvHeads = static_cast<int>(options_.numKVHeads);
  const int numHeads = static_cast<int>(options_.numHeads);
  const int headDim = (numHeads > 0) ? static_cast<int>(hiddenSize / numHeads) : static_cast<int>(options_.ropeDim);

  MLKVBackendAdapter kvAdapter(ctx.getBackend());
  duorou::kvcache::Context kvCtx(&kvAdapter);

  std::vector<int> kvShape = { static_cast<int>(seqLen), kvHeads, headDim };
  duorou::kvcache::Tensor key(kvShape, duorou::kvcache::DType::FLOAT32, &kvAdapter);
  duorou::kvcache::Tensor value(kvShape, duorou::kvcache::DType::FLOAT32, &kvAdapter);

  // Map hidden states to K/V: simple slice of last hidden elements per token
  const size_t kvElems = static_cast<size_t>(kvHeads) * static_cast<size_t>(headDim);
  std::vector<float> kvBuffer(seqLen * kvElems, 0.0f);
  for (size_t t = 0; t < seqLen; ++t) {
    const size_t hBase = t * hiddenSize;
    const size_t kvBase = t * kvElems;
    const size_t copyCount = std::min(kvElems, hiddenSize);
    // Copy the first copyCount elements from hidden to K and V buffers
    std::memcpy(kvBuffer.data() + kvBase, input.data() + hBase, copyCount * sizeof(float));
    // If kvElems > hiddenSize, remaining stays zero
  }

  // Write buffer to key and value
  kvAdapter.copy(key.data(), kvBuffer.data(), key.bytesSize());
  kvAdapter.copy(value.data(), kvBuffer.data(), value.bytesSize());

  // Optionally read past cache (no-op for now)
  try {
    (void)cache->get(kvCtx, /*seq*/0, /*startPos*/0, /*endPos*/ static_cast<int32_t>(seqLen > 0 ? (seqLen - 1) : 0));
  } catch (...) {
    // Ignore cache miss or errors in placeholder integration
  }

  // Store current K/V
  cache->put(kvCtx, key, value);

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
  (void)attentionMask; // unused placeholder

  // KV cache handling is performed inside SelfAttention::forward

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
    std::cerr << "Error: Tokenizer not initialized. Cannot encode text." << std::endl;
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
    std::cerr << "Error: Tokenizer not initialized. Cannot decode tokens." << std::endl;
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

bool QwenTextModel::initialize(const std::string& configPath) {
    return initialize(configPath, false);
}

bool QwenTextModel::initialize(const std::string& configPath, bool skipVocabInit) {
    if (!loadConfig(configPath)) {
        std::cerr << "[ERROR] Failed to load config from: " << configPath << std::endl;
        return false;
    }
    
    if (!skipVocabInit) {
        // Load vocabulary and tokenizer from GGUF
        try {
            duorou::extensions::ollama::GGUFParser parser(/*verbose=*/true);
            if (!parser.parseFile(configPath)) {
                std::cerr << "[ERROR] Failed to parse GGUF: " << configPath << std::endl;
                return false;
            }
            auto vocab = duorou::model::createVocabularyFromGGUF(parser);
            if (!vocab) {
                std::cerr << "[ERROR] Failed to create vocabulary from GGUF: " << configPath << std::endl;
                return false;
            }
            TokenizerFactoryOptions opts; // defaults
            tokenizer_ = duorou::model::createTextProcessorFromGGUF(parser, vocab, opts);
            if (!tokenizer_) {
                std::cerr << "[ERROR] Failed to create tokenizer from GGUF: " << configPath << std::endl;
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception creating tokenizer: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Initialize transformer layers
    layers_.clear();
    layers_.reserve(options_.blockCount);
    for (size_t i = 0; i < options_.blockCount; ++i) {
        layers_.push_back(std::make_unique<TransformerLayer>(options_));
    }
    
    initialized_ = true;
    return true;
}

// Implement BaseModel pure virtual methods
std::string QwenTextModel::getModelType() const {
    return "qwen-text";
}

bool QwenTextModel::isInitialized() const {
    return initialized_;
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

std::vector<float> QwenTextModel::layerNorm(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    float eps) {
  const size_t hidden = options_.hiddenSize;
  if (hidden == 0) return input;
  if (input.empty()) return {};
  if (input.size() % hidden != 0) {
    // Fallback: return input if shape is inconsistent
    return input;
  }
  const size_t seq_len = input.size() / hidden;
  std::vector<float> out(input.size());
  const bool has_scale = (weights.size() == hidden);

  for (size_t t = 0; t < seq_len; ++t) {
    const size_t base = t * hidden;
    // Compute mean
    double mean = 0.0;
    for (size_t i = 0; i < hidden; ++i) {
      mean += static_cast<double>(input[base + i]);
    }
    mean /= static_cast<double>(hidden);

    // Compute variance
    double var = 0.0;
    for (size_t i = 0; i < hidden; ++i) {
      double d = static_cast<double>(input[base + i]) - mean;
      var += d * d;
    }
    var /= static_cast<double>(hidden);
    float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);

    // Normalize and scale (no bias for text model LN here)
    for (size_t i = 0; i < hidden; ++i) {
      float norm = (input[base + i] - static_cast<float>(mean)) * inv_std;
      float scale = has_scale ? weights[i] : 1.0f;
      out[base + i] = norm * scale;
    }
  }
  return out;
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
  // Return logits computed from last token hidden state
  return computeLogitsFromHidden(hiddenStates);
}

// Forward overload with Context/Tensor and KV Cache support (returns hidden states as Tensor)
duorou::ml::Tensor QwenTextModel::forward(
    duorou::ml::Context& ctx,
    const duorou::ml::Tensor& inputIds,
    duorou::kvcache::Cache* cache) {
  size_t n = static_cast<size_t>(inputIds.numel());
  if (n == 0) {
    return duorou::ml::Tensor();
  }
  if (inputIds.dtype() != duorou::ml::DataType::INT32) {
    std::cerr << "[WARN] QwenTextModel::forward expected INT32 inputIds; proceeding with reinterpretation" << std::endl;
  }
  std::vector<int32_t> ids(n, 0);
  inputIds.copyToHost(ids.data(), n * sizeof(int32_t));

  // Embedding + positional encoding
  auto hiddenStates = embedTokens(ids);
  hiddenStates = applyPositionalEncoding(hiddenStates, ids.size());

  // If KV Cache is provided, start forward with batch metadata
  if (cache) {
    MLKVBackendAdapter kvAdapter(ctx.getBackend());
    duorou::kvcache::Context kvCtx(&kvAdapter);
    duorou::kvcache::Batch batch;
    batch.seqs = {0};
    batch.seqLens = {static_cast<int>(ids.size())};
    batch.positions = {static_cast<int>(ids.size() > 0 ? ids.size() - 1 : 0)};
    batch.batchSize = 1;
    try { cache->startForward(kvCtx, batch, false); } catch (...) {}
  }

  // Transformer layers with potential KV Cache usage
  std::vector<float> attentionMask; // placeholder
  for (size_t li = 0; li < layers_.size(); ++li) {
    if (cache) cache->setLayer(static_cast<int>(li));
    hiddenStates = layers_[li]->forward(ctx, hiddenStates, attentionMask, cache);
  }

  // Output normalization
  hiddenStates = layerNorm(hiddenStates, outputNormWeights_, options_.eps);

  // Compute logits from last token hidden and return as Tensor [vocab_size]
  std::vector<float> logits = computeLogitsFromHidden(hiddenStates);
  std::vector<int64_t> shape = {static_cast<int64_t>(logits.size())};
  duorou::ml::Tensor out = duorou::ml::Tensor::zeros(shape, duorou::ml::DataType::FLOAT32);
  float* outData = out.data<float>();
  if (outData && !logits.empty()) {
    std::copy(logits.begin(), logits.end(), outData);
  }
  return out;
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

// Duplicate initialize overload removed to avoid redefinition

::duorou::ml::Tensor QwenTextModel::stepDecode(
    ::duorou::ml::Context& ctx,
    const ::duorou::ml::Tensor& lastTokenId,
    ::duorou::kvcache::Cache* cache) {
  // Expect a single token id
  size_t seqLen = static_cast<size_t>(lastTokenId.numel());
  if (seqLen == 0) {
    return ::duorou::ml::Tensor();
  }
  if (lastTokenId.dtype() != ::duorou::ml::DataType::INT32) {
    std::cerr << "[WARN] QwenTextModel::stepDecode expected INT32 id; proceeding with reinterpretation" << std::endl;
  }
  // Read token id from tensor
  std::vector<int32_t> ids(seqLen, 0);
  lastTokenId.copyToHost(ids.data(), seqLen * sizeof(int32_t));

  // Embedding + positional encoding for this step
  auto hiddenStates = embedTokens(ids);
  hiddenStates = applyPositionalEncoding(hiddenStates, ids.size());

  // If KV Cache is provided, start forward with batch metadata for this step
  if (cache) {
    MLKVBackendAdapter kvAdapter(ctx.getBackend());
    ::duorou::kvcache::Context kvCtx(&kvAdapter);
    ::duorou::kvcache::Batch batch;
    batch.seqs = {0};
    batch.seqLens = {static_cast<int>(ids.size())};
    batch.positions = {static_cast<int>(ids.size() > 0 ? ids.size() - 1 : 0)};
    batch.batchSize = 1;
    try { cache->startForward(kvCtx, batch, false); } catch (...) {}
  }

  // Transformer layers with potential KV Cache usage
  std::vector<float> attentionMask; // placeholder
  for (size_t li = 0; li < layers_.size(); ++li) {
    if (cache) cache->setLayer(static_cast<int>(li));
    hiddenStates = layers_[li]->forward(ctx, hiddenStates, attentionMask, cache);
  }

  // Output normalization
  hiddenStates = layerNorm(hiddenStates, outputNormWeights_, options_.eps);

  // Compute logits and return as tensor
  auto logits = computeLogitsFromHidden(hiddenStates);
  std::vector<int64_t> shape = {static_cast<int64_t>(logits.size())};
  ::duorou::ml::Tensor out = ::duorou::ml::Tensor::zeros(shape, ::duorou::ml::DataType::FLOAT32);
  float* outData = out.data<float>();
  if (outData && !logits.empty()) {
    std::copy(logits.begin(), logits.end(), outData);
  }
  return out;
}

// New helper exposures
size_t QwenTextModel::getHiddenSize() const {
  return options_.hiddenSize;
}

std::vector<float> QwenTextModel::computeLogitsFromHidden(const std::vector<float>& hidden) {
  // hidden is [seq_len * hidden_size]
  size_t hidden_size = options_.hiddenSize;
  if (hidden.size() < hidden_size) {
    return {};
  }
  size_t seq_len = hidden.size() / hidden_size;
  size_t vocab = getVocabSize();
  std::vector<float> logits(vocab, 0.0f);
  if (outputWeights_.size() != hidden_size * vocab) {
    return logits;
  }
  size_t last_offset = (seq_len > 0 ? (seq_len - 1) * hidden_size : 0);
  const float* hptr = hidden.data() + last_offset;
  // outputWeights_ flattened with vocab major: [vocab][hidden]
  for (size_t v = 0; v < vocab; ++v) {
    float sum = 0.0f;
    size_t woff = v * hidden_size;
    for (size_t i = 0; i < hidden_size; ++i) {
      sum += hptr[i] * outputWeights_[woff + i];
    }
    logits[v] = sum;
  }
  return logits;
}

// New: nextToken helper using stepDecode with temperature and top-p sampling
int32_t QwenTextModel::nextToken(
    duorou::ml::Context& ctx,
    const duorou::ml::Tensor& lastTokenId,
    duorou::kvcache::Cache* cache,
    float temperature,
    float topP) {
  // Run stepDecode to get logits
  ::duorou::ml::Tensor logits_tensor = stepDecode(ctx, lastTokenId, cache);
  if (logits_tensor.numel() == 0) {
    return -1;
  }
  std::vector<float> logits(static_cast<size_t>(logits_tensor.numel()));
  logits_tensor.copyToHost(logits.data(), logits.size() * sizeof(float));

  // Temperature scaling
  if (temperature > 0.0f) {
    for (auto &x : logits) x /= temperature;
  }

  // Softmax
  std::vector<float> probs;
  if (!logits.empty()) {
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    probs.resize(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
      double e = std::exp(static_cast<double>(logits[i] - maxLogit));
      probs[i] = static_cast<float>(e);
      sum += e;
    }
    if (sum > 0.0) {
      for (auto &p : probs) p = static_cast<float>(p / sum);
    }
  }

  // Top-p sampling
  if (probs.empty()) return -1;
  if (topP >= 1.0f) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    static thread_local std::mt19937 rng(std::random_device{}());
    return static_cast<int32_t>(dist(rng));
  }
  std::vector<int> idx(probs.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });
  std::vector<int> kept;
  std::vector<float> kept_probs;
  float acc = 0.0f;
  for (int id : idx) {
    kept.push_back(id);
    kept_probs.push_back(probs[id]);
    acc += probs[id];
    if (acc >= topP) break;
  }
  if (acc > 0.0f) {
    for (auto &p : kept_probs) p = p / acc;
  }
  static thread_local std::mt19937 rng(std::random_device{}());
  std::discrete_distribution<int> dist(kept_probs.begin(), kept_probs.end());
  int pick = dist(rng);
  return static_cast<int32_t>(kept[pick]);
}

std::unique_ptr<BaseModel> createQwenTextModel(const std::string& configPath) {
  auto model = std::make_unique<QwenTextModel>();
  if (!model->initialize(configPath)) {
    return nullptr;
  }
  return model;
}

} // namespace model
} // namespace duorou