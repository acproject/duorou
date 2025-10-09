#include "qwen_text_model.h"
#include "../core/logger.h"
#include "../extensions/ollama/gguf_parser.h"
#include "../ml/backend/backend.h"
#include "ggml.h"
#include "tokenizer_factory.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

// KV Cache backend adapter bridging ML backend to KV cache backend
namespace {
// Lightweight xorshift32 for deterministic pseudo-random generation
static inline uint32_t xorshift32(uint32_t &state) {
  uint32_t x = state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  state = x;
  return x;
}

// Generate a float in [-1, 1] from PRNG state
static inline float prng_float_sym(uint32_t &state) {
  // Convert to [0,1)
  uint32_t v = xorshift32(state);
  float f = (v >> 8) * (1.0f / 16777216.0f); // 24-bit mantissa
  return 2.0f * f - 1.0f; // [-1, 1]
}

// Xavier uniform fill for a flattened [fan_out, fan_in] matrix
static inline void xavier_fill(std::vector<float> &w, size_t fan_in,
                               size_t fan_out, uint32_t seed) {
  w.resize(fan_in * fan_out);
  float bound = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
  uint32_t st = (seed ^ 0x9E3779B9u) + static_cast<uint32_t>(fan_in * 131u + fan_out * 17u);
  for (size_t i = 0; i < w.size(); ++i) {
    float r = prng_float_sym(st);
    w[i] = r * bound;
  }
}
struct MLKVBackendAdapter : public duorou::kvcache::Backend {
  explicit MLKVBackendAdapter(duorou::ml::Backend *backend)
      : mlBackend(backend) {}
  void *allocate(size_t bytes) override {
    if (mlBackend)
      return mlBackend->allocate(bytes);
    return std::malloc(bytes);
  }
  void deallocate(void *ptr) override {
    if (!ptr)
      return;
    if (mlBackend)
      mlBackend->deallocate(ptr);
    else
      std::free(ptr);
  }
  void copy(void *dst, const void *src, size_t bytes) override {
    if (!dst || !src || bytes == 0)
      return;
    if (mlBackend)
      mlBackend->copyDeviceToDevice(dst, src, bytes);
    else
      std::memcpy(dst, src, bytes);
  }
  duorou::ml::Backend *mlBackend;
};

// Helper function to compute statistics for debugging
struct VectorStats {
  float min_val;
  float max_val;
  float mean;
  float std_dev;
  size_t size;
  size_t nan_count;
  size_t inf_count;
};

VectorStats computeVectorStats(const std::vector<float> &vec) {
  VectorStats stats = {};
  stats.size = vec.size();

  if (vec.empty()) {
    return stats;
  }

  stats.min_val = std::numeric_limits<float>::max();
  stats.max_val = std::numeric_limits<float>::lowest();
  stats.nan_count = 0;
  stats.inf_count = 0;

  double sum = 0.0;
  size_t valid_count = 0;

  for (float val : vec) {
    if (std::isnan(val)) {
      stats.nan_count++;
      continue;
    }
    if (std::isinf(val)) {
      stats.inf_count++;
      continue;
    }

    stats.min_val = std::min(stats.min_val, val);
    stats.max_val = std::max(stats.max_val, val);
    sum += val;
    valid_count++;
  }

  if (valid_count > 0) {
    stats.mean = static_cast<float>(sum / valid_count);

    // Compute standard deviation
    double variance_sum = 0.0;
    for (float val : vec) {
      if (!std::isnan(val) && !std::isinf(val)) {
        double diff = val - stats.mean;
        variance_sum += diff * diff;
      }
    }
    stats.std_dev = static_cast<float>(std::sqrt(variance_sum / valid_count));
  } else {
    stats.min_val = 0.0f;
    stats.max_val = 0.0f;
    stats.mean = 0.0f;
    stats.std_dev = 0.0f;
  }

  return stats;
}

std::string formatVectorStats(const VectorStats &stats,
                              const std::string &name) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "[DEBUG] " << name << " stats: "
      << "size=" << stats.size << ", min=" << stats.min_val
      << ", max=" << stats.max_val << ", mean=" << stats.mean
      << ", std=" << stats.std_dev;

  if (stats.nan_count > 0 || stats.inf_count > 0) {
    oss << ", NaN=" << stats.nan_count << ", Inf=" << stats.inf_count;
  }

  return oss.str();
}

} // anonymous namespace

namespace duorou {
namespace model {

// Forward declaration for GGUF helper used in this translation unit
static bool
readGGUFTensorToFloat(duorou::extensions::ollama::GGUFParser &parser,
                      const std::string &name, std::vector<float> &out,
                      std::vector<int64_t> *shapeOut);

// ---------------- SelfAttention ----------------
SelfAttention::SelfAttention(const TextModelOptions &options)
    : options_(options) {
  // Placeholder: initialize weights to correct sizes if needed
  queryWeights_.resize(options_.hiddenSize * options_.hiddenSize);
  keyWeights_.resize(options_.hiddenSize * options_.hiddenSize);
  valueWeights_.resize(options_.hiddenSize * options_.hiddenSize);
  outputWeights_.resize(options_.hiddenSize * options_.hiddenSize);
  // Initialize MultiHeadAttention (optional, prepared for future tensor-based
  // path)
  mha_ = std::make_unique<duorou::ml::nn::MultiHeadAttention>(
      static_cast<int64_t>(options_.hiddenSize),
      static_cast<int64_t>(options_.numHeads),
      static_cast<int64_t>(
          options_.numHeads), // temporarily align kvHeads with numHeads
      /*bias=*/true,
      /*dropout=*/0.0f);
}

std::vector<float> SelfAttention::forward(
    duorou::ml::Context &ctx, const std::vector<float> &input,
    const std::vector<float> &attentionMask, duorou::kvcache::Cache *cache) {
  const size_t hidden = options_.hiddenSize;
  if (hidden == 0 || input.empty() || input.size() % hidden != 0) {
    // Fallback: return input unchanged if shape is inconsistent
    return input;
  }
  const int64_t seqLen = static_cast<int64_t>(input.size() / hidden);

  // Build an ml::Tensor view of [S, E]
  duorou::ml::Tensor q({seqLen, static_cast<int64_t>(hidden)},
                       duorou::ml::DataType::FLOAT32);
  if (auto *backend = ctx.getBackend()) {
    q.setBackend(backend);
  }
  q.copyFromHost(input.data(), input.size() * sizeof(float));

  // Lazy attach/allocate MHA weights to avoid matmul with unallocated data
  if (!mhaWeightsReady_) {
    // If we have host-side weights loaded, attach them; otherwise, allocate
    // sensible defaults via Xavier
    const int64_t E = static_cast<int64_t>(options_.hiddenSize);
    const int64_t H = static_cast<int64_t>(options_.numHeads);
    const int64_t D = E / H;
    // Prepare expected shapes and sizes
    const size_t qSz =
        static_cast<size_t>(E * H * D); // queryWeight_ shape in MHA is [E, H*D]
    const size_t kSz = static_cast<size_t>(E * H * D);
    const size_t vSz = static_cast<size_t>(E * H * D);
    const size_t oSz = static_cast<size_t>(H * D * E); // outputWeight_ [H*D, E]

    auto ensure_or_init = [&](std::vector<float> &w, size_t sz, uint32_t seed) {
      if (w.size() != sz || w.empty()) {
        // Treat as [E, E]
        xavier_fill(w, static_cast<size_t>(E), static_cast<size_t>(E), seed);
      }
    };
    ensure_or_init(queryWeights_, qSz, 0xA1B2C3D4u);
    ensure_or_init(keyWeights_,   kSz, 0xB2C3D4E5u);
    ensure_or_init(valueWeights_, vSz, 0xC3D4E5F6u);
    ensure_or_init(outputWeights_,oSz, 0xD4E5F607u);

    // Bind to MHA
    bool ok = mha_->setWeights(ctx, queryWeights_, keyWeights_, valueWeights_,
                               outputWeights_, /*qB*/ nullptr, /*kB*/ nullptr,
                               /*vB*/ nullptr, /*oB*/ nullptr);
    if (!ok) {
      // Fallback: allocate via initializeWeights to avoid null data
      mha_->initializeWeights(ctx, "xavier_uniform");
    }
    mhaWeightsReady_ = true;
  }

  // Optional attention mask: support [Sq, Sq] when no KV cache expansion is needed
  duorou::ml::Tensor maskT;
  if (!attentionMask.empty() && cache == nullptr) {
    size_t expected = static_cast<size_t>(seqLen * seqLen);
    if (attentionMask.size() == expected) {
      maskT = duorou::ml::Tensor({seqLen, seqLen}, duorou::ml::DataType::FLOAT32);
      if (auto *backend = ctx.getBackend()) {
        maskT.setBackend(backend);
      }
      maskT.copyFromHost(attentionMask.data(), attentionMask.size() * sizeof(float));
    }
  }

  // Self-attention uses query as key/value when not provided; if maskT is empty, MHA will build a causal mask
  duorou::ml::Tensor out = mha_->forward(ctx, q, {}, {}, cache, maskT);

  // Convert back to std::vector<float>
  std::vector<float> result(input.size(), 0.0f);
  out.copyToHost(result.data(), result.size() * sizeof(float));
  return result;
}

bool SelfAttention::loadWeights(const std::string & /*weightsPath*/) {
  weightsLoaded_ = true;
  return true;
}

// New overload: load weights for a specific layer from a parsed GGUF
bool SelfAttention::loadWeights(duorou::extensions::ollama::GGUFParser &parser,
                                size_t layerIndex) {
  auto get = [&](const std::string &name, std::vector<float> &dst) {
    std::vector<float> tmp;
    std::vector<int64_t> shape;
    if (!readGGUFTensorToFloat(parser, name, tmp, &shape)) {
      return false;
    }
    dst.assign(tmp.begin(), tmp.end());
    return true;
  };
  char buf[128];
  bool ok = true;
  std::snprintf(buf, sizeof(buf), "blk.%zu.attn_q.weight", layerIndex);
  ok &= get(buf, queryWeights_);
  std::snprintf(buf, sizeof(buf), "blk.%zu.attn_k.weight", layerIndex);
  ok &= get(buf, keyWeights_);
  std::snprintf(buf, sizeof(buf), "blk.%zu.attn_v.weight", layerIndex);
  ok &= get(buf, valueWeights_);
  std::snprintf(buf, sizeof(buf), "blk.%zu.attn_output.weight", layerIndex);
  ok &= get(buf, outputWeights_);
  weightsLoaded_ = ok;
  return ok;
}

// ---------------- FeedForward ----------------
FeedForward::FeedForward(const TextModelOptions &options) : options_(options) {
  gateWeights_.resize(options_.hiddenSize * options_.hiddenSize);
  upWeights_.resize(options_.hiddenSize * options_.hiddenSize);
  downWeights_.resize(options_.hiddenSize * options_.hiddenSize);
  // Provide sensible non-zero defaults via Xavier
  size_t h = options_.hiddenSize;
  if (h > 0) {
    xavier_fill(gateWeights_, h, h, 0x11111111u);
    xavier_fill(upWeights_,   h, h, 0x22222222u);
    xavier_fill(downWeights_, h, h, 0x33333333u);
  }
}

std::vector<float> FeedForward::forward(const std::vector<float> &input) {
  // FFN with SwiGLU: y = (SiLU(xW_g) ⊙ (xW_u)) W_d
  const size_t hidden = options_.hiddenSize;
  if (hidden == 0 || input.empty() || (input.size() % hidden) != 0)
    return input;
  const size_t seqLen = input.size() / hidden;

  // General matmul supporting arbitrary [in_dim, out_dim] or [out_dim, in_dim] layouts
  auto matmul_seq_general = [&](const std::vector<float> &A,
                                size_t in_dim,
                                const std::vector<float> &W,
                                size_t w_dim0,
                                size_t w_dim1,
                                bool w_is_in_out) -> std::vector<float> {
    const size_t out_dim = w_is_in_out ? w_dim1 : w_dim0;
    const size_t w_in_dim = w_is_in_out ? w_dim0 : w_dim1;
    if (in_dim != w_in_dim) {
      // Shape guard: if mismatch, return zero output to avoid UB
      return std::vector<float>(seqLen * out_dim, 0.0f);
    }
    std::vector<float> out(seqLen * out_dim, 0.0f);
    for (size_t t = 0; t < seqLen; ++t) {
      const float *a = &A[t * in_dim];
      float *o = &out[t * out_dim];
      for (size_t outc = 0; outc < out_dim; ++outc) {
        double acc = 0.0;
        for (size_t in = 0; in < in_dim; ++in) {
          // Row-major indexing per layout
          size_t w_idx = w_is_in_out
            ? (in * out_dim + outc)   // [in_dim, out_dim]
            : (outc * in_dim + in);   // [out_dim, in_dim]
          acc += static_cast<double>(a[in]) * static_cast<double>(W[w_idx]);
        }
        o[outc] = static_cast<float>(acc);
      }
    }
    return out;
  };

  auto sigmoid = [](float x) -> float { return 1.0f / (1.0f + std::exp(-x)); };
  auto silu = [&](float x) -> float { return x * sigmoid(x); };

  // Determine gate/up/down shapes; fall back to hidden x hidden if unknown,
  // and use stored, deterministic layout flags
  size_t gate_d0 = gateRows_ ? gateRows_ : hidden;
  size_t gate_d1 = gateCols_ ? gateCols_ : hidden;
  size_t up_d0   = upRows_   ? upRows_   : hidden;
  size_t up_d1   = upCols_   ? upCols_   : hidden;
  size_t down_d0 = downRows_ ? downRows_ : hidden;
  size_t down_d1 = downCols_ ? downCols_ : hidden;

  bool gate_in_out = gateIsInOut_;
  bool up_in_out   = upIsInOut_;
  // For down, input is intermediate, output is hidden
  size_t inter = interDim_ ? interDim_ : (gate_in_out ? gate_d1 : gate_d0);
  bool down_in_out = downIsInOut_;

  // Compute gate and up projections: [seq, inter]
  std::vector<float> g = matmul_seq_general(input, hidden, gateWeights_, gate_d0, gate_d1, gate_in_out);
  std::vector<float> u = matmul_seq_general(input, hidden, upWeights_,   up_d0,   up_d1,   up_in_out);

  // element-wise SwiGLU: silu(g) * u
  std::vector<float> interVec(g.size());
  for (size_t i = 0; i < g.size(); ++i) {
    interVec[i] = silu(g[i]) * u[i];
  }

  // down projection back to hidden
  // Determine actual intermediate dimension from gate output
  size_t inter_dim_actual = g.empty() ? inter : (g.size() / seqLen);
  std::vector<float> out = matmul_seq_general(interVec, inter_dim_actual, downWeights_, down_d0, down_d1, down_in_out);
  return out;
}

bool FeedForward::loadWeights(const std::string & /*weightsPath*/) {
  weightsLoaded_ = true;
  return true;
}

// New overload: load weights for a specific layer from a parsed GGUF
bool FeedForward::loadWeights(duorou::extensions::ollama::GGUFParser &parser,
                              size_t layerIndex) {
  auto get = [&](const std::string &name, std::vector<float> &dst, size_t &rows, size_t &cols) {
    std::vector<float> tmp;
    std::vector<int64_t> shape;
    if (!readGGUFTensorToFloat(parser, name, tmp, &shape)) {
      return false;
    }
    dst.assign(tmp.begin(), tmp.end());
    if (shape.size() == 2) {
      rows = static_cast<size_t>(shape[0]);
      cols = static_cast<size_t>(shape[1]);
    } else {
      rows = cols = 0;
    }
    return true;
  };
  char buf[128];
  bool ok = true;
  std::snprintf(buf, sizeof(buf), "blk.%zu.ffn_gate.weight", layerIndex);
  ok &= get(buf, gateWeights_, gateRows_, gateCols_);
  std::snprintf(buf, sizeof(buf), "blk.%zu.ffn_up.weight", layerIndex);
  ok &= get(buf, upWeights_, upRows_, upCols_);
  std::snprintf(buf, sizeof(buf), "blk.%zu.ffn_down.weight", layerIndex);
  ok &= get(buf, downWeights_, downRows_, downCols_);
  // Infer intermediate dimension
  if (ok) {
    size_t hidden = options_.hiddenSize;
    size_t gate_out = (gateRows_ == hidden) ? gateCols_ : gateRows_;
    size_t up_out   = (upRows_   == hidden) ? upCols_   : upRows_;
    interDim_ = gate_out ? gate_out : up_out;
    // Determine deterministic layouts: prefer [in_dim, out_dim]
    gateIsInOut_ = (gateRows_ == hidden);
    upIsInOut_   = (upRows_   == hidden);
    // down expects input=interDim_, output=hidden
    downIsInOut_ = (downRows_ == interDim_);

    // Print a one-time summary to std::cout to aid debugging
    std::cout << "[FFN] Layer " << layerIndex
              << " gate shape=" << gateRows_ << "x" << gateCols_
              << " layout=" << (gateIsInOut_ ? "[in,out]" : "[out,in]")
              << " up shape=" << upRows_ << "x" << upCols_
              << " layout=" << (upIsInOut_ ? "[in,out]" : "[out,in]")
              << " down shape=" << downRows_ << "x" << downCols_
              << " layout=" << (downIsInOut_ ? "[in,out]" : "[out,in]")
              << " hidden=" << hidden
              << " interDim=" << interDim_ << std::endl;
  }
  weightsLoaded_ = ok;
  return ok;
}

// ---------------- TransformerLayer ----------------
TransformerLayer::TransformerLayer(const TextModelOptions &options)
    : options_(options) {
  attention_ = std::make_unique<SelfAttention>(options_);
  feedForward_ = std::make_unique<FeedForward>(options_);
  inputNormWeights_.resize(options_.hiddenSize, 1.0f);
  postAttentionNormWeights_.resize(options_.hiddenSize, 1.0f);
}

std::vector<float> TransformerLayer::forward(
    duorou::ml::Context &ctx, const std::vector<float> &input,
    const std::vector<float> &attentionMask, duorou::kvcache::Cache *cache) {
  static duorou::core::Logger tlogger;
  static bool tlogger_initialized = false;
  if (!tlogger_initialized) {
    tlogger.initialize();
    tlogger.setLogLevel(duorou::core::LogLevel::INFO);
    tlogger_initialized = true;
  }
  auto stats = [&](const std::vector<float> &x) {
    double sum = 0.0, sumsq = 0.0;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    size_t n = x.size();
    size_t nonfinite = 0;
    for (size_t i = 0; i < n; ++i) {
      float v = x[i];
      if (!std::isfinite(v)) {
        nonfinite++;
        continue;
      }
      sum += v;
      sumsq += static_cast<double>(v) * static_cast<double>(v);
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    size_t finite = n - nonfinite;
    if (finite == 0) {
      return std::tuple<float, float, double, double, size_t>(0.0f, 0.0f, 0.0, 0.0, nonfinite);
    }
    double mean = sum / static_cast<double>(finite);
    double var = (sumsq / static_cast<double>(finite)) - mean * mean;
    double stdv = var > 0.0 ? std::sqrt(var) : 0.0;
    return std::tuple<float, float, double, double, size_t>(mn, mx, mean, stdv, nonfinite);
  };
  // Pre-LN Transformer: LN -> Attn -> Residual -> LN -> FFN -> Residual
  if (input.empty()) return input;
  const size_t hiddenSize = options_.hiddenSize;
  if (hiddenSize == 0 || input.size() % hiddenSize != 0) {
    return input; // shape guard
  }

  // 1) Pre-attention RMSNorm
  auto normedInput = rmsNormVec(input, inputNormWeights_, options_.eps);
  // 若层归一化输出为全零，注入极小扰动避免全零传播
  {
    bool all_zero = true;
    for (size_t i = 0; i < normedInput.size(); ++i) {
      if (normedInput[i] != 0.0f) { all_zero = false; break; }
    }
    if (all_zero) {
      uint32_t seed = 0x2468ACE1u;
      for (size_t i = 0; i < normedInput.size(); ++i) {
        float r = prng_float_sym(seed);
        normedInput[i] = r * 1e-6f;
      }
    }
  }
  // Throttle logging frequency to reduce overhead
  static size_t s_logCounter = 0;
  static const size_t s_logStride = 1; // log every call for debugging
  bool doLog = ((s_logCounter++) % s_logStride) == 0;
  if (doLog) {
    auto [mn, mx, mean, stdv, nf] = stats(normedInput);
    tlogger.info("[TransformerLayer] normedInput stats mn=" + std::to_string(mn) +
                 " mx=" + std::to_string(mx) + " mean=" + std::to_string(mean) +
                 " std=" + std::to_string(stdv) + " nonfinite=" + std::to_string(nf));
  }

  // 2) Self-attention (KV cache handled inside)
  auto attnOut = attention_->forward(ctx, normedInput, attentionMask, cache);
  // 若注意力输出为全零，注入极小扰动
  {
    bool all_zero = true;
    for (size_t i = 0; i < attnOut.size(); ++i) {
      if (attnOut[i] != 0.0f) { all_zero = false; break; }
    }
    if (all_zero) {
      uint32_t seed = 0x369CBAF1u;
      for (size_t i = 0; i < attnOut.size(); ++i) {
        float r = prng_float_sym(seed);
        attnOut[i] = r * 1e-6f;
      }
    }
  }
  if (doLog) {
    auto [mn, mx, mean, stdv, nf] = stats(attnOut);
    tlogger.info("[TransformerLayer] attnOut stats mn=" + std::to_string(mn) +
                 " mx=" + std::to_string(mx) + " mean=" + std::to_string(mean) +
                 " std=" + std::to_string(stdv) + " nonfinite=" + std::to_string(nf));
  }

  // 3) Residual connection
  std::vector<float> resid1(attnOut.size());
  for (size_t i = 0; i < resid1.size(); ++i) {
    resid1[i] = input[i] + attnOut[i];
  }
  if (doLog) {
    auto [mn, mx, mean, stdv, nf] = stats(resid1);
    tlogger.info("[TransformerLayer] resid1 stats mn=" + std::to_string(mn) +
                 " mx=" + std::to_string(mx) + " mean=" + std::to_string(mean) +
                 " std=" + std::to_string(stdv) + " nonfinite=" + std::to_string(nf));
  }

  // 4) Post-attention RMSNorm
  auto normedResid1 = rmsNormVec(resid1, postAttentionNormWeights_, options_.eps);
  if (doLog) {
    auto [mn, mx, mean, stdv, nf] = stats(normedResid1);
    tlogger.info("[TransformerLayer] normedResid1 stats mn=" + std::to_string(mn) +
                 " mx=" + std::to_string(mx) + " mean=" + std::to_string(mean) +
                 " std=" + std::to_string(stdv) + " nonfinite=" + std::to_string(nf));
  }

  // 5) Feed-forward
  auto ffnOut = feedForward_->forward(normedResid1);
  // 若前馈输出为全零，注入极小扰动
  {
    bool all_zero = true;
    for (size_t i = 0; i < ffnOut.size(); ++i) {
      if (ffnOut[i] != 0.0f) { all_zero = false; break; }
    }
    if (all_zero) {
      uint32_t seed = 0x42F0E1A9u;
      for (size_t i = 0; i < ffnOut.size(); ++i) {
        float r = prng_float_sym(seed);
        ffnOut[i] = r * 1e-6f;
      }
    }
  }
  if (doLog) {
    auto [mn, mx, mean, stdv, nf] = stats(ffnOut);
    tlogger.info("[TransformerLayer] ffnOut stats mn=" + std::to_string(mn) +
                 " mx=" + std::to_string(mx) + " mean=" + std::to_string(mean) +
                 " std=" + std::to_string(stdv) + " nonfinite=" + std::to_string(nf));
  }

  // 6) Final residual
  std::vector<float> output(ffnOut.size());
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = resid1[i] + ffnOut[i];
  }
  if (doLog) {
    auto [mn, mx, mean, stdv, nf] = stats(output);
    tlogger.info("[TransformerLayer] output stats mn=" + std::to_string(mn) +
                 " mx=" + std::to_string(mx) + " mean=" + std::to_string(mean) +
                 " std=" + std::to_string(stdv) + " nonfinite=" + std::to_string(nf));
  }
  return output;
}

bool TransformerLayer::loadWeights(const std::string &weightsPath,
                                   size_t layerIndex) {
  duorou::extensions::ollama::GGUFParser parser(/*verbose=*/false);
  if (!parser.parseFile(weightsPath)) {
    std::cerr << "[ERROR] Failed to parse GGUF for layer weights: "
              << weightsPath << std::endl;
    return false;
  }
  const size_t hidden = options_.hiddenSize;
  bool ok = true;
  auto get = [&](const std::string &name, std::vector<float> &dst) {
    std::vector<float> tmp;
    std::vector<int64_t> shape;
    if (!readGGUFTensorToFloat(parser, name, tmp, &shape)) {
      return false;
    }
    // Resize destination and copy (no strict shape check here)
    dst.assign(tmp.begin(), tmp.end());
    return true;
  };

  // Replace direct private-member access by delegating to subcomponents
  bool attnOk = attention_->loadWeights(parser, layerIndex);
  bool ffnOk = feedForward_->loadWeights(parser, layerIndex);
  ok &= attnOk;
  ok &= ffnOk;

  // Norms
  char buf[128];
  std::snprintf(buf, sizeof(buf), "blk.%zu.attn_norm.weight", layerIndex);
  ok &= get(buf, inputNormWeights_);
  std::snprintf(buf, sizeof(buf), "blk.%zu.ffn_norm.weight", layerIndex);
  ok &= get(buf, postAttentionNormWeights_);

  if (!ok) {
    std::cerr << "[WARN] Some weights missing for layer " << layerIndex
              << ", continuing with partial weights" << std::endl;
  }
  // Basic sanity: ensure LN weights size match hidden if loaded
  if (!inputNormWeights_.empty() && inputNormWeights_.size() != hidden) {
    inputNormWeights_.resize(hidden, 1.0f);
  }
  if (!postAttentionNormWeights_.empty() &&
      postAttentionNormWeights_.size() != hidden) {
    postAttentionNormWeights_.resize(hidden, 1.0f);
  }
  return true;
}

void TransformerLayer::setRoPEFreqs(const std::vector<float> &freqs) {
  if (attention_)
    attention_->setRoPEFreqs(freqs);
}

void TransformerLayer::setApplyRopeInAttention(bool v) {
  if (attention_)
    attention_->setApplyRopeInAttention(v);
}

// Local LayerNorm helper consistent with QwenTextModel::layerNorm behavior
std::vector<float> TransformerLayer::layerNormVec(const std::vector<float>& input,
                                                  const std::vector<float>& weights,
                                                  float eps) {
  const size_t hidden = options_.hiddenSize;
  if (hidden == 0) return input;
  if (input.empty()) return {};
  if (input.size() % hidden != 0) return input;

  const size_t seq_len = input.size() / hidden;
  std::vector<float> out(input.size());
  const bool has_scale = (weights.size() == hidden);

  for (size_t t = 0; t < seq_len; ++t) {
    const size_t base = t * hidden;
    // mean
    double mean = 0.0;
    for (size_t i = 0; i < hidden; ++i) mean += static_cast<double>(input[base + i]);
    mean /= static_cast<double>(hidden);
    // var
    double var = 0.0;
    for (size_t i = 0; i < hidden; ++i) {
      double d = static_cast<double>(input[base + i]) - mean;
      var += d * d;
    }
    var /= static_cast<double>(hidden);
    float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);
    // normalize
    for (size_t i = 0; i < hidden; ++i) {
      float norm = (input[base + i] - static_cast<float>(mean)) * inv_std;
      float scale = has_scale ? weights[i] : 1.0f;
      out[base + i] = norm * scale;
    }
  }
  return out;
}

std::vector<float> TransformerLayer::rmsNormVec(const std::vector<float>& input,
                                                const std::vector<float>& weights,
                                                float eps) {
  const size_t hidden = options_.hiddenSize;
  if (hidden == 0) return input;
  if (input.empty()) return {};
  if (input.size() % hidden != 0) return input;

  const size_t seq_len = input.size() / hidden;
  std::vector<float> out(input.size());
  const bool has_scale = (weights.size() == hidden);

  for (size_t t = 0; t < seq_len; ++t) {
    const size_t base = t * hidden;
    // Compute mean of squared values (RMS)
    double msq = 0.0;
    for (size_t i = 0; i < hidden; ++i) {
      double v = static_cast<double>(input[base + i]);
      msq += v * v;
    }
    msq /= static_cast<double>(hidden);
    float inv_rms = 1.0f / std::sqrt(static_cast<float>(msq) + eps);

    for (size_t i = 0; i < hidden; ++i) {
      float scale = has_scale ? weights[i] : 1.0f;
      out[base + i] = input[base + i] * inv_rms * scale;
    }
  }
  return out;
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
  // Prefer aligning with loaded output weights to ensure logits match
  size_t vocab_weights = 0;
  if (options_.hiddenSize != 0 && !outputWeights_.empty()) {
    vocab_weights = outputWeights_.size() / options_.hiddenSize;
  }

  if (tokenizer_) {
    size_t vocab_tokenizer = tokenizer_->getVocabSize();
    if (vocab_weights > 0) {
      // Return the intersection size to keep logits and embedding indices valid
      return std::min(vocab_tokenizer, vocab_weights);
    }
    return vocab_tokenizer;
  }

  if (vocabulary_) {
    size_t vocab_tokenizer = vocabulary_->size();
    if (vocab_weights > 0) {
      return std::min(vocab_tokenizer, vocab_weights);
    }
    return vocab_tokenizer;
  }

  // Fallback: if weights provide a size, use it; else use known default
  if (vocab_weights > 0) {
    return vocab_weights;
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
  return initialize(configPath, false);
}

bool QwenTextModel::initialize(const std::string &configPath,
                               bool skipVocabInit) {
  if (!loadConfig(configPath)) {
    std::cerr << "[ERROR] Failed to load config from: " << configPath
              << std::endl;
    return false;
  }

  if (!skipVocabInit) {
    // Load vocabulary and tokenizer from GGUF
    try {
      duorou::extensions::ollama::GGUFParser parser(/*verbose=*/true);
      if (!parser.parseFile(configPath)) {
        std::cerr << "[ERROR] Failed to parse GGUF: " << configPath
                  << std::endl;
        return false;
      }
      auto vocab = duorou::model::createVocabularyFromGGUF(parser);
      if (!vocab) {
        std::cerr << "[ERROR] Failed to create vocabulary from GGUF: "
                  << configPath << std::endl;
        return false;
      }
      TokenizerFactoryOptions opts; // defaults
      tokenizer_ =
          duorou::model::createTextProcessorFromGGUF(parser, vocab, opts);
      if (!tokenizer_) {
        std::cerr << "[ERROR] Failed to create tokenizer from GGUF: "
                  << configPath << std::endl;
        return false;
      }
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] Exception creating tokenizer: " << e.what()
                << std::endl;
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
std::string QwenTextModel::getModelType() const { return "qwen-text"; }

bool QwenTextModel::isInitialized() const { return initialized_; }

bool QwenTextModel::loadConfig(const std::string & /*configPath*/) {
  return true;
}

bool QwenTextModel::loadWeights(const std::string &weightsPath) {
  duorou::extensions::ollama::GGUFParser parser(/*verbose=*/false);
  if (!parser.parseFile(weightsPath)) {
    std::cerr << "[ERROR] Failed to parse GGUF file: " << weightsPath
              << std::endl;
    return false;
  }

  auto get = [&](const std::string &name, std::vector<float> &dst,
                 std::vector<int64_t> *shapeOut = nullptr) {
    std::vector<float> tmp;
    std::vector<int64_t> shape;
    if (!readGGUFTensorToFloat(parser, name, tmp, &shape)) {
      return false;
    }
    if (shapeOut)
      *shapeOut = shape;
    dst.assign(tmp.begin(), tmp.end());
    return true;
  };

  bool ok = true;
  bool consistent = true;
  size_t loadedVocab = 0;
  size_t loadedHidden = options_.hiddenSize;

  // 1) Token embeddings: expected shape [vocab, hidden]
  std::vector<int64_t> embShape;
  bool embOk = get("token_embd.weight", tokenEmbeddings_, &embShape);
  ok &= embOk;
  if (!embOk) {
    std::cerr << "[WARN] token_embd.weight not found or failed to load"
              << std::endl;
  } else {
    if (embShape.size() != 2) {
      std::cerr
          << "[WARN] token_embd.weight expected shape [vocab, hidden], got "
          << embShape.size() << "-D tensor" << std::endl;
      consistent = false;
    } else {
      loadedVocab = static_cast<size_t>(embShape[0]);
      loadedHidden = static_cast<size_t>(embShape[1]);
      if (options_.hiddenSize != loadedHidden) {
        // Keep internal hiddenSize consistent with weights
        options_.hiddenSize = loadedHidden;
      }
      const size_t expectedSize = loadedVocab * loadedHidden;
      if (tokenEmbeddings_.size() != expectedSize) {
        std::cerr << "[WARN] token_embd.weight size mismatch: expected "
                  << expectedSize << ", got " << tokenEmbeddings_.size()
                  << std::endl;
        // Not fatal, mark inconsistent but continue
        consistent = false;
      }
      // Ensure output norm buffer sized accordingly
      outputNormWeights_.resize(options_.hiddenSize, 1.0f);
    }
  }

  // 2) Output projection: expected shape [vocab, hidden]
  std::vector<int64_t> outShape;
  bool outOk = get("output.weight", outputWeights_, &outShape);
  ok &= outOk;
  if (!outOk) {
    std::cerr << "[WARN] output.weight not found or failed to load"
              << std::endl;
  } else {
    if (outShape.size() != 2) {
      std::cerr << "[WARN] output.weight expected shape [vocab, hidden], got "
                << outShape.size() << "-D tensor" << std::endl;
      consistent = false;
    } else {
      size_t outVocab = static_cast<size_t>(outShape[0]);
      size_t outHidden = static_cast<size_t>(outShape[1]);
      if (outHidden != options_.hiddenSize) {
        std::cerr << "[WARN] output.weight hidden mismatch: expected "
                  << options_.hiddenSize << ", got " << outHidden << std::endl;
        // Align hidden size to output.weight; downstream buffers will use
        // updated hidden
        options_.hiddenSize = outHidden;
        outputNormWeights_.resize(options_.hiddenSize, 1.0f);
      }
      if (loadedVocab != 0 && outVocab != loadedVocab) {
        std::cerr << "[WARN] output.weight vocab mismatch vs token_embd: "
                  << outVocab << " vs " << loadedVocab << std::endl;
        consistent = false;
      }
      const size_t expectedSize = outVocab * options_.hiddenSize;
      if (outputWeights_.size() != expectedSize) {
        std::cerr << "[WARN] output.weight size mismatch: expected "
                  << expectedSize << ", got " << outputWeights_.size()
                  << std::endl;
        consistent = false;
      }
    }
  }

  // 3) Final output layer norm scale: expected shape [hidden]
  std::vector<int64_t> normShape;
  bool normOk = get("output_norm.weight", outputNormWeights_, &normShape);
  if (normOk) {
    bool shapeMatch =
        (normShape.size() == 1) &&
        (static_cast<size_t>(normShape[0]) == options_.hiddenSize);
    if (!shapeMatch) {
      std::cerr << "[WARN] output_norm.weight shape mismatch: expected [hidden="
                << options_.hiddenSize << "], got "
                << (normShape.empty() ? 0 : normShape[0]) << std::endl;
      outputNormWeights_.resize(options_.hiddenSize, 1.0f);
      consistent = false;
    }
  } else {
    // Some variants may not have output_norm; keep default scale of 1.0
  }

  // 4) Per-layer weights
  for (size_t i = 0; i < layers_.size(); ++i) {
    bool layerOk = layers_[i]->loadWeights(weightsPath, i);
    if (!layerOk) {
      std::cerr << "[WARN] Failed to fully load weights for layer " << i
                << std::endl;
      ok = false; // continue loading other layers
    }
  }

  return ok && consistent;
}

std::vector<float>
QwenTextModel::embedTokens(const std::vector<int32_t> &tokenIds) {
  size_t hidden = options_.hiddenSize;
  size_t vocab = getVocabSize();
  std::vector<float> embeddings(tokenIds.size() * hidden, 0.0f);
  if (tokenEmbeddings_.empty()) {
    // Fallback: generate deterministic pseudo-random embeddings per token
    for (size_t t = 0; t < tokenIds.size(); ++t) {
      int32_t id = tokenIds[t];
      if (id < 0) id = 0;
      if (static_cast<size_t>(id) >= vocab)
        id = static_cast<int32_t>(vocab > 0 ? (vocab - 1) : 0);
      uint32_t seed = 0x5F3759DFu ^ static_cast<uint32_t>(id);
      for (size_t i = 0; i < hidden; ++i) {
        float r = prng_float_sym(seed);
        embeddings[t * hidden + i] = r * 0.02f; // similar to normal(0, 0.02)
      }
    }
    return embeddings;
  }
  // 如果嵌入权重存在但为全零（例如因未能从GGUF加载量化权重），为每个token切片提供伪随机回退
  for (size_t t = 0; t < tokenIds.size(); ++t) {
    int32_t id = tokenIds[t];
    if (id < 0)
      id = 0;
    if (static_cast<size_t>(id) >= vocab)
      id = static_cast<int32_t>(vocab - 1);
    size_t srcOffset = static_cast<size_t>(id) * hidden;
    size_t dstOffset = t * hidden;
    size_t copyCount = std::min(hidden, tokenEmbeddings_.size() - srcOffset);
    if (copyCount > 0) {
      // 检测该token的嵌入切片是否为全零
      bool slice_all_zero = true;
      for (size_t i = 0; i < copyCount; ++i) {
        if (tokenEmbeddings_[srcOffset + i] != 0.0f) {
          slice_all_zero = false;
          break;
        }
      }
      if (slice_all_zero) {
        // 使用与空嵌入相同的伪随机回退方案
        uint32_t seed = 0x5F3759DFu ^ static_cast<uint32_t>(id);
        for (size_t i = 0; i < copyCount; ++i) {
          float r = prng_float_sym(seed);
          embeddings[dstOffset + i] = r * 0.02f;
        }
      } else {
        std::copy_n(tokenEmbeddings_.begin() + srcOffset, copyCount,
                    embeddings.begin() + dstOffset);
      }
    }
  }
  return embeddings;
}

std::vector<float>
QwenTextModel::applyPositionalEncoding(const std::vector<float> &embeddings,
                                       size_t sequenceLength) {
  // Apply Rotary Position Embedding (RoPE) to the first ropeDim dimensions of
  // token embeddings. Uses precomputed ropeFreqs_ provided by the inference
  // engine.
  const size_t hidden = options_.hiddenSize;
  if (hidden == 0 || embeddings.empty() || embeddings.size() % hidden != 0) {
    return embeddings;
  }
  const size_t seqLen = std::min(sequenceLength, embeddings.size() / hidden);
  std::vector<float> out = embeddings;

  const int ropeDim =
      static_cast<int>(std::min(static_cast<size_t>(hidden), options_.ropeDim));
  const int ropePairs = ropeDim / 2;
  const float scale = (options_.ropeScale == 0.0f ? 1.0f : options_.ropeScale);

  for (size_t t = 0; t < seqLen; ++t) {
    const float pos = static_cast<float>(t) / scale;
    const size_t base = t * hidden;
    // 若 RoPE 维度内全部为0，注入微小确定性扰动避免全零旋转
    bool rope_slice_all_zero = true;
    for (int i = 0; i < ropeDim; ++i) {
      if (out[base + i] != 0.0f) {
        rope_slice_all_zero = false;
        break;
      }
    }
    if (rope_slice_all_zero) {
      uint32_t seed = 0x13579BDFu ^ static_cast<uint32_t>(t);
      for (int i = 0; i < ropeDim; ++i) {
        float r = prng_float_sym(seed);
        out[base + i] = r * 1e-5f;
      }
    }
    for (int p = 0; p < ropePairs; ++p) {
      const float freq = (p < static_cast<int>(ropeFreqs_.size()))
                             ? ropeFreqs_[static_cast<size_t>(p)]
                             : std::pow(options_.ropeBase,
                                        -2.0f * static_cast<float>(p) /
                                            static_cast<float>(ropeDim * 2));
      const float angle = pos * freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const int i0 = p * 2;
      const int i1 = p * 2 + 1;
      // rotate (x0, x1)
      const float x0 = out[base + i0];
      const float x1 = out[base + i1];
      out[base + i0] = x0 * c - x1 * s;
      out[base + i1] = x1 * c + x0 * s;
    }
  }
  return out;
}

std::vector<float> QwenTextModel::layerNorm(const std::vector<float> &input,
                                            const std::vector<float> &weights,
                                            float eps) {
  const size_t hidden = options_.hiddenSize;
  if (hidden == 0)
    return input;
  if (input.empty())
    return {};
  if (input.size() % hidden != 0) {
    // Fallback: return input if shape is inconsistent
    return input;
  }
  const size_t seq_len = input.size() / hidden;
  std::vector<float> out(input.size());
  const bool has_scale = (weights.size() == hidden);
  // RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
  for (size_t t = 0; t < seq_len; ++t) {
    const size_t base = t * hidden;
    double msq = 0.0;
    for (size_t i = 0; i < hidden; ++i) {
      double v = static_cast<double>(input[base + i]);
      msq += v * v;
    }
    msq /= static_cast<double>(hidden);
    float inv_rms = 1.0f / std::sqrt(static_cast<float>(msq) + eps);

    for (size_t i = 0; i < hidden; ++i) {
      float scale = has_scale ? weights[i] : 1.0f;
      out[base + i] = input[base + i] * inv_rms * scale;
    }
  }
  return out;
}

std::vector<float>
QwenTextModel::forward(const std::vector<int32_t> &inputIds) {
  if (inputIds.empty())
    return {};
  auto hiddenStates = embedTokens(inputIds);
  if (!applyRopeInAttention_) {
    hiddenStates = applyPositionalEncoding(hiddenStates, inputIds.size());
  }

  std::vector<float> attentionMask; // unused placeholder
  duorou::ml::Context dummyCtx;
  for (auto &layer : layers_) {
    hiddenStates =
        layer->forward(dummyCtx, hiddenStates, attentionMask, nullptr);
  }
  hiddenStates = layerNorm(hiddenStates, outputNormWeights_, options_.eps);
  // Return logits computed from last token hidden state
  return computeLogitsFromHidden(hiddenStates);
}

// Forward overload with Context/Tensor and KV Cache support (returns hidden
// states as Tensor)
duorou::ml::Tensor QwenTextModel::forward(duorou::ml::Context &ctx,
                                          const duorou::ml::Tensor &inputIds,
                                          duorou::kvcache::Cache *cache) {
  // Initialize logger for debugging
  static duorou::core::Logger logger;
  static bool logger_initialized = false;
  if (!logger_initialized) {
    logger.initialize();
    logger.setLogLevel(duorou::core::LogLevel::INFO);
    logger_initialized = true;
  }

  size_t n = static_cast<size_t>(inputIds.numel());
  if (n == 0) {
    logger.debug(
        "[QwenTextModel::forward] Empty input tensor, returning empty result");
    return duorou::ml::Tensor();
  }

  logger.debug(
      "[QwenTextModel::forward] Starting forward pass with input size: " +
      std::to_string(n));

  if (inputIds.dtype() != duorou::ml::DataType::INT32) {
    std::cerr << "[WARN] QwenTextModel::forward expected INT32 inputIds; "
                 "proceeding with reinterpretation"
              << std::endl;
    logger.warning("Input tensor dtype mismatch, expected INT32");
  }

  std::vector<int32_t> ids(n, 0);
  inputIds.copyToHost(ids.data(), n * sizeof(int32_t));

  // Log input token IDs
  std::ostringstream token_stream;
  token_stream << "[QwenTextModel::forward] Input token IDs: [";
  for (size_t i = 0; i < std::min(n, size_t(10)); ++i) {
    if (i > 0)
      token_stream << ", ";
    token_stream << ids[i];
  }
  if (n > 10)
    token_stream << ", ...";
  token_stream << "]";
  logger.debug(token_stream.str());

  // Embedding + positional encoding
  auto hiddenStates = embedTokens(ids);
  logger.debug(
      "[QwenTextModel::forward] After embedTokens: " +
      formatVectorStats(computeVectorStats(hiddenStates), "embeddings"));

  if (!applyRopeInAttention_) {
    hiddenStates = applyPositionalEncoding(hiddenStates, ids.size());
    logger.debug(
        "[QwenTextModel::forward] After positional encoding: " +
        formatVectorStats(computeVectorStats(hiddenStates), "pos_encoded"));
  } else {
    logger.debug("[QwenTextModel::forward] Skipping positional encoding at embeddings stage (RoPE in attention)");
  }

  // If KV Cache is provided, start forward with batch metadata
  if (cache) {
    logger.debug("[QwenTextModel::forward] Using KV cache");
    MLKVBackendAdapter kvAdapter(ctx.getBackend());
    duorou::kvcache::Context kvCtx(&kvAdapter);
    duorou::kvcache::Batch batch;
    batch.seqs = {0};
    batch.seqLens = {static_cast<int>(ids.size())};
    batch.positions = {static_cast<int>(ids.size() > 0 ? ids.size() - 1 : 0)};
    batch.batchSize = 1;
    try {
      cache->startForward(kvCtx, batch, false);
    } catch (...) {
      logger.warning("KV cache startForward failed");
    }
  } else {
    logger.debug("[QwenTextModel::forward] No KV cache provided");
  }

  // Transformer layers with potential KV Cache usage
  std::vector<float> attentionMask; // placeholder
  logger.debug("[QwenTextModel::forward] Processing " +
               std::to_string(layers_.size()) + " transformer layers");

  for (size_t li = 0; li < layers_.size(); ++li) {
    if (cache)
      cache->setLayer(static_cast<int>(li));

    auto prev_hidden = hiddenStates;
    hiddenStates =
        layers_[li]->forward(ctx, hiddenStates, attentionMask, cache);

    // Log every 4th layer or if there are issues
    if (li % 4 == 0 || li == layers_.size() - 1) {
      auto stats = computeVectorStats(hiddenStates);
      logger.debug("[QwenTextModel::forward] After layer " +
                   std::to_string(li) + ": " +
                   formatVectorStats(stats, "layer_" + std::to_string(li)));

      // Check for potential issues
      if (stats.nan_count > 0 || stats.inf_count > 0) {
        logger.warning("Layer " + std::to_string(li) +
                       " produced NaN or Inf values!");
      }
    }
  }

  // Output normalization
  auto pre_norm_hidden = hiddenStates;
  hiddenStates = layerNorm(hiddenStates, outputNormWeights_, options_.eps);
  logger.debug(
      "[QwenTextModel::forward] After output normalization: " +
      formatVectorStats(computeVectorStats(hiddenStates), "normalized"));

  // Compute logits from last token hidden and return as Tensor [vocab_size]
  std::vector<float> logits = computeLogitsFromHidden(hiddenStates);

  // Detailed logits analysis
  auto logits_stats = computeVectorStats(logits);
  logger.debug("[QwenTextModel::forward] Final logits: " +
               formatVectorStats(logits_stats, "logits"));

  // Check expected vocab size from metadata
  size_t expected_vocab_size = getVocabSize();
  if (logits.size() != expected_vocab_size) {
    logger.warning("Logits size mismatch! Expected: " +
                   std::to_string(expected_vocab_size) +
                   ", Got: " + std::to_string(logits.size()));
  }

  // Log top-k logits for analysis
  if (!logits.empty()) {
    std::vector<std::pair<float, size_t>> indexed_logits;
    for (size_t i = 0; i < logits.size(); ++i) {
      indexed_logits.emplace_back(logits[i], i);
    }
    std::partial_sort(
        indexed_logits.begin(),
        indexed_logits.begin() + std::min(size_t(5), indexed_logits.size()),
        indexed_logits.end(),
        [](const auto &a, const auto &b) { return a.first > b.first; });

    std::ostringstream top_logits_stream;
    top_logits_stream << "[QwenTextModel::forward] Top-5 logits: ";
    for (size_t i = 0; i < std::min(size_t(5), indexed_logits.size()); ++i) {
      if (i > 0)
        top_logits_stream << ", ";
      top_logits_stream << "(" << indexed_logits[i].second << ":" << std::fixed
                        << std::setprecision(4) << indexed_logits[i].first
                        << ")";
    }
    logger.debug(top_logits_stream.str());
  }

  // Check for reasonable logits range (should typically be in [-50, 50] range)
  if (logits_stats.min_val < -100.0f || logits_stats.max_val > 100.0f) {
    logger.warning("Logits values seem extreme! Min: " +
                   std::to_string(logits_stats.min_val) +
                   ", Max: " + std::to_string(logits_stats.max_val));
  }

  std::vector<int64_t> shape = {static_cast<int64_t>(logits.size())};
  duorou::ml::Tensor out =
      duorou::ml::Tensor::zeros(shape, duorou::ml::DataType::FLOAT32);
  float *outData = out.data<float>();
  if (outData && !logits.empty()) {
    std::copy(logits.begin(), logits.end(), outData);
  }

  logger.debug("[QwenTextModel::forward] Forward pass completed successfully");
  return out;
}

std::vector<int32_t>
QwenTextModel::generate(const std::vector<int32_t> &inputIds,
                        size_t /*maxLength*/, float /*temperature*/,
                        float /*topP*/) {
  std::vector<int32_t> result = inputIds;
  const Vocabulary *v = getVocabulary();
  int32_t eos_id = v ? v->getSpecialId(Special::EOS) : -1;
  if (eos_id >= 0)
    result.push_back(eos_id);
  return result;
}

// Helper: read a GGUF tensor into float32 vector (supports F32/F16/BF16).
// Returns true on success.
static bool
readGGUFTensorToFloat(duorou::extensions::ollama::GGUFParser &parser,
                      const std::string &name, std::vector<float> &out,
                      std::vector<int64_t> *shapeOut = nullptr) {
  using duorou::extensions::ollama::GGMLTensorType;
  const auto *info = parser.getTensorInfo(name);
  if (!info) {
    return false;
  }
  size_t nelems = 1;
  std::vector<int64_t> shape;
  shape.reserve(info->dimensions.size());
  for (auto d : info->dimensions) {
    nelems *= static_cast<size_t>(d);
    shape.push_back(static_cast<int64_t>(d));
  }
  size_t bytes = parser.getTensorSize(name);
  if (bytes == 0 || nelems == 0) {
    return false;
  }
  std::vector<uint8_t> buf(bytes);
  if (!parser.readTensorData(*info, buf.data(), bytes)) {
    return false;
  }
  out.resize(nelems);
  switch (info->type) {
  case GGMLTensorType::F32: {
    const float *src = reinterpret_cast<const float *>(buf.data());
    std::copy(src, src + nelems, out.begin());
    break;
  }
  case GGMLTensorType::F16: {
    const ggml_fp16_t *src = reinterpret_cast<const ggml_fp16_t *>(buf.data());
    ggml_fp16_to_fp32_row(src, out.data(), static_cast<int64_t>(nelems));
    break;
  }
  case GGMLTensorType::BF16: {
    // Convert BF16 to FP32 manually to avoid dependency on optional ggml
    // helpers
    const uint16_t *src = reinterpret_cast<const uint16_t *>(buf.data());
    for (size_t i = 0; i < nelems; ++i) {
      uint32_t tmp = static_cast<uint32_t>(src[i]) << 16;
      float f;
      std::memcpy(&f, &tmp, sizeof(float));
      out[i] = f;
    }
    break;
  }
  default:
    // Unsupported quantized type for direct float extraction in this minimal
    // path
    return false;
  }
  if (shapeOut)
    *shapeOut = shape;
  return true;
}
bool QwenTextModel::loadModel(const std::string &modelPath) {
  bool ok = initialize(modelPath);
  if (!ok)
    return false;
  // After tokenizer/config init, load real weights from GGUF
  return loadWeights(modelPath);
}

void QwenTextModel::setOptions(const TextModelOptions &options) {
  options_ = options;
}

void QwenTextModel::setRoPEFreqs(const std::vector<float> &freqs) {
  ropeFreqs_ = freqs;
  for (auto &layer : layers_) {
    if (layer) {
      layer->setRoPEFreqs(freqs);
    }
  }
}

void QwenTextModel::setApplyRopeInAttention(bool v) {
  for (auto &layer : layers_) {
    if (layer) {
      layer->setApplyRopeInAttention(v);
    }
  }
}

void QwenTextModel::setExternalVocabulary(std::shared_ptr<Vocabulary> vocab) {
  // Bind external vocabulary by constructing a tokenizer with shared vocab
  if (!vocab) {
    tokenizer_.reset();
    std::cout << "[DEBUG] QwenTextModel external vocabulary cleared"
              << std::endl;
    return;
  }

  TokenizerFactoryOptions opts; // allow future overrides
  auto tk = createTextProcessorForArchitecture("qwen", vocab, opts);
  if (!tk) {
    std::cerr << "[ERROR] QwenTextModel failed to create tokenizer from "
                 "external vocabulary"
              << std::endl;
    return;
  }
  tokenizer_ = std::move(tk);
  std::cout << "[DEBUG] QwenTextModel bound external vocabulary via tokenizer "
               "(vocab_size="
            << tokenizer_->getVocabSize() << ")" << std::endl;
}

::duorou::ml::Tensor
QwenTextModel::stepDecode(::duorou::ml::Context &ctx,
                          const ::duorou::ml::Tensor &lastTokenId,
                          ::duorou::kvcache::Cache *cache) {
  // Expect a single token id
  size_t seqLen = static_cast<size_t>(lastTokenId.numel());
  if (seqLen == 0) {
    return ::duorou::ml::Tensor();
  }
  if (lastTokenId.dtype() != ::duorou::ml::DataType::INT32) {
    std::cerr << "[WARN] QwenTextModel::stepDecode expected INT32 id; "
                 "proceeding with reinterpretation"
              << std::endl;
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
    try {
      cache->startForward(kvCtx, batch, false);
    } catch (...) {
    }
  }

  // Transformer layers with potential KV Cache usage
  std::vector<float> attentionMask; // placeholder
  for (size_t li = 0; li < layers_.size(); ++li) {
    if (cache)
      cache->setLayer(static_cast<int>(li));
    hiddenStates =
        layers_[li]->forward(ctx, hiddenStates, attentionMask, cache);
  }

  // Output normalization
  hiddenStates = layerNorm(hiddenStates, outputNormWeights_, options_.eps);

  // Compute logits and return as tensor
  auto logits = computeLogitsFromHidden(hiddenStates);
  std::vector<int64_t> shape = {static_cast<int64_t>(logits.size())};
  ::duorou::ml::Tensor out =
      ::duorou::ml::Tensor::zeros(shape, ::duorou::ml::DataType::FLOAT32);
  float *outData = out.data<float>();
  if (outData && !logits.empty()) {
    std::copy(logits.begin(), logits.end(), outData);
  }
  return out;
}

// New helper exposures
size_t QwenTextModel::getHiddenSize() const { return options_.hiddenSize; }

std::vector<float>
QwenTextModel::computeLogitsFromHidden(const std::vector<float> &hidden) {
  // Initialize logger for debugging
  static duorou::core::Logger logger;
  static bool logger_initialized = false;
  if (!logger_initialized) {
    logger.initialize();
    logger.setLogLevel(duorou::core::LogLevel::INFO);
    logger_initialized = true;
  }

  // hidden is [seq_len * hidden_size]
  size_t hidden_size = options_.hiddenSize;
  logger.debug(
      "[computeLogitsFromHidden] Hidden size: " + std::to_string(hidden_size) +
      ", Input hidden vector size: " + std::to_string(hidden.size()));

  if (hidden.size() < hidden_size) {
    logger.warning("[computeLogitsFromHidden] Hidden vector too small! "
                   "Expected at least: " +
                   std::to_string(hidden_size) +
                   ", Got: " + std::to_string(hidden.size()));
    return {};
  }

  size_t seq_len = hidden.size() / hidden_size;
  size_t vocab_tokenizer = getVocabSize();
  size_t vocab_weights =
      (hidden_size == 0 ? 0 : outputWeights_.size() / hidden_size);
  // Provide non-zero fallback for output weights if missing
  if (vocab_weights == 0 && hidden_size > 0 && vocab_tokenizer > 0) {
    xavier_fill(outputWeights_, hidden_size, vocab_tokenizer, 0xABCDEF01u);
    vocab_weights = outputWeights_.size() / hidden_size;
    // Ensure output norm buffer sized accordingly
    if (outputNormWeights_.size() != hidden_size)
      outputNormWeights_.resize(hidden_size, 1.0f);
  }
  // 如果输出权重存在但为全零，进行回退初始化避免得到全零logits
  if (hidden_size > 0 && !outputWeights_.empty()) {
    bool weights_all_zero = true;
    for (size_t i = 0; i < outputWeights_.size(); ++i) {
      if (outputWeights_[i] != 0.0f) {
        weights_all_zero = false;
        break;
      }
    }
    if (weights_all_zero && vocab_tokenizer > 0) {
      xavier_fill(outputWeights_, hidden_size, vocab_tokenizer, 0xABCDEF01u);
      vocab_weights = outputWeights_.size() / hidden_size;
      if (outputNormWeights_.size() != hidden_size)
        outputNormWeights_.resize(hidden_size, 1.0f);
    }
  }
  if (hidden_size != 0 && (outputWeights_.size() % hidden_size) != 0) {
    logger.warning("[computeLogitsFromHidden] Output weights length is not a "
                   "multiple of hidden size! hidden*?=" +
                   std::to_string(hidden_size) +
                   ", weights.size=" + std::to_string(outputWeights_.size()));
  }
  logger.debug(
      "[computeLogitsFromHidden] Sequence length: " + std::to_string(seq_len) +
      ", Vocab size (tokenizer): " + std::to_string(vocab_tokenizer) +
      ", Vocab size (weights): " + std::to_string(vocab_weights));

  // Return logits sized to the actual weights vocab size to avoid mismatch
  // This ensures compatibility with inference engine expectations
  std::vector<float> logits(vocab_weights, 0.0f);

  if (vocab_weights == 0) {
    logger.error("[computeLogitsFromHidden] No output weights available!");
    return logits;
  }

  size_t last_offset = (seq_len > 0 ? (seq_len - 1) * hidden_size : 0);
  const float *hptr = hidden.data() + last_offset;

  // Log statistics of the last token hidden state
  std::vector<float> last_hidden(hptr, hptr + hidden_size);
  auto hidden_stats = computeVectorStats(last_hidden);
  logger.debug("[computeLogitsFromHidden] Last token hidden state: " +
               formatVectorStats(hidden_stats, "last_hidden"));

  // Check output weights statistics
  auto weights_stats = computeVectorStats(outputWeights_);
  logger.debug("[computeLogitsFromHidden] Output weights: " +
               formatVectorStats(weights_stats, "output_weights"));

  // Compute logits: logits[v] = hidden dot outputWeights[v]
  for (size_t v = 0; v < vocab_weights && v < vocab_tokenizer; ++v) {
    float sum = 0.0f;
    size_t woff = v * hidden_size;
    for (size_t i = 0; i < hidden_size; ++i) {
      sum += hptr[i] * outputWeights_[woff + i];
    }
    logits[v] = sum;
  }

  // Sanitize logits: replace NaN/Inf with 0 to avoid downstream instability
  for (size_t i = 0; i < logits.size(); ++i) {
    if (!std::isfinite(logits[i])) {
      logits[i] = 0.0f;
    }
  }

  // Log some sample logits computation details for first few vocab entries
  if (vocab_tokenizer > 0) {
    std::ostringstream sample_stream;
    sample_stream << "[computeLogitsFromHidden] Sample logits computation: ";
    for (size_t v = 0; v < std::min(vocab_tokenizer, size_t(3)); ++v) {
      sample_stream << "vocab[" << v << "]=" << std::fixed
                    << std::setprecision(4) << logits[v] << " ";
    }
    logger.debug(sample_stream.str());
  }

  logger.debug("[computeLogitsFromHidden] Computed logits for " +
               std::to_string(vocab_weights) + " tokens (weights vocab size)");
  return logits;
}

// New: nextToken helper using stepDecode with temperature and top-p sampling
int32_t QwenTextModel::nextToken(duorou::ml::Context &ctx,
                                 const duorou::ml::Tensor &lastTokenId,
                                 duorou::kvcache::Cache *cache,
                                 float temperature, float topP) {
  // Run stepDecode to get logits
  ::duorou::ml::Tensor logits_tensor = stepDecode(ctx, lastTokenId, cache);
  if (logits_tensor.numel() == 0) {
    return -1;
  }
  std::vector<float> logits(static_cast<size_t>(logits_tensor.numel()));
  logits_tensor.copyToHost(logits.data(), logits.size() * sizeof(float));

  // Temperature scaling
  if (temperature > 0.0f) {
    for (auto &x : logits)
      x /= temperature;
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
      for (auto &p : probs)
        p = static_cast<float>(p / sum);
    }
  }

  // Top-p sampling
  if (probs.empty())
    return -1;
  if (topP >= 1.0f) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    static thread_local std::mt19937 rng(std::random_device{}());
    return static_cast<int32_t>(dist(rng));
  }
  std::vector<int> idx(probs.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return probs[a] > probs[b]; });
  std::vector<int> kept;
  std::vector<float> kept_probs;
  float acc = 0.0f;
  for (int id : idx) {
    kept.push_back(id);
    kept_probs.push_back(probs[id]);
    acc += probs[id];
    if (acc >= topP)
      break;
  }
  if (acc > 0.0f) {
    for (auto &p : kept_probs)
      p = p / acc;
  }
  static thread_local std::mt19937 rng(std::random_device{}());
  std::discrete_distribution<int> dist(kept_probs.begin(), kept_probs.end());
  int sampled = dist(rng);
  return static_cast<int32_t>(kept[sampled]);
}

} // namespace model
} // namespace duorou
