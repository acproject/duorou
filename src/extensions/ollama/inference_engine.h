// Inference engine using llama.cpp for real text generation
#ifndef DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H
#define DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H

// Only enable C++ constructs when compiled as C++.
#ifdef __cplusplus

#include <string>
#include <vector>

// llama.cpp headers (relative to project root)
#include "../../third_party/llama.cpp/src/models/models.h"
#include "../../third_party/llama.cpp/include/llama.h"

namespace duorou {
namespace extensions {
namespace ollama {

class InferenceEngine {
public:
  virtual ~InferenceEngine() {}
  virtual bool initialize() = 0;
  virtual bool isReady() const = 0;
  virtual std::string generateText(const std::string &prompt,
                                   unsigned int max_tokens,
                                   float temperature,
                                   float top_p) = 0;\
  virtual std::string generateTextWithImages(const std::string &prompt,
                                           const std::vector<std::vector<float>> &image_features,
                                           unsigned int max_tokens,
                                           float temperature,
                                           float top_p) = 0;
};

class MLInferenceEngine : public InferenceEngine {
public:
  MLInferenceEngine(const std::string &model_id, const std::string &gguf_path)
      : model_id_(model_id), gguf_path_(gguf_path), ready_(false), model_(nullptr),
        ctx_(nullptr) {}

  ~MLInferenceEngine() override {
    // Cleanup resources
    if (ctx_) {
      llama_free(ctx_);
      ctx_ = nullptr;
    }
    if (model_) {
      llama_model_free(model_);
      model_ = nullptr;
    }
  }

  bool initialize() override {
    // Load all dynamic backends if available
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    // Heuristic GPU layers; adjust based on environment
    mparams.n_gpu_layers = 99;

    model_ = llama_model_load_from_file(gguf_path_.c_str(), mparams);
    if (!model_) {
      return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048; // Default context size
    cparams.n_batch = 64; // Reasonable batch size
    cparams.no_perf = true; // Disable perf logs in production path

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
      llama_model_free(model_);
      model_ = nullptr;
      return false;
    }

    ready_ = true;
    return true;
  }

  bool isReady() const override { return ready_ && model_ && ctx_; }

  std::string generateText(const std::string &prompt, unsigned int max_tokens,
                           float temperature, float top_p) override {
    if (!isReady()) {
      return std::string("[Engine not ready] ") + prompt;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model_);
    if (!vocab) {
      return std::string("[Vocab error] ") + prompt;
    }

    // Tokenize prompt
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                   nullptr, 0, /*add_special=*/true,
                                   /*parse_special=*/true);
    if (n_prompt <= 0) {
      return std::string("[Tokenize error] ") + prompt;
    }
    std::vector<llama_token> prompt_tokens((size_t)n_prompt);
    int n_tok = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                               prompt_tokens.data(), (int)prompt_tokens.size(),
                               /*add_special=*/true, /*parse_special=*/true);
    if (n_tok < 0) {
      return std::string("[Tokenize error] ") + prompt;
    }

    // Build batch from prompt
    llama_batch batch =
        llama_batch_get_one(prompt_tokens.data(), (int)prompt_tokens.size());

    // If the model has an encoder, run encode first
    if (llama_model_has_encoder(model_)) {
      if (llama_encode(ctx_, batch)) {
        return std::string("[Encode error] ") + prompt;
      }
      llama_token decoder_start_token_id = llama_model_decoder_start_token(model_);
      if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
        decoder_start_token_id = llama_vocab_bos(vocab);
      }
      // 使用持久 token 缓冲避免悬空指针
      std::vector<llama_token> token_buf(1);
      token_buf[0] = decoder_start_token_id;
      batch = llama_batch_get_one(token_buf.data(), 1);
    }

    // Initialize sampler chain per call so temperature/top_p can vary
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    if (temperature > 0.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    }
    if (top_p > 0.0f && top_p < 1.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, /*min_keep*/ 1));
    }
    // 保证链上始终有“最终选择器”以设置 cur_p.selected
    // - 当无随机性设置时使用贪心选择
    // - 当使用温度/Top-P 等随机性时，使用分布采样选择
    if (temperature <= 0.0f && !(top_p > 0.0f && top_p < 1.0f)) {
      // 无随机性：贪心
      llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
      // 有随机性：按概率分布采样（使用默认种子以获得合理随机性）
      llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    std::string output;
    unsigned int to_predict = max_tokens == 0 ? 64u : max_tokens;
    unsigned int produced = 0;

    // Main generation loop
    while (produced < to_predict) {
      if (llama_decode(ctx_, batch)) {
        break; // decode error
      }
      // position advance is implicit; not used for output here

      // 首次从提示词采样应使用最后一个被标记输出的位置（n_tokens-1）
      // 后续单 token batch 的采样索引恒为 0
      llama_token new_token_id = llama_sampler_sample(sampler, ctx_, batch.n_tokens - 1);
      if (llama_vocab_is_eog(vocab, new_token_id)) {
        break;
      }

      // Append piece for the sampled token
      char buf[256];
      const int n = llama_token_to_piece(vocab, new_token_id, buf,
                                         sizeof(buf), 0, true);
      if (n > 0) {
        output.append(buf, (size_t)n);
      }

      // Next step batch uses the sampled token（持久缓冲）
      // 复用或创建持久缓冲以避免传入局部地址
      static thread_local std::vector<llama_token> next_token_buf(1);
      next_token_buf[0] = new_token_id;
      batch = llama_batch_get_one(next_token_buf.data(), 1);
      produced += 1;
    }

    llama_sampler_free(sampler);
    return output.empty() ? std::string("") : output;
  }

  std::string generateTextWithImages(
      const std::string &prompt,
      const std::vector<std::vector<float>> &image_features,
      unsigned int max_tokens, float temperature, float top_p) override {
    if (!isReady()) {
      return std::string("[Engine not ready] ") + prompt;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model_);
    if (!vocab) {
      return std::string("[Vocab error] ") + prompt;
    }

    if (!llama_model_has_encoder(model_)) {
      return std::string("[No encoder - vision unsupported] ") + prompt;
    }

    // Validate image features
    if (image_features.empty()) {
      return std::string("[No image features provided] ") + prompt;
    }

    const int32_t embd_dim = llama_model_n_embd(model_);

    // Concatenate all image feature vectors, ensuring they align with embd_dim
    std::vector<float> visual_concat;
    int32_t total_img_tokens = 0;
    for (const auto &feat : image_features) {
      if (feat.empty()) {
        continue;
      }
      if (feat.size() % embd_dim != 0) {
        return std::string("[Image feature size mismatch to n_embd] ") +
               std::to_string(feat.size()) + " vs " + std::to_string(embd_dim);
      }
      total_img_tokens += static_cast<int32_t>(feat.size() / embd_dim);
      visual_concat.insert(visual_concat.end(), feat.begin(), feat.end());
    }

    if (total_img_tokens <= 0) {
      return std::string("[No valid image tokens computed] ") + prompt;
    }

    // Guard against ubatch constraints on encoder
    const uint32_t n_ubatch = llama_n_ubatch(ctx_);
    if (static_cast<uint32_t>(total_img_tokens) > n_ubatch) {
      // Conservative handling for Milestone 1: reject to avoid assert in encode
      return std::string("[Too many image tokens for n_ubatch] tokens=") +
             std::to_string(total_img_tokens) + ", n_ubatch=" +
             std::to_string(n_ubatch);
    }

    // Build embedding-only batch for encoder input
    llama_batch enc_batch;
    enc_batch.n_tokens = total_img_tokens;
    enc_batch.token = nullptr;
    enc_batch.embd = visual_concat.data();
    enc_batch.pos = nullptr;
    enc_batch.n_seq_id = nullptr;
    enc_batch.seq_id = nullptr;
    enc_batch.logits = nullptr;

    if (llama_encode(ctx_, enc_batch)) {
      return std::string("[Encode error] ") + prompt;
    }

    // Prepare decoder initial tokens: decoder start + prompt tokens (without special)
    llama_token decoder_start_token_id = llama_model_decoder_start_token(model_);
    if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
      decoder_start_token_id = llama_vocab_bos(vocab);
    }

    // Tokenize prompt without adding special tokens to avoid double BOS
    std::vector<llama_token> prompt_tokens;
    if (!prompt.empty()) {
      int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                     nullptr, 0, /*add_special=*/false,
                                     /*parse_special=*/true);
      if (n_prompt < 0) {
        return std::string("[Tokenize error] ") + prompt;
      }
      prompt_tokens.resize((size_t)n_prompt);
      const int n_tok = llama_tokenize(
          vocab, prompt.c_str(), (int)prompt.size(), prompt_tokens.data(),
          (int)prompt_tokens.size(), /*add_special=*/false,
          /*parse_special=*/true);
      if (n_tok < 0) {
        return std::string("[Tokenize error] ") + prompt;
      }
    }

    std::vector<llama_token> dec_init_tokens;
    dec_init_tokens.reserve(1 + prompt_tokens.size());
    dec_init_tokens.push_back(decoder_start_token_id);
    dec_init_tokens.insert(dec_init_tokens.end(), prompt_tokens.begin(),
                           prompt_tokens.end());

    llama_batch dec_batch = llama_batch_get_one(dec_init_tokens.data(),
                                                (int32_t)dec_init_tokens.size());

    // Initialize sampler chain per call so temperature/top_p can vary
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    if (temperature > 0.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    }
    if (top_p > 0.0f && top_p < 1.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, /*min_keep*/ 1));
    }
    if (temperature <= 0.0f && !(top_p > 0.0f && top_p < 1.0f)) {
      llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
      llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    std::string output;
    unsigned int to_predict = max_tokens == 0 ? 64u : max_tokens;
    unsigned int produced = 0;

    // Prime decoder with initial sequence (start + prompt)
    if (llama_decode(ctx_, dec_batch)) {
      llama_sampler_free(sampler);
      return std::string("[Decode error (init)] ") + prompt;
    }

    // Main generation loop
    while (produced < to_predict) {
      // Sample from the last position of the current batch
      llama_token new_token_id =
          llama_sampler_sample(sampler, ctx_, dec_batch.n_tokens - 1);
      if (llama_vocab_is_eog(vocab, new_token_id)) {
        break;
      }

      char buf[256];
      const int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf),
                                         0, true);
      if (n > 0) {
        output.append(buf, (size_t)n);
      }

      // Next step uses only the newly sampled token
      static thread_local std::vector<llama_token> next_token_buf(1);
      next_token_buf[0] = new_token_id;
      llama_batch next_batch = llama_batch_get_one(next_token_buf.data(), 1);

      if (llama_decode(ctx_, next_batch)) {
        break;
      }

      dec_batch = next_batch;
      produced += 1;
    }

    llama_sampler_free(sampler);
    return output.empty() ? std::string("") : output;
  }

private:
  std::string model_id_;
  std::string gguf_path_;
  bool ready_;
  llama_model *model_;
  llama_context *ctx_;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#else
// Non-C++ compilation units: keep minimal to avoid diagnostics.
#endif // __cplusplus

#endif // DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H