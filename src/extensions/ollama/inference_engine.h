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
      batch = llama_batch_get_one(&decoder_start_token_id, 1);
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

      // 使用默认序列 ID 0；传入 -1 会触发内部断言失败
      llama_token new_token_id = llama_sampler_sample(sampler, ctx_, 0);
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

      // Next step batch uses the sampled token
      batch = llama_batch_get_one(&new_token_id, 1);
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