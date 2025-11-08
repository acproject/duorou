// Inference engine using llama.cpp for real text generation
#ifndef DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H
#define DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H

// Only enable C++ constructs when compiled as C++.
#ifdef __cplusplus

#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <cctype>
#include <algorithm>
#include <filesystem>
#include <cstdio>

// llama.cpp & ggml headers (resolved via target include directories)
// Explicit relative include to satisfy IDE/LSP resolution
#include "../../../third_party/llama.cpp/include/llama.h"
#include "ggml-backend.h"
// mtmd helpers for multimodal image injection (llama.cpp tools)
#include "../../../third_party/llama.cpp/tools/mtmd/mtmd.h"
#include "../../../third_party/llama.cpp/tools/mtmd/mtmd-helper.h"

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
  virtual std::string generateTextWithImages(const std::string &prompt,
                                           const std::vector<std::vector<float>> &image_features,
                                           unsigned int max_tokens,
                                           float temperature,
                                           float top_p) = 0;
  // Expose embedding dimension for pre-validation at manager level
  virtual int32_t getEmbeddingDim() const = 0;
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
    // Increase batch to accommodate prompt + last-logits without assert
    cparams.n_ctx = 2048; // Default context size
    cparams.n_batch = 512; // Larger batch to avoid n_tokens + n_outputs overflow
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

    // --------- Prompt heuristics for text path (handle media inlined with text) ---------
    auto trim_copy = [](const std::string &s) -> std::string {
      size_t start = 0, end = s.size();
      while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
      while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
      return s.substr(start, end - start);
    };

    auto contains_md_image_any = [](const std::string &s) -> bool {
      static const std::regex re("!\\[[^\\]]*\\]\\(([^)]+)\\)");
      return std::regex_search(s, re);
    };

    auto contains_data_image = [](const std::string &s) -> bool {
      static const std::regex re_data("data:image/[^;]+;base64,");
      return std::regex_search(s, re_data);
    };

    auto contains_file_url = [](const std::string &s) -> bool {
      static const std::regex re_file("file://[^\\s)]+");
      return std::regex_search(s, re_file);
    };

    auto contains_http_image_url = [](const std::string &s) -> bool {
      static const std::regex re_http_img("https?://[^\\s)]+\\.(png|jpg|jpeg|gif|webp|bmp|tiff|svg)(\\?[^\\s)]*)?", std::regex::icase);
      return std::regex_search(s, re_http_img);
    };

    std::string prompt_effective = trim_copy(prompt);
    const bool prompt_contains_media = contains_md_image_any(prompt) || contains_data_image(prompt) ||
                                       contains_file_url(prompt) || contains_http_image_url(prompt);
    // 如果提示中包含任何图片引用（即便混合文本），转到 generateTextWithImages
    if (prompt_contains_media) {
      return generateTextWithImages(prompt, /*image_features=*/{}, max_tokens, temperature, top_p);
    }

    // 打印原始与处理后的提示词，便于诊断
    std::cout << "Text Raw Prompt: " << prompt << std::endl;
    std::cout << "Text Prompt: " << prompt_effective << std::endl;

    // Tokenize prompt
    int n_prompt = -llama_tokenize(vocab, prompt_effective.c_str(), (int)prompt_effective.size(),
                                   nullptr, 0, /*add_special=*/true,
                                   /*parse_special=*/true);
    if (n_prompt <= 0) {
      return std::string("[Tokenize error] ") + prompt_effective;
    }
    std::vector<llama_token> prompt_tokens((size_t)n_prompt);
    int n_tok = llama_tokenize(vocab, prompt_effective.c_str(), (int)prompt_effective.size(),
                               prompt_tokens.data(), (int)prompt_tokens.size(),
                               /*add_special=*/true, /*parse_special=*/true);
    if (n_tok < 0) {
      return std::string("[Tokenize error] ") + prompt_effective;
    }

    // Build batch from prompt; request logits only for last token by default
    llama_batch batch =
        llama_batch_get_one(prompt_tokens.data(), (int)prompt_tokens.size());
    batch.logits = nullptr; // ensure only last token outputs logits

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
      batch.logits = nullptr; // single-token decode outputs logits for that token
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
      batch.logits = nullptr; // ensure single-token decode outputs logits
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

    const bool has_encoder = llama_model_has_encoder(model_);
    // 允许在无视觉编码器的情况下走 MTMD 注入路径；
    // 仅当用户提供图像特征时才需要视觉编码器。

    const int32_t embd_dim = llama_model_n_embd(model_);

    // If no features provided, attempt to inject image via mtmd using the prompt reference
    if (image_features.empty()) {
      // Heuristics to detect and extract media path even when mixed with text
      auto trim_copy_local = [](const std::string &s) -> std::string {
        size_t start = 0, end = s.size();
        while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
        while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
        return s.substr(start, end - start);
      };
      auto contains_md_image_any_local = [](const std::string &s) -> bool {
        static const std::regex re("!\\[[^\\]]*\\]\\(([^)]+)\\)");
        return std::regex_search(s, re);
      };
      auto extract_md_image_url_any = [](const std::string &s, std::smatch &m_out) -> bool {
        static const std::regex re("!\\[[^\\]]*\\]\\(([^)]+)\\)");
        return std::regex_search(s, m_out, re) && m_out.size() >= 2;
      };
      auto extract_first_file_url = [](const std::string &s, std::smatch &m_out) -> bool {
        static const std::regex re("file://[^\\s)]+");
        return std::regex_search(s, m_out, re);
      };
      auto extract_first_http_image = [](const std::string &s, std::smatch &m_out) -> bool {
        static const std::regex re("https?://[^\\s)]+\\.(png|jpg|jpeg|gif|webp|bmp|tiff|svg)(\\?[^\\s)]*)?", std::regex::icase);
        return std::regex_search(s, m_out, re);
      };
      auto contains_data_image_local = [](const std::string &s) -> bool {
        static const std::regex re("data:image/[^;]+;base64,");
        return std::regex_search(s, re);
      };

      // Determine media presence
      const bool has_media = contains_md_image_any_local(prompt) || contains_data_image_local(prompt) ||
                             std::regex_search(prompt, std::regex("file://")) ||
                             std::regex_search(prompt, std::regex("https?://", std::regex::icase));
      if (!has_media) {
        return std::string("[No image features provided] ") + prompt;
      }

      // Extract media URL/path and user text part
      std::string media_path;
      std::string user_text;
      std::smatch m;
      if (extract_md_image_url_any(prompt, m)) {
        media_path = trim_copy_local(m[1].str());
        user_text = trim_copy_local(prompt.substr(0, m.position(0)) + " " +
                                    prompt.substr(m.position(0) + m.length(0)));
      } else if (extract_first_file_url(prompt, m)) {
        media_path = trim_copy_local(m.str());
        user_text = trim_copy_local(prompt.substr(0, m.position(0)) + " " +
                                    prompt.substr(m.position(0) + m.length(0)));
      } else if (extract_first_http_image(prompt, m)) {
        media_path = trim_copy_local(m.str());
        user_text = trim_copy_local(prompt.substr(0, m.position(0)) + " " +
                                    prompt.substr(m.position(0) + m.length(0)));
      } else {
        // Fallback: data:image or local path mixed; use full prompt as text, try to pick first token as path
        user_text = trim_copy_local(prompt);
        media_path = user_text; // will be handled below for local path
      }

      // 对本地路径执行存在性校验；对 http(s)/data URI 不进行文件系统校验
      if (media_path.empty()) {
        return std::string("[Media path invalid or not found] ") + media_path;
      }
      const bool is_http = media_path.rfind("http://", 0) == 0 || media_path.rfind("https://", 0) == 0;
      const bool is_data = media_path.rfind("data:image/", 0) == 0;
      if (!is_http && !is_data) {
        std::string local_path = media_path;
        if (local_path.rfind("file://", 0) == 0) {
          // 解码 file:// URI -> 本地路径
          auto percent_decode_local = [](const std::string &s) -> std::string {
            std::string out; out.reserve(s.size());
            for (size_t i = 0; i < s.size(); ++i) {
              if (s[i] == '%' && i + 2 < s.size()) {
                auto hex = s.substr(i + 1, 2);
                char *endp = nullptr;
                long v = std::strtol(hex.c_str(), &endp, 16);
                if (endp && *endp == '\0') { out.push_back(static_cast<char>(v)); i += 2; continue; }
              }
              out.push_back(s[i]);
            }
            return out;
          };
          local_path = percent_decode_local(local_path.substr(7));
        }
        // local path should exist; otherwise bail out early
        if (!std::filesystem::exists(local_path)) {
          return std::string("[Media path invalid or not found] ") + media_path;
        }
      }

      // Build effective prompt with media marker (allow override)
      std::string prompt_media;
      const char *marker = mtmd_default_marker();
      const char *env_prompt = std::getenv("OVERRIDE_IMAGE_PROMPT");
      if (env_prompt && *env_prompt) {
        prompt_media = std::string(env_prompt);
        if (prompt_media.find(marker) == std::string::npos) {
          prompt_media += marker;
        }
      } else {
        if (!user_text.empty()) {
          prompt_media = user_text;
          if (prompt_media.find(marker) == std::string::npos) {
            prompt_media += marker;
          }
        } else {
          prompt_media = std::string("请详细用中文描述这张图片：") + marker + std::string("。要求简洁准确。");
        }
      }
      

      

      // Locate mmproj.gguf near the base gguf path
      auto filename_is_mmproj = [](const std::string &filename) {
        std::string lower = filename;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c){ return std::tolower(c); });
        return lower.rfind("mmproj-", 0) == 0 || lower.find("-mmproj-") != std::string::npos || lower.find("mmproj") != std::string::npos;
      };
      auto find_mmproj_near = [&](const std::filesystem::path &dir) -> std::string {
        try {
          if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
            for (const auto &entry : std::filesystem::recursive_directory_iterator(dir)) {
              if (entry.is_regular_file()) {
                const auto &p = entry.path();
                if (p.has_extension() && p.extension() == ".gguf") {
                  if (filename_is_mmproj(p.filename().string())) return p.string();
                }
              }
            }
          }
        } catch (...) {}
        return std::string();
      };
      std::filesystem::path gguf(gguf_path_);
      // Try OVERRIDE_MMPROJ_PATH first; if dir, scan for mmproj gguf, else fallback near model path
      std::string mmproj_path;
      if (const char *env_mm = std::getenv("OVERRIDE_MMPROJ_PATH")) {
        try {
          std::filesystem::path p(env_mm);
          if (std::filesystem::exists(p)) {
            if (std::filesystem::is_regular_file(p)) {
              mmproj_path = p.string();
            } else if (std::filesystem::is_directory(p)) {
              mmproj_path = find_mmproj_near(p);
            }
          }
        } catch (...) {}
      }
      if (mmproj_path.empty()) {
        mmproj_path = find_mmproj_near(gguf.parent_path());
      }
      if (mmproj_path.empty()) {
        return std::string("[mmproj not found near model path] ") + gguf.parent_path().string();
      }

      // Initialize mtmd context and tokenize text+image
      auto mparams_mtmd = mtmd_context_params_default();
      mparams_mtmd.use_gpu = true;
      mparams_mtmd.media_marker = mtmd_default_marker();
      std::cout << "mtmd_init_from_file: " << mmproj_path << std::endl;
      mtmd_context *mctx = mtmd_init_from_file(mmproj_path.c_str(), model_, mparams_mtmd);
      if (!mctx) {
        return std::string("[mtmd_init_from_file failed] ") + mmproj_path;
      }
      // 根据路径或URL创建位图：支持本地文件、data:image/base64、http(s) URL
      auto base64_decode_local = [](const std::string &in) -> std::vector<unsigned char> {
        static int B64_INDEX[256];
        static bool init = false;
        if (!init) {
          for (int i = 0; i < 256; ++i) B64_INDEX[i] = -1;
          for (int i = 'A'; i <= 'Z'; ++i) B64_INDEX[i] = i - 'A';
          for (int i = 'a'; i <= 'z'; ++i) B64_INDEX[i] = i - 'a' + 26;
          for (int i = '0'; i <= '9'; ++i) B64_INDEX[i] = i - '0' + 52;
          B64_INDEX[(int)'+'] = 62; B64_INDEX[(int)'/'] = 63;
          init = true;
        }
        std::vector<unsigned char> out;
        int val = 0, valb = -8;
        for (unsigned char c : in) {
          if (c == '=') break;
          int d = (c < 256) ? B64_INDEX[c] : -1;
          if (d == -1) continue;
          val = (val << 6) + d;
          valb += 6;
          if (valb >= 0) {
            out.push_back((unsigned char)((val >> valb) & 0xFF));
            valb -= 8;
          }
        }
        return out;
      };
      auto read_url_to_buffer = [](const std::string &url) -> std::vector<unsigned char> {
        std::vector<unsigned char> buf;
        FILE *pipe = popen((std::string("curl -sL ") + url).c_str(), "r");
        if (!pipe) return buf;
        unsigned char tmp[4096];
        size_t nread = 0;
        while ((nread = fread(tmp, 1, sizeof(tmp), pipe)) > 0) {
          buf.insert(buf.end(), tmp, tmp + nread);
        }
        pclose(pipe);
        return buf;
      };
      // 辅助：对 file:// URI 进行百分号解码，得到可用的本地路径
      auto percent_decode = [](const std::string &s) -> std::string {
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
          if (s[i] == '%' && i + 2 < s.size()) {
            auto hex = s.substr(i + 1, 2);
            char *endp = nullptr;
            long v = std::strtol(hex.c_str(), &endp, 16);
            if (endp && *endp == '\0') {
              out.push_back(static_cast<char>(v));
              i += 2;
              continue;
            }
          }
          out.push_back(s[i]);
        }
        return out;
      };
      auto file_uri_to_path = [&](const std::string &uri) -> std::string {
        // 仅当是 file:// 时进行处理
        if (uri.rfind("file://", 0) == 0) {
          // 去掉协议前缀并对百分号编码解码
          return percent_decode(uri.substr(7));
        }
        return uri;
      };

      mtmd_bitmap *bitmap = nullptr;
      if (media_path.rfind("data:image/", 0) == 0) {
        size_t comma_pos = media_path.find(',');
        if (comma_pos == std::string::npos) {
          mtmd_free(mctx);
          return std::string("[data URL invalid] ") + media_path;
        }
        std::string b64 = media_path.substr(comma_pos + 1);
        std::vector<unsigned char> bytes = base64_decode_local(b64);
        if (bytes.empty()) {
          mtmd_free(mctx);
          return std::string("[base64 decode failed] ") + std::to_string(b64.size());
        }
        bitmap = mtmd_helper_bitmap_init_from_buf(mctx, bytes.data(), bytes.size());
      } else if (media_path.rfind("http://", 0) == 0 || media_path.rfind("https://", 0) == 0) {
        std::vector<unsigned char> bytes = read_url_to_buffer(media_path);
        if (bytes.empty()) {
          mtmd_free(mctx);
          return std::string("[download failed or empty] ") + media_path;
        }
        bitmap = mtmd_helper_bitmap_init_from_buf(mctx, bytes.data(), bytes.size());
      } else if (media_path.rfind("file://", 0) == 0) {
        const std::string local_path = file_uri_to_path(media_path);
        bitmap = mtmd_helper_bitmap_init_from_file(mctx, local_path.c_str());
      } else {
        // 处理纯本地路径（可能包含百分号编码的边缘情况，但通常不需要）
        bitmap = mtmd_helper_bitmap_init_from_file(mctx, media_path.c_str());
      }
      if (!bitmap) {
        mtmd_free(mctx);
        return std::string("[mtmd_bitmap init failed] ") + media_path;
      }
      std::cout << "Bitmap created from media path successfully" << std::endl;
      mtmd_input_chunks *mm_chunks = mtmd_input_chunks_init();
      mtmd_input_text txt{prompt_media.c_str(), /*add_special=*/true, /*parse_special=*/true};
      const mtmd_bitmap *bitmaps[1] = {bitmap};
      int32_t tok_res = mtmd_tokenize(mctx, mm_chunks, &txt, bitmaps, 1);
      mtmd_bitmap_free(bitmap);
      if (tok_res != 0) {
        mtmd_input_chunks_free(mm_chunks);
        mtmd_free(mctx);
        return std::string("[mtmd_tokenize failed] code=") + std::to_string(tok_res);
      }

      llama_pos n_past_out = 0;
      int32_t eval_res = mtmd_helper_eval_chunks(
          mctx, ctx_, mm_chunks,
          /*n_past=*/0,
          /*seq_id=*/0,
          /*n_batch=*/llama_n_batch(ctx_),
          /*logits_last=*/true, &n_past_out);
      if (eval_res != 0) {
        mtmd_input_chunks_free(mm_chunks);
        mtmd_free(mctx);
        return std::string("[mtmd_helper_eval_chunks failed] code=") + std::to_string(eval_res);
      }

      // 日志输出以便确认模板生效
      std::cout << "Image Raw Prompt: " << media_path << std::endl;
      std::cout << "Image Prompt: " << prompt_media << std::endl;

      // Initialize sampler and sample following tokens
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

      // 从已评估的最后 logits 采样第一个 token
      llama_token new_token_id = llama_sampler_sample(sampler, ctx_, -1);
      if (llama_vocab_is_eog(vocab, new_token_id)) {
        llama_sampler_free(sampler);
        mtmd_input_chunks_free(mm_chunks);
        mtmd_free(mctx);
        return std::string("");
      }
      // 生成循环：单 token 步进
      static thread_local std::vector<llama_token> next_token_buf(1);
      while (produced < to_predict) {
        // 输出片段
        char buf[256];
        const int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) output.append(buf, (size_t)n);

        // 准备下一个解码 batch
        next_token_buf[0] = new_token_id;
        llama_batch next_batch = llama_batch_get_one(next_token_buf.data(), 1);
        next_batch.logits = nullptr;
        if (llama_decode(ctx_, next_batch)) {
          break;
        }
        new_token_id = llama_sampler_sample(sampler, ctx_, 0);
        if (llama_vocab_is_eog(vocab, new_token_id)) {
          break;
        }
        produced += 1;
      }

      llama_sampler_free(sampler);
      mtmd_input_chunks_free(mm_chunks);
      mtmd_free(mctx);
      return output.empty() ? std::string("") : output;
    }

    // -------- Features provided path (original flow) --------
    // 当用户直接提供图像特征时，必须有视觉编码器
    if (!has_encoder) {
      return std::string("[No encoder - vision unsupported for feature input] ") + prompt;
    }
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

    // --------- Prompt heuristics: detect Markdown image, URLs, or local image path ---------
    auto trim_copy = [](const std::string &s) -> std::string {
      size_t start = 0, end = s.size();
      while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
      while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
      return s.substr(start, end - start);
    };

    auto is_markdown_image_line = [](const std::string &s) -> bool {
      static const std::regex re("^\\s*!\\[[^\\]]*\\]\\(([^)]+)\\)\\s*$");
      return std::regex_match(s, re);
    };

    auto looks_like_url = [&](const std::string &s) -> bool {
      const std::string t = trim_copy(s);
      static const std::regex re_scheme("^([a-zA-Z][a-zA-Z0-9+.-]*):\\/\\/");
      if (std::regex_search(t, re_scheme)) return true;
      // data URI for images
      static const std::regex re_data("^data:image/[^;]+;base64,");
      if (std::regex_search(t, re_data)) return true;
      return false;
    };

    auto looks_like_image_reference = [&](const std::string &s) -> bool {
      const std::string t = trim_copy(s);
      if (t.empty()) return false;
      if (is_markdown_image_line(t)) return true;
      if (looks_like_url(t)) {
        static const std::regex re_img_ext("\\.(png|jpg|jpeg|gif|webp|bmp|tiff|svg)(\\?.*)?$", std::regex::icase);
        return std::regex_search(t, re_img_ext) || t.rfind("data:image/", 0) == 0;
      }
      // Local path with image extension (absolute/relative) and no spaces
      static const std::regex re_local_img("\\.(png|jpg|jpeg|gif|webp|bmp|tiff|svg)$", std::regex::icase);
      if (std::regex_search(t, re_local_img)) {
        if (t.find('/') != std::string::npos || t.find('\\') != std::string::npos) {
          return t.find_first_of(" \t\r\n") == std::string::npos;
        }
      }
      return false;
    };

    // Fallback/template: treat prompt as empty when it's a media reference
    std::string prompt_effective = trim_copy(prompt);
    const bool prompt_is_media_ref = looks_like_image_reference(prompt_effective);
    const bool should_use_default_template = prompt_effective.empty() || prompt_is_media_ref;
    if (should_use_default_template) {
      const char *env_prompt = std::getenv("OVERRIDE_IMAGE_PROMPT");
      if (env_prompt && *env_prompt) {
        prompt_effective = env_prompt;
      } else {
        // 默认模板：前缀 + 媒体标记 + 后缀（与 demo/test 保持一致）
        static constexpr const char * media_marker = "<__media__>";
        prompt_effective = std::string("请详细用中文描述这张图片：") + media_marker +
                           std::string("。要求简洁准确。");
      }
    }

    // Ensure media marker is present for multimodal prompt; do not duplicate
    {
      static constexpr const char * media_marker = "<__media__>";
      if (prompt_effective.find(media_marker) == std::string::npos) {
        // Append marker at the end to signal image content presence
        prompt_effective += media_marker;
      }
    }

    // 统一打印提示词（图像路径）
    std::cout << "Prompt: " << prompt_effective << std::endl;

    // Tokenize prompt without adding special tokens to avoid double BOS
    std::vector<llama_token> prompt_tokens;
    if (!prompt_effective.empty()) {
      int n_prompt = -llama_tokenize(vocab, prompt_effective.c_str(), (int)prompt_effective.size(),
                                     nullptr, 0, /*add_special=*/false,
                                     /*parse_special=*/true);
      if (n_prompt < 0) {
        return std::string("[Tokenize error] ") + prompt_effective;
      }
      prompt_tokens.resize((size_t)n_prompt);
      const int n_tok = llama_tokenize(
          vocab, prompt_effective.c_str(), (int)prompt_effective.size(), prompt_tokens.data(),
          (int)prompt_tokens.size(), /*add_special=*/false,
          /*parse_special=*/true);
      if (n_tok < 0) {
        return std::string("[Tokenize error] ") + prompt_effective;
      }
    }

    std::vector<llama_token> dec_init_tokens;
    dec_init_tokens.reserve(1 + prompt_tokens.size());
    dec_init_tokens.push_back(decoder_start_token_id);
    dec_init_tokens.insert(dec_init_tokens.end(), prompt_tokens.begin(),
                           prompt_tokens.end());

    llama_batch dec_batch = llama_batch_get_one(dec_init_tokens.data(),
                                                (int32_t)dec_init_tokens.size());
    dec_batch.logits = nullptr; // only last token logits by default

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
      next_batch.logits = nullptr;

      if (llama_decode(ctx_, next_batch)) {
        break;
      }

      dec_batch = next_batch;
      produced += 1;
    }

    llama_sampler_free(sampler);
    return output.empty() ? std::string("") : output;
  }

  int32_t getEmbeddingDim() const override {
    return model_ ? llama_model_n_embd(model_) : 0;
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