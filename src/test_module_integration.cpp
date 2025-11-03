#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// 包含 ollama 扩展的头文件
#include "../../third_party/llama.cpp/src/models/models.h"
#include "../../third_party/llama.cpp/include/llama.h"
#include "extensions/ollama/gguf_parser.h"
#include "extensions/ollama/ollama_model_manager.h"
#include "extensions/ollama/ollama_path_resolver.h"
using namespace duorou::extensions::ollama;

// 测试输入文本
const std::string TEST_INPUT = "你好，马上是中秋节了，帮我写一首诗。";

// 模型文件路径
static std::string getModelPath() {
  const char *env = std::getenv("OVERRIDE_MODEL_PATH");
  if (env && std::filesystem::exists(env)) {
    return std::string(env);
  }
  return std::string(
      "/Users/acproject/.ollama/models/blobs/"

      // "sha256-"
      // "9c60bdd691c1897bbfe5ddbc67336848e18c346b7ee2ab8541b135f208e5bb38"
      "sha256-3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f"
    
    );
}

static void print_piece(const llama_vocab * vocab, llama_token id) {
  char buf[256];
  const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
  if (n > 0) {
    std::string s(buf, n);
    std::cout << s;
  }
}

int main() {
  // 加载所有动态后端（如有）
  ggml_backend_load_all();

  // 初始化模型参数并加载模型
  std::string model_path = getModelPath();
  std::cout << "Using model: " << model_path << std::endl;

  llama_model_params mparams = llama_model_default_params();
  // 可选：将部分层下放到GPU（根据环境调整）
  mparams.n_gpu_layers = 99;

  llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
  if (!model) {
    std::cerr << "Failed to load model from: " << model_path << std::endl;
    return 1;
  }

  const llama_vocab * vocab = llama_model_get_vocab(model);

  // 处理提示词：中文诗歌请求
  const std::string prompt = TEST_INPUT;

  // 计算提示词的 token 数量
  const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
  if (n_prompt <= 0) {
    std::cerr << "Tokenization returned invalid length: " << n_prompt << std::endl;
    llama_model_free(model);
    return 1;
  }

  std::vector<llama_token> prompt_tokens(n_prompt);
  const int n_tok = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), prompt_tokens.data(), (int)prompt_tokens.size(), /*add_special=*/true, /*parse_special=*/true);
  if (n_tok < 0) {
    std::cerr << "Failed to tokenize prompt" << std::endl;
    llama_model_free(model);
    return 1;
  }

  // 初始化上下文参数
  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = std::max(1024, n_prompt + 128);
  cparams.n_batch = std::max(32, n_prompt);
  cparams.no_perf = false;

  llama_context * ctx = llama_init_from_model(model, cparams);
  if (!ctx) {
    std::cerr << "Failed to create llama_context" << std::endl;
    llama_model_free(model);
    return 1;
  }

  // 初始化采样器（贪心或可配置）
  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = false;
  llama_sampler * smpl = llama_sampler_chain_init(sparams);
  llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

  // 打印提示词原文（按 token）
  std::cout << "Prompt: ";
  for (auto id : prompt_tokens) {
    print_piece(vocab, id);
  }
  std::cout << std::endl;

  // 为提示词创建 batch
  llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int)prompt_tokens.size());

  // 如果模型包含编码器，先进行 encode，然后从解码器开始符号继续
  if (llama_model_has_encoder(model)) {
    if (llama_encode(ctx, batch)) {
      std::cerr << "llama_encode failed" << std::endl;
      llama_sampler_free(smpl);
      llama_free(ctx);
      llama_model_free(model);
      return 1;
    }

    llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
    if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
      decoder_start_token_id = llama_vocab_bos(vocab);
    }
    batch = llama_batch_get_one(&decoder_start_token_id, 1);
  }

  // 主生成循环
  const auto t_start = ggml_time_us();
  int n_decode = 0;
  const int n_predict = 64;
  std::cout << "\nOutput: ";

  for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
    if (llama_decode(ctx, batch)) {
      std::cerr << "llama_decode failed" << std::endl;
      break;
    }

    n_pos += batch.n_tokens;

    // 采样下一个 token
    llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
    if (llama_vocab_is_eog(vocab, new_token_id)) {
      break;
    }

    print_piece(vocab, new_token_id);
    std::fflush(stdout);

    // 为下一个循环构造 batch
    batch = llama_batch_get_one(&new_token_id, 1);
    n_decode += 1;
  }

  const auto t_end = ggml_time_us();
  const float dt = (t_end - t_start) / 1000000.0f;
  std::cout << "\n\nDecoded " << n_decode << " tokens in " << dt << " s" << std::endl;

  // 性能数据输出
  llama_perf_sampler_print(smpl);
  llama_perf_context_print(ctx);

  // 清理资源
  llama_sampler_free(smpl);
  llama_free(ctx);
  llama_model_free(model);

  return 0;
}
