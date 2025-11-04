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
const std::string TEST_INPUT = "你好，你有名字吗？";

// 默认的 Hugging Face 模型目录（作为未设置 OVERRIDE_MODEL_DIR 时的默认值）
static constexpr const char * DEFAULT_OVERRIDE_MODEL_DIR = 
    "/Volumes/data/models/qwen3-vl-4b-instruct/";

// 模型文件路径
static std::string find_first_gguf_in_dir(const std::filesystem::path &dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    return std::string();
  }
  // 递归查找首个 .gguf 文件
  try {
    for (const auto &entry : std::filesystem::recursive_directory_iterator(dir)) {
      if (entry.is_regular_file()) {
        const auto &p = entry.path();
        if (p.has_extension() && p.extension() == ".gguf") {
          // 跳过 mmproj 文件（多模态视觉投影权重），避免误选
          const std::string fname = p.filename().string();
          if (fname.find("mmproj") != std::string::npos) {
            continue;
          }
          return p.string();
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "扫描目录失败: " << dir << ", error: " << e.what() << std::endl;
  }
  return std::string();
}

// 查找 safetensors（HF 原始权重），用于提示用户转换到 GGUF
static std::string find_first_safetensors_in_dir(const std::filesystem::path &dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    return std::string();
  }
  try {
    for (const auto &entry : std::filesystem::recursive_directory_iterator(dir)) {
      if (entry.is_regular_file()) {
        const auto &p = entry.path();
        if (p.has_extension() && p.extension() == ".safetensors") {
          return p.string();
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "扫描目录失败: " << dir << ", error: " << e.what() << std::endl;
  }
  return std::string();
}

static std::string getModelPath() {
  // 优先读取 OVERRIDE_MODEL_PATH。如果它是文件，直接返回；如果是目录，扫描 .gguf 文件。
  if (const char *env = std::getenv("OVERRIDE_MODEL_PATH")) {
    std::filesystem::path p(env);
    if (std::filesystem::exists(p)) {
      if (std::filesystem::is_regular_file(p)) {
        return p.string();
      }
      if (std::filesystem::is_directory(p)) {
        std::string found = find_first_gguf_in_dir(p);
        if (!found.empty()) return found;
        // 如果没有 gguf，尝试找到 safetensors 以提示转换
        std::string hf = find_first_safetensors_in_dir(p);
        if (!hf.empty()) return hf;
      }
    }
  }

  // 其次支持 OVERRIDE_MODEL_DIR：作为目录扫描 .gguf 文件
  if (const char *env_dir = std::getenv("OVERRIDE_MODEL_DIR")) {
    std::filesystem::path d(env_dir);
    std::string found = find_first_gguf_in_dir(d);
    if (!found.empty()) return found;
    std::string hf = find_first_safetensors_in_dir(d);
    if (!hf.empty()) return hf;
  }

  // 若未设置环境变量 OVERRIDE_MODEL_DIR，则使用默认目录
  {
    std::filesystem::path d(DEFAULT_OVERRIDE_MODEL_DIR);
    if (std::filesystem::exists(d) && std::filesystem::is_directory(d)) {
      std::string found = find_first_gguf_in_dir(d);
      if (!found.empty()) return found;
      std::string hf = find_first_safetensors_in_dir(d);
      if (!hf.empty()) return hf;
    }
  }

  // 默认为当前硬编码的 Ollama 路径
  return std::string(
      "/Users/acproject/.ollama/models/blobs/"
      "sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"
      // "sha256-9c60bdd691c1897bbfe5ddbc67336848e18c346b7ee2ab8541b135f208e5bb38"
      // "sha256-3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f"
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

// 简单的GGUF元数据dump，重点打印架构与RoPE相关键
static void dump_gguf_metadata(const std::string & model_path) {
  std::cout << "\n[GGUF Metadata Dump]" << std::endl;
  GGUFParser parser(true);
  // 可选：开启mmap读取以提升读取性能
  parser.setUseMmap(true);
  if (!parser.parseFile(model_path)) {
    std::cerr << "GGUF 解析失败：" << model_path << std::endl;
    return;
  }

  const auto & header = parser.getHeader();
  std::cout << "Header: version=" << header.version
            << ", tensors=" << header.tensor_count
            << ", kv_count=" << header.metadata_kv_count << std::endl;

  const auto & arch = parser.getArchitecture();
  std::cout << "Architecture: name='" << arch.name << "'"
            << ", ctx_len=" << arch.context_length
            << ", emb_len=" << arch.embedding_length
            << ", blocks=" << arch.block_count
            << ", rope_dim_cnt=" << arch.rope_dimension_count
            << ", rope_freq_base=" << arch.rope_freq_base
            << std::endl;

  // 打印RoPE分段信息
  std::cout << "RoPE dimension sections (" << arch.rope_dimension_sections.size() << ") : ";
  if (!arch.rope_dimension_sections.empty()) {
    for (size_t i = 0; i < arch.rope_dimension_sections.size(); ++i) {
      std::cout << arch.rope_dimension_sections[i];
      if (i + 1 < arch.rope_dimension_sections.size()) std::cout << ",";
    }
  } else {
    std::cout << "<empty>";
  }
  std::cout << std::endl;

  if (arch.has_vision) {
    std::cout << "Vision: patch_size=" << arch.vision_patch_size
              << ", spatial_patch_size=" << arch.vision_spatial_patch_size
              << ", fullatt_blocks=" << arch.vision_fullatt_block_indexes.size() << std::endl;
  }

  // 关键键存在性检查
  // 优先按架构名组装目标键，其次兼容常见别名
  const std::vector<std::string> candidate_arches = {
    arch.name,
    std::string("qwen3vl"),
    std::string("qwen2vl"),
    std::string("qwen25vl")
  };
  bool found_dimension_sections = false;
  for (const auto & a : candidate_arches) {
    const std::string key = a + ".rope.dimension_sections";
    if (const GGUFKeyValue * kv = parser.getMetadata(key)) {
      std::cout << "Found key: '" << key << "' -> [";
      auto arr = kv->asUInt64Array();
      for (size_t i = 0; i < arr.size(); ++i) {
        std::cout << arr[i];
        if (i + 1 < arr.size()) std::cout << ",";
      }
      std::cout << "]" << std::endl;
      found_dimension_sections = true;
      break;
    }
  }
  if (!found_dimension_sections) {
    std::cout << "Missing key: '<arch>.rope.dimension_sections' (尝试架构：";
    for (size_t i = 0; i < candidate_arches.size(); ++i) {
      std::cout << candidate_arches[i];
      if (i + 1 < candidate_arches.size()) std::cout << ",";
    }
    std::cout << ")" << std::endl;
  }

  // 额外打印与RoPE相关的其他键，便于诊断
  const std::vector<std::string> rope_related_keys = {
    "rope.dimension_count",
    "rope.freq_base",
    "rope.mrope_section", // 一些旧转换脚本可能写这个
  };
  for (const auto & rk : rope_related_keys) {
    if (const GGUFKeyValue * kv = parser.getMetadata(rk)) {
      std::cout << "Found key: '" << rk << "'";
      if (rk == "rope.dimension_count") {
        std::cout << " -> " << kv->asUInt32();
      } else if (rk == "rope.freq_base") {
        std::cout << " -> " << kv->asFloat32();
      } else if (rk == "rope.mrope_section") {
        auto arr = kv->asUInt64Array();
        std::cout << " -> [";
        for (size_t i = 0; i < arr.size(); ++i) {
          std::cout << arr[i];
          if (i + 1 < arr.size()) std::cout << ",";
        }
        std::cout << "]";
      }
      std::cout << std::endl;
    }
  }

  // 打印所有元数据键的数量与部分示例（避免过长）
  auto keys = parser.listMetadataKeys();
  std::cout << "Total metadata keys: " << keys.size() << std::endl;
  size_t print_n = std::min<size_t>(keys.size(), 32);
  if (print_n > 0) {
    std::cout << "Sample keys (" << print_n << ") : ";
    for (size_t i = 0; i < print_n; ++i) {
      std::cout << keys[i];
      if (i + 1 < print_n) std::cout << ", ";
    }
    std::cout << std::endl;
  }
}

int main() {
  // 加载所有动态后端（如有）
  ggml_backend_load_all();

  // 初始化模型参数并加载模型
  std::string model_path = getModelPath();
  std::cout << "Using model: " << model_path << std::endl;

  // 如果是 safetensors，提示用户先转换为 GGUF 并退出
  if (!model_path.empty()) {
    std::filesystem::path mp(model_path);
    if (mp.has_extension() && mp.extension() == ".safetensors") {
      const std::string dir = mp.parent_path().string();
      std::cerr << "\n检测到 Hugging Face safetensors 权重，llama.cpp 不能直接加载。" << std::endl;
      std::cerr << "请先转换为 GGUF 格式。推荐命令如下：" << std::endl;
      std::cerr << "\n  python3 third_party/llama.cpp/convert_hf_to_gguf.py \"" << dir << "\" \\\n+  --outfile \"" << dir << "/gguf\" \\\n+  --outtype f16" << std::endl;
      std::cerr << "\n说明：" << std::endl;
      std::cerr << "- 将在 \"" << dir << "/gguf\" 目录下生成文本模型 GGUF 和 mmproj GGUF" << std::endl;
      std::cerr << "- 转换脚本会自动为 Qwen3-VL 写入必需的 qwen3vl.rope.dimension_sections 元数据" << std::endl;
      std::cerr << "- 生成后，使用： OVERRIDE_MODEL_DIR=\"" << dir << "/gguf\" ./module_integration_test" << std::endl;
      std::cerr << "\n如果希望更小体积，可将 --outtype 改为 q8_0（精度稍降）。" << std::endl;
      return 2;
    }
  }

  // 在加载模型前先dump一次GGUF元数据，便于诊断
  dump_gguf_metadata(model_path);

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
  // 使用持久 token 缓冲，避免传入局部变量地址导致悬空指针
  std::vector<llama_token> token_buf(1);

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
    token_buf[0] = decoder_start_token_id;
    batch = llama_batch_get_one(token_buf.data(), 1);
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

    // 采样下一个 token：取当前 batch 中被标记输出的最后一个位置
    // 初次解码（提示词）仅为最后一个token计算logits，因此需要使用 n_tokens-1
    llama_token new_token_id = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1);
    if (llama_vocab_is_eog(vocab, new_token_id)) {
      break;
    }

    print_piece(vocab, new_token_id);
    std::fflush(stdout);

    // 为下一个循环构造 batch（使用持久缓冲）
    token_buf[0] = new_token_id;
    batch = llama_batch_get_one(token_buf.data(), 1);
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
