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

// llama.cpp & mtmd 头文件
#include "../extensions/ggml/ggml-backend.h"
#include "llama.h"
#include "../../third_party/llama.cpp/tools/mtmd/mtmd.h"
#include "../../third_party/llama.cpp/tools/mtmd/mtmd-helper.h"

// 项目扩展
#include "../extensions/ollama/gguf_parser.h"

using namespace duorou::extensions::ollama;

// 测试输入文本
static const std::string DEMO_TEST_INPUT = "你好，你有名字吗？";

// 默认的 Hugging Face 模型目录（作为未设置 OVERRIDE_MODEL_DIR 时的默认值）
// 默认改为项目内的 models 目录，免去每次设置环境变量
static constexpr const char * DEFAULT_OVERRIDE_MODEL_DIR =
    "/Users/acproject/workspace/cpp_projects/duorou/models";

static std::string find_first_mmproj_gguf_in_dir(const std::filesystem::path &dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    return std::string();
  }
  try {
    for (const auto &entry : std::filesystem::recursive_directory_iterator(dir)) {
      if (entry.is_regular_file()) {
        const auto &p = entry.path();
        if (p.has_extension() && p.extension() == ".gguf") {
          const std::string fname = p.filename().string();
          if (fname.find("mmproj") != std::string::npos) {
            return p.string();
          }
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "扫描目录失败: " << dir << ", error: " << e.what() << std::endl;
  }
  return std::string();
}

static std::string find_first_gguf_in_dir(const std::filesystem::path &dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    return std::string();
  }
  try {
    for (const auto &entry : std::filesystem::recursive_directory_iterator(dir)) {
      if (entry.is_regular_file()) {
        const auto &p = entry.path();
        if (p.has_extension() && p.extension() == ".gguf") {
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
  if (const char *env = std::getenv("OVERRIDE_MODEL_PATH")) {
    std::filesystem::path p(env);
    if (std::filesystem::exists(p)) {
      if (std::filesystem::is_regular_file(p)) {
        return p.string();
      }
      if (std::filesystem::is_directory(p)) {
        std::string found = find_first_gguf_in_dir(p);
        if (!found.empty()) return found;
        std::string hf = find_first_safetensors_in_dir(p);
        if (!hf.empty()) return hf;
      }
    }
  }

  if (const char *env_dir = std::getenv("OVERRIDE_MODEL_DIR")) {
    std::filesystem::path d(env_dir);
    std::string found = find_first_gguf_in_dir(d);
    if (!found.empty()) return found;
    std::string hf = find_first_safetensors_in_dir(d);
    if (!hf.empty()) return hf;
  }

  {
    std::filesystem::path d(DEFAULT_OVERRIDE_MODEL_DIR);
    if (std::filesystem::exists(d) && std::filesystem::is_directory(d)) {
      std::string found = find_first_gguf_in_dir(d);
      if (!found.empty()) return found;
      std::string hf = find_first_safetensors_in_dir(d);
      if (!hf.empty()) return hf;
    }
  }

  return std::string(
      "/Users/acproject/.ollama/models/blobs/"
      "sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"
  );
}

static std::string getMmprojPathFallback(const std::string &model_path) {
  if (const char *env = std::getenv("OVERRIDE_MMPROJ_PATH")) {
    std::filesystem::path p(env);
    if (std::filesystem::exists(p)) {
      if (std::filesystem::is_regular_file(p)) {
        return p.string();
      }
      if (std::filesystem::is_directory(p)) {
        std::string found = find_first_mmproj_gguf_in_dir(p);
        if (!found.empty()) return found;
      }
    }
  }

  std::filesystem::path mp(model_path);
  std::filesystem::path d = std::filesystem::is_regular_file(mp) ? mp.parent_path() : mp;
  if (std::filesystem::exists(d) && std::filesystem::is_directory(d)) {
    std::string found = find_first_mmproj_gguf_in_dir(d);
    if (!found.empty()) return found;
  }

  {
    std::filesystem::path d2(DEFAULT_OVERRIDE_MODEL_DIR);
    if (std::filesystem::exists(d2) && std::filesystem::is_directory(d2)) {
      std::string found = find_first_mmproj_gguf_in_dir(d2);
      if (!found.empty()) return found;
    }
  }

  return std::string();
}

static std::string getImagePath() {
  if (const char *env = std::getenv("OVERRIDE_IMAGE_PATH")) {
    std::filesystem::path p(env);
    if (std::filesystem::exists(p) && std::filesystem::is_regular_file(p)) {
      return p.string();
    }
  }
  return std::string();
}

static void print_piece(const llama_vocab * vocab, llama_token id) {
  char buf[256];
  const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
  if (n > 0) {
    std::string s(buf, n);
    std::cout << s;
  }
}

static void dump_gguf_metadata(const std::string & model_path) {
  std::cout << "\n[GGUF Metadata Dump]" << std::endl;
  GGUFParser parser(true);
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

  const std::vector<std::string> rope_related_keys = {
    "rope.dimension_count",
    "rope.freq_base",
    "rope.mrope_section",
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

int run_mtmd_demo() {
  ggml_backend_load_all();

  std::string model_path = getModelPath();
  std::cout << "Using model: " << model_path << std::endl;

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
      std::cerr << "- 生成后，使用： OVERRIDE_MODEL_DIR=\"" << dir << "/gguf\" ./duorou" << std::endl;
      std::cerr << "\n如果希望更小体积，可将 --outtype 改为 q8_0（精度稍降）。" << std::endl;
      return 2;
    }
  }

  dump_gguf_metadata(model_path);

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 99;

  llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
  if (!model) {
    std::cerr << "Failed to load model from: " << model_path << std::endl;
    return 1;
  }

  const llama_vocab * vocab = llama_model_get_vocab(model);

  std::string mmproj_path = getMmprojPathFallback(model_path);
  std::string image_path  = getImagePath();
  bool use_mtmd = false;
  mtmd_context * mctx = nullptr;
  mtmd_input_chunks * mm_chunks = nullptr;
  std::string prompt_media;
  llama_pos mm_required_pos = 0;

  if (!mmproj_path.empty() && !image_path.empty()) {
    std::cout << "Using mmproj: " << mmproj_path << std::endl;
    std::cout << "Using image : " << image_path  << std::endl;

    auto mparams_mtmd = mtmd_context_params_default();
    mparams_mtmd.use_gpu = true;
    mparams_mtmd.media_marker = mtmd_default_marker();

    mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams_mtmd);
    if (!mctx) {
      std::cerr << "mtmd_init_from_file 失败，回退到纯文本推理" << std::endl;
    } else {
      const char * marker = mtmd_default_marker();
      const char * env_prompt = std::getenv("OVERRIDE_IMAGE_PROMPT");
      if (env_prompt && *env_prompt) {
        prompt_media = std::string(env_prompt);
        if (prompt_media.find(marker) == std::string::npos) {
          prompt_media += marker;
        }
      } else {
        prompt_media = std::string("请详细用中文描述这张图片：") + marker + std::string("。要求简洁准确。");
      }

      mtmd_bitmap * bitmap = mtmd_helper_bitmap_init_from_file(mctx, image_path.c_str());
      if (!bitmap) {
        std::cerr << "加载图片失败，回退到纯文本推理" << std::endl;
        mtmd_free(mctx);
        mctx = nullptr;
      } else {
        mm_chunks = mtmd_input_chunks_init();
        mtmd_input_text txt { prompt_media.c_str(), /*add_special=*/true, /*parse_special=*/true };
        const mtmd_bitmap * bitmaps[1] = { bitmap };
        int32_t tok_res = mtmd_tokenize(mctx, mm_chunks, &txt, bitmaps, 1);
        mtmd_bitmap_free(bitmap);
        if (tok_res != 0) {
          std::cerr << "mtmd_tokenize 失败（返回码 " << tok_res << ")，回退到纯文本推理" << std::endl;
          mtmd_input_chunks_free(mm_chunks);
          mm_chunks = nullptr;
          mtmd_free(mctx);
          mctx = nullptr;
        } else {
          use_mtmd = true;
          mm_required_pos = mtmd_helper_get_n_pos(mm_chunks);
        }
      }
    }
  } else {
    if (mmproj_path.empty()) {
      std::cout << "未找到 mmproj GGUF，保持纯文本推理。可设置 OVERRIDE_MMPROJ_PATH 或将 mmproj 放在模型目录。" << std::endl;
    }
    if (image_path.empty()) {
      std::cout << "未设置图片路径。请通过 OVERRIDE_IMAGE_PATH 提供图片文件（jpg/png/gif/bmp）。" << std::endl;
    }
  }

  llama_context_params cparams = llama_context_default_params();
  if (use_mtmd) {
    const int n_predict = 64;
    cparams.n_ctx = std::max(2048, (int)mm_required_pos + n_predict + 128);
    cparams.n_batch = 512;
  } else {
    const std::string prompt = DEMO_TEST_INPUT;
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
    cparams.n_ctx = std::max(1024, n_prompt + 128);
    cparams.n_batch = std::max(32, n_prompt);
  }
  cparams.no_perf = false;

  llama_context * ctx = llama_init_from_model(model, cparams);
  if (!ctx) {
    std::cerr << "Failed to create llama_context" << std::endl;
    llama_model_free(model);
    return 1;
  }

  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = false;
  llama_sampler * smpl = llama_sampler_chain_init(sparams);
  llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

  const int n_predict = 64;
  std::vector<llama_token> token_buf(1);
  llama_batch batch = {};

  if (use_mtmd) {
    std::cout << "Prompt: " << prompt_media << std::endl;

    llama_pos n_past_out = 0;
    int32_t eval_res = mtmd_helper_eval_chunks(mctx, ctx, mm_chunks,
                                               /*n_past=*/0,
                                               /*seq_id=*/0,
                                               /*n_batch=*/cparams.n_batch,
                                               /*logits_last=*/true,
                                               &n_past_out);
    if (eval_res != 0) {
      std::cerr << "mtmd_helper_eval_chunks 失败（返回码 " << eval_res << ")" << std::endl;
      if (mm_chunks) mtmd_input_chunks_free(mm_chunks);
      if (mctx) mtmd_free(mctx);
      llama_sampler_free(smpl);
      llama_free(ctx);
      llama_model_free(model);
      return 1;
    }

    llama_token first_token = llama_sampler_sample(smpl, ctx, -1);
    if (llama_vocab_is_eog(vocab, first_token)) {
      std::cout << "\n<eog>" << std::endl;
      if (mm_chunks) mtmd_input_chunks_free(mm_chunks);
      if (mctx) mtmd_free(mctx);
      llama_sampler_free(smpl);
      llama_free(ctx);
      llama_model_free(model);
      return 0;
    }
    print_piece(vocab, first_token);
    std::fflush(stdout);
    token_buf[0] = first_token;
    batch = llama_batch_get_one(token_buf.data(), 1);
  } else {
    const std::string prompt = DEMO_TEST_INPUT;
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    const int n_tok = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), prompt_tokens.data(), (int)prompt_tokens.size(), /*add_special=*/true, /*parse_special=*/true);
    if (n_tok < 0) {
      std::cerr << "Failed to tokenize prompt" << std::endl;
      llama_sampler_free(smpl);
      llama_free(ctx);
      llama_model_free(model);
      return 1;
    }

    std::cout << "Prompt: ";
    for (auto id : prompt_tokens) { print_piece(vocab, id); }
    std::cout << std::endl;

    batch = llama_batch_get_one(prompt_tokens.data(), (int)prompt_tokens.size());

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
  }

  const auto t_start = ggml_time_us();
  int n_decode = use_mtmd ? 1 : 0;
  std::cout << "\nOutput: ";

  for (; n_decode < n_predict; ) {
    if (llama_decode(ctx, batch)) {
      std::cerr << "llama_decode failed" << std::endl;
      break;
    }

    llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
    if (llama_vocab_is_eog(vocab, new_token_id)) {
      break;
    }

    print_piece(vocab, new_token_id);
    std::fflush(stdout);

    token_buf[0] = new_token_id;
    batch = llama_batch_get_one(token_buf.data(), 1);
    n_decode += 1;
  }

  const auto t_end = ggml_time_us();
  const float dt = (t_end - t_start) / 1000000.0f;
  std::cout << "\n\nDecoded " << n_decode << " tokens in " << dt << " s" << std::endl;

  llama_perf_sampler_print(smpl);
  llama_perf_context_print(ctx);

  llama_sampler_free(smpl);
  llama_free(ctx);
  if (mm_chunks) mtmd_input_chunks_free(mm_chunks);
  if (mctx) mtmd_free(mctx);
  llama_model_free(model);

  return 0;
}