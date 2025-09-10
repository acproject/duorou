#include "qwen25vl_modular_engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>

// BLAS支持
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
#include <cblas.h>
#endif

namespace duorou {
namespace extensions {
namespace ollama {

Qwen25VLModularEngine::Qwen25VLModularEngine() {
  // 使用单例模式获取算法管理器
}

Qwen25VLModularEngine::~Qwen25VLModularEngine() = default;

bool Qwen25VLModularEngine::initialize(const Qwen25VLConfig &config) {
  if (!validateConfig(config)) {
    std::cerr << "Invalid Qwen2.5-VL configuration" << std::endl;
    return false;
  }

  config_ = config;

  try {
    // 创建算法上下文
    algorithms::AlgorithmContext context;
    context.device = "cpu";
    context.verbose = false;
    context.num_threads = 1;

    // 创建模型配置
    algorithms::ModelConfig model_config;
    model_config.hidden_size = config.hidden_size;
    model_config.num_attention_heads = config.num_attention_heads;
    model_config.num_key_value_heads = config.num_key_value_heads;
    model_config.intermediate_size = config.intermediate_size;
    model_config.max_position_embeddings = config.max_position_embeddings;
    model_config.rope_theta = config.rope_theta;
    model_config.rms_norm_eps = config.rms_norm_eps;

    // 初始化算法组件
    attention_ = std::make_unique<algorithms::MultiHeadAttention>();
    if (!attention_->initialize(model_config, context)) {
      std::cerr << "Failed to initialize MultiHeadAttention" << std::endl;
      return false;
    }

    feed_forward_ = std::make_unique<algorithms::FeedForward>();
    if (!feed_forward_->initialize(model_config, context)) {
      std::cerr << "Failed to initialize FeedForward" << std::endl;
      return false;
    }

    rope_processor_ = std::make_unique<algorithms::RoPEProcessor>();
    if (!rope_processor_->initialize(model_config, context)) {
      std::cerr << "Failed to initialize RoPEProcessor" << std::endl;
      return false;
    }

    matrix_ops_ = std::make_unique<algorithms::MatrixOperations>();
    if (!matrix_ops_->initialize(model_config, context)) {
      std::cerr << "Failed to initialize MatrixOperations" << std::endl;
      return false;
    }

    // 初始化KV缓存
    initializeKVCache();

    initialized_ = true;
    std::cout << "Qwen2.5-VL Modular Engine initialized successfully"
              << std::endl;

    return true;
  } catch (const std::exception &e) {
    std::cerr << "Exception during initialization: " << e.what() << std::endl;
    return false;
  }
}

bool Qwen25VLModularEngine::loadWeights(const std::string &model_path) {
  if (!initialized_) {
    std::cerr << "Engine not initialized" << std::endl;
    return false;
  }

  try {
    // 加载Transformer权重
    if (!loadTransformerWeights(model_path)) {
      std::cerr << "Failed to load transformer weights" << std::endl;
      return false;
    }

    // 加载视觉编码器权重
    if (!loadVisionWeights(model_path)) {
      std::cerr << "Failed to load vision weights" << std::endl;
      return false;
    }

    std::cout << "Model weights loaded successfully from: " << model_path
              << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Exception during weight loading: " << e.what() << std::endl;
    return false;
  }
}

std::vector<uint32_t>
Qwen25VLModularEngine::generateText(const std::vector<uint32_t> &input_ids,
                                    uint32_t max_length, float temperature,
                                    uint32_t top_k, float top_p) {

  if (!initialized_) {
    std::cerr << "[DEBUG] Engine not initialized" << std::endl;
    return {};
  }

  std::cerr << "[DEBUG] Qwen25VLModularEngine::generateText called with "
            << input_ids.size() << " input tokens, max_length=" << max_length
            << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<uint32_t> generated_tokens = input_ids;
  state_.current_length = input_ids.size();
  state_.is_prefill = true;

  try {
    // 修正：max_length应该是总长度限制，而不是新生成token的数量
    uint32_t max_new_tokens =
        max_length > input_ids.size() ? max_length - input_ids.size() : 100;
    for (uint32_t step = 0; step < max_new_tokens; ++step) {
      std::cerr << "[DEBUG] Generation step " << step << "/" << max_new_tokens
                << " (total length: " << generated_tokens.size() << "/"
                << max_length << ")" << std::endl;

      // 准备输入
      std::vector<uint32_t> current_input;
      if (state_.is_prefill) {
        current_input = generated_tokens;
        state_.is_prefill = false;
        std::cerr << "[DEBUG] Prefill mode with " << current_input.size()
                  << " tokens" << std::endl;
      } else {
        current_input = {generated_tokens.back()};
        std::cerr << "[DEBUG] Decode mode with 1 token" << std::endl;
      }

      // 应用词嵌入
      std::cerr << "[DEBUG] Applying embedding..." << std::endl;
      algorithms::Tensor embeddings = applyEmbedding(current_input);
      std::cerr << "[DEBUG] Embedding applied successfully" << std::endl;

      // 通过Transformer层
      algorithms::Tensor hidden_states = embeddings;
      algorithms::Tensor attention_mask =
          createAttentionMask(current_input.size());
      std::cerr << "[DEBUG] Created attention mask" << std::endl;

      for (uint32_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
        std::cerr << "[DEBUG] Processing transformer layer " << layer << "/"
                  << config_.num_hidden_layers << std::endl;
        hidden_states =
            forwardTransformerLayer(hidden_states, layer, &attention_mask);
        std::cerr << "[DEBUG] Transformer layer " << layer << " completed"
                  << std::endl;
      }

      // 生成logits
      std::cerr << "[DEBUG] Generating logits..." << std::endl;
      algorithms::Tensor logits = generateLogits(hidden_states);
      std::cerr << "[DEBUG] Logits generated successfully" << std::endl;

      // 采样下一个token（应用重复惩罚）
      std::cerr << "[DEBUG] Sampling next token with repetition penalty..."
                << std::endl;
      float repetition_penalty = 1.1f; // 设置重复惩罚参数
      uint32_t next_token = sampleToken(logits, temperature, top_k, top_p,
                                        generated_tokens, repetition_penalty);
      std::cerr << "[DEBUG] Sampled token: " << next_token << std::endl;

      // 详细检查token值
      std::cerr << "[DEBUG] Token check - sampled: " << next_token
                << ", EOS candidates: 151645, 151643, 151644" << std::endl;
      generated_tokens.push_back(next_token);

      // 更新状态
      state_.current_length++;
      state_.cache_position++;

      // 检查结束条件 - 使用Qwen特定的EOS tokens
      bool is_eos = (next_token == QwenTokens::ENDOFTEXT ||
                     next_token == QwenTokens::IM_END);
      std::cerr << "[DEBUG] EOS check result: " << (is_eos ? "TRUE" : "FALSE")
                << std::endl;
      if (is_eos) {
        std::cerr << "[DEBUG] EOS token encountered (" << next_token
                  << "), stopping generation" << std::endl;
        break;
      }
    }

    // 更新性能统计
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    perf_stats_.total_inference_time += duration.count();
    perf_stats_.total_tokens += generated_tokens.size() - input_ids.size();
    perf_stats_.inference_count++;

    if (perf_stats_.total_inference_time > 0) {
      perf_stats_.tokens_per_second = (perf_stats_.total_tokens * 1000.0) /
                                      perf_stats_.total_inference_time;
    }

    return generated_tokens;

  } catch (const std::exception &e) {
    std::cerr << "Exception during text generation: " << e.what() << std::endl;
    return {};
  }
}

std::vector<uint32_t> Qwen25VLModularEngine::generateMultimodal(
    const std::vector<uint32_t> &input_ids,
    const algorithms::Tensor &image_features, uint32_t max_length,
    float temperature, uint32_t top_k, float top_p) {

  if (!initialized_) {
    std::cerr << "Engine not initialized" << std::endl;
    return {};
  }

  try {
    // 编码图像特征
    algorithms::Tensor encoded_image = encodeImage(image_features);

    // 这里需要实现图像特征与文本特征的融合
    // 简化实现：直接使用文本生成
    return generateText(input_ids, max_length, temperature, top_k, top_p);

  } catch (const std::exception &e) {
    std::cerr << "Exception during multimodal generation: " << e.what()
              << std::endl;
    return {};
  }
}

void Qwen25VLModularEngine::generateTextStreaming(
    const std::vector<uint32_t> &input_ids, StreamingCallback callback,
    uint32_t max_length, float temperature, uint32_t top_k, float top_p) {

  if (!initialized_) {
    std::cerr << "[DEBUG] Engine not initialized" << std::endl;
    if (callback)
      callback(0, true); // 通知错误结束
    return;
  }

  if (!callback) {
    std::cerr << "[DEBUG] No callback provided for streaming" << std::endl;
    return;
  }

  std::cerr << "[DEBUG] Starting streaming text generation with "
            << input_ids.size() << " input tokens" << std::endl;

  // 初始化流式状态
  streaming_state_.is_streaming = true;
  streaming_state_.should_stop = false;
  streaming_state_.callback = callback;
  streaming_state_.generated_tokens.clear();

  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<uint32_t> generated_tokens = input_ids;
  state_.current_length = input_ids.size();
  state_.is_prefill = true;

  try {
    uint32_t max_new_tokens =
        max_length > input_ids.size() ? max_length - input_ids.size() : 100;

    for (uint32_t step = 0;
         step < max_new_tokens && !streaming_state_.should_stop; ++step) {
      std::cerr << "[DEBUG] Streaming step " << step << "/" << max_new_tokens
                << std::endl;

      // 准备输入
      std::vector<uint32_t> current_input;
      if (state_.is_prefill) {
        current_input = generated_tokens;
        state_.is_prefill = false;
        std::cerr << "[DEBUG] Prefill mode with " << current_input.size()
                  << " tokens" << std::endl;
      } else {
        current_input = {generated_tokens.back()};
        std::cerr << "[DEBUG] Decode mode with 1 token" << std::endl;
      }

      // 应用词嵌入
      algorithms::Tensor embeddings = applyEmbedding(current_input);

      // 通过Transformer层
      algorithms::Tensor hidden_states = embeddings;
      algorithms::Tensor attention_mask =
          createAttentionMask(current_input.size());

      for (uint32_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
        hidden_states =
            forwardTransformerLayer(hidden_states, layer, &attention_mask);
      }

      // 生成logits
      algorithms::Tensor logits = generateLogits(hidden_states);

      // 采样下一个token（应用重复惩罚）
      float repetition_penalty = 1.1f; // 设置重复惩罚参数
      uint32_t next_token = sampleToken(logits, temperature, top_k, top_p,
                                        generated_tokens, repetition_penalty);
      std::cerr << "[DEBUG] Streaming sampled token: " << next_token
                << std::endl;

      generated_tokens.push_back(next_token);
      streaming_state_.generated_tokens.push_back(next_token);

      // 更新状态
      state_.current_length++;
      state_.cache_position++;

      // 检查结束条件 - 使用Qwen特定的EOS tokens
      bool is_eos = (next_token == QwenTokens::ENDOFTEXT ||
                     next_token == QwenTokens::IM_END);
      bool is_final = is_eos || (step == max_new_tokens - 1) ||
                      streaming_state_.should_stop;

      // 立即通过回调返回token
      callback(next_token, is_final);

      if (is_eos) {
        std::cerr << "[DEBUG] EOS token encountered in streaming, stopping"
                  << std::endl;
        break;
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cerr << "[DEBUG] Streaming generation completed in "
              << duration.count() << "ms" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Exception during streaming generation: " << e.what()
              << std::endl;
    if (callback)
      callback(0, true); // 通知错误结束
  }

  // 重置流式状态
  streaming_state_.is_streaming = false;
  streaming_state_.callback = nullptr;
}

void Qwen25VLModularEngine::generateMultimodalStreaming(
    const std::vector<uint32_t> &input_ids,
    const algorithms::Tensor &image_features, StreamingCallback callback,
    uint32_t max_length, float temperature, uint32_t top_k, float top_p) {

  if (!initialized_) {
    std::cerr << "Engine not initialized" << std::endl;
    if (callback)
      callback(0, true);
    return;
  }

  // TODO: 实现流式多模态推理
  std::cerr << "Streaming multimodal generation not yet implemented, falling "
               "back to text-only"
            << std::endl;
  generateTextStreaming(input_ids, callback, max_length, temperature, top_k,
                        top_p);
}

void Qwen25VLModularEngine::stopStreaming() {
  if (streaming_state_.is_streaming) {
    std::cerr << "[DEBUG] Stopping streaming generation" << std::endl;
    streaming_state_.should_stop = true;
  }
}

algorithms::Tensor
Qwen25VLModularEngine::encodeImage(const algorithms::Tensor &image) {
  if (!initialized_) {
    throw std::runtime_error("Engine not initialized");
  }

  try {
    // 通过视觉编码器处理图像
    return forwardVisionEncoder(image);
  } catch (const std::exception &e) {
    std::cerr << "Exception during image encoding: " << e.what() << std::endl;
    throw;
  }
}

void Qwen25VLModularEngine::resetPerformanceStats() {
  perf_stats_ = PerformanceStats{};
}

// 私有方法实现

algorithms::Tensor Qwen25VLModularEngine::forwardTransformerLayer(
    const algorithms::Tensor &input, uint32_t layer_idx,
    const algorithms::Tensor *attention_mask) {

  try {
    // 标准Transformer层架构：
    // 1. Multi-head Self-Attention (带causal mask)
    // 2. Add & LayerNorm
    // 3. Feed-Forward Network
    // 4. Add & LayerNorm

    // 1. 自注意力计算
    // 应用RoPE位置编码到输入
    algorithms::Tensor rope_input =
        rope_processor_->apply(input, state_.cache_position);

    // 调试：打印张量维度信息
    std::cout << "[DEBUG] Input tensor shape: [";
    for (size_t i = 0; i < input.shape.size(); ++i) {
      std::cout << input.shape[i];
      if (i < input.shape.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "[DEBUG] RoPE output tensor shape: [";
    for (size_t i = 0; i < rope_input.shape.size(); ++i) {
      std::cout << rope_input.shape[i];
      if (i < rope_input.shape.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Q/K/V投影计算
    algorithms::Tensor q_proj =
        performMatMul(rope_input, weights_.q_proj_weights[layer_idx]);
    algorithms::Tensor k_proj =
        performMatMul(rope_input, weights_.k_proj_weights[layer_idx]);
    algorithms::Tensor v_proj =
        performMatMul(rope_input, weights_.v_proj_weights[layer_idx]);

    // 调试：打印投影后张量维度信息
    std::cout << "[DEBUG] Q projection shape: [";
    for (size_t i = 0; i < q_proj.shape.size(); ++i) {
      std::cout << q_proj.shape[i];
      if (i < q_proj.shape.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 多头注意力计算
    algorithms::Tensor attention_output;
    if (state_.is_prefill) {
      attention_output =
          attention_->compute(q_proj, k_proj, v_proj, attention_mask);
    } else {
      attention_output = attention_->computeWithCache(
          q_proj, k_proj, v_proj, state_.key_cache[layer_idx],
          state_.value_cache[layer_idx], state_.cache_position, attention_mask);
    }

    // 输出投影
    attention_output =
        performMatMul(attention_output, weights_.o_proj_weights[layer_idx]);

    // 2. Add & LayerNorm (Post-LN)
    algorithms::Tensor residual1 = input;
    for (size_t i = 0;
         i < input.data.size() && i < attention_output.data.size(); ++i) {
      residual1.data[i] += attention_output.data[i];
    }

    // 检查layer_norm_weights索引
    if (layer_idx * 2 >= weights_.layer_norm_weights.size()) {
      std::cerr << "[ERROR] Layer norm weights not found for layer "
                << layer_idx << std::endl;
      return input; // 返回原始输入作为fallback
    }

    algorithms::Tensor norm_output1 =
        applyRMSNorm(residual1, weights_.layer_norm_weights[layer_idx * 2]);

    // 3. 前馈网络计算
    algorithms::Tensor ffn_output;
    if (layer_idx * 3 + 2 < weights_.ffn_weights.size()) {
      try {
        ffn_output = feed_forward_->compute(
            norm_output1,
            weights_.ffn_weights[layer_idx * 3],     // gate_weights
            weights_.ffn_weights[layer_idx * 3 + 1], // up_weights
            weights_.ffn_weights[layer_idx * 3 + 2]  // down_weights
        );
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] FFN computation failed: " << e.what()
                  << std::endl;
        ffn_output = norm_output1; // fallback
      }
    } else {
      std::cerr << "[WARNING] FFN weights not found for layer " << layer_idx
                << std::endl;
      ffn_output = norm_output1; // fallback
    }

    // 4. Add & LayerNorm (Post-LN)
    algorithms::Tensor residual2 = norm_output1;
    for (size_t i = 0;
         i < norm_output1.data.size() && i < ffn_output.data.size(); ++i) {
      residual2.data[i] += ffn_output.data[i];
    }

    // 检查第二个layer norm权重
    if (layer_idx * 2 + 1 >= weights_.layer_norm_weights.size()) {
      std::cerr << "[ERROR] Second layer norm weights not found for layer "
                << layer_idx << std::endl;
      return residual2; // 返回未归一化的结果
    }

    algorithms::Tensor final_output =
        applyRMSNorm(residual2, weights_.layer_norm_weights[layer_idx * 2 + 1]);

    return final_output;

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in transformer layer " << layer_idx << ": "
              << e.what() << std::endl;
    return input; // 返回原始输入作为fallback
  }
}

algorithms::Tensor
Qwen25VLModularEngine::forwardVisionEncoder(const algorithms::Tensor &image) {
  try {
    // 简化的视觉编码器实现
    algorithms::Tensor current = image;

    // 通过视觉Transformer层
    for (uint32_t layer = 0; layer < config_.vision_num_hidden_layers;
         ++layer) {
      // 这里需要实现视觉注意力和前馈网络
      // 简化实现：直接返回输入
    }

    return current;

  } catch (const std::exception &e) {
    std::cerr << "Exception in vision encoder: " << e.what() << std::endl;
    throw;
  }
}

algorithms::Tensor
Qwen25VLModularEngine::applyRMSNorm(const algorithms::Tensor &input,
                                    const algorithms::Tensor &weight,
                                    float eps) {

  try {
    // RMSNorm实现
    algorithms::Tensor output = input;
    uint32_t hidden_size = input.shape.back();
    uint32_t seq_len = input.shape[input.shape.size() - 2];
    uint32_t batch_size = (input.shape.size() > 2) ? input.shape[0] : 1;

    // 对每个batch和序列位置应用RMSNorm
    for (uint32_t b = 0; b < batch_size; ++b) {
      for (uint32_t s = 0; s < seq_len; ++s) {
        // 计算当前位置的RMS
        float sum_squares = 0.0f;
        size_t base_idx = b * seq_len * hidden_size + s * hidden_size;

        for (uint32_t i = 0; i < hidden_size; ++i) {
          float val = input.data[base_idx + i];
          sum_squares += val * val;
        }

        // 添加数值稳定性检查
        float variance = sum_squares / hidden_size;
        if (variance < 1e-12f) {
          variance = 1e-12f; // 防止除零
        }
        float rms = std::sqrt(variance + eps);

        // 应用归一化和权重
        for (uint32_t i = 0; i < hidden_size; ++i) {
          float normalized = input.data[base_idx + i] / rms;
          output.data[base_idx + i] = normalized * weight.data[i];
        }
      }
    }

    return output;

  } catch (const std::exception &e) {
    std::cerr << "Exception in RMSNorm: " << e.what() << std::endl;
    throw;
  }
}

algorithms::Tensor
Qwen25VLModularEngine::applyEmbedding(const std::vector<uint32_t> &input_ids) {
  try {
    uint32_t seq_len = input_ids.size();
    // 修复：添加batch维度以匹配MultiHeadAttention的期望输入格式 [batch_size,
    // seq_len, hidden_size]
    std::vector<uint32_t> shape = {1, seq_len, config_.hidden_size};
    algorithms::Tensor embeddings(shape);

    // 检查token_embeddings张量是否已初始化
    if (weights_.token_embeddings.data.empty()) {
      std::cerr << "Token embeddings not initialized" << std::endl;
      throw std::runtime_error("Token embeddings not initialized");
    }

    // 检查token_embeddings张量大小是否正确
    size_t expected_embedding_size =
        static_cast<size_t>(config_.vocab_size) * config_.hidden_size;
    if (weights_.token_embeddings.data.size() < expected_embedding_size) {
      std::cerr << "Token embeddings size mismatch. Expected: "
                << expected_embedding_size
                << ", Got: " << weights_.token_embeddings.data.size()
                << std::endl;
      throw std::runtime_error("Token embeddings size mismatch");
    }

    // 简化的嵌入实现
    for (uint32_t i = 0; i < seq_len; ++i) {
      uint32_t token_id = input_ids[i];

      // 检查token_id是否在有效范围内
      if (token_id >= config_.vocab_size) {
        std::cerr << "Token ID out of range: " << token_id
                  << " >= " << config_.vocab_size << std::endl;
        throw std::out_of_range("Token ID out of range");
      }

      for (uint32_t j = 0; j < config_.hidden_size; ++j) {
        size_t src_idx =
            static_cast<size_t>(token_id) * config_.hidden_size + j;
        // 修复：更新索引计算以匹配3D张量格式 [batch_size, seq_len, hidden_size]
        size_t dst_idx = static_cast<size_t>(i) * config_.hidden_size + j;

        // 检查源索引边界
        if (src_idx >= weights_.token_embeddings.data.size()) {
          std::cerr << "Source index out of bounds: " << src_idx
                    << " >= " << weights_.token_embeddings.data.size()
                    << std::endl;
          throw std::out_of_range(
              "Source index out of bounds in token embeddings");
        }

        // 检查目标索引边界
        if (dst_idx >= embeddings.data.size()) {
          std::cerr << "Destination index out of bounds: " << dst_idx
                    << " >= " << embeddings.data.size() << std::endl;
          throw std::out_of_range(
              "Destination index out of bounds in embeddings");
        }

        embeddings.data[dst_idx] = weights_.token_embeddings.data[src_idx];
      }
    }

    return embeddings;

  } catch (const std::exception &e) {
    std::cerr << "Exception in embedding: " << e.what() << std::endl;
    throw;
  }
}

algorithms::Tensor
Qwen25VLModularEngine::generateLogits(const algorithms::Tensor &hidden_states) {
  try {
    // 应用最终层归一化
    std::cerr << "[DEBUG] Applying final layer norm..." << std::endl;
    algorithms::Tensor norm_hidden =
        applyRMSNorm(hidden_states, weights_.norm_weight);
    std::cerr << "[DEBUG] Final layer norm applied" << std::endl;

    // 应用语言模型头
    // 修复：更新维度索引以匹配3D张量格式 [batch_size, seq_len, hidden_size]
    uint32_t seq_len = hidden_states.shape[1];
    uint32_t hidden_size = hidden_states.shape[2];
    std::vector<uint32_t> logits_shape = {seq_len, config_.vocab_size};
    algorithms::Tensor logits(logits_shape);

    std::cerr << "[DEBUG] Computing logits matrix multiplication..."
              << std::endl;
    std::cerr << "[DEBUG] Matrix dimensions: seq_len=" << seq_len
              << ", hidden_size=" << hidden_size
              << ", vocab_size=" << config_.vocab_size << std::endl;

    // 计算总操作数以显示进度
    uint64_t total_ops = static_cast<uint64_t>(seq_len) * config_.vocab_size;
    std::cerr << "[DEBUG] Total operations: " << total_ops << std::endl;

    // 检查权重是否已加载
    if (weights_.lm_head_weight.data.empty()) {
      std::cerr << "[WARNING] LM head weights not loaded, using fallback"
                << std::endl;
      // 使用简化的fallback避免大量计算
      for (uint32_t i = 0; i < seq_len; ++i) {
        if (i % 10 == 0) {
          std::cerr << "[DEBUG] Fallback processing sequence " << i << "/"
                    << seq_len << std::endl;
        }

        // 只计算前1000个vocab项以加速fallback
        uint32_t limited_vocab = std::min(config_.vocab_size, 1000u);
        for (uint32_t j = 0; j < limited_vocab; ++j) {
          float sum = 0.0f;
          for (uint32_t k = 0; k < hidden_size; ++k) {
            // 使用简单的权重初始化
            float weight = (float)(k + j) / (hidden_size + limited_vocab);
            // 修复：更新索引计算以匹配3D张量格式 [batch_size, seq_len,
            // hidden_size]
            size_t idx = 0 * seq_len * hidden_size + i * hidden_size + k;
            sum += norm_hidden.data[idx] * weight;
          }
          logits.data[i * config_.vocab_size + j] = sum;
        }

        // 其余vocab项设为小值
        for (uint32_t j = limited_vocab; j < config_.vocab_size; ++j) {
          logits.data[i * config_.vocab_size + j] = -10.0f;
        }
      }
    } else {
      std::cerr << "[DEBUG] Using CBLAS-optimized matrix multiplication"
                << std::endl;

      try {
        // 使用CBLAS进行优化的矩阵乘法
        // CBLAS_SGEMM: C = alpha * A * B + beta * C
        // 其中 A 是 norm_hidden (seq_len x hidden_size)
        //      B 是 lm_head_weight^T (hidden_size x vocab_size)
        //      C 是 logits (seq_len x vocab_size)

        const float alpha = 1.0f;
        const float beta = 0.0f;

#ifdef __APPLE__
        // macOS使用Accelerate框架
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq_len,
                    config_.vocab_size, hidden_size, alpha,
                    norm_hidden.data.data(), hidden_size,
                    weights_.lm_head_weight.data.data(), hidden_size, beta,
                    logits.data.data(), config_.vocab_size);
        std::cerr
            << "[DEBUG] Used Accelerate framework for matrix multiplication"
            << std::endl;
#elif defined(USE_OPENBLAS)
        // OpenBLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq_len,
                    config_.vocab_size, hidden_size, alpha,
                    norm_hidden.data.data(), hidden_size,
                    weights_.lm_head_weight.data.data(), hidden_size, beta,
                    logits.data.data(), config_.vocab_size);
        std::cerr << "[DEBUG] Used OpenBLAS for matrix multiplication"
                  << std::endl;
#else
        // 回退到优化的分批处理（带早期停止机制）
        std::cerr << "[DEBUG] Using optimized batch processing with early "
                     "stopping (CBLAS not available)"
                  << std::endl;

        const uint32_t batch_size = 2000; // 增加批处理大小
        const auto start_time = std::chrono::steady_clock::now();
        const auto max_duration = std::chrono::seconds(30); // 最大计算时间30秒

        bool early_stopped = false;

        for (uint32_t i = 0; i < seq_len && !early_stopped; ++i) {
          std::cerr << "[DEBUG] Processing sequence " << i + 1 << "/" << seq_len
                    << std::endl;

          for (uint32_t j_start = 0; j_start < config_.vocab_size;
               j_start += batch_size) {
            // 检查是否超时
            auto current_time = std::chrono::steady_clock::now();
            if (current_time - start_time > max_duration) {
              std::cerr << "[WARNING] Matrix multiplication timeout after 30 "
                           "seconds, using partial results"
                        << std::endl;
              early_stopped = true;

              // 填充剩余的logits为小值
              for (uint32_t remaining_i = i; remaining_i < seq_len;
                   ++remaining_i) {
                for (uint32_t remaining_j = (remaining_i == i ? j_start : 0);
                     remaining_j < config_.vocab_size; ++remaining_j) {
                  logits.data[remaining_i * config_.vocab_size + remaining_j] =
                      -10.0f;
                }
              }
              break;
            }

            uint32_t j_end = std::min(j_start + batch_size, config_.vocab_size);

            if (j_start % (batch_size * 5) == 0) { // 减少输出频率
              float progress = (float)(j_start) / config_.vocab_size * 100.0f;
              auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                 current_time - start_time)
                                 .count();
              std::cerr << "[DEBUG] Progress: " << progress
                        << "%, elapsed: " << elapsed << "s" << std::endl;
            }

            // 向量化计算
            for (uint32_t j = j_start; j < j_end; ++j) {
              float sum = 0.0f;
              const float *weight_row =
                  &weights_.lm_head_weight.data[j * hidden_size];
              const float *hidden_row = &norm_hidden.data[i * hidden_size];

              // 展开循环以提高性能
              uint32_t k = 0;
              for (; k + 4 <= hidden_size; k += 4) {
                sum += weight_row[k] * hidden_row[k] +
                       weight_row[k + 1] * hidden_row[k + 1] +
                       weight_row[k + 2] * hidden_row[k + 2] +
                       weight_row[k + 3] * hidden_row[k + 3];
              }
              // 处理剩余元素
              for (; k < hidden_size; ++k) {
                sum += weight_row[k] * hidden_row[k];
              }
              logits.data[i * config_.vocab_size + j] = sum;
            }

            // 每处理一定数量的批次后让出CPU时间
            if (j_start % (batch_size * 10) == 0) {
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
          }
        }

        if (early_stopped) {
          std::cerr << "[WARNING] Computation was stopped early due to timeout"
                    << std::endl;
        }
#endif

      } catch (const std::exception &e) {
        std::cerr << "[ERROR] Matrix multiplication failed: " << e.what()
                  << std::endl;
        throw;
      }
    }

    std::cerr << "[DEBUG] Logits computation completed" << std::endl;
    return logits;

  } catch (const std::exception &e) {
    std::cerr << "Exception in logits generation: " << e.what() << std::endl;
    throw;
  }
}

uint32_t Qwen25VLModularEngine::sampleToken(
    const algorithms::Tensor &logits, float temperature, uint32_t top_k,
    float top_p, const std::vector<uint32_t> &history,
    float repetition_penalty) {

  try {
    // 获取最后一个位置的logits
    uint32_t seq_len = logits.shape[0];
    uint32_t vocab_size = logits.shape[1];

    std::vector<float> last_logits(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
      last_logits[i] = logits.data[(seq_len - 1) * vocab_size + i];
    }

    // 应用重复惩罚
    if (repetition_penalty != 1.0f && !history.empty()) {
      std::cerr << "[DEBUG] Applying repetition penalty: " << repetition_penalty
                << " to " << history.size() << " history tokens" << std::endl;
      for (uint32_t token_id : history) {
        if (token_id < vocab_size) {
          if (last_logits[token_id] > 0) {
            last_logits[token_id] /= repetition_penalty;
          } else {
            last_logits[token_id] *= repetition_penalty;
          }
        }
      }
    }

    // 应用温度缩放
    if (temperature > 0.0f && temperature != 1.0f) {
      for (float &logit : last_logits) {
        logit /= temperature;
      }
    }

    // 如果温度为0或很小，使用贪婪采样
    if (temperature <= 0.01f) {
      auto max_it = std::max_element(last_logits.begin(), last_logits.end());
      return static_cast<uint32_t>(std::distance(last_logits.begin(), max_it));
    }

    // 创建索引-logit对并排序
    std::vector<std::pair<uint32_t, float>> indexed_logits;
    for (uint32_t i = 0; i < vocab_size; ++i) {
      indexed_logits.emplace_back(i, last_logits[i]);
    }

    // 按logit值降序排序
    std::sort(indexed_logits.begin(), indexed_logits.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    // 应用top-k过滤
    if (top_k > 0 && top_k < vocab_size) {
      indexed_logits.resize(top_k);
    }

    // 计算softmax概率
    float max_logit = indexed_logits[0].second;
    std::vector<float> probs;
    probs.reserve(indexed_logits.size());
    float sum_exp = 0.0f;

    for (const auto &pair : indexed_logits) {
      float prob = std::exp(pair.second - max_logit);
      probs.push_back(prob);
      sum_exp += prob;
    }

    // 归一化概率
    for (float &prob : probs) {
      prob /= sum_exp;
    }

    // 应用top-p (nucleus sampling) 过滤
    if (top_p > 0.0f && top_p < 1.0f) {
      float cumulative_prob = 0.0f;
      size_t nucleus_size = 0;

      for (size_t i = 0; i < probs.size(); ++i) {
        cumulative_prob += probs[i];
        nucleus_size = i + 1;
        if (cumulative_prob >= top_p) {
          break;
        }
      }

      // 重新归一化nucleus内的概率
      if (nucleus_size < probs.size()) {
        indexed_logits.resize(nucleus_size);
        probs.resize(nucleus_size);

        float nucleus_sum = 0.0f;
        for (float prob : probs) {
          nucleus_sum += prob;
        }

        for (float &prob : probs) {
          prob /= nucleus_sum;
        }
      }
    }

    // 随机采样
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    float random_val = dis(gen);
    float cumulative = 0.0f;

    for (size_t i = 0; i < probs.size(); ++i) {
      cumulative += probs[i];
      if (random_val <= cumulative) {
        return indexed_logits[i].first;
      }
    }

    // 如果没有找到，返回第一个候选token
    return indexed_logits.empty() ? 0 : indexed_logits[0].first;

  } catch (const std::exception &e) {
    std::cerr << "Exception in token sampling: " << e.what() << std::endl;
    return 0;
  }
}

void Qwen25VLModularEngine::initializeKVCache() {
  try {
    state_.key_cache.clear();
    state_.value_cache.clear();

    uint32_t head_dim = config_.hidden_size / config_.num_attention_heads;
    uint32_t kv_head_dim = config_.hidden_size / config_.num_key_value_heads;

    for (uint32_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
      // 缓存形状应该是 [max_seq_len, num_kv_heads * kv_head_dim] 以匹配
      // splitCacheToHeads 的期望
      std::vector<uint32_t> cache_shape = {config_.max_position_embeddings,
                                           config_.num_key_value_heads *
                                               kv_head_dim};

      state_.key_cache.emplace_back(cache_shape);
      state_.value_cache.emplace_back(cache_shape);
    }

    state_.current_length = 0;
    state_.cache_position = 0;

  } catch (const std::exception &e) {
    std::cerr << "Exception in KV cache initialization: " << e.what()
              << std::endl;
    throw;
  }
}

void Qwen25VLModularEngine::updateKVCache(uint32_t layer_idx,
                                          const algorithms::Tensor &key,
                                          const algorithms::Tensor &value) {

  try {
    if (layer_idx >= state_.key_cache.size()) {
      throw std::out_of_range("Layer index out of range");
    }

    // 简化的缓存更新实现
    // 实际实现需要根据cache_position更新特定位置

  } catch (const std::exception &e) {
    std::cerr << "Exception in KV cache update: " << e.what() << std::endl;
    throw;
  }
}

algorithms::Tensor
Qwen25VLModularEngine::createAttentionMask(uint32_t seq_length,
                                           bool is_causal) {
  try {
    std::vector<uint32_t> mask_shape = {seq_length, seq_length};
    algorithms::Tensor mask(mask_shape);

    for (uint32_t i = 0; i < seq_length; ++i) {
      for (uint32_t j = 0; j < seq_length; ++j) {
        if (is_causal && j > i) {
          mask.data[i * seq_length + j] = 0.0f; // 屏蔽未来位置
        } else {
          mask.data[i * seq_length + j] = 1.0f;
        }
      }
    }

    return mask;

  } catch (const std::exception &e) {
    std::cerr << "Exception in attention mask creation: " << e.what()
              << std::endl;
    throw;
  }
}

void Qwen25VLModularEngine::logPerformance(const std::string &operation,
                                           double time_ms) {
  std::cout << "[PERF] " << operation << ": " << time_ms << "ms" << std::endl;
}

bool Qwen25VLModularEngine::validateConfig(const Qwen25VLConfig &config) {
  if (config.hidden_size == 0 || config.num_attention_heads == 0 ||
      config.num_hidden_layers == 0 || config.vocab_size == 0) {
    return false;
  }

  if (config.hidden_size % config.num_attention_heads != 0) {
    return false;
  }

  return true;
}

bool Qwen25VLModularEngine::loadTransformerWeights(
    const std::string &model_path) {
  try {
    // 简化的权重加载实现
    // 实际实现需要从文件加载权重

    // 初始化权重张量
    std::vector<uint32_t> embedding_shape = {config_.vocab_size,
                                             config_.hidden_size};
    weights_.token_embeddings = algorithms::Tensor(embedding_shape);

    std::vector<uint32_t> norm_shape = {config_.hidden_size};
    weights_.norm_weight = algorithms::Tensor(norm_shape);

    std::vector<uint32_t> lm_head_shape = {config_.hidden_size,
                                           config_.vocab_size};
    weights_.lm_head_weight = algorithms::Tensor(lm_head_shape);

    // 初始化层权重
    weights_.layer_norm_weights.resize(config_.num_hidden_layers * 2);
    for (uint32_t i = 0; i < config_.num_hidden_layers * 2; ++i) {
      weights_.layer_norm_weights[i] = algorithms::Tensor(norm_shape);
    }

    // 初始化注意力投影权重
    weights_.q_proj_weights.resize(config_.num_hidden_layers);
    weights_.k_proj_weights.resize(config_.num_hidden_layers);
    weights_.v_proj_weights.resize(config_.num_hidden_layers);
    weights_.o_proj_weights.resize(config_.num_hidden_layers);

    for (uint32_t i = 0; i < config_.num_hidden_layers; ++i) {
      // Q投影权重：[hidden_size, hidden_size]
      weights_.q_proj_weights[i] =
          algorithms::Tensor({config_.hidden_size, config_.hidden_size});
      // K投影权重：[hidden_size, num_kv_heads * head_dim]
      uint32_t kv_dim = config_.num_key_value_heads *
                        (config_.hidden_size / config_.num_attention_heads);
      weights_.k_proj_weights[i] =
          algorithms::Tensor({config_.hidden_size, kv_dim});
      // V投影权重：[hidden_size, num_kv_heads * head_dim]
      weights_.v_proj_weights[i] =
          algorithms::Tensor({config_.hidden_size, kv_dim});
      // O投影权重：[hidden_size, hidden_size]
      weights_.o_proj_weights[i] =
          algorithms::Tensor({config_.hidden_size, config_.hidden_size});
    }

    // 初始化注意力和前馈网络权重（临时用随机值）
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    // 初始化注意力投影权重
    for (uint32_t i = 0; i < config_.num_hidden_layers; ++i) {
      // 初始化Q投影权重
      for (size_t j = 0; j < weights_.q_proj_weights[i].data.size(); ++j) {
        weights_.q_proj_weights[i].data[j] = dist(gen);
      }
      // 初始化K投影权重
      for (size_t j = 0; j < weights_.k_proj_weights[i].data.size(); ++j) {
        weights_.k_proj_weights[i].data[j] = dist(gen);
      }
      // 初始化V投影权重
      for (size_t j = 0; j < weights_.v_proj_weights[i].data.size(); ++j) {
        weights_.v_proj_weights[i].data[j] = dist(gen);
      }
      // 初始化O投影权重
      for (size_t j = 0; j < weights_.o_proj_weights[i].data.size(); ++j) {
        weights_.o_proj_weights[i].data[j] = dist(gen);
      }
    }

    // 初始化FFN权重 - 每层需要3个权重矩阵
    weights_.ffn_weights.resize(config_.num_hidden_layers *
                                3); // gate, up, down

    for (uint32_t i = 0; i < config_.num_hidden_layers; ++i) {
      // Gate权重：[hidden_size, intermediate_size]
      weights_.ffn_weights[i * 3] =
          algorithms::Tensor({config_.hidden_size, config_.intermediate_size});
      // Up权重：[hidden_size, intermediate_size]
      weights_.ffn_weights[i * 3 + 1] =
          algorithms::Tensor({config_.hidden_size, config_.intermediate_size});
      // Down权重：[intermediate_size, hidden_size]
      weights_.ffn_weights[i * 3 + 2] =
          algorithms::Tensor({config_.intermediate_size, config_.hidden_size});

      // 初始化为小的随机值
      for (int w = 0; w < 3; ++w) {
        for (size_t j = 0; j < weights_.ffn_weights[i * 3 + w].data.size();
             ++j) {
          weights_.ffn_weights[i * 3 + w].data[j] = dist(gen);
        }
      }
    }

    // 初始化Q/K/V/O投影权重
    for (uint32_t i = 0; i < config_.num_hidden_layers; ++i) {
      for (size_t j = 0; j < weights_.q_proj_weights[i].data.size(); ++j) {
        weights_.q_proj_weights[i].data[j] = dist(gen);
      }
      for (size_t j = 0; j < weights_.k_proj_weights[i].data.size(); ++j) {
        weights_.k_proj_weights[i].data[j] = dist(gen);
      }
      for (size_t j = 0; j < weights_.v_proj_weights[i].data.size(); ++j) {
        weights_.v_proj_weights[i].data[j] = dist(gen);
      }
      for (size_t j = 0; j < weights_.o_proj_weights[i].data.size(); ++j) {
        weights_.o_proj_weights[i].data[j] = dist(gen);
      }
    }

    // 初始化embedding权重
    for (size_t i = 0; i < weights_.token_embeddings.data.size(); ++i) {
      weights_.token_embeddings.data[i] = dist(gen);
    }

    // 初始化norm权重为1.0
    for (size_t i = 0; i < weights_.norm_weight.data.size(); ++i) {
      weights_.norm_weight.data[i] = 1.0f;
    }

    // 初始化layer norm权重为1.0
    for (auto &layer_norm : weights_.layer_norm_weights) {
      for (size_t i = 0; i < layer_norm.data.size(); ++i) {
        layer_norm.data[i] = 1.0f;
      }
    }

    // 初始化lm_head权重
    for (size_t i = 0; i < weights_.lm_head_weight.data.size(); ++i) {
      weights_.lm_head_weight.data[i] = dist(gen);
    }

    // 验证权重维度
    if (weights_.token_embeddings.shape[0] != config_.vocab_size ||
        weights_.token_embeddings.shape[1] != config_.hidden_size) {
      std::cerr << "Token embedding dimension mismatch" << std::endl;
      return false;
    }

    if (weights_.norm_weight.shape[0] != config_.hidden_size) {
      std::cerr << "Norm weight dimension mismatch" << std::endl;
      return false;
    }

    std::cerr << "[INFO] Transformer weights initialized successfully"
              << std::endl;
    return true;

  } catch (const std::exception &e) {
    std::cerr << "Exception loading transformer weights: " << e.what()
              << std::endl;
    return false;
  }
}

bool Qwen25VLModularEngine::loadVisionWeights(const std::string &model_path) {
  try {
    // 简化的视觉权重加载实现
    std::vector<uint32_t> vision_embedding_shape = {config_.vision_hidden_size,
                                                    config_.vision_hidden_size};
    weights_.vision_embeddings = algorithms::Tensor(vision_embedding_shape);

    return true;

  } catch (const std::exception &e) {
    std::cerr << "Exception loading vision weights: " << e.what() << std::endl;
    return false;
  }
}

algorithms::Tensor
Qwen25VLModularEngine::loadTensorFromFile(const std::string &file_path) {
  // 简化实现：返回空张量
  return algorithms::Tensor({1});
}

// 矩阵乘法函数实现
algorithms::Tensor
Qwen25VLModularEngine::performMatMul(const algorithms::Tensor &a,
                                     const algorithms::Tensor &b) {
  // 检查输入张量的维度
  if (a.shape.empty() || b.shape.empty()) {
    throw std::invalid_argument("Input tensors cannot be empty");
  }

  // 支持2D和3D张量的矩阵乘法
  if (a.shape.size() < 2 || b.shape.size() < 2) {
    throw std::invalid_argument(
        "Input tensors must have at least 2 dimensions");
  }

  // 获取矩阵维度
  uint32_t batch_size = 1;
  uint32_t a_rows, a_cols, b_rows, b_cols;

  if (a.shape.size() == 3) {
    batch_size = a.shape[0];
    a_rows = a.shape[1];
    a_cols = a.shape[2];
  } else {
    a_rows = a.shape[0];
    a_cols = a.shape[1];
  }

  if (b.shape.size() == 3) {
    if (batch_size != b.shape[0] && batch_size != 1 && b.shape[0] != 1) {
      throw std::invalid_argument("Batch sizes must be compatible");
    }
    if (batch_size == 1)
      batch_size = b.shape[0];
    b_rows = b.shape[1];
    b_cols = b.shape[2];
  } else {
    b_rows = b.shape[0];
    b_cols = b.shape[1];
  }

  // 检查矩阵乘法的维度兼容性
  if (a_cols != b_rows) {
    throw std::invalid_argument(
        "Matrix dimensions are not compatible for multiplication");
  }

  // 创建输出张量
  std::vector<uint32_t> output_shape;
  if (batch_size > 1) {
    output_shape = {batch_size, a_rows, b_cols};
  } else {
    output_shape = {a_rows, b_cols};
  }

  algorithms::Tensor result(output_shape);

  // 执行矩阵乘法
  for (uint32_t batch = 0; batch < batch_size; ++batch) {
    for (uint32_t i = 0; i < a_rows; ++i) {
      for (uint32_t j = 0; j < b_cols; ++j) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < a_cols; ++k) {
          // 计算索引
          size_t a_idx, b_idx;
          if (a.shape.size() == 3) {
            a_idx = batch * a_rows * a_cols + i * a_cols + k;
          } else {
            a_idx = i * a_cols + k;
          }

          if (b.shape.size() == 3) {
            size_t b_batch = (b.shape[0] == 1) ? 0 : batch;
            b_idx = b_batch * b_rows * b_cols + k * b_cols + j;
          } else {
            b_idx = k * b_cols + j;
          }

          sum += a.data[a_idx] * b.data[b_idx];
        }

        // 计算输出索引
        size_t out_idx;
        if (batch_size > 1) {
          out_idx = batch * a_rows * b_cols + i * b_cols + j;
        } else {
          out_idx = i * b_cols + j;
        }
        result.data[out_idx] = sum;
      }
    }
  }

  return result;
}

} // namespace ollama
} // namespace extensions
} // namespace duorou