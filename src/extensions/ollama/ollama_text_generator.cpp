#include "ollama_text_generator.h"
#include <iostream>
#include <chrono>
#include <algorithm>

namespace duorou {
namespace extensions {
namespace ollama {

OllamaTextGenerator::OllamaTextGenerator(llama_model* model)
    : model_(model), context_(nullptr) {
    if (!model_) {
        std::cerr << "[ERROR] OllamaTextGenerator: Invalid model pointer" << std::endl;
        return;
    }
    
    if (!initializeContext()) {
        std::cerr << "[ERROR] OllamaTextGenerator: Failed to initialize context" << std::endl;
    }
}

OllamaTextGenerator::~OllamaTextGenerator() {
    cleanup();
}

bool OllamaTextGenerator::initializeContext() {
    if (!model_) {
        return false;
    }
    
    // 设置上下文参数
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;  // 上下文长度
    ctx_params.n_batch = 512; // 批处理大小
    ctx_params.no_perf = false; // 启用性能计数器
    
    // 创建上下文
    context_ = llama_init_from_model(model_, ctx_params);
    if (!context_) {
        std::cerr << "[ERROR] Failed to create llama context" << std::endl;
        return false;
    }
    
    std::cout << "[DEBUG] OllamaTextGenerator context initialized successfully" << std::endl;
    return true;
}

void OllamaTextGenerator::cleanup() {
    if (context_) {
        llama_free(context_);
        context_ = nullptr;
    }
}

duorou::core::GenerationResult OllamaTextGenerator::generate(
    const std::string& prompt,
    const duorou::core::GenerationParams& params) {
    
    duorou::core::GenerationResult result;
    
    if (!canGenerate()) {
        result.text = "";
        result.finished = true;
        result.stop_reason = "Model not ready";
        return result;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 获取词汇表
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    
    // 计算提示词的token数量
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    if (n_prompt <= 0) {
        result.text = "";
        result.finished = true;
        result.stop_reason = "Failed to tokenize prompt";
        return result;
    }
    
    // 分配并tokenize提示词
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        result.text = "";
        result.finished = true;
        result.stop_reason = "Failed to tokenize prompt";
        return result;
    }
    
    result.prompt_tokens = prompt_tokens.size();
    std::cout << "[DEBUG] Prompt tokens: " << result.prompt_tokens << std::endl;
    
    // 初始化采样器
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    
    // 添加采样策略
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.temperature != 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }
    
    // 如果没有其他采样策略，使用贪心采样
    if (params.top_k <= 0 && params.top_p >= 1.0f && params.temperature == 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    }
    
    // 准备批处理
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    
    std::string generated_text;
    int n_decode = 0;
    llama_token new_token_id;
    
    // 生成循环
    for (int n_pos = 0; n_pos + batch.n_tokens < (int)prompt_tokens.size() + params.max_tokens; ) {
        // 评估当前批次
        if (llama_decode(context_, batch)) {
            result.stop_reason = "Decode error";
            break;
        }
        
        n_pos += batch.n_tokens;
        
        // 采样下一个token
        new_token_id = llama_sampler_sample(sampler, context_, -1);
        
        // 检查是否为结束token
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            result.stop_reason = "EOS token";
            break;
        }
        
        // 将token转换为文本
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            result.stop_reason = "Token conversion error";
            break;
        }
        
        std::string token_text(buf, n);
        generated_text += token_text;
        
        // 检查停止序列
        bool should_stop = false;
        for (const auto& stop_seq : params.stop_sequences) {
            if (generated_text.find(stop_seq) != std::string::npos) {
                result.stop_reason = "Stop sequence: " + stop_seq;
                should_stop = true;
                break;
            }
        }
        
        if (should_stop) {
            break;
        }
        
        // 准备下一个批次
        batch = llama_batch_get_one(&new_token_id, 1);
        n_decode += 1;
        
        if (n_decode >= params.max_tokens) {
            result.stop_reason = "Max tokens reached";
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    result.text = generated_text;
    result.generated_tokens = n_decode;
    result.generation_time = duration.count() / 1000.0;
    result.finished = true;
    
    if (result.stop_reason.empty()) {
        result.stop_reason = "Generation completed";
    }
    
    std::cout << "[DEBUG] Generated " << result.generated_tokens << " tokens in " 
              << result.generation_time << " seconds" << std::endl;
    
    // 清理采样器
    llama_sampler_free(sampler);
    
    return result;
}

duorou::core::GenerationResult OllamaTextGenerator::generateStream(
    const std::string& prompt,
    duorou::core::StreamCallback callback,
    const duorou::core::GenerationParams& params) {
    
    // 对于流式生成，我们可以在生成过程中调用回调函数
    // 这里先实现一个简化版本，基于generate方法
    duorou::core::GenerationResult result = generate(prompt, params);
    
    // 调用回调函数返回最终结果
    if (callback) {
        callback(-1, result.text, true);
    }
    
    return result;
}

size_t OllamaTextGenerator::countTokens(const std::string& text) const {
    if (!model_) {
        return 0;
    }
    
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    const int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, false, true);
    return n_tokens > 0 ? n_tokens : 0;
}

bool OllamaTextGenerator::canGenerate() const {
    return model_ != nullptr && context_ != nullptr;
}

void OllamaTextGenerator::reset() {
    // 在新的API中，我们可能需要重新初始化上下文或清理状态
    // 这里暂时不做任何操作，因为每次生成都是独立的
}

int OllamaTextGenerator::getContextSize() const {
    if (!context_) {
        return 0;
    }
    return llama_n_ctx(context_);
}

int OllamaTextGenerator::getVocabSize() const {
    if (!model_) {
        return 0;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    return llama_vocab_n_tokens(vocab);
}

std::vector<llama_token> OllamaTextGenerator::tokenize(const std::string& text, bool add_bos) const {
    if (!model_) {
        return {};
    }
    
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    const int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, add_bos, true);
    
    if (n_tokens <= 0) {
        return {};
    }
    
    std::vector<llama_token> tokens(n_tokens);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true) < 0) {
        return {};
    }
    
    return tokens;
}

std::string OllamaTextGenerator::detokenize(const std::vector<llama_token>& tokens) const {
    if (!model_ || tokens.empty()) {
        return "";
    }
    
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::string result;
    
    for (llama_token token : tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            result.append(buf, n);
        }
    }
    
    return result;
}

llama_token OllamaTextGenerator::sampleToken(llama_token_data_array* candidates, 
                                           const duorou::core::GenerationParams& params) {
    // 这个方法在新的API中不再需要，因为采样由llama_sampler处理
    // 保留这个方法以保持接口兼容性，但实际上不会被调用
    return 0;
}

} // namespace ollama
} // namespace extensions
} // namespace duorou