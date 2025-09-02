#include "text_generator.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <sstream>
#include <iostream>

namespace duorou {
namespace core {

TextGenerator::TextGenerator(llama_model* model, llama_context* context)
    : model_(model), context_(context), vocab_(nullptr) {
    if (!model_ || !context_) {
        throw std::invalid_argument("Model and context cannot be null");
    }
    
    // 获取vocab对象
    if (model_) {
        vocab_ = llama_model_get_vocab(model_);
    }
    
    // 获取模型信息
    context_size_ = llama_n_ctx(context_);
    vocab_size_ = vocab_ ? llama_vocab_n_tokens(vocab_) : 0;
    bos_token_ = vocab_ ? llama_vocab_bos(vocab_) : -1;
    eos_token_ = vocab_ ? llama_vocab_eos(vocab_) : -1;
    
    // 初始化随机数生成器
    initializeRNG(-1);
}

TextGenerator::~TextGenerator() {
    // 清理资源
    generated_tokens_.clear();
}

GenerationResult TextGenerator::generate(const std::string& prompt, const GenerationParams& params) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    GenerationResult result;
    
    try {
        // 将提示词转换为tokens
        auto prompt_tokens = textToTokens(prompt, true);
        result.prompt_tokens = prompt_tokens.size();
        
        if (prompt_tokens.empty()) {
            result.stop_reason = "Empty prompt";
            result.finished = true;
            return result;
        }
        
        // 检查提示词长度
        if (static_cast<int>(prompt_tokens.size()) >= context_size_) {
            result.stop_reason = "Prompt too long";
            result.finished = true;
            return result;
        }
        
        // 清空KV缓存
        llama_memory_t memory = llama_get_memory(context_);
        if (memory) {
            llama_memory_clear(memory, true);
        }
        
        // 处理提示词
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            if (llama_decode(context_, llama_batch_get_one(&prompt_tokens[i], 1)) != 0) {
                result.stop_reason = "Failed to process prompt";
                result.finished = true;
                return result;
            }
        }
        
        // 生成tokens
        std::vector<llama_token> generated;
        std::string generated_text;
        
        for (int i = 0; i < params.max_tokens; ++i) {
            // 获取logits
            float* logits = llama_get_logits_ith(context_, -1);
            if (!logits) {
                result.stop_reason = "Failed to get logits";
                break;
            }
            
            // 采样下一个token
            llama_token next_token = sampleToken(logits, params);
            
            // 检查是否为结束token
            if (next_token == eos_token_) {
                result.stop_reason = "EOS token";
                break;
            }
            
            generated.push_back(next_token);
            
            // 转换token为文本
            std::string token_text;
            if (vocab_) {
                char buffer[256];
                int len = llama_token_to_piece(vocab_, next_token, buffer, sizeof(buffer), 0, false);
                if (len > 0) {
                    token_text.assign(buffer, len);
                }
            }
            generated_text += token_text;
            
            // 检查停止序列
            if (shouldStop(generated_text, params.stop_sequences)) {
                result.stop_reason = "Stop sequence";
                break;
            }
            
            // 继续解码
            if (llama_decode(context_, llama_batch_get_one(&next_token, 1)) != 0) {
                result.stop_reason = "Decode error";
                break;
            }
        }
        
        if (generated.size() >= static_cast<size_t>(params.max_tokens)) {
            result.stop_reason = "Max tokens reached";
        }
        
        result.text = generated_text;
        result.tokens = generated;
        result.generated_tokens = generated.size();
        result.finished = true;
        
    } catch (const std::exception& e) {
        result.stop_reason = std::string("Exception: ") + e.what();
        result.finished = true;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.generation_time = duration.count() / 1000.0;
    
    return result;
}

GenerationResult TextGenerator::generateStream(const std::string& prompt, 
                                             StreamCallback callback,
                                             const GenerationParams& params) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    GenerationResult result;
    
    try {
        // 将提示词转换为tokens
        auto prompt_tokens = textToTokens(prompt, true);
        result.prompt_tokens = prompt_tokens.size();
        
        if (prompt_tokens.empty()) {
            result.stop_reason = "Empty prompt";
            result.finished = true;
            callback(0, "", true);
            return result;
        }
        
        // 检查提示词长度
        if (static_cast<int>(prompt_tokens.size()) >= context_size_) {
            result.stop_reason = "Prompt too long";
            result.finished = true;
            callback(0, "", true);
            return result;
        }
        
        // 清空KV缓存
        llama_memory_t memory = llama_get_memory(context_);
        if (memory) {
            llama_memory_clear(memory, true);
        }
        
        // 处理提示词
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            if (llama_decode(context_, llama_batch_get_one(&prompt_tokens[i], 1)) != 0) {
                result.stop_reason = "Failed to process prompt";
                result.finished = true;
                callback(0, "", true);
                return result;
            }
        }
        
        // 生成tokens
        std::vector<llama_token> generated;
        std::string generated_text;
        
        for (int i = 0; i < params.max_tokens; ++i) {
            // 获取logits
            float* logits = llama_get_logits_ith(context_, -1);
            if (!logits) {
                result.stop_reason = "Failed to get logits";
                break;
            }
            
            // 采样下一个token
            llama_token next_token = sampleToken(logits, params);
            
            // 检查是否为结束token
            if (next_token == eos_token_) {
                result.stop_reason = "EOS token";
                break;
            }
            
            generated.push_back(next_token);
            
            // 转换token为文本
            std::string token_text;
            if (vocab_) {
                char buffer[256];
                int32_t len = llama_token_to_piece(vocab_, next_token, buffer, sizeof(buffer), 0, true);
                if (len > 0) {
                    token_text = std::string(buffer, len);
                }
            }
            generated_text += token_text;
            
            // 流式回调
            callback(next_token, token_text, false);
            
            // 检查停止序列
            if (shouldStop(generated_text, params.stop_sequences)) {
                result.stop_reason = "Stop sequence";
                break;
            }
            
            // 继续解码
            if (llama_decode(context_, llama_batch_get_one(&next_token, 1)) != 0) {
                result.stop_reason = "Decode error";
                break;
            }
        }
        
        if (generated.size() >= static_cast<size_t>(params.max_tokens)) {
            result.stop_reason = "Max tokens reached";
        }
        
        result.text = generated_text;
        result.tokens = generated;
        result.generated_tokens = generated.size();
        result.finished = true;
        
        // 最终回调
        callback(0, "", true);
        
    } catch (const std::exception& e) {
        result.stop_reason = std::string("Exception: ") + e.what();
        result.finished = true;
        callback(0, "", true);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.generation_time = duration.count() / 1000.0;
    
    return result;
}

size_t TextGenerator::countTokens(const std::string& text) const {
    auto tokens = textToTokens(text, false);
    return tokens.size();
}

std::string TextGenerator::tokensToText(const std::vector<llama_token>& tokens) const {
    if (!vocab_) {
        return "";
    }
    
    std::string result;
    char buffer[256];
    for (llama_token token : tokens) {
        int len = llama_token_to_piece(vocab_, token, buffer, sizeof(buffer), 0, false);
        if (len > 0) {
            result.append(buffer, len);
        }
    }
    return result;
}

std::vector<llama_token> TextGenerator::textToTokens(const std::string& text, bool add_bos) const {
    if (!vocab_) {
        return {};
    }
    
    std::vector<llama_token> tokens;
    tokens.resize(text.length() + (add_bos ? 1 : 0));
    
    int n_tokens = llama_tokenize(vocab_, text.c_str(), text.length(), 
                                 tokens.data(), tokens.size(), add_bos, false);
    
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab_, text.c_str(), text.length(), 
                                 tokens.data(), tokens.size(), add_bos, false);
    }
    
    tokens.resize(n_tokens);
    return tokens;
}

bool TextGenerator::canGenerate() const {
    return model_ != nullptr && context_ != nullptr;
}

void TextGenerator::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    generated_tokens_.clear();
    if (context_) {
        llama_memory_t memory = llama_get_memory(context_);
        if (memory) {
            llama_memory_clear(memory, true);
        }
    }
}

int TextGenerator::getContextSize() const {
    return context_size_;
}

int TextGenerator::getVocabSize() const {
    return vocab_size_;
}

llama_token TextGenerator::sampleToken(float* logits, const GenerationParams& params) {
    // 应用重复惩罚
    if (params.repeat_penalty != 1.0f && !generated_tokens_.empty()) {
        int start_idx = std::max(0, static_cast<int>(generated_tokens_.size()) - params.repeat_last_n);
        std::vector<llama_token> last_tokens(generated_tokens_.begin() + start_idx, generated_tokens_.end());
        applyRepeatPenalty(logits, last_tokens, params.repeat_penalty);
    }
    
    // 应用温度
    if (params.temperature != 1.0f) {
        applyTemperature(logits, params.temperature);
    }
    
    // 应用top-k
    if (params.top_k > 0) {
        applyTopK(logits, params.top_k);
    }
    
    // 应用top-p
    if (params.top_p < 1.0f) {
        applyTopP(logits, params.top_p);
    }
    
    // 计算概率分布
    std::vector<float> probs(vocab_size_);
    float max_logit = *std::max_element(logits, logits + vocab_size_);
    float sum = 0.0f;
    
    for (int i = 0; i < vocab_size_; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    
    // 归一化
    for (int i = 0; i < vocab_size_; ++i) {
        probs[i] /= sum;
    }
    
    // 采样
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float cumsum = 0.0f;
    
    for (int i = 0; i < vocab_size_; ++i) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return i;
        }
    }
    
    return vocab_size_ - 1;
}

void TextGenerator::applyRepeatPenalty(float* logits, 
                                     const std::vector<llama_token>& last_tokens, 
                                     float penalty) {
    for (llama_token token : last_tokens) {
        if (token >= 0 && token < vocab_size_) {
            if (logits[token] > 0) {
                logits[token] /= penalty;
            } else {
                logits[token] *= penalty;
            }
        }
    }
}

void TextGenerator::applyTopK(float* logits, int k) {
    if (k >= vocab_size_) return;
    
    std::vector<std::pair<float, int>> logit_pairs;
    for (int i = 0; i < vocab_size_; ++i) {
        logit_pairs.emplace_back(logits[i], i);
    }
    
    std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + k, logit_pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    for (int i = k; i < vocab_size_; ++i) {
        logits[logit_pairs[i].second] = -INFINITY;
    }
}

void TextGenerator::applyTopP(float* logits, float p) {
    std::vector<std::pair<float, int>> logit_pairs;
    for (int i = 0; i < vocab_size_; ++i) {
        logit_pairs.emplace_back(logits[i], i);
    }
    
    std::sort(logit_pairs.begin(), logit_pairs.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 计算softmax概率
    float max_logit = logit_pairs[0].first;
    float sum = 0.0f;
    for (auto& pair : logit_pairs) {
        pair.first = std::exp(pair.first - max_logit);
        sum += pair.first;
    }
    
    // 归一化并计算累积概率
    float cumsum = 0.0f;
    for (size_t i = 0; i < logit_pairs.size(); ++i) {
        logit_pairs[i].first /= sum;
        cumsum += logit_pairs[i].first;
        
        if (cumsum > p) {
            // 将剩余的logits设为负无穷
            for (size_t j = i + 1; j < logit_pairs.size(); ++j) {
                logits[logit_pairs[j].second] = -INFINITY;
            }
            break;
        }
    }
}

void TextGenerator::applyTemperature(float* logits, float temperature) {
    if (temperature <= 0.0f) {
        // 贪婪采样
        int max_idx = 0;
        for (int i = 1; i < vocab_size_; ++i) {
            if (logits[i] > logits[max_idx]) {
                max_idx = i;
            }
        }
        
        for (int i = 0; i < vocab_size_; ++i) {
            logits[i] = (i == max_idx) ? 0.0f : -INFINITY;
        }
    } else {
        for (int i = 0; i < vocab_size_; ++i) {
            logits[i] /= temperature;
        }
    }
}

bool TextGenerator::shouldStop(const std::string& generated_text, 
                              const std::vector<std::string>& stop_sequences) const {
    for (const auto& stop_seq : stop_sequences) {
        if (generated_text.find(stop_seq) != std::string::npos) {
            return true;
        }
    }
    return false;
}

void TextGenerator::initializeRNG(int64_t seed) {
    if (seed == -1) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(static_cast<unsigned int>(seed));
    }
}

// TextGeneratorFactory implementation
std::unique_ptr<TextGenerator> TextGeneratorFactory::create(llama_model* model, llama_context* context) {
    if (!model || !context) {
        return nullptr;
    }
    
    try {
        return std::make_unique<TextGenerator>(model, context);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create TextGenerator: " << e.what() << std::endl;
        return nullptr;
    }
}

} // namespace core
} // namespace duorou