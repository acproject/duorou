#include "text_generator.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <random>
#include <sstream>
#include <thread>

#ifdef DUOROU_ENABLE_MNN
#include <MNN/llm/llm.hpp>
#endif

namespace duorou {
namespace core {

#ifdef DUOROU_ENABLE_MNN
static std::string readFilePrefix(const std::string &path, size_t max_bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::string();
  std::string content;
  content.resize(max_bytes);
  in.read(content.data(), static_cast<std::streamsize>(content.size()));
  content.resize(static_cast<size_t>(in.gcount()));
  return content;
}

static bool detectOmniRuntimeConfig(const std::string &config_path) {
  const std::string content = readFilePrefix(config_path, 256 * 1024);
  if (content.empty()) return false;
  if (content.find("\"visual_model\"") != std::string::npos) return true;
  if (content.find("\"audio_model\"") != std::string::npos) return true;
  if (content.find("\"global_image\"") != std::string::npos) return true;
  if (content.find("\"vision_start\"") != std::string::npos) return true;
  if (content.find("\"image_pad\"") != std::string::npos) return true;
  return false;
}

static int hexToInt(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
  if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
  return -1;
}

static std::string urlDecodePercent(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c == '%' && i + 2 < s.size()) {
      int hi = hexToInt(s[i + 1]);
      int lo = hexToInt(s[i + 2]);
      if (hi >= 0 && lo >= 0) {
        out.push_back(static_cast<char>((hi << 4) | lo));
        i += 2;
        continue;
      }
    }
    out.push_back(c);
  }
  return out;
}

static std::string trim(std::string s) {
  auto notSpace = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
  s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
  return s;
}

static std::string stripAngleOrQuotes(std::string s) {
  s = trim(std::move(s));
  if (s.size() >= 2 && ((s.front() == '<' && s.back() == '>') ||
                        (s.front() == '"' && s.back() == '"') ||
                        (s.front() == '\'' && s.back() == '\''))) {
    s = s.substr(1, s.size() - 2);
  }
  return trim(std::move(s));
}

static std::string fileUriToPath(const std::string &uri) {
  std::string s = stripAngleOrQuotes(uri);
  const std::string prefix = "file://";
  if (s.rfind(prefix, 0) != 0) {
    return s;
  }
  s.erase(0, prefix.size());
  const std::string localhost = "localhost/";
  if (s.rfind(localhost, 0) == 0) {
    s.erase(0, localhost.size() - 1);
  }
  return urlDecodePercent(s);
}

static std::string preprocessMnnOmniPrompt(const std::string &prompt) {
  static const std::regex md_img(R"(!\[[^\]]*\]\(([^)]+)\))");
  std::string out;
  out.reserve(prompt.size());
  std::sregex_iterator it(prompt.begin(), prompt.end(), md_img);
  std::sregex_iterator end;
  size_t last = 0;
  for (; it != end; ++it) {
    const std::smatch &m = *it;
    out.append(prompt, last, static_cast<size_t>(m.position()) - last);
    std::string target = fileUriToPath(m.str(1));
    out.append("<img>");
    out.append(target);
    out.append("</img>");
    last = static_cast<size_t>(m.position() + m.length());
  }
  out.append(prompt, last, std::string::npos);
  return out;
}
#endif

// Default constructor implementation
TextGenerator::TextGenerator(const std::string &model_path)
    : context_size_(2048), vocab_size_(32000), use_ollama_(false)
#ifdef DUOROU_ENABLE_MNN
    , use_mnn_(false)
#endif
{
  // Initialize random number generator
  std::random_device rd;
  rng_.seed(rd());
}

#ifdef DUOROU_ENABLE_MNN
void TextGenerator::MnnLlmDeleter::operator()(MNN::Transformer::Llm* p) const {
  if (!p) return;
  MNN::Transformer::Llm::destroy(p);
}

TextGenerator::TextGenerator(MnnBackendTag, const std::string &config_path)
    : context_size_(2048), vocab_size_(32000), use_ollama_(false),
      use_mnn_(true), mnn_config_path_(config_path) {
  std::random_device rd;
  rng_.seed(rd());
  mnn_is_omni_ = detectOmniRuntimeConfig(mnn_config_path_);
  mnn_llm_.reset(MNN::Transformer::Llm::createLLM(mnn_config_path_));
  if (mnn_llm_) {
    if (!mnn_llm_->load()) {
      mnn_llm_.reset();
      mnn_llm_.reset(MNN::Transformer::Llm::createLLM(mnn_config_path_));
      if (mnn_llm_) {
        mnn_llm_->set_config(R"({"has_talker":false})");
        if (!mnn_llm_->load()) {
          mnn_llm_.reset();
        }
      }
    }
  }
}
#endif

// Ollama model manager constructor
TextGenerator::TextGenerator(
    std::shared_ptr<duorou::extensions::ollama::OllamaModelManager>
        model_manager,
    const std::string &model_id)
    : context_size_(2048), vocab_size_(32000), model_manager_(model_manager),
      model_id_(normalizeModelId(model_id)), use_ollama_(true)
#ifdef DUOROU_ENABLE_MNN
    , use_mnn_(false)
#endif
{
  // Initialize random number generator
  std::random_device rd;
  rng_.seed(rd());
}

// Destructor
TextGenerator::~TextGenerator() {
  // Empty destructor implementation
}

// Generate text
GenerationResult TextGenerator::generate(const std::string &prompt,
                                         const GenerationParams &params) {

  std::lock_guard<std::mutex> lock(mutex_);
  GenerationResult result;

#ifdef DUOROU_ENABLE_MNN
  if (use_mnn_) {
    if (!mnn_llm_) {
      result.text = "Error: MNN LLM not initialized";
      result.finished = true;
      result.stop_reason = "error";
      result.prompt_tokens = countTokens(prompt);
      result.generated_tokens = 0;
      result.generation_time = 0.0;
      return result;
    }

    try {
      const std::string effective_prompt =
          mnn_is_omni_ ? preprocessMnnOmniPrompt(prompt) : prompt;
      std::ostringstream oss;
      auto start_time = std::chrono::high_resolution_clock::now();
      mnn_llm_->response(effective_prompt, &oss, nullptr, params.max_tokens);
      auto end_time = std::chrono::high_resolution_clock::now();

      result.text = oss.str();
      result.finished = true;
      result.stop_reason = "completed";
      result.prompt_tokens = countTokens(prompt);
      result.generated_tokens = countTokens(result.text);
      result.generation_time =
          std::chrono::duration<double>(end_time - start_time).count();
      return result;
    } catch (const std::exception &e) {
      result.text = std::string("Error: MNN inference exception: ") + e.what();
      result.finished = true;
      result.stop_reason = "exception";
      result.prompt_tokens = countTokens(prompt);
      result.generated_tokens = 0;
      result.generation_time = 0.0;
      return result;
    }
  }
#endif

  if (use_ollama_ && model_manager_) {

    try {
      // Create inference request
      duorou::extensions::ollama::InferenceRequest request;
      request.model_id = model_id_;
      request.prompt = prompt;
      request.max_tokens = params.max_tokens;
      request.temperature = params.temperature;
      request.top_p = params.top_p;

      // Execute inference
      auto start_time = std::chrono::high_resolution_clock::now();
      auto response = model_manager_->generateText(request);
      auto end_time = std::chrono::high_resolution_clock::now();

      if (response.success) {
        result.text = response.generated_text;
        result.finished = true;
        result.stop_reason = "completed";
        result.prompt_tokens = countTokens(prompt);
        result.generated_tokens = response.tokens_generated;
        result.generation_time =
            std::chrono::duration<double>(end_time - start_time).count();

        // Inference successful; returning result
      } else {
        // Inference failed; populate error result
        result.text = "Sorry, an error occurred during inference: " +
                      response.error_message;
        result.finished = true;
        result.stop_reason = "error";
        result.prompt_tokens = countTokens(prompt);
        result.generated_tokens = 0;
        result.generation_time = 0.0;
      }
    } catch (const std::exception &e) {
      // Exception during inference; populate error result
      result.text = "Sorry, an exception occurred during inference: " +
                    std::string(e.what());
      result.finished = true;
      result.stop_reason = "exception";
      result.prompt_tokens = countTokens(prompt);
      result.generated_tokens = 0;
      result.generation_time = 0.0;
    }
  } else {
    // Using fallback mock implementation

    // Simple mock response
    if (prompt.find("你好") != std::string::npos ||
        prompt.find("hello") != std::string::npos) {
      result.text = "Hello! I am Duorou AI assistant, happy to serve you. How "
                    "can I help you?";
    } else {
      result.text =
          "Thank you for your question. This is a simulated text generation "
          "response. "
          "The current version uses a simplified implementation, and will "
          "integrate full llama.cpp functionality in the future.";
    }
    result.finished = true;
    result.stop_reason = "completed";
    result.prompt_tokens = countTokens(prompt);
    result.generated_tokens = countTokens(result.text);
    result.generation_time = 0.5; // Simulated generation time
  }

  return result;
}

// Stream text generation
GenerationResult TextGenerator::generateStream(const std::string &prompt,
                                               StreamCallback callback,
                                               const GenerationParams &params) {

  std::lock_guard<std::mutex> lock(mutex_);
  GenerationResult result;

  if (!callback) {
    result.text = "Error: No callback provided for streaming";
    result.finished = true;
    result.stop_reason = "error";
    result.prompt_tokens = countTokens(prompt);
    result.generated_tokens = 0;
    result.generation_time = 0.0;
    return result;
  }

#ifdef DUOROU_ENABLE_MNN
  if (use_mnn_) {
    if (!mnn_llm_) {
      std::string error_msg = "Error: MNN LLM not initialized";
      callback(0, error_msg, true);
      result.text = error_msg;
      result.finished = true;
      result.stop_reason = "error";
      result.prompt_tokens = countTokens(prompt);
      result.generated_tokens = 0;
      result.generation_time = 0.0;
      return result;
    }

    struct CallbackStreambuf : public std::streambuf {
      StreamCallback cb;
      std::string buffer;
      size_t token_index = 0;

      explicit CallbackStreambuf(StreamCallback callback) : cb(std::move(callback)) {}

      std::streamsize xsputn(const char* s, std::streamsize count) override {
        if (count > 0) {
          buffer.append(s, static_cast<size_t>(count));
          cb(static_cast<int>(token_index++), std::string(s, static_cast<size_t>(count)), false);
        }
        return count;
      }

      int overflow(int ch) override {
        if (ch == EOF) return EOF;
        char c = static_cast<char>(ch);
        buffer.push_back(c);
        cb(static_cast<int>(token_index++), std::string(1, c), false);
        return ch;
      }
    };

    CallbackStreambuf cb_buf(callback);
    std::ostream os(&cb_buf);

    try {
      const std::string effective_prompt =
          mnn_is_omni_ ? preprocessMnnOmniPrompt(prompt) : prompt;
      auto start_time = std::chrono::high_resolution_clock::now();
      mnn_llm_->response(effective_prompt, &os, nullptr, params.max_tokens);
      auto end_time = std::chrono::high_resolution_clock::now();

      callback(static_cast<int>(cb_buf.token_index), "", true);

      result.text = cb_buf.buffer;
      result.finished = true;
      result.stop_reason = "completed";
      result.prompt_tokens = countTokens(prompt);
      result.generated_tokens = countTokens(result.text);
      result.generation_time =
          std::chrono::duration<double>(end_time - start_time).count();
      return result;
    } catch (const std::exception &e) {
      std::string error_msg = std::string("Error: MNN streaming inference exception: ") + e.what();
      callback(static_cast<int>(cb_buf.token_index), error_msg, true);
      result.text = error_msg;
      result.finished = true;
      result.stop_reason = "exception";
      result.prompt_tokens = countTokens(prompt);
      result.generated_tokens = 0;
      result.generation_time = 0.0;
      return result;
    }
  }
#endif

  if (use_ollama_ && model_manager_) {

    try {
      // Create streaming inference request
      duorou::extensions::ollama::InferenceRequest request;
      request.model_id = model_id_;
      request.prompt = prompt;
      request.max_tokens = params.max_tokens;
      request.temperature = params.temperature;
      request.top_p = params.top_p;

      auto start_time = std::chrono::high_resolution_clock::now();
      auto response = model_manager_->generateText(request);
      auto end_time = std::chrono::high_resolution_clock::now();

      if (response.success) {
        // Simulate streaming output, send complete response in chunks
        std::string full_text = response.generated_text;
        const size_t chunk_size = 10; // Send 10 characters each time

        for (size_t i = 0; i < full_text.length(); i += chunk_size) {
          std::string chunk = full_text.substr(i, chunk_size);
          bool is_final = (i + chunk_size >= full_text.length());
          callback(i / chunk_size, chunk, is_final);

          // Simulate streaming delay
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        result.text = full_text;
        result.finished = true;
        result.stop_reason = "completed";
        result.prompt_tokens = countTokens(prompt);
        result.generated_tokens = response.tokens_generated;
        result.generation_time =
            std::chrono::duration<double>(end_time - start_time).count();

        // Streaming inference successful
      } else {
        std::string error_msg =
            "Sorry, an error occurred during streaming inference: " +
            response.error_message;
        callback(0, error_msg, true);

        result.text = error_msg;
        result.finished = true;
        result.stop_reason = "error";
        result.prompt_tokens = countTokens(prompt);
        result.generated_tokens = 0;
        result.generation_time = 0.0;
      }
    } catch (const std::exception &e) {
      std::string error_msg =
          "Sorry, an exception occurred during streaming inference: " +
          std::string(e.what());
      callback(0, error_msg, true);

      result.text = error_msg;
      result.finished = true;
      result.stop_reason = "exception";
      result.prompt_tokens = countTokens(prompt);
      result.generated_tokens = 0;
      result.generation_time = 0.0;
    }
  } else {
    // Using fallback mock streaming implementation

    // Simulate streaming generation
    std::string response_text;
    if (prompt.find("你好") != std::string::npos ||
        prompt.find("hello") != std::string::npos) {
      response_text = "Hello! I am Duorou AI assistant, happy to serve you. "
                      "How can I help you?";
    } else {
      response_text =
          "Thank you for your question. This is a simulated streaming text "
          "generation response. "
          "The current version uses a simplified implementation, and will "
          "integrate full llama.cpp functionality in the future.";
    }

    // Send response in chunks
    const size_t chunk_size = 8;
    for (size_t i = 0; i < response_text.length(); i += chunk_size) {
      std::string chunk = response_text.substr(i, chunk_size);
      bool is_final = (i + chunk_size >= response_text.length());
      callback(i / chunk_size, chunk, is_final);

      // Simulate generation delay
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    result.text = response_text;
    result.finished = true;
    result.stop_reason = "completed";
    result.prompt_tokens = countTokens(prompt);
    result.generated_tokens = countTokens(response_text);
    result.generation_time = 0.5;
  }

  return result;
}

// Calculate the number of tokens in text
size_t TextGenerator::countTokens(const std::string &text) const {
  // Simple estimation: average 4 characters per token
  return text.length() / 4 + 1;
}

// Check if generation is possible
bool TextGenerator::canGenerate() const {
#ifdef DUOROU_ENABLE_MNN
  if (use_mnn_) {
    return static_cast<bool>(mnn_llm_);
  }
#endif
  if (use_ollama_) {
    return static_cast<bool>(model_manager_);
  }
  return true;
}

// Reset generator state
void TextGenerator::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  // Reset internal state
}

// Get context size
int TextGenerator::getContextSize() const { return context_size_; }

// Get vocabulary size
int TextGenerator::getVocabSize() const { return vocab_size_; }

// Apply Top-K sampling
void TextGenerator::applyTopK(float *logits, int k) {
  if (k <= 0 || !logits)
    return;

  // Simple implementation: set all values except the top k largest to negative
  // infinity
  std::vector<std::pair<float, int>> logit_pairs;
  for (int i = 0; i < vocab_size_; ++i) {
    logit_pairs.emplace_back(logits[i], i);
  }

  std::partial_sort(
      logit_pairs.begin(), logit_pairs.begin() + k, logit_pairs.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  for (int i = k; i < vocab_size_; ++i) {
    logits[logit_pairs[i].second] = -INFINITY;
  }
}

// Apply Top-P sampling
void TextGenerator::applyTopP(float *logits, float p) {
  if (p <= 0.0f || p >= 1.0f || !logits)
    return;

  // Calculate softmax probabilities
  std::vector<std::pair<float, int>> prob_pairs;
  float max_logit = *std::max_element(logits, logits + vocab_size_);

  float sum = 0.0f;
  for (int i = 0; i < vocab_size_; ++i) {
    float prob = std::exp(logits[i] - max_logit);
    prob_pairs.emplace_back(prob, i);
    sum += prob;
  }

  // Normalize
  for (auto &pair : prob_pairs) {
    pair.first /= sum;
  }

  // Sort by probability in descending order
  std::sort(prob_pairs.begin(), prob_pairs.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  // Calculate cumulative probability and truncate
  float cumulative = 0.0f;
  for (size_t i = 0; i < prob_pairs.size(); ++i) {
    cumulative += prob_pairs[i].first;
    if (cumulative > p) {
      // Set remaining token probabilities to 0
      for (size_t j = i + 1; j < prob_pairs.size(); ++j) {
        logits[prob_pairs[j].second] = -INFINITY;
      }
      break;
    }
  }
}

// Apply temperature sampling
void TextGenerator::applyTemperature(float *logits, float temperature) {
  if (temperature <= 0.0f || !logits)
    return;

  for (int i = 0; i < vocab_size_; ++i) {
    logits[i] /= temperature;
  }
}

// Check if generation should stop
bool TextGenerator::shouldStop(
    const std::string &generated_text,
    const std::vector<std::string> &stop_sequences) const {
  for (const auto &stop_seq : stop_sequences) {
    if (generated_text.find(stop_seq) != std::string::npos) {
      return true;
    }
  }
  return false;
}

// Initialize random number generator
void TextGenerator::initializeRNG(int64_t seed) {
  if (seed == -1) {
    std::random_device rd;
    rng_.seed(rd());
  } else {
    rng_.seed(static_cast<unsigned int>(seed));
  }
}

std::string
TextGenerator::normalizeModelId(const std::string &model_name) const {
  // Trim whitespace to be consistent with OllamaModelManager
  auto trim = [](const std::string &s) {
    auto begin = std::find_if(
        s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); });
    auto end = std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                 return !std::isspace(ch);
               }).base();
    if (begin >= end)
      return std::string();
    return std::string(begin, end);
  };
  std::string model_id = trim(model_name);
  // Allow the same character set as OllamaModelManager: alnum, '_', '-', '.',
  // ':', '/'
  for (char &c : model_id) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-' &&
        c != '.' && c != ':' && c != '/') {
      c = '_';
    }
  }
  return model_id;
}

} // namespace core
} // namespace duorou
