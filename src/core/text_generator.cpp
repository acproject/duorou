#include "text_generator.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>

namespace duorou {
namespace core {

// Default constructor implementation
TextGenerator::TextGenerator(const std::string &model_path)
    : context_size_(2048), vocab_size_(32000), use_ollama_(false) {
  // Initialize random number generator
  std::random_device rd;
  rng_.seed(rd());
}

// Ollama model manager constructor
TextGenerator::TextGenerator(
    std::shared_ptr<duorou::extensions::ollama::OllamaModelManager>
        model_manager,
    const std::string &model_id)
    : context_size_(2048), vocab_size_(32000), model_manager_(model_manager),
      model_id_(normalizeModelId(model_id)), use_ollama_(true) {
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
  std::cout << "[DEBUG] TextGenerator::generate() called with prompt: "
            << prompt.substr(0, 50) << "..." << std::endl;

  std::lock_guard<std::mutex> lock(mutex_);
  GenerationResult result;

  if (use_ollama_ && model_manager_) {
    std::cout << "[DEBUG] Using Ollama model manager for inference"
              << std::endl;

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

        std::cout << "[DEBUG] Ollama inference successful: "
                  << result.text.substr(0, 50) << "..." << std::endl;
      } else {
        std::cout << "[DEBUG] Ollama inference failed: "
                  << response.error_message << std::endl;
        result.text = "Sorry, an error occurred during inference: " +
                      response.error_message;
        result.finished = true;
        result.stop_reason = "error";
        result.prompt_tokens = countTokens(prompt);
        result.generated_tokens = 0;
        result.generation_time = 0.0;
      }
    } catch (const std::exception &e) {
      std::cout << "[DEBUG] Exception during Ollama inference: " << e.what()
                << std::endl;
      result.text = "Sorry, an exception occurred during inference: " +
                    std::string(e.what());
      result.finished = true;
      result.stop_reason = "exception";
      result.prompt_tokens = countTokens(prompt);
      result.generated_tokens = 0;
      result.generation_time = 0.0;
    }
  } else {
    std::cout << "[DEBUG] Using fallback mock implementation" << std::endl;

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

  std::cout << "[DEBUG] TextGenerator returning result: "
            << result.text.substr(0, 30) << "..." << std::endl;
  return result;
}

// Stream text generation
GenerationResult TextGenerator::generateStream(const std::string &prompt,
                                               StreamCallback callback,
                                               const GenerationParams &params) {
  std::cout << "[DEBUG] TextGenerator::generateStream() called with prompt: "
            << prompt.substr(0, 50) << "..." << std::endl;

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

  if (use_ollama_ && model_manager_) {
    std::cout << "[DEBUG] Using Ollama model manager for streaming inference"
              << std::endl;

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

        std::cout << "[DEBUG] Ollama streaming inference successful"
                  << std::endl;
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
    std::cout << "[DEBUG] Using fallback mock streaming implementation"
              << std::endl;

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

  std::cout << "[DEBUG] TextGenerator returning streaming result" << std::endl;
  return result;
}

// Calculate the number of tokens in text
size_t TextGenerator::countTokens(const std::string &text) const {
  // Simple estimation: average 4 characters per token
  return text.length() / 4 + 1;
}

// Check if generation is possible
bool TextGenerator::canGenerate() const {
  std::cout << "[DEBUG] TextGenerator::canGenerate() called - returning true "
               "(functionality enabled)"
            << std::endl;
  // Enable text generation functionality
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