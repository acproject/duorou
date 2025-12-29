#pragma once

#ifdef __cplusplus

#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>
// #include "../../third_party/llama.cpp/include/llama.h"  //
// Note: Temporarily disable llama-related functionality
#include "../extensions/ollama/ollama_model_manager.h"

#ifdef DUOROU_ENABLE_MNN
namespace MNN {
namespace Transformer {
class Llm;
}
}
#endif

namespace duorou {
namespace core {

/**
 * @brief Text generation parameters structure
 */
struct GenerationParams {
  int max_tokens = 100;                    ///< Maximum number of tokens to generate
  float temperature = 0.8f;                ///< Temperature parameter, controls randomness
  float top_p = 0.9f;                      ///< Top-p sampling parameter
  int top_k = 40;                          ///< Top-k sampling parameter
  float repeat_penalty = 1.1f;             ///< Repetition penalty
  int repeat_last_n = 64;                  ///< Number of tokens to consider for repetition penalty
  int64_t seed = -1;                       ///< Random seed, -1 means random
  std::vector<std::string> stop_sequences; ///< Stop sequences
  bool stream = false;                     ///< Whether to use streaming output

  GenerationParams() = default;
};

/**
 * @brief Generation result structure
 */
struct GenerationResult {
  std::string text; ///< Generated text
  // std::vector<llama_token> tokens;  ///< Generated token sequence - temporarily disabled
  bool finished;           ///< Whether generation is finished
  std::string stop_reason; ///< Stop reason
  size_t prompt_tokens;    ///< Number of prompt tokens
  size_t generated_tokens; ///< Number of generated tokens
  double generation_time;  ///< Generation time (seconds)

  GenerationResult()
      : finished(false), prompt_tokens(0), generated_tokens(0),
        generation_time(0.0) {}
};

/**
 * @brief Streaming generation callback function type
 * @param token Newly generated token
 * @param text Corresponding text fragment
 * @param finished Whether finished
 */
// typedef std::function<void(llama_token token, const std::string& text, bool
// finished)> StreamCallback;  // Temporarily disabled
typedef std::function<void(int token, const std::string &text, bool finished)>
    StreamCallback;

/**
 * @brief Text generator class
 *
 * Responsible for text generation using llama models, supports multiple sampling strategies and parameter configurations
 */
class TextGenerator {
public:
  struct MnnBackendTag {};

  /**
   * @brief Constructor
   * @param model llama model pointer
   * @param context llama context pointer
   */
  // TextGenerator(llama_model* model, llama_context* context);  // Temporarily disabled

  /**
   * @brief Default constructor (for Ollama models)
   * @param model_path Model path
   */
  TextGenerator(const std::string &model_path = "");

  TextGenerator(MnnBackendTag, const std::string &config_path);

  /**
   * @brief Constructor (using OllamaModelManager)
   * @param model_manager Ollama model manager pointer
   * @param model_id Model ID
   */
  TextGenerator(std::shared_ptr<duorou::extensions::ollama::OllamaModelManager>
                    model_manager,
                const std::string &model_id);

  /**
   * @brief Destructor
   */
  ~TextGenerator();

  /**
   * @brief Generate text
   * @param prompt Input prompt
   * @param params Generation parameters
   * @return Generation result
   */
  GenerationResult
  generate(const std::string &prompt,
           const GenerationParams &params = GenerationParams());

  /**
   * @brief Generate text with streaming
   * @param prompt Input prompt
   * @param callback Streaming callback function
   * @param params Generation parameters
   * @return Generation result
   */
  GenerationResult
  generateStream(const std::string &prompt, StreamCallback callback,
                 const GenerationParams &params = GenerationParams());

  /**
   * @brief Count tokens in text
   * @param text Input text
   * @return Number of tokens
   */
  size_t countTokens(const std::string &text) const;

  /**
   * @brief Convert text to token sequence
   * @param text Input text
   * @param add_bos Whether to add beginning token
   * @return Token sequence
   */
  // std::vector<llama_token> textToTokens(const std::string& text, bool add_bos
  // = true) const;  // Temporarily disabled

  /**
   * @brief Convert token sequence to text
   * @param tokens Token sequence
   * @return Text
   */
  // std::string tokensToText(const std::vector<llama_token>& tokens) const;  //
  // Temporarily disabled

  /**
   * @brief Check if generation is possible
   * @return Whether generation is possible
   */
  bool canGenerate() const;

  /**
   * @brief Reset generator state
   */
  void reset();

  /**
   * @brief Get context size
   * @return Context size
   */
  int getContextSize() const;

  /**
   * @brief Get vocabulary size
   * @return Vocabulary size
   */
  int getVocabSize() const;

private:
  /**
   * @brief Apply Top-K sampling
   * @param logits Logits array
   * @param k Top-K parameter
   */
  void applyTopK(float *logits, int k);

  /**
   * @brief Apply Top-P sampling
   * @param logits Logits array
   * @param p Top-P parameter
   */
  void applyTopP(float *logits, float p);

  /**
   * @brief Apply temperature sampling
   * @param logits Logits array
   * @param temperature Temperature parameter
   */
  void applyTemperature(float *logits, float temperature);

  /**
   * @brief Check if generation should stop
   * @param generated_text Generated text so far
   * @param stop_sequences Stop sequences
   * @return Whether generation should stop
   */
  bool shouldStop(const std::string &generated_text,
                  const std::vector<std::string> &stop_sequences) const;

  /**
   * @brief Initialize random number generator
   * @param seed Random seed
   */
  void initializeRNG(int64_t seed);

  /**
   * @brief Normalize model ID, consistent with OllamaModelManager
   * @param model_name Original model name
   * @return Normalized model ID
   */
  std::string normalizeModelId(const std::string &model_name) const;

private:
  std::mt19937 rng_; ///< Random number generator

  mutable std::mutex mutex_; ///< Thread-safe mutex

  // Model information
  int context_size_; ///< Context size
  int vocab_size_;   ///< Vocabulary size

  // Ollama model manager
  std::shared_ptr<duorou::extensions::ollama::OllamaModelManager>
      model_manager_;
  std::string model_id_; ///< Currently used model ID
  bool use_ollama_;      ///< Whether to use Ollama model

#ifdef DUOROU_ENABLE_MNN
  struct MnnLlmDeleter {
    void operator()(MNN::Transformer::Llm* p) const;
  };
  std::unique_ptr<MNN::Transformer::Llm, MnnLlmDeleter> mnn_llm_;
  bool use_mnn_;
  std::string mnn_config_path_;
#endif
};

/**
 * @brief Text generator factory class
 */
class TextGeneratorFactory {
public:
  /**
   * @brief Create text generator
   * @param model Llama model pointer
   * @param context Llama context pointer
   * @return Text generator pointer
   */
  // static std::unique_ptr<TextGenerator> create(llama_model* model,
  // llama_context* context);  // Temporarily disabled
};

} // namespace core
} // namespace duorou

#endif // __cplusplus
