// Minimal stub for InferenceEngine to satisfy build and diagnostics
#ifndef DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H
#define DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H

// Only enable C++ constructs when compiled as C++.
#ifdef __cplusplus

#include <string>

namespace duorou {
namespace extensions {
namespace ollama {

class InferenceEngine {
public:
  virtual ~InferenceEngine() {}
  virtual bool initialize() { return true; }
  virtual bool isReady() const { return true; }
  virtual std::string generateText(const std::string &prompt,
                                   unsigned int max_tokens,
                                   float temperature,
                                   float top_p) {
    (void)max_tokens; (void)temperature; (void)top_p;
    return std::string("[Stub] ") + prompt;
  }
};

class MLInferenceEngine : public InferenceEngine {
public:
  explicit MLInferenceEngine(const std::string &model_id)
      : model_id_(model_id), ready_(false) {}

  bool initialize() override { ready_ = true; return true; }
  bool isReady() const override { return ready_; }
  std::string generateText(const std::string &prompt,
                           unsigned int max_tokens,
                           float temperature,
                           float top_p) override {
    (void)max_tokens; (void)temperature; (void)top_p;
    return std::string("[Model ") + model_id_ + "] " + prompt;
  }

private:
  std::string model_id_;
  bool ready_;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#else
// Non-C++ compilation units: intentionally left minimal to avoid diagnostics.
#endif // __cplusplus

#endif // DUOROU_EXTENSIONS_OLLAMA_INFERENCE_ENGINE_H