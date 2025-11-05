#ifndef DUOROU_ML_CONTEXT_H
#define DUOROU_ML_CONTEXT_H

#include <ggml.h>
#include "tensor.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace duorou {
namespace ml {

// Forward declarations
class Backend;
class Tensor;
enum class DataType;

// Computation context class
class Context {
public:
  Context(Backend *backend = nullptr);
  ~Context();

  // Backend management
  void setBackend(Backend *backend);
  Backend *getBackend() const { return backend_; }

  // ggml compute graph
  ggml_context *ggml_ctx() const;      // get ggml_context*
  void compute(ggml_cgraph *gf) const; // process graph

  // Memory management
  void *allocate(size_t bytes);
  void deallocate(void *ptr);

  // Temporary tensor management
  Tensor createTempTensor(const std::vector<int64_t> &shape, DataType dtype);
  void releaseTempTensors();

  // Computation graph management (optional, for automatic differentiation)
  void enableGradient(bool enable = true);
  bool isGradientEnabled() const { return gradientEnabled_; }

  // Synchronization operations
  void synchronize();

  // Performance profiling
  void enableProfiling(bool enable = true);
  bool isProfilingEnabled() const { return profilingEnabled_; }
  void printProfilingInfo() const;

  // Configuration parameters
  void setParameter(const std::string &key, const std::string &value);
  std::string getParameter(const std::string &key) const;

private:
  Backend *backend_{nullptr};
  bool gradientEnabled_;
  bool profilingEnabled_;
  std::vector<std::unique_ptr<Tensor>> tempTensors_;
  std::unordered_map<std::string, std::string> parameters_;

  // Performance statistics
  mutable std::unordered_map<std::string, double> timingStats_;
};

} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_CONTEXT_H