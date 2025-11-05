#include "context.h"
#include "backend/backend.h"
#include "backend/ggml_backend.h"
#include <ggml.h>
#include <ggml-cpu.h>
#include "tensor.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <string>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace duorou {
namespace ml {

Context::Context(Backend *backend)
    : backend_(backend), gradientEnabled_(false), profilingEnabled_(false) {}

Context::~Context() { releaseTempTensors(); }

void Context::setBackend(Backend *backend) { backend_ = backend; }

ggml_context *Context::ggml_ctx() const {
  auto *gb = dynamic_cast<GGMLBackend *>(backend_);
  return gb ? gb->ggml_ctx() : nullptr;
}

void Context::compute(ggml_cgraph *gf) const {
  unsigned n_threads = 4;
  // 允许通过环境变量覆盖线程数
  if (const char *env = std::getenv("DUOROU_NUM_THREADS")) {
    try {
      int v = std::stoi(std::string(env));
      if (v > 0) n_threads = static_cast<unsigned>(v);
    } catch (...) {
      // ignore
    }
  } else {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw > 0) n_threads = hw;
  }
  const bool debug_timing = std::getenv("DUOROU_DEBUG_TIMING") != nullptr;
  auto t0 = std::chrono::steady_clock::now();
  if (debug_timing) {
    std::cout << "[DEBUG] [Context::compute] start: threads=" << n_threads << std::endl;
  }

  // 使用 plan/work_data，避免将工作区占用到上下文内存池，降低 OOM 风险
  ggml_cplan plan = ggml_graph_plan(gf, (int)n_threads, /*threadpool*/ nullptr);
  std::unique_ptr<uint8_t, void(*)(void*)> work_buf(nullptr, [](void* p){ if (p) std::free(p); });
  if (plan.work_size > 0) {
    void *buf = std::malloc(plan.work_size);
    if (!buf) {
      throw std::runtime_error("Context::compute: failed to allocate work buffer");
    }
    plan.work_data = reinterpret_cast<uint8_t*>(buf);
    work_buf.reset(reinterpret_cast<uint8_t*>(buf));
  }

  ggml_status st = ggml_graph_compute(gf, &plan);
  if (st != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("Context::compute: ggml_graph_compute failed");
  }

  auto t1 = std::chrono::steady_clock::now();
  if (debug_timing) {
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[DEBUG] [Context::compute] done in " << ms << " ms" << std::endl;
  }
}

void *Context::allocate(size_t bytes) {
  if (backend_) {
    return backend_->allocate(bytes);
  } else {
    // Cross-platform aligned memory allocation
#ifdef _WIN32
    return _aligned_malloc(bytes, 32);
#else
// For platforms that support C++17 aligned_alloc
#if __cplusplus >= 201703L && defined(__GLIBC__) && __GLIBC__ >= 2 &&          \
    __GLIBC_MINOR__ >= 16
    return std::aligned_alloc(32, bytes);
#else
    // Use posix_memalign as fallback
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 32, bytes) == 0) {
      return ptr;
    }
    return nullptr;
#endif
#endif
  }
}

void Context::deallocate(void *ptr) {
  if (backend_) {
    backend_->deallocate(ptr);
  } else {
    if (ptr) {
#ifdef _WIN32
      _aligned_free(ptr);
#else
      std::free(ptr);
#endif
    }
  }
}

Tensor Context::createTempTensor(const std::vector<int64_t> &shape,
                                 DataType dtype) {
  auto tensor = std::make_unique<Tensor>(shape, dtype);
  tensor->setBackend(backend_);
  tensor->allocate();

  Tensor result = *tensor;
  tempTensors_.push_back(std::move(tensor));
  return result;
}

void Context::releaseTempTensors() { tempTensors_.clear(); }

void Context::enableGradient(bool enable) { gradientEnabled_ = enable; }

void Context::synchronize() {
  if (backend_) {
    backend_->synchronize();
  }
}

void Context::enableProfiling(bool enable) { profilingEnabled_ = enable; }

void Context::printProfilingInfo() const {
  if (!profilingEnabled_) {
    std::cout << "Profiling is not enabled" << std::endl;
    return;
  }

  std::cout << "=== Profiling Information ===" << std::endl;
  for (const auto &[operation, time] : timingStats_) {
    std::cout << operation << ": " << time << " ms" << std::endl;
  }
  std::cout << "=============================" << std::endl;
}

void Context::setParameter(const std::string &key, const std::string &value) {
  parameters_[key] = value;
}

std::string Context::getParameter(const std::string &key) const {
  auto it = parameters_.find(key);
  if (it != parameters_.end()) {
    return it->second;
  }
  return "";
}

} // namespace ml
} // namespace duorou