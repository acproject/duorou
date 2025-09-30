#include "ggml_backend.h"
// #include "../../../third_party/llama.cpp/ggml/include/ggml.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace duorou {
namespace ml {

GGMLBackend::GGMLBackend() : initialized_(false), currentDeviceId_(0) {}

GGMLBackend::~GGMLBackend() { cleanup(); }

bool GGMLBackend::initialize() {
  if (initialized_)
    return true;
  // For now, treat as always available (CPU-backed ggml by default)

  // create ggml context
  ggml_init_params params{
      .mem_size = kGgmlCtxSize,
      .mem_buffer = nullptr,
      .no_alloc = false,
  };
  ggml_ctx_.reset(ggml_init(params));
  if (!ggml_ctx_) {
    std::cerr << "[GGMLBackend] ggml_init failed\n";
    return false;
  }
  initialized_ = true;
  std::cout << "GGML Backend initialized" << std::endl;
  return true;
}

void GGMLBackend::cleanup() {
  initialized_ = false;
  ggml_ctx_.reset(); // auto run ggml_free
}

std::vector<DeviceInfo> GGMLBackend::getAvailableDevices() const {
  std::vector<DeviceInfo> devices;
  // ggml 通常在 CPU 上运行，也可以通过 Metal/CUDA 后端，本骨架先报 1 个逻辑设备
  devices.emplace_back(DeviceType::GGML, "GGML-Logical", 0, 0);
  return devices;
}

bool GGMLBackend::setDevice(int deviceId) {
  // 单一逻辑设备
  currentDeviceId_ = (deviceId == 0) ? 0 : 0;
  return deviceId == 0;
}

int GGMLBackend::getCurrentDevice() const { return currentDeviceId_; }

void *GGMLBackend::allocate(size_t bytes) {
  std::lock_guard<std::mutex> lock(allocMutex_);
#ifdef _WIN32
  return _aligned_malloc(bytes, 32);
#else
  void *ptr = nullptr;
  if (posix_memalign(&ptr, 32, bytes) != 0)
    return nullptr;
  return ptr;
#endif
}

void GGMLBackend::deallocate(void *ptr) {
  std::lock_guard<std::mutex> lock(allocMutex_);
#ifdef _WIN32
  if (ptr)
    _aligned_free(ptr);
#else
  if (ptr)
    free(ptr);
#endif
}

void GGMLBackend::copyToDevice(void *dst, const void *src, size_t bytes) {
  std::memcpy(dst, src, bytes);
}

void GGMLBackend::copyFromDevice(void *dst, const void *src, size_t bytes) {
  std::memcpy(dst, src, bytes);
}

void GGMLBackend::copyDeviceToDevice(void *dst, const void *src, size_t bytes) {
  std::memcpy(dst, src, bytes);
}

void GGMLBackend::synchronize() {
  // no-op for now
}

bool GGMLBackend::isAvailable() const { return true; }

} // namespace ml
} // namespace duorou