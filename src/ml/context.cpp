#include "context.h"
#include "tensor.h"
#include "backend/backend.h"
#include <iostream>
#include <chrono>
#include <cstdlib>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace duorou {
namespace ml {

Context::Context(Backend* backend) 
    : backend_(backend), gradientEnabled_(false), profilingEnabled_(false) {
}

Context::~Context() {
    releaseTempTensors();
}

void Context::setBackend(Backend* backend) {
    backend_ = backend;
}

void* Context::allocate(size_t bytes) {
    if (backend_) {
        return backend_->allocate(bytes);
    } else {
        // 跨平台的内存对齐分配
#ifdef _WIN32
        return _aligned_malloc(bytes, 32);
#else
        // 对于支持 C++17 aligned_alloc 的平台
        #if __cplusplus >= 201703L && defined(__GLIBC__) && __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 16
            return std::aligned_alloc(32, bytes);
        #else
            // 使用 posix_memalign 作为后备方案
            void* ptr = nullptr;
            if (posix_memalign(&ptr, 32, bytes) == 0) {
                return ptr;
            }
            return nullptr;
        #endif
#endif
    }
}

void Context::deallocate(void* ptr) {
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

Tensor Context::createTempTensor(const std::vector<int64_t>& shape, DataType dtype) {
    auto tensor = std::make_unique<Tensor>(shape, dtype);
    tensor->setBackend(backend_);
    tensor->allocate();
    
    Tensor result = *tensor;
    tempTensors_.push_back(std::move(tensor));
    return result;
}

void Context::releaseTempTensors() {
    tempTensors_.clear();
}

void Context::enableGradient(bool enable) {
    gradientEnabled_ = enable;
}

void Context::synchronize() {
    if (backend_) {
        backend_->synchronize();
    }
}

void Context::enableProfiling(bool enable) {
    profilingEnabled_ = enable;
}

void Context::printProfilingInfo() const {
    if (!profilingEnabled_) {
        std::cout << "Profiling is not enabled" << std::endl;
        return;
    }
    
    std::cout << "=== Profiling Information ===" << std::endl;
    for (const auto& [operation, time] : timingStats_) {
        std::cout << operation << ": " << time << " ms" << std::endl;
    }
    std::cout << "=============================" << std::endl;
}

void Context::setParameter(const std::string& key, const std::string& value) {
    parameters_[key] = value;
}

std::string Context::getParameter(const std::string& key) const {
    auto it = parameters_.find(key);
    if (it != parameters_.end()) {
        return it->second;
    }
    return "";
}

} // namespace ml
} // namespace duorou