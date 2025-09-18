#include "context.h"
#include "tensor.h"
#include "backend/backend.h"
#include <iostream>
#include <chrono>

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
        return std::aligned_alloc(32, bytes);
    }
}

void Context::deallocate(void* ptr) {
    if (backend_) {
        backend_->deallocate(ptr);
    } else {
        std::free(ptr);
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