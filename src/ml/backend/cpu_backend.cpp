#include "cpu_backend.h"
#include <cstdlib>
#include <cstring>
#include <thread>
#include <iostream>

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace duorou {
namespace ml {

CPUBackend::CPUBackend() 
    : initialized_(false), numThreads_(std::thread::hardware_concurrency()) {
}

CPUBackend::~CPUBackend() {
    cleanup();
}

bool CPUBackend::initialize() {
    if (initialized_) {
        return true;
    }
    
    // 检查CPU是否可用（总是可用）
    initialized_ = true;
    
    std::cout << "CPU Backend initialized with " << numThreads_ << " threads" << std::endl;
    return true;
}

void CPUBackend::cleanup() {
    initialized_ = false;
}

std::vector<DeviceInfo> CPUBackend::getAvailableDevices() const {
    std::vector<DeviceInfo> devices;
    
    // CPU设备信息
    size_t memorySize = 0; // 可以通过系统调用获取实际内存大小
    devices.emplace_back(DeviceType::CPU, "CPU", memorySize, 0);
    
    return devices;
}

bool CPUBackend::setDevice(int deviceId) {
    // CPU只有一个设备
    return deviceId == 0;
}

int CPUBackend::getCurrentDevice() const {
    return 0; // CPU设备ID总是0
}

void* CPUBackend::allocate(size_t bytes) {
    std::lock_guard<std::mutex> lock(allocMutex_);
    return alignedAlloc(bytes);
}

void CPUBackend::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(allocMutex_);
    alignedFree(ptr);
}

void CPUBackend::copyToDevice(void* dst, const void* src, size_t bytes) {
    // CPU上的内存拷贝
    std::memcpy(dst, src, bytes);
}

void CPUBackend::copyFromDevice(void* dst, const void* src, size_t bytes) {
    // CPU上的内存拷贝
    std::memcpy(dst, src, bytes);
}

void CPUBackend::copyDeviceToDevice(void* dst, const void* src, size_t bytes) {
    // CPU上的内存拷贝
    std::memcpy(dst, src, bytes);
}

void CPUBackend::synchronize() {
    // CPU后端不需要同步操作
}

bool CPUBackend::isAvailable() const {
    return true; // CPU总是可用
}

void CPUBackend::setNumThreads(int numThreads) {
    if (numThreads > 0) {
        numThreads_ = numThreads;
    }
}

int CPUBackend::getNumThreads() const {
    return numThreads_;
}

void* CPUBackend::alignedAlloc(size_t bytes, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(bytes, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void CPUBackend::alignedFree(void* ptr) {
    if (ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

} // namespace ml
} // namespace duorou