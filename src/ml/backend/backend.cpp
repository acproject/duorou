#include "backend.h"
#include "cpu_backend.h"
#include <iostream>
#include <algorithm>
#include "ggml_backend.h"

namespace duorou {
namespace ml {

// BackendFactory implementation
BackendFactory& BackendFactory::getInstance() {
    static BackendFactory instance;
    return instance;
}

void BackendFactory::registerBackend(DeviceType type, CreateBackendFunc createFunc) {
    backends_[type] = createFunc;
}

std::unique_ptr<Backend> BackendFactory::createBackend(DeviceType type) {
    auto it = backends_.find(type);
    if (it != backends_.end()) {
        return it->second();
    }
    return nullptr;
}

std::vector<DeviceType> BackendFactory::getAvailableBackendTypes() const {
    std::vector<DeviceType> types;
    for (const auto& pair : backends_) {
        types.push_back(pair.first);
    }
    return types;
}

std::unique_ptr<Backend> BackendFactory::createBestBackend() {
    // Priority order: CUDA > METAL > GGML > CPU
    std::vector<DeviceType> priority = {
        DeviceType::CUDA,
        DeviceType::METAL,
        DeviceType::GGML,
        DeviceType::CPU
    };
    
    for (DeviceType type : priority) {
        auto backend = createBackend(type);
        if (backend && backend->isAvailable()) {
            return backend;
        }
    }
    
    return nullptr;
}

// BackendManager implementation
BackendManager& BackendManager::getInstance() {
    static BackendManager instance;
    return instance;
}

bool BackendManager::initializeBackends() {
    auto& factory = BackendFactory::getInstance();
    
    // Register CPU backend
    factory.registerBackend(DeviceType::CPU, []() {
        return std::make_unique<CPUBackend>();
    });

    // Register GGML backend
    factory.registerBackend(DeviceType::GGML, []() {
        return std::make_unique<GGMLBackend>();
    });
    
    // Try to create all available backends
    auto types = factory.getAvailableBackendTypes();
    for (DeviceType type : types) {
        auto backend = factory.createBackend(type);
        if (backend && backend->initialize()) {
            backends_.push_back(std::move(backend));
        }
    }
    
    // Set default backend
    if (!backends_.empty()) {
        currentBackend_ = backends_[0].get();
        return true;
    }
    
    return false;
}

Backend* BackendManager::getCurrentBackend() const {
    return currentBackend_;
}

bool BackendManager::setCurrentBackend(DeviceType type) {
    for (auto& backend : backends_) {
        if (backend->getType() == type) {
            currentBackend_ = backend.get();
            return true;
        }
    }
    return false;
}

const std::vector<std::unique_ptr<Backend>>& BackendManager::getBackends() const {
    return backends_;
}

void BackendManager::cleanup() {
    for (auto& backend : backends_) {
        backend->cleanup();
    }
    backends_.clear();
    currentBackend_ = nullptr;
}

// Utility functions implementation
std::string deviceTypeToString(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::CUDA: return "CUDA";
        case DeviceType::METAL: return "METAL";
        case DeviceType::OPENCL: return "OPENCL";
        case DeviceType::VULKAN: return "VULKAN";
        case DeviceType::GGML: return "GGML";
        default: return "UNKNOWN";
    }
}

DeviceType stringToDeviceType(const std::string& typeStr) {
    if (typeStr == "CPU") return DeviceType::CPU;
    if (typeStr == "CUDA") return DeviceType::CUDA;
    if (typeStr == "METAL") return DeviceType::METAL;
    if (typeStr == "OPENCL") return DeviceType::OPENCL;
    if (typeStr == "VULKAN") return DeviceType::VULKAN;
    if (typeStr == "GGML") return DeviceType::GGML;
    return DeviceType::CPU; // Default to CPU
}

} // namespace ml
} // namespace duorou