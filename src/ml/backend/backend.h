#ifndef DUOROU_ML_BACKEND_H
#define DUOROU_ML_BACKEND_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

namespace duorou {
namespace ml {

// Forward declarations
class Tensor;
class Context;

// Device type enumeration
enum class DeviceType {
    CPU,
    CUDA,
    METAL,
    OPENCL,
    VULKAN
};

// Backend device information
struct DeviceInfo {
    DeviceType type;
    std::string name;
    size_t memorySize;
    int deviceId;
    bool isAvailable;
    
    DeviceInfo(DeviceType t, const std::string& n, size_t mem, int id)
        : type(t), name(n), memorySize(mem), deviceId(id), isAvailable(true) {}
};

// Backend interface base class
class Backend {
public:
    virtual ~Backend() = default;
    
    // Initialization and cleanup
    virtual bool initialize() = 0;
    virtual void cleanup() = 0;
    
    // Device management
    virtual std::vector<DeviceInfo> getAvailableDevices() const = 0;
    virtual bool setDevice(int deviceId) = 0;
    virtual int getCurrentDevice() const = 0;
    
    // Memory management
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void copyToDevice(void* dst, const void* src, size_t bytes) = 0;
    virtual void copyFromDevice(void* dst, const void* src, size_t bytes) = 0;
    virtual void copyDeviceToDevice(void* dst, const void* src, size_t bytes) = 0;
    
    // Synchronization
    virtual void synchronize() = 0;
    
    // Backend information
    virtual DeviceType getType() const = 0;
    virtual std::string getName() const = 0;
    virtual bool isAvailable() const = 0;
};

// Backend factory for creating backends
class BackendFactory {
public:
    using CreateBackendFunc = std::function<std::unique_ptr<Backend>()>;
    
    static BackendFactory& getInstance();
    
    // Register a backend creation function
    void registerBackend(DeviceType type, CreateBackendFunc createFunc);
    
    // Create a backend of specified type
    std::unique_ptr<Backend> createBackend(DeviceType type);
    
    // Get available backend types
    std::vector<DeviceType> getAvailableBackendTypes() const;
    
    // Create the best available backend
    std::unique_ptr<Backend> createBestBackend();

private:
    BackendFactory() = default;
    std::unordered_map<DeviceType, CreateBackendFunc> backends_;
};

// Backend manager for managing multiple backends
class BackendManager {
public:
    static BackendManager& getInstance();
    
    // Initialize all available backends
    bool initializeBackends();
    
    // Get current active backend
    Backend* getCurrentBackend() const;
    
    // Set current backend by type
    bool setCurrentBackend(DeviceType type);
    
    // Get all initialized backends
    const std::vector<std::unique_ptr<Backend>>& getBackends() const;
    
    // Cleanup all backends
    void cleanup();

private:
    BackendManager() = default;
    std::vector<std::unique_ptr<Backend>> backends_;
    Backend* currentBackend_ = nullptr;
};

// Utility functions
std::string deviceTypeToString(DeviceType type);
DeviceType stringToDeviceType(const std::string& typeStr);

} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_BACKEND_H