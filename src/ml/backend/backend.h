#ifndef DUOROU_ML_BACKEND_H
#define DUOROU_ML_BACKEND_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

namespace duorou {
namespace ml {

// 前向声明
class Tensor;
class Context;

// 设备类型枚举
enum class DeviceType {
    CPU,
    CUDA,
    METAL,
    OPENCL,
    VULKAN
};

// 后端设备信息
struct DeviceInfo {
    DeviceType type;
    std::string name;
    size_t memorySize;
    int deviceId;
    bool isAvailable;
    
    DeviceInfo(DeviceType t, const std::string& n, size_t mem, int id)
        : type(t), name(n), memorySize(mem), deviceId(id), isAvailable(true) {}
};

// 后端接口基类
class Backend {
public:
    virtual ~Backend() = default;
    
    // 初始化和清理
    virtual bool initialize() = 0;
    virtual void cleanup() = 0;
    
    // 设备管理
    virtual std::vector<DeviceInfo> getAvailableDevices() const = 0;
    virtual bool setDevice(int deviceId) = 0;
    virtual int getCurrentDevice() const = 0;
    
    // 内存管理
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void copyToDevice(void* dst, const void* src, size_t bytes) = 0;
    virtual void copyFromDevice(void* dst, const void* src, size_t bytes) = 0;
    virtual void copyDeviceToDevice(void* dst, const void* src, size_t bytes) = 0;
    
    // 同步操作
    virtual void synchronize() = 0;
    
    // 后端信息
    virtual DeviceType getType() const = 0;
    virtual std::string getName() const = 0;
    virtual bool isAvailable() const = 0;
};

// 后端工厂类
class BackendFactory {
public:
    using CreateBackendFunc = std::function<std::unique_ptr<Backend>()>;
    
    static BackendFactory& getInstance();
    
    // 注册后端
    void registerBackend(DeviceType type, CreateBackendFunc createFunc);
    
    // 创建后端
    std::unique_ptr<Backend> createBackend(DeviceType type);
    
    // 获取可用后端类型
    std::vector<DeviceType> getAvailableBackendTypes() const;
    
    // 自动选择最佳后端
    std::unique_ptr<Backend> createBestBackend();

private:
    BackendFactory() = default;
    std::unordered_map<DeviceType, CreateBackendFunc> backends_;
};

// 后端管理器
class BackendManager {
public:
    static BackendManager& getInstance();
    
    // 初始化所有可用后端
    bool initializeBackends();
    
    // 获取当前后端
    Backend* getCurrentBackend() const;
    
    // 设置当前后端
    bool setCurrentBackend(DeviceType type);
    
    // 获取所有可用后端
    const std::vector<std::unique_ptr<Backend>>& getBackends() const;
    
    // 清理所有后端
    void cleanup();

private:
    BackendManager() = default;
    std::vector<std::unique_ptr<Backend>> backends_;
    Backend* currentBackend_ = nullptr;
};

// 工具函数
std::string deviceTypeToString(DeviceType type);
DeviceType stringToDeviceType(const std::string& typeStr);

} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_BACKEND_H