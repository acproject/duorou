#ifndef DUOROU_ML_CPU_BACKEND_H
#define DUOROU_ML_CPU_BACKEND_H

#include "backend.h"
#include <thread>
#include <mutex>

namespace duorou {
namespace ml {

// CPU backend implementation
class CPUBackend : public Backend {
public:
    CPUBackend();
    virtual ~CPUBackend();
    
    // Backend interface implementation
    bool initialize() override;
    void cleanup() override;
    
    std::vector<DeviceInfo> getAvailableDevices() const override;
    bool setDevice(int deviceId) override;
    int getCurrentDevice() const override;
    
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    void copyToDevice(void* dst, const void* src, size_t bytes) override;
    void copyFromDevice(void* dst, const void* src, size_t bytes) override;
    void copyDeviceToDevice(void* dst, const void* src, size_t bytes) override;
    
    void synchronize() override;
    
    DeviceType getType() const override { return DeviceType::CPU; }
    std::string getName() const override { return "CPU Backend"; }
    bool isAvailable() const override;
    
    // CPU specific functionality
    void setNumThreads(int numThreads);
    int getNumThreads() const;
    
private:
    bool initialized_;
    int numThreads_;
    std::mutex allocMutex_;
    
    // Memory aligned allocation
    void* alignedAlloc(size_t bytes, size_t alignment = 32);
    void alignedFree(void* ptr);
};

} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_CPU_BACKEND_H