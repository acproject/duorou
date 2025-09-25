#ifdef __cplusplus
#ifndef DUOROU_ML_GGML_BACKEND_H
#define DUOROU_ML_GGML_BACKEND_H

#include "backend.h"
#include <string>
#include <vector>
#include <mutex>
#include <cstddef>
#include <cstring>

namespace duorou {
namespace ml {

// GGML backend skeleton implementation (no ggml dependency for now)
class GGMLBackend : public Backend {
public:
    GGMLBackend();
    ~GGMLBackend() override;

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

    DeviceType getType() const override { return DeviceType::GGML; }
    std::string getName() const override { return "GGML Backend"; }
    bool isAvailable() const override;

private:
    bool initialized_;
    int currentDeviceId_;
    std::mutex allocMutex_;
};

} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_GGML_BACKEND_H
#endif // __cplusplus