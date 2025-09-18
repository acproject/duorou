#ifndef DUOROU_ML_CONTEXT_H
#define DUOROU_ML_CONTEXT_H

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

namespace duorou {
namespace ml {

// 前向声明
class Backend;
class Tensor;
enum class DataType;

// 计算上下文类
class Context {
public:
    Context(Backend* backend = nullptr);
    ~Context();
    
    // 后端管理
    void setBackend(Backend* backend);
    Backend* getBackend() const { return backend_; }
    
    // 内存管理
    void* allocate(size_t bytes);
    void deallocate(void* ptr);
    
    // 临时张量管理
    Tensor createTempTensor(const std::vector<int64_t>& shape, DataType dtype);
    void releaseTempTensors();
    
    // 计算图管理（可选，用于自动微分）
    void enableGradient(bool enable = true);
    bool isGradientEnabled() const { return gradientEnabled_; }
    
    // 同步操作
    void synchronize();
    
    // 性能分析
    void enableProfiling(bool enable = true);
    bool isProfilingEnabled() const { return profilingEnabled_; }
    void printProfilingInfo() const;
    
    // 配置参数
    void setParameter(const std::string& key, const std::string& value);
    std::string getParameter(const std::string& key) const;
    
private:
    Backend* backend_;
    bool gradientEnabled_;
    bool profilingEnabled_;
    std::vector<std::unique_ptr<Tensor>> tempTensors_;
    std::unordered_map<std::string, std::string> parameters_;
    
    // 性能统计
    mutable std::unordered_map<std::string, double> timingStats_;
};

} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_CONTEXT_H