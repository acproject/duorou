#ifndef DUOROU_KVCACHE_CACHE_H
#define DUOROU_KVCACHE_CACHE_H

#include <vector>
#include <string>
#include <tuple>
#include <stdexcept>
#include <cstdint>

namespace duorou {
namespace kvcache {

// 错误类型定义
class CacheError : public std::runtime_error {
public:
    explicit CacheError(const std::string& message) : std::runtime_error(message) {}
};

class OutOfMemoryError : public CacheError {
public:
    explicit OutOfMemoryError(const std::string& message) : CacheError(message) {}
};

// 数据类型枚举
enum class DType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT64
};

// 缓存配置结构
struct CacheConfig {
    int maxSeqLen;
    int maxBatchSize;
    int numLayers;
    int numHeads;
    int headDim;
    DType dtype;
    
    CacheConfig() : maxSeqLen(2048), maxBatchSize(32), numLayers(32), 
                   numHeads(32), headDim(128), dtype(DType::FLOAT32) {}
};

// 批处理输入结构
struct Batch {
    std::vector<int> seqs;
    std::vector<int> seqLens;
    std::vector<int> positions;
    int batchSize;
    
    Batch() : batchSize(0) {}
};

// 前向声明
class Context;
class Backend;

// 张量类
class Tensor {
private:
    std::vector<int> shape_;
    DType dtype_;
    void* data_;
    size_t size_;

public:
    Tensor(const std::vector<int>& shape, DType dtype);
    ~Tensor();
    
    const std::vector<int>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    void* data() const { return data_; }
    size_t size() const { return size_; }
    
    size_t totalElements() const;
    size_t bytesSize() const;
};

// 上下文类
class Context {
private:
    Backend* backend_;
    
public:
    explicit Context(Backend* backend) : backend_(backend) {}
    Backend* backend() const { return backend_; }
};

// 后端类
class Backend {
public:
    virtual ~Backend() = default;
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void copy(void* dst, const void* src, size_t bytes) = 0;
};

// 缓存接口
class Cache {
public:
    virtual ~Cache() = default;
    
    // 初始化缓存
    virtual void init(Context& ctx, const CacheConfig& config) = 0;
    
    // 关闭缓存
    virtual void close() = 0;
    
    // 设置当前层
    virtual void setLayer(int layer) = 0;
    
    // 获取缓存
    virtual std::tuple<Tensor, Tensor> get(Context& ctx, int seq, int32_t startPos, int32_t endPos) = 0;
    
    // 存储缓存
    virtual void put(Context& ctx, const Tensor& key, const Tensor& value) = 0;
    
    // 开始前向传播
    virtual void startForward(Context& ctx, const Batch& batch, bool reserve) = 0;
    
    // 复制前缀
    virtual void copyPrefix(Context& ctx, int srcSeq, int dstSeq, int32_t length) = 0;
    
    // 检查是否可以恢复
    virtual bool canResume(int seq, int32_t pos) const = 0;
    
    // 移除缓存条目
    virtual void remove(int seq, int32_t beginIndex, int32_t endIndex) = 0;
    
    // 构建输出张量
    virtual std::tuple<Tensor, Tensor, Tensor> buildOutputTensors(Context& ctx, const std::vector<int>& activeSeqs) = 0;
};

} // namespace kvcache
} // namespace duorou

#endif // DUOROU_KVCACHE_CACHE_H