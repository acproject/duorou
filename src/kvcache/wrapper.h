#ifndef DUOROU_KVCACHE_WRAPPER_H
#define DUOROU_KVCACHE_WRAPPER_H

#include <vector>
#include <string>
#include <cstdint>
#include "cache.h"

namespace duorou {
namespace kvcache {

// 缓存类型枚举
enum class CacheType {
    ENCODER,
    CAUSAL,
    BIDIRECTIONAL
};

// 缓存包装器类
class CacheWrapper {
private:
    CacheType type_;
    Cache* cache_;
    bool ownsCache_;

public:
    // 构造函数
    explicit CacheWrapper(CacheType type);
    CacheWrapper(CacheType type, Cache* cache);
    
    // 析构函数
    ~CacheWrapper();
    
    // 禁用拷贝构造和赋值
    CacheWrapper(const CacheWrapper&) = delete;
    CacheWrapper& operator=(const CacheWrapper&) = delete;
    
    // 移动构造和赋值
    CacheWrapper(CacheWrapper&& other) noexcept;
    CacheWrapper& operator=(CacheWrapper&& other) noexcept;
    
    // 缓存操作接口
    void init(Context& ctx, const CacheConfig& config);
    void close();
    void setLayer(int layer);
    std::tuple<Tensor, Tensor> get(Context& ctx, int seq, int32_t startPos, int32_t endPos);
    void put(Context& ctx, const Tensor& key, const Tensor& value);
    void startForward(Context& ctx, const Batch& batch, bool reserve = false);
    void copyPrefix(Context& ctx, int srcSeq, int dstSeq, int32_t length);
    bool canResume(int seq, int32_t pos) const;
    void remove(int seq, int32_t beginIndex, int32_t endIndex);
    std::tuple<Tensor, Tensor, Tensor> buildOutputTensors(Context& ctx, const std::vector<int>& activeSeqs);
    
    // 包装器特有方法
    CacheType getType() const;
    Cache* getCache() const;
    bool isValid() const;
    void reset();
    
    // 静态工厂方法
    static CacheWrapper createEncoder();
    static CacheWrapper createCausal();
    
private:
    // 内部辅助方法
    void createCache();
    void destroyCache();
    void validateCache() const;
};

// 全局函数
CacheWrapper createCacheWrapper(CacheType type);
std::string cacheTypeToString(CacheType type);
CacheType stringToCacheType(const std::string& typeStr);

} // namespace kvcache
} // namespace duorou

#endif // DUOROU_KVCACHE_WRAPPER_H