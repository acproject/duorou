#ifndef DUOROU_KVCACHE_WRAPPER_H
#define DUOROU_KVCACHE_WRAPPER_H

#include <vector>
#include <string>
#include <cstdint>
#include "cache.h"

namespace duorou {
namespace kvcache {

// Cache type enumeration
enum class CacheType {
    ENCODER,
    CAUSAL,
    BIDIRECTIONAL
};

// Cache wrapper class
class CacheWrapper {
private:
    CacheType type_;
    Cache* cache_;
    bool ownsCache_;

public:
    // Constructors
    explicit CacheWrapper(CacheType type);
    CacheWrapper(CacheType type, Cache* cache);
    
    // Destructor
    ~CacheWrapper();
    
    // Copy semantics (disabled)
    CacheWrapper(const CacheWrapper&) = delete;
    CacheWrapper& operator=(const CacheWrapper&) = delete;
    
    // Move semantics
    CacheWrapper(CacheWrapper&& other) noexcept;
    CacheWrapper& operator=(CacheWrapper&& other) noexcept;
    
    // Cache operations
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
    
    // Getters and utilities
    CacheType getType() const;
    Cache* getCache() const;
    bool isValid() const;
    void reset();
    
    // Static factory methods
    static CacheWrapper createEncoder();
    static CacheWrapper createCausal();
    
private:
    // Helper methods
    void createCache();
    void destroyCache();
    void validateCache() const;
};

// Utility functions
CacheWrapper createCacheWrapper(CacheType type);
std::string cacheTypeToString(CacheType type);
CacheType stringToCacheType(const std::string& typeStr);

} // namespace kvcache
} // namespace duorou

#endif // DUOROU_KVCACHE_WRAPPER_H