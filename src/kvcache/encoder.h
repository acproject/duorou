#ifndef DUOROU_KVCACHE_ENCODER_H
#define DUOROU_KVCACHE_ENCODER_H

#include "cache.h"
#include <vector>
#include <unordered_map>
#include <tuple>
#include <cstdint>

namespace duorou {
namespace kvcache {

// 编码器缓存配置
struct EncoderConfig {
    int maxSeqLen;
    int numLayers;
    int numHeads;
    int headDim;
    bool enableOptimization;
    
    EncoderConfig() : maxSeqLen(512), numLayers(12), numHeads(12), 
                     headDim(64), enableOptimization(true) {}
};

// 编码器缓存类
class EncoderCache : public Cache {
private:
    EncoderConfig encoderConfig_;
    CacheConfig config_;
    int currentLayer_;
    bool initialized_;
    
    // 缓存存储
    std::unordered_map<int, std::vector<Tensor>> keyCache_;
    std::unordered_map<int, std::vector<Tensor>> valueCache_;

public:
    // 构造函数
    explicit EncoderCache(const EncoderConfig& config = EncoderConfig());
    
    // 析构函数
    virtual ~EncoderCache() = default;
    
    // Cache接口实现
    virtual void init(Context& ctx, const CacheConfig& config) override;
    virtual void close() override;
    virtual void setLayer(int layer) override;
    virtual std::tuple<Tensor, Tensor> get(Context& ctx, int seq, int32_t startPos, int32_t endPos) override;
    virtual void put(Context& ctx, const Tensor& key, const Tensor& value) override;
    virtual void startForward(Context& ctx, const Batch& batch, bool reserve) override;
    virtual void copyPrefix(Context& ctx, int srcSeq, int dstSeq, int32_t length) override;
    virtual bool canResume(int seq, int32_t pos) const override;
    virtual void remove(int seq, int32_t beginIndex, int32_t endIndex) override;
    virtual std::tuple<Tensor, Tensor, Tensor> buildOutputTensors(Context& ctx, const std::vector<int>& activeSeqs) override;
    
    // EncoderCache特有方法
    void setEncoderConfig(const EncoderConfig& config);
    const EncoderConfig& getEncoderConfig() const;
    void clearCache();
    size_t getCacheSize() const;
    
private:
    // 内部辅助方法
    void validateLayer(int layer) const;
    void ensureLayerExists(int layer);
    void allocateLayerCache(int layer);
};

} // namespace kvcache
} // namespace duorou

#endif // DUOROU_KVCACHE_ENCODER_H