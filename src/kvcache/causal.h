#ifndef DUOROU_KVCACHE_CAUSAL_H
#define DUOROU_KVCACHE_CAUSAL_H

#include <vector>
#include <unordered_map>
#include <limits>
#include "cache.h"

namespace duorou {
namespace kvcache {

// Causal cache options
struct CausalOptions {
    int slidingWindow;
    bool enableSlidingWindow;
    
    CausalOptions() : slidingWindow(-1), enableSlidingWindow(false) {}
};

// Sequence information structure
struct SequenceInfo {
    int32_t length;
    int32_t capacity;
    bool active;
    
    SequenceInfo() : length(0), capacity(0), active(false) {}
};

// Causal attention cache class
class CausalCache : public Cache {
private:
    CausalOptions options_;
    std::unordered_map<int, SequenceInfo> sequences_;
    int currentLayer_;
    CacheConfig config_;
    bool initialized_;

public:
    // Constructor
    explicit CausalCache(const CausalOptions& options = CausalOptions());
    
    // Destructor
    virtual ~CausalCache() = default;
    
    // Cache interface implementation
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
    
    // CausalCache-specific methods
    void setSlidingWindow(int window);
    int getSlidingWindow() const;
    void enableSlidingWindow(bool enable);
    bool isSlidingWindowEnabled() const;
    
    // Sequence management
    void addSequence(int seq, int32_t capacity);
    void removeSequence(int seq);
    bool hasSequence(int seq) const;
    int32_t getSequenceLength(int seq) const;
    
private:
    // Internal helper methods
    void validateSequence(int seq) const;
    void updateSequenceLength(int seq, int32_t newLength);
    bool isWithinSlidingWindow(int seq, int32_t pos) const;
};

} // namespace kvcache
} // namespace duorou

#endif // DUOROU_KVCACHE_CAUSAL_H