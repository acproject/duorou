#include "causal.h"

namespace duorou {
namespace kvcache {

CausalCache::CausalCache(const CausalOptions& options) 
    : options_(options), currentLayer_(0), initialized_(false) {
}

void CausalCache::init(Context& ctx, const CacheConfig& config) {
    config_ = config;
    initialized_ = true;
}

void CausalCache::close() {
    sequences_.clear();
    initialized_ = false;
}

void CausalCache::setLayer(int layer) {
    currentLayer_ = layer;
}

std::tuple<Tensor, Tensor> CausalCache::get(Context& ctx, int seq, int32_t startPos, int32_t endPos) {
    // Determine previous length for the sequence
    int32_t prevLen = 0;
    auto it = sequences_.find(seq);
    if (it != sequences_.end()) {
        prevLen = it->second.length;
        if (options_.enableSlidingWindow && options_.slidingWindow > 0) {
            // Apply sliding window constraint
            if (prevLen > options_.slidingWindow) {
                prevLen = options_.slidingWindow;
            }
        }
    }
    // Build 4D shape: [B=1, S=prevLen, H=numHeads, D=headDim]
    std::vector<int> shape = {1, prevLen, config_.numHeads, config_.headDim};
    Tensor key(shape, DType::FLOAT32, ctx.backend());
    Tensor value(shape, DType::FLOAT32, ctx.backend());
    return std::make_tuple(key, value);
}

void CausalCache::put(Context& ctx, const Tensor& key, const Tensor& value) {
    // Update sequence length for seq 0 by default (single-seq execution)
    int seq = 0;
    auto& info = sequences_[seq];
    info.active = true;
    info.capacity = config_.maxSeqLen;
    // shape: [B, newSk, H, D]
    const auto& s = key.shape();
    int newSk = (s.size() >= 2) ? s[1] : 0;
    info.length = std::min(info.length + newSk, info.capacity);
}

void CausalCache::startForward(Context& ctx, const Batch& batch, bool reserve) {
    // Implementation for starting forward propagation
    for (int seq : batch.seqs) {
        if (sequences_.find(seq) == sequences_.end()) {
            addSequence(seq, config_.maxSeqLen);
        }
        sequences_[seq].active = true;
        sequences_[seq].capacity = config_.maxSeqLen;
        // If reserve is set, optionally pre-set length
        if (reserve && seq < static_cast<int>(batch.seqLens.size())) {
            sequences_[seq].length = batch.seqLens[seq];
        }
    }
}

void CausalCache::copyPrefix(Context& ctx, int srcSeq, int dstSeq, int32_t length) {
    // Implementation for copying prefix
    if (sequences_.find(srcSeq) != sequences_.end() && 
        sequences_.find(dstSeq) != sequences_.end()) {
        // Simplified implementation: copy sequence information
        sequences_[dstSeq].length = std::min(length, sequences_[srcSeq].length);
    }
}

bool CausalCache::canResume(int seq, int32_t pos) const {
    auto it = sequences_.find(seq);
    if (it == sequences_.end()) {
        return false;
    }
    return pos <= it->second.length;
}

void CausalCache::remove(int seq, int32_t beginIndex, int32_t endIndex) {
    auto it = sequences_.find(seq);
    if (it != sequences_.end()) {
        if (endIndex == std::numeric_limits<int32_t>::max()) {
            it->second.length = beginIndex;
        } else {
            // Simplified implementation: adjust sequence length
            it->second.length = std::max(0, static_cast<int32_t>(it->second.length) - (endIndex - beginIndex));
        }
    }
}

std::tuple<Tensor, Tensor, Tensor> CausalCache::buildOutputTensors(Context& ctx, const std::vector<int>& activeSeqs) {
    // Implementation for building output tensors
    std::vector<int> shape = {static_cast<int>(activeSeqs.size()), config_.numHeads, config_.headDim};
    Tensor key(shape, DType::FLOAT32, ctx.backend());
    Tensor value(shape, DType::FLOAT32, ctx.backend());
    Tensor mask(shape, DType::FLOAT32, ctx.backend());
    return std::make_tuple(key, value, mask);
}

// Implementation of CausalCache-specific methods
void CausalCache::setSlidingWindow(int window) {
    options_.slidingWindow = window;
}

int CausalCache::getSlidingWindow() const {
    return options_.slidingWindow;
}

void CausalCache::enableSlidingWindow(bool enable) {
    options_.enableSlidingWindow = enable;
}

bool CausalCache::isSlidingWindowEnabled() const {
    return options_.enableSlidingWindow;
}

// Sequence management
void CausalCache::addSequence(int seqId, int32_t capacity) {
    SequenceInfo info;
    info.length = 0;
    info.capacity = capacity;
    info.active = true;
    sequences_[seqId] = info;
}

void CausalCache::removeSequence(int seqId) {
    sequences_.erase(seqId);
}

void CausalCache::clearSequences() {
    sequences_.clear();
}

} // namespace kvcache
} // namespace duorou