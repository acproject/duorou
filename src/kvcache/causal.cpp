#include "causal.h"

namespace duorou {
namespace kvcache {

CausalCache::CausalCache(const CausalOptions& options) 
    : options_(options), currentLayer_(0), initialized_(false) {
}

void CausalCache::init(Context& ctx, const CacheConfig& config) {
    config_ = config;
    initialized_ = true;
    kv_store_.clear();
    kv_stride_ = config_.numHeads * config_.headDim;
}

void CausalCache::close() {
    sequences_.clear();
    initialized_ = false;
}

void CausalCache::setLayer(int layer) {
    currentLayer_ = layer;
}

std::tuple<Tensor, Tensor> CausalCache::get(Context& ctx, int seq, int32_t startPos, int32_t endPos) {
    // Fetch layer KV for current layer
    auto &layers = kv_store_[seq];
    auto itLayer = layers.find(currentLayer_);
    int32_t totalLen = 0;
    if (itLayer != layers.end()) {
        totalLen = itLayer->second.length;
    }
    // Apply sliding window
    int32_t win = totalLen;
    if (options_.enableSlidingWindow && options_.slidingWindow > 0) {
        win = std::min(win, options_.slidingWindow);
    }
    // Compute slice range
    int32_t sPos = 0;
    int32_t ePos = win;
    if (endPos != std::numeric_limits<int32_t>::max()) {
        ePos = std::min(endPos, win);
    }
    if (startPos > 0) {
        sPos = std::min(startPos, ePos);
    }
    int32_t outLen = std::max(0, ePos - sPos);
    // Allocate tensors for output [B=1, S=outLen, H, D]
    std::vector<int> shape = {1, outLen, config_.numHeads, config_.headDim};
    Tensor key(shape, DType::FLOAT32, ctx.backend());
    Tensor value(shape, DType::FLOAT32, ctx.backend());
    // Copy data from kv_store_ into tensors
    if (outLen > 0 && itLayer != layers.end()) {
        const LayerKV &kv = itLayer->second;
        size_t perTok = static_cast<size_t>(kv_stride_);
        size_t copyBytes = static_cast<size_t>(outLen) * perTok * sizeof(float);
        // Source offset in tokens
        int32_t srcTok = totalLen - win + sPos;
        srcTok = std::max(0, srcTok);
        const float *kSrc = kv.k.data() + static_cast<size_t>(srcTok) * perTok;
        const float *vSrc = kv.v.data() + static_cast<size_t>(srcTok) * perTok;
        if (key.data() && value.data()) {
            if (ctx.backend()) {
                ctx.backend()->copy(key.data(), kSrc, copyBytes);
                ctx.backend()->copy(value.data(), vSrc, copyBytes);
            } else {
                std::memcpy(key.data(), kSrc, copyBytes);
                std::memcpy(value.data(), vSrc, copyBytes);
            }
        }
    }
    return std::make_tuple(key, value);
}

void CausalCache::put(Context& ctx, const Tensor& key, const Tensor& value) {
    // Append new K/V segment to kv_store_ for current layer
    // NOTE: current implementation assumes single-sequence (seq=0).
    // Future work: plumb sequence id through call sites.
    int seq = 0;
    auto &layers = kv_store_[seq];
    LayerKV &kv = layers[currentLayer_];
    const auto &s = key.shape();
    int newSk = (s.size() >= 2) ? s[1] : 0;
    if (newSk <= 0) return;
    size_t perTok = static_cast<size_t>(kv_stride_);
    size_t add = static_cast<size_t>(newSk) * perTok;
    size_t old = kv.k.size();
    kv.k.resize(old + add);
    kv.v.resize(old + add);
    // Copy from tensor buffers
    size_t copyBytes = add * sizeof(float);
    if (key.data() && value.data()) {
        if (ctx.backend()) {
            ctx.backend()->copy(kv.k.data() + old, key.data(), copyBytes);
            ctx.backend()->copy(kv.v.data() + old, value.data(), copyBytes);
        } else {
            std::memcpy(kv.k.data() + old, key.data(), copyBytes);
            std::memcpy(kv.v.data() + old, value.data(), copyBytes);
        }
    }
    kv.length += newSk;
    // Update sequence length bookkeeping before eviction
    auto &info = sequences_[seq];
    info.active = true;
    info.capacity = config_.maxSeqLen;
    info.length = std::min(kv.length, info.capacity);
    // Apply sliding window eviction to limit memory growth
    if (options_.enableSlidingWindow && options_.slidingWindow > 0 && kv.length > options_.slidingWindow) {
        int32_t evictTok = kv.length - options_.slidingWindow;
        if (evictTok > 0) {
            // Drop prefix tokens
            size_t evict = static_cast<size_t>(evictTok) * perTok;
            if (evict < kv.k.size()) {
                kv.k.erase(kv.k.begin(), kv.k.begin() + static_cast<std::ptrdiff_t>(evict));
                kv.v.erase(kv.v.begin(), kv.v.begin() + static_cast<std::ptrdiff_t>(evict));
                kv.length = options_.slidingWindow;
                // Keep sequence bookkeeping consistent with the windowed length
                info.length = std::min(kv.length, info.capacity);
            }
        }
    }
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
    auto it = kv_store_.find(seq);
    if (it == kv_store_.end()) return;
    auto &layers = it->second;
    for (auto &p : layers) {
        auto &kv = p.second;
        // Remove tokens from [beginIndex, endIndex)
        if (endIndex == std::numeric_limits<int32_t>::max()) {
            // truncate at beginIndex
            int32_t keep = std::max(0, beginIndex);
            size_t perTok = static_cast<size_t>(kv_stride_);
            size_t keepElems = static_cast<size_t>(keep) * perTok;
            if (keepElems < kv.k.size()) {
                kv.k.resize(keepElems);
                kv.v.resize(keepElems);
                kv.length = keep;
            }
        } else if (beginIndex < endIndex) {
            int32_t rem = std::min(kv.length, endIndex) - std::max(0, beginIndex);
            if (rem > 0) {
                // We simply truncate for now to avoid fragmentation
                int32_t keep = std::max(0, kv.length - rem);
                size_t perTok = static_cast<size_t>(kv_stride_);
                size_t keepElems = static_cast<size_t>(keep) * perTok;
                if (keepElems < kv.k.size()) {
                    kv.k.resize(keepElems);
                    kv.v.resize(keepElems);
                    kv.length = keep;
                }
            }
        }
    }
    // Update sequence bookkeeping for consistency
    auto itSeq = sequences_.find(seq);
    if (itSeq != sequences_.end()) {
        itSeq->second.length = std::min(itSeq->second.length, it->second[currentLayer_].length);
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
    kv_store_[seqId].clear();
}

void CausalCache::removeSequence(int seqId) {
    sequences_.erase(seqId);
}

void CausalCache::clearSequences() {
    sequences_.clear();
}

bool CausalCache::hasSequence(int seq) const {
    return sequences_.find(seq) != sequences_.end();
}

int32_t CausalCache::getSequenceLength(int seq) const {
    auto it = sequences_.find(seq);
    if (it == sequences_.end()) return 0;
    return it->second.length;
}

void CausalCache::validateSequence(int seq) const {
    if (sequences_.find(seq) == sequences_.end()) {
        // Basic validation: sequence must exist
        // In production, we might throw a more specific error type
        throw CacheError("Invalid sequence id");
    }
}

void CausalCache::updateSequenceLength(int seq, int32_t newLength) {
    auto &info = sequences_[seq];
    info.length = std::min(newLength, info.capacity);
}

bool CausalCache::isWithinSlidingWindow(int seq, int32_t pos) const {
    auto itLayer = kv_store_.find(seq);
    if (itLayer == kv_store_.end()) {
        return pos == 0;
    }
    const auto &layers = itLayer->second;
    auto itKV = layers.find(currentLayer_);
    int32_t totalLen = (itKV != layers.end()) ? itKV->second.length : 0;
    int32_t win = totalLen;
    if (options_.enableSlidingWindow && options_.slidingWindow > 0) {
        win = std::min(win, options_.slidingWindow);
    }
    return pos <= win;
}

} // namespace kvcache
} // namespace duorou