#include "encoder.h"

namespace duorou {
namespace kvcache {

EncoderCache::EncoderCache(const EncoderConfig& config) 
    : encoderConfig_(config), currentLayer_(0), initialized_(false) {
}

void EncoderCache::init(Context& ctx, const CacheConfig& config) {
    config_ = config;
    initialized_ = true;
    
    // Clear existing cache
    keyCache_.clear();
    valueCache_.clear();
}

void EncoderCache::close() {
    keyCache_.clear();
    valueCache_.clear();
    initialized_ = false;
}

void EncoderCache::setLayer(int layer) {
    validateLayer(layer);
    currentLayer_ = layer;
    ensureLayerExists(layer);
}

std::tuple<Tensor, Tensor> EncoderCache::get(Context& ctx, int seq, int32_t startPos, int32_t endPos) {
    ensureLayerExists(currentLayer_);
    
    // Create empty key-value tensors as placeholders
    std::vector<int> shape = {1, encoderConfig_.numHeads, encoderConfig_.headDim};
    Tensor key(shape, DType::FLOAT32);
    Tensor value(shape, DType::FLOAT32);
    
    // If cache data exists, return cached tensors
    auto& keyLayer = keyCache_[currentLayer_];
    auto& valueLayer = valueCache_[currentLayer_];
    
    if (seq < static_cast<int>(keyLayer.size()) && seq < static_cast<int>(valueLayer.size())) {
        return std::make_tuple(keyLayer[seq], valueLayer[seq]);
    }
    
    return std::make_tuple(key, value);
}

void EncoderCache::put(Context& ctx, const Tensor& key, const Tensor& value) {
    ensureLayerExists(currentLayer_);
    
    // Simplified implementation: assume sequence ID is 0
    int seq = 0;
    auto& keyLayer = keyCache_[currentLayer_];
    auto& valueLayer = valueCache_[currentLayer_];
    
    // Ensure vector size is sufficient
    if (seq >= static_cast<int>(keyLayer.size())) {
        keyLayer.resize(seq + 1, Tensor({1, 1, 1}, DType::FLOAT32));
    }
    if (seq >= static_cast<int>(valueLayer.size())) {
        valueLayer.resize(seq + 1, Tensor({1, 1, 1}, DType::FLOAT32));
    }
    
    keyLayer[seq] = key;
    valueLayer[seq] = value;
}

void EncoderCache::startForward(Context& ctx, const Batch& batch, bool reserve) {
    // Ensure current layer exists
    ensureLayerExists(currentLayer_);
    
    // Prepare cache space for all sequences in the batch
    for (int seq : batch.seqs) {
        auto& keyLayer = keyCache_[currentLayer_];
        auto& valueLayer = valueCache_[currentLayer_];
        
        if (seq >= static_cast<int>(keyLayer.size())) {
            keyLayer.resize(seq + 1, Tensor({1, 1, 1}, DType::FLOAT32));
        }
        if (seq >= static_cast<int>(valueLayer.size())) {
            valueLayer.resize(seq + 1, Tensor({1, 1, 1}, DType::FLOAT32));
        }
    }
}

void EncoderCache::copyPrefix(Context& ctx, int srcSeq, int dstSeq, int32_t length) {
    ensureLayerExists(currentLayer_);
    
    auto& keyLayer = keyCache_[currentLayer_];
    auto& valueLayer = valueCache_[currentLayer_];
    
    // Ensure both source and destination sequences exist
    if (srcSeq < static_cast<int>(keyLayer.size()) && srcSeq < static_cast<int>(valueLayer.size())) {
        if (dstSeq >= static_cast<int>(keyLayer.size())) {
            keyLayer.resize(dstSeq + 1, Tensor({1, 1, 1}, DType::FLOAT32));
        }
        if (dstSeq >= static_cast<int>(valueLayer.size())) {
            valueLayer.resize(dstSeq + 1, Tensor({1, 1, 1}, DType::FLOAT32));
        }
        
        // Copy cached data
        keyLayer[dstSeq] = keyLayer[srcSeq];
        valueLayer[dstSeq] = valueLayer[srcSeq];
    }
}

bool EncoderCache::canResume(int seq, int32_t pos) const {
    if (!initialized_) {
        return false;
    }
    
    // Check if cache data exists for this sequence
    auto keyIt = keyCache_.find(currentLayer_);
    auto valueIt = valueCache_.find(currentLayer_);
    
    if (keyIt != keyCache_.end() && valueIt != valueCache_.end()) {
        const auto& keyLayer = keyIt->second;
        const auto& valueLayer = valueIt->second;
        
        return seq < static_cast<int>(keyLayer.size()) && seq < static_cast<int>(valueLayer.size());
    }
    
    return false;
}

void EncoderCache::remove(int seq, int32_t beginIndex, int32_t endIndex) {
    if (!initialized_) {
        return;
    }
    
    // For encoder cache, delete operation usually means clearing the cache for that sequence
    for (auto& [layer, keyLayer] : keyCache_) {
        if (seq < static_cast<int>(keyLayer.size())) {
            keyLayer[seq] = Tensor({1, 1, 1}, DType::FLOAT32); // Reset to empty tensor
        }
    }
    
    for (auto& [layer, valueLayer] : valueCache_) {
        if (seq < static_cast<int>(valueLayer.size())) {
            valueLayer[seq] = Tensor({1, 1, 1}, DType::FLOAT32); // Reset to empty tensor
        }
    }
}

std::tuple<Tensor, Tensor, Tensor> EncoderCache::buildOutputTensors(Context& ctx, const std::vector<int>& activeSeqs) {
    // Implementation for building output tensors
    std::vector<int> shape = {static_cast<int>(activeSeqs.size()), encoderConfig_.numHeads, encoderConfig_.headDim};
    Tensor key(shape, DType::FLOAT32);
    Tensor value(shape, DType::FLOAT32);
    Tensor mask(shape, DType::FLOAT32);
    return std::make_tuple(key, value, mask);
}

// Implementation of EncoderCache-specific methods
void EncoderCache::setEncoderConfig(const EncoderConfig& config) {
    encoderConfig_ = config;
}

const EncoderConfig& EncoderCache::getEncoderConfig() const {
    return encoderConfig_;
}

void EncoderCache::clearCache() {
    keyCache_.clear();
    valueCache_.clear();
}

size_t EncoderCache::getCacheSize() const {
    size_t totalSize = 0;
    for (const auto& [layer, keyLayer] : keyCache_) {
        totalSize += keyLayer.size();
    }
    for (const auto& [layer, valueLayer] : valueCache_) {
        totalSize += valueLayer.size();
    }
    return totalSize;
}

// Implementation of private methods
void EncoderCache::validateLayer(int layer) const {
    if (layer < 0 || layer >= encoderConfig_.numLayers) {
        throw CacheError("Invalid layer index: " + std::to_string(layer));
    }
}

void EncoderCache::ensureLayerExists(int layer) {
    if (keyCache_.find(layer) == keyCache_.end()) {
        allocateLayerCache(layer);
    }
}

void EncoderCache::allocateLayerCache(int layer) {
    keyCache_[layer] = std::vector<Tensor>();
    valueCache_[layer] = std::vector<Tensor>();
}

} // namespace kvcache
} // namespace duorou