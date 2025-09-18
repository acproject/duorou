#include "encoder.h"

namespace duorou {
namespace kvcache {

EncoderCache::EncoderCache(const EncoderConfig& config) 
    : encoderConfig_(config), currentLayer_(0), initialized_(false) {
}

void EncoderCache::init(Context& ctx, const CacheConfig& config) {
    config_ = config;
    initialized_ = true;
    
    // 清空现有缓存
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
    
    // 创建空的键值张量作为占位符
    std::vector<int> shape = {1, encoderConfig_.numHeads, encoderConfig_.headDim};
    Tensor key(shape, DType::FLOAT32);
    Tensor value(shape, DType::FLOAT32);
    
    // 如果有缓存数据，返回缓存的张量
    auto& keyLayer = keyCache_[currentLayer_];
    auto& valueLayer = valueCache_[currentLayer_];
    
    if (seq < static_cast<int>(keyLayer.size()) && seq < static_cast<int>(valueLayer.size())) {
        return std::make_tuple(keyLayer[seq], valueLayer[seq]);
    }
    
    return std::make_tuple(key, value);
}

void EncoderCache::put(Context& ctx, const Tensor& key, const Tensor& value) {
    ensureLayerExists(currentLayer_);
    
    // 简化实现：假设序列ID为0
    int seq = 0;
    auto& keyLayer = keyCache_[currentLayer_];
    auto& valueLayer = valueCache_[currentLayer_];
    
    // 确保向量大小足够
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
    // 确保当前层存在
    ensureLayerExists(currentLayer_);
    
    // 为批次中的所有序列准备缓存空间
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
    
    // 确保源序列和目标序列都存在
    if (srcSeq < static_cast<int>(keyLayer.size()) && srcSeq < static_cast<int>(valueLayer.size())) {
        if (dstSeq >= static_cast<int>(keyLayer.size())) {
            keyLayer.resize(dstSeq + 1, Tensor({1, 1, 1}, DType::FLOAT32));
        }
        if (dstSeq >= static_cast<int>(valueLayer.size())) {
            valueLayer.resize(dstSeq + 1, Tensor({1, 1, 1}, DType::FLOAT32));
        }
        
        // 复制缓存数据
        keyLayer[dstSeq] = keyLayer[srcSeq];
        valueLayer[dstSeq] = valueLayer[srcSeq];
    }
}

bool EncoderCache::canResume(int seq, int32_t pos) const {
    if (!initialized_) {
        return false;
    }
    
    // 检查是否有该序列的缓存数据
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
    
    // 对于编码器缓存，删除操作通常意味着清空该序列的缓存
    for (auto& [layer, keyLayer] : keyCache_) {
        if (seq < static_cast<int>(keyLayer.size())) {
            keyLayer[seq] = Tensor({1, 1, 1}, DType::FLOAT32); // 重置为空张量
        }
    }
    
    for (auto& [layer, valueLayer] : valueCache_) {
        if (seq < static_cast<int>(valueLayer.size())) {
            valueLayer[seq] = Tensor({1, 1, 1}, DType::FLOAT32); // 重置为空张量
        }
    }
}

std::tuple<Tensor, Tensor, Tensor> EncoderCache::buildOutputTensors(Context& ctx, const std::vector<int>& activeSeqs) {
    // 构建输出张量的实现
    std::vector<int> shape = {static_cast<int>(activeSeqs.size()), encoderConfig_.numHeads, encoderConfig_.headDim};
    Tensor key(shape, DType::FLOAT32);
    Tensor value(shape, DType::FLOAT32);
    Tensor mask(shape, DType::FLOAT32);
    return std::make_tuple(key, value, mask);
}

// EncoderCache特有方法实现
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

// 私有方法实现
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