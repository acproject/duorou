#include "wrapper.h"
#include "causal.h"
#include "encoder.h"
#include <stdexcept>

namespace duorou {
namespace kvcache {

CacheWrapper::CacheWrapper(CacheType type) 
    : type_(type), cache_(nullptr), ownsCache_(true) {
    createCache();
}

CacheWrapper::CacheWrapper(CacheType type, Cache* cache)
    : type_(type), cache_(cache), ownsCache_(false) {
}

CacheWrapper::~CacheWrapper() {
    if (ownsCache_) {
        destroyCache();
    }
}

CacheWrapper::CacheWrapper(CacheWrapper&& other) noexcept
    : type_(other.type_), cache_(other.cache_), ownsCache_(other.ownsCache_) {
    other.cache_ = nullptr;
    other.ownsCache_ = false;
}

CacheWrapper& CacheWrapper::operator=(CacheWrapper&& other) noexcept {
    if (this != &other) {
        if (ownsCache_) {
            destroyCache();
        }
        
        type_ = other.type_;
        cache_ = other.cache_;
        ownsCache_ = other.ownsCache_;
        
        other.cache_ = nullptr;
        other.ownsCache_ = false;
    }
    return *this;
}

void CacheWrapper::init(Context& ctx, const CacheConfig& config) {
    validateCache();
    cache_->init(ctx, config);
}

void CacheWrapper::close() {
    if (cache_) {
        cache_->close();
    }
}

void CacheWrapper::setLayer(int layer) {
    validateCache();
    cache_->setLayer(layer);
}

std::tuple<Tensor, Tensor> CacheWrapper::get(Context& ctx, int seq, int32_t startPos, int32_t endPos) {
    validateCache();
    return cache_->get(ctx, seq, startPos, endPos);
}

void CacheWrapper::put(Context& ctx, const Tensor& key, const Tensor& value) {
    validateCache();
    cache_->put(ctx, key, value);
}

void CacheWrapper::startForward(Context& ctx, const Batch& batch, bool reserve) {
    validateCache();
    cache_->startForward(ctx, batch, reserve);
}

void CacheWrapper::copyPrefix(Context& ctx, int srcSeq, int dstSeq, int32_t length) {
    validateCache();
    cache_->copyPrefix(ctx, srcSeq, dstSeq, length);
}

bool CacheWrapper::canResume(int seq, int32_t pos) const {
    if (!cache_) return false;
    return cache_->canResume(seq, pos);
}

void CacheWrapper::remove(int seq, int32_t beginIndex, int32_t endIndex) {
    validateCache();
    cache_->remove(seq, beginIndex, endIndex);
}

std::tuple<Tensor, Tensor, Tensor> CacheWrapper::buildOutputTensors(Context& ctx, const std::vector<int>& activeSeqs) {
    validateCache();
    return cache_->buildOutputTensors(ctx, activeSeqs);
}

CacheType CacheWrapper::getType() const {
    return type_;
}

Cache* CacheWrapper::getCache() const {
    return cache_;
}

bool CacheWrapper::isValid() const {
    return cache_ != nullptr;
}

void CacheWrapper::reset() {
    if (ownsCache_) {
        destroyCache();
    }
    cache_ = nullptr;
    createCache();
}

CacheWrapper CacheWrapper::createEncoder() {
    return CacheWrapper(CacheType::ENCODER);
}

CacheWrapper CacheWrapper::createCausal() {
    return CacheWrapper(CacheType::CAUSAL);
}

void CacheWrapper::createCache() {
    switch (type_) {
        case CacheType::ENCODER:
            cache_ = new EncoderCache();
            break;
        case CacheType::CAUSAL:
            cache_ = new CausalCache();
            break;
        case CacheType::BIDIRECTIONAL:
            // 暂时使用因果缓存作为双向缓存的实现
            cache_ = new CausalCache();
            break;
        default:
            throw std::invalid_argument("Unknown cache type");
    }
}

void CacheWrapper::destroyCache() {
    delete cache_;
    cache_ = nullptr;
}

void CacheWrapper::validateCache() const {
    if (!cache_) {
        throw std::runtime_error("Cache is not initialized");
    }
}

// 全局函数实现
CacheWrapper createCacheWrapper(CacheType type) {
    return CacheWrapper(type);
}

std::string cacheTypeToString(CacheType type) {
    switch (type) {
        case CacheType::ENCODER:
            return "ENCODER";
        case CacheType::CAUSAL:
            return "CAUSAL";
        case CacheType::BIDIRECTIONAL:
            return "BIDIRECTIONAL";
        default:
            return "UNKNOWN";
    }
}

CacheType stringToCacheType(const std::string& typeStr) {
    if (typeStr == "ENCODER") {
        return CacheType::ENCODER;
    } else if (typeStr == "CAUSAL") {
        return CacheType::CAUSAL;
    } else if (typeStr == "BIDIRECTIONAL") {
        return CacheType::BIDIRECTIONAL;
    } else {
        throw std::invalid_argument("Unknown cache type string: " + typeStr);
    }
}

} // namespace kvcache
} // namespace duorou