#ifndef DUOROU_KVCACHE_CACHE_H
#define DUOROU_KVCACHE_CACHE_H

#include <vector>
#include <string>
#include <tuple>
#include <stdexcept>
#include <cstdint>

namespace duorou {
namespace kvcache {

// Error type definitions
class CacheError : public std::runtime_error {
public:
    explicit CacheError(const std::string& message) : std::runtime_error(message) {}
};

class OutOfMemoryError : public CacheError {
public:
    explicit OutOfMemoryError(const std::string& message) : CacheError(message) {}
};

// Data type enumeration
enum class DType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT64
};

// Cache configuration structure
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

// Batch input structure
struct Batch {
    std::vector<int> seqs;
    std::vector<int> seqLens;
    std::vector<int> positions;
    int batchSize;
    
    Batch() : batchSize(0) {}
};

// Forward declarations
class Context;
class Backend;

// Tensor class
class Tensor {
private:
    std::vector<int> shape_;
    DType dtype_;
    void* data_;
    size_t size_;
    Backend* backend_;

public:
    Tensor(const std::vector<int>& shape, DType dtype);
    Tensor(const std::vector<int>& shape, DType dtype, Backend* backend);
    // Copy / Move semantics
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();
    
    const std::vector<int>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    void* data() const { return data_; }
    size_t size() const { return size_; }
    
    size_t totalElements() const;
    size_t bytesSize() const;
};

// Context class
class Context {
private:
    Backend* backend_;
    
public:
    explicit Context(Backend* backend) : backend_(backend) {}
    Backend* backend() const { return backend_; }
};

// Backend class
class Backend {
public:
    virtual ~Backend() = default;
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void copy(void* dst, const void* src, size_t bytes) = 0;
};

// Cache interface
class Cache {
public:
    virtual ~Cache() = default;
    
    // Initialize cache
    virtual void init(Context& ctx, const CacheConfig& config) = 0;
    
    // Close cache
    virtual void close() = 0;
    
    // Set current layer
    virtual void setLayer(int layer) = 0;
    
    // Get cache
    virtual std::tuple<Tensor, Tensor> get(Context& ctx, int seq, int32_t startPos, int32_t endPos) = 0;
    
    // Store cache
    virtual void put(Context& ctx, const Tensor& key, const Tensor& value) = 0;
    
    // Start forward propagation
    virtual void startForward(Context& ctx, const Batch& batch, bool reserve) = 0;
    
    // Copy prefix
    virtual void copyPrefix(Context& ctx, int srcSeq, int dstSeq, int32_t length) = 0;
    
    // Check if can resume
    virtual bool canResume(int seq, int32_t pos) const = 0;
    
    // Remove cache entries
    virtual void remove(int seq, int32_t beginIndex, int32_t endIndex) = 0;
    
    // Build output tensors
    virtual std::tuple<Tensor, Tensor, Tensor> buildOutputTensors(Context& ctx, const std::vector<int>& activeSeqs) = 0;
};

} // namespace kvcache
} // namespace duorou

#endif // DUOROU_KVCACHE_CACHE_H