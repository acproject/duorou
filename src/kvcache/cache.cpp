#include "cache.h"
#include <cstring>
#include <cstdlib>

namespace duorou {
namespace kvcache {

// Tensor类实现
Tensor::Tensor(const std::vector<int>& shape, DType dtype) 
    : shape_(shape), dtype_(dtype), data_(nullptr), size_(0) {
    
    // 计算总元素数量
    size_t totalElements = 1;
    for (int dim : shape) {
        totalElements *= dim;
    }
    
    // 根据数据类型计算字节大小
    size_t elementSize = 0;
    switch (dtype) {
        case DType::FLOAT32:
            elementSize = 4;
            break;
        case DType::FLOAT16:
            elementSize = 2;
            break;
        case DType::INT32:
            elementSize = 4;
            break;
        case DType::INT64:
            elementSize = 8;
            break;
    }
    
    size_ = totalElements * elementSize;
    
    // 分配内存
    if (size_ > 0) {
        data_ = std::malloc(size_);
        if (data_) {
            std::memset(data_, 0, size_);
        }
    }
}

Tensor::~Tensor() {
    if (data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

size_t Tensor::totalElements() const {
    size_t total = 1;
    for (int dim : shape_) {
        total *= dim;
    }
    return total;
}

size_t Tensor::bytesSize() const {
    return size_;
}

} // namespace kvcache
} // namespace duorou