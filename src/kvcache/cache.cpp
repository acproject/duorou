#include "cache.h"
#include <cstring>
#include <cstdlib>

namespace duorou {
namespace kvcache {

// Tensor class implementation
Tensor::Tensor(const std::vector<int>& shape, DType dtype) 
    : shape_(shape), dtype_(dtype), data_(nullptr), size_(0), backend_(nullptr) {
    
    // Calculate total number of elements
    size_t totalElements = 1;
    for (int dim : shape) {
        totalElements *= dim;
    }
    
    // Calculate byte size based on data type
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
    
    // Allocate memory
    if (size_ > 0) {
        data_ = std::malloc(size_);
        if (data_) {
            std::memset(data_, 0, size_);
        }
    }
}

Tensor::Tensor(const std::vector<int>& shape, DType dtype, Backend* backend)
    : shape_(shape), dtype_(dtype), data_(nullptr), size_(0), backend_(backend) {
    // Calculate total number of elements
    size_t totalElements = 1;
    for (int dim : shape) {
        totalElements *= dim;
    }

    // Calculate byte size based on data type
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

    // Allocate using backend if provided, otherwise fallback to malloc
    if (size_ > 0) {
        if (backend_) {
            data_ = backend_->allocate(size_);
            if (data_) {
                std::memset(data_, 0, size_);
            }
        } else {
            data_ = std::malloc(size_);
            if (data_) {
                std::memset(data_, 0, size_);
            }
        }
    }
}

// Added copy/move semantics to avoid double-free and dangling pointers
Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), data_(nullptr), size_(other.size_), backend_(other.backend_) {
    if (size_ > 0) {
        if (backend_) {
            data_ = backend_->allocate(size_);
        } else {
            data_ = std::malloc(size_);
        }
        if (!data_) {
            throw OutOfMemoryError("Tensor copy constructor: allocation failed");
        }
        if (other.data_) {
            std::memcpy(data_, other.data_, size_);
        } else {
            std::memset(data_, 0, size_);
        }
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), dtype_(other.dtype_), data_(other.data_), size_(other.size_), backend_(other.backend_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.backend_ = nullptr;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        // First deallocate current data
        if (data_) {
            if (backend_) {
                backend_->deallocate(data_);
            } else {
                std::free(data_);
            }
            data_ = nullptr;
        }
        // Copy metadata
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        size_ = other.size_;
        backend_ = other.backend_;
        // Allocate and copy data
        if (size_ > 0) {
            if (backend_) {
                data_ = backend_->allocate(size_);
            } else {
                data_ = std::malloc(size_);
            }
            if (!data_) {
                throw OutOfMemoryError("Tensor copy assignment: allocation failed");
            }
            if (other.data_) {
                std::memcpy(data_, other.data_, size_);
            } else {
                std::memset(data_, 0, size_);
            }
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Deallocate current data
        if (data_) {
            if (backend_) {
                backend_->deallocate(data_);
            } else {
                std::free(data_);
            }
        }
        // Move metadata and pointer
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        data_ = other.data_;
        size_ = other.size_;
        backend_ = other.backend_;
        // Null out other's pointer
        other.data_ = nullptr;
        other.size_ = 0;
        other.backend_ = nullptr;
    }
    return *this;
}

Tensor::~Tensor() {
    if (data_) {
        if (backend_) {
            backend_->deallocate(data_);
        } else {
            std::free(data_);
        }
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