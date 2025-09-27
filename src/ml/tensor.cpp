#include "tensor.h"
#include "backend/backend.h"
#include "context.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>
#include <random>

namespace duorou {
namespace ml {

// Constructor implementation
Tensor::Tensor() 
    : dtype_(DataType::FLOAT32), data_(nullptr), backend_(nullptr), ownsData_(false) {
}

Tensor::Tensor(const std::vector<int64_t>& shape, DataType dtype)
    : shape_(shape), dtype_(dtype), data_(nullptr), backend_(nullptr), ownsData_(false) {
    validateShape(shape);
}

Tensor::Tensor(std::initializer_list<int64_t> shape, DataType dtype)
    : Tensor(std::vector<int64_t>(shape), dtype) {
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), backend_(other.backend_), ownsData_(false) {
    if (other.data_ && other.numel() > 0) {
        allocate(backend_);
        copyFrom(other);
    } else {
        data_ = nullptr;
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), dtype_(other.dtype_), 
      data_(other.data_), backend_(other.backend_), ownsData_(other.ownsData_) {
    other.data_ = nullptr;
    other.ownsData_ = false;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate();
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        backend_ = other.backend_;
        ownsData_ = false;
        
        if (other.data_ && other.numel() > 0) {
            allocate(backend_);
            copyFrom(other);
        } else {
            data_ = nullptr;
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        data_ = other.data_;
        backend_ = other.backend_;
        ownsData_ = other.ownsData_;
        
        other.data_ = nullptr;
        other.ownsData_ = false;
    }
    return *this;
}

Tensor::~Tensor() {
    deallocate();
}

// Basic properties implementation
int64_t Tensor::dim(int index) const {
    if (index < 0) {
        index += static_cast<int>(shape_.size());
    }
    if (index < 0 || index >= static_cast<int>(shape_.size())) {
        throw std::out_of_range("Dimension index out of range");
    }
    return shape_[index];
}

int64_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int64_t>());
}

size_t Tensor::itemSize() const {
    return getDataTypeSize(dtype_);
}

size_t Tensor::nbytes() const {
    return numel() * itemSize();
}

// Memory management implementation
void Tensor::allocate(Backend* backend) {
    if (data_ && ownsData_) {
        deallocate();
    }
    
    if (backend) {
        backend_ = backend;
    }
    
    size_t bytes = nbytes();
    if (bytes > 0) {
        if (backend_) {
            data_ = backend_->allocate(bytes);
        } else {
            // Use malloc instead of aligned_alloc to avoid compatibility issues
            data_ = std::malloc(bytes);
            if (!data_) {
                throw std::runtime_error("Failed to allocate memory for tensor");
            }
        }
        ownsData_ = true;
    }
}

void Tensor::deallocate() {
    if (data_ && ownsData_) {
        if (backend_) {
            backend_->deallocate(data_);
        } else {
            std::free(data_);
        }
        data_ = nullptr;
        ownsData_ = false;
    }
}

// Data copying implementation
void Tensor::copyFrom(const Tensor& other) {
    if (numel() != other.numel()) {
        throw std::invalid_argument("Tensor sizes must match for copying");
    }
    
    if (!data_) {
        allocate(backend_);
    }
    
    size_t bytes = nbytes();
    if (backend_ && other.backend_) {
        backend_->copyDeviceToDevice(data_, other.data_, bytes);
    } else {
        std::memcpy(data_, other.data_, bytes);
    }
}

void Tensor::copyTo(Tensor& other) const {
    other.copyFrom(*this);
}

void Tensor::copyFromHost(const void* hostData, size_t bytes) {
    if (!data_) {
        allocate(backend_);
    }
    
    if (backend_) {
        backend_->copyToDevice(data_, hostData, bytes);
    } else {
        std::memcpy(data_, hostData, bytes);
    }
}

void Tensor::copyToHost(void* hostData, size_t bytes) const {
    if (!data_) {
        throw std::runtime_error("Tensor data is not allocated");
    }
    
    if (backend_) {
        backend_->copyFromDevice(hostData, data_, bytes);
    } else {
        std::memcpy(hostData, data_, bytes);
    }
}

bool Tensor::isValid() const {
    // A tensor is valid if it has a non-empty shape, a positive number of elements,
    // and its data buffer has been allocated.
    if (shape_.empty()) return false;
    if (numel() <= 0) return false;
    return data_ != nullptr;
}

// Static factory methods implementation
Tensor Tensor::zeros(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor tensor(shape, dtype);
    tensor.allocate();
    
    size_t bytes = tensor.nbytes();
    if (bytes > 0) {
        std::memset(tensor.data_, 0, bytes);
    }
    
    return tensor;
}

Tensor Tensor::ones(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor tensor(shape, dtype);
    tensor.allocate();
    
    // Fill with 1 based on data type
    int64_t numel = tensor.numel();
    switch (dtype) {
        case DataType::FLOAT32: {
            float* data = tensor.data<float>();
            std::fill(data, data + numel, 1.0f);
            break;
        }
        case DataType::INT32: {
            int32_t* data = tensor.data<int32_t>();
            std::fill(data, data + numel, 1);
            break;
        }
        // Other data types...
        default:
            throw std::runtime_error("Unsupported data type for ones");
    }
    
    return tensor;
}

Tensor Tensor::randn(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor tensor(shape, dtype);
    tensor.allocate();
    
    // Generate random numbers using normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    int64_t numel = tensor.numel();
    switch (dtype) {
        case DataType::FLOAT32: {
            float* data = tensor.data<float>();
            for (int64_t i = 0; i < numel; ++i) {
                data[i] = dis(gen);
            }
            break;
        }
        case DataType::INT32: {
            int32_t* data = tensor.data<int32_t>();
            for (int64_t i = 0; i < numel; ++i) {
                data[i] = static_cast<int32_t>(dis(gen));
            }
            break;
        }
        // Other data types...
        default:
            throw std::runtime_error("Unsupported data type for randn");
    }
    
    return tensor;
}

// Tensor operations implementation
Tensor Tensor::add(Context& /*ctx*/, const Tensor& other) const {
    if (dtype_ != DataType::FLOAT32 || other.dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("add: only FLOAT32 supported in current implementation");
    }
    // Compute broadcasted result shape (align from the right)
    const std::vector<int64_t>& aShape = shape_;
    const std::vector<int64_t>& bShape = other.shape_;
    int ndA = static_cast<int>(aShape.size());
    int ndB = static_cast<int>(bShape.size());
    int ndR = std::max(ndA, ndB);
    std::vector<int64_t> rShape(ndR, 1);
    for (int i = 0; i < ndR; ++i) {
        int aIdx = ndA - 1 - i;
        int bIdx = ndB - 1 - i;
        int64_t aDim = (aIdx >= 0 ? aShape[aIdx] : 1);
        int64_t bDim = (bIdx >= 0 ? bShape[bIdx] : 1);
        if (aDim != bDim && aDim != 1 && bDim != 1) {
            throw std::invalid_argument("add: shapes not broadcastable");
        }
        rShape[ndR - 1 - i] = std::max(aDim, bDim);
    }

    Tensor result(rShape, dtype_);
    result.allocate();

    auto computeStrides = [](const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    };

    // Align shapes by prepending 1s
    std::vector<int64_t> aAligned = aShape, bAligned = bShape;
    if (static_cast<int>(aAligned.size()) < ndR) aAligned.insert(aAligned.begin(), ndR - aAligned.size(), 1);
    if (static_cast<int>(bAligned.size()) < ndR) bAligned.insert(bAligned.begin(), ndR - bAligned.size(), 1);

    auto aStrides = computeStrides(aAligned);
    auto bStrides = computeStrides(bAligned);
    auto rStrides = computeStrides(rShape);

    const float* aData = static_cast<const float*>(data_);
    const float* bData = static_cast<const float*>(other.data_);
    float* out = static_cast<float*>(result.data_);

    int64_t total = result.numel();
    for (int64_t linear = 0; linear < total; ++linear) {
        // Decompose linear index into multi-index for result
        int64_t tmp = linear;
        int64_t aOffset = 0;
        int64_t bOffset = 0;
        for (int d = 0; d < ndR; ++d) {
            int64_t idx = tmp / rStrides[d];
            tmp = tmp % rStrides[d];
            int64_t aIdx = (aAligned[d] == 1) ? 0 : idx;
            int64_t bIdx = (bAligned[d] == 1) ? 0 : idx;
            aOffset += aIdx * aStrides[d];
            bOffset += bIdx * bStrides[d];
        }
        out[linear] = aData[aOffset] + bData[bOffset];
    }

    return result;
}

Tensor Tensor::sub(Context& /*ctx*/, const Tensor& other) const {
    if (dtype_ != DataType::FLOAT32 || other.dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("sub: only FLOAT32 supported in current implementation");
    }
    // Reuse add's broadcasting logic
    const std::vector<int64_t>& aShape = shape_;
    const std::vector<int64_t>& bShape = other.shape_;
    int ndA = static_cast<int>(aShape.size());
    int ndB = static_cast<int>(bShape.size());
    int ndR = std::max(ndA, ndB);
    std::vector<int64_t> rShape(ndR, 1);
    for (int i = 0; i < ndR; ++i) {
        int aIdx = ndA - 1 - i;
        int bIdx = ndB - 1 - i;
        int64_t aDim = (aIdx >= 0 ? aShape[aIdx] : 1);
        int64_t bDim = (bIdx >= 0 ? bShape[bIdx] : 1);
        if (aDim != bDim && aDim != 1 && bDim != 1) {
            throw std::invalid_argument("sub: shapes not broadcastable");
        }
        rShape[ndR - 1 - i] = std::max(aDim, bDim);
    }

    Tensor result(rShape, dtype_);
    result.allocate();

    auto computeStrides = [](const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    };

    std::vector<int64_t> aAligned = aShape, bAligned = bShape;
    if (static_cast<int>(aAligned.size()) < ndR) aAligned.insert(aAligned.begin(), ndR - aAligned.size(), 1);
    if (static_cast<int>(bAligned.size()) < ndR) bAligned.insert(bAligned.begin(), ndR - bAligned.size(), 1);

    auto aStrides = computeStrides(aAligned);
    auto bStrides = computeStrides(bAligned);
    auto rStrides = computeStrides(rShape);

    const float* aData = static_cast<const float*>(data_);
    const float* bData = static_cast<const float*>(other.data_);
    float* out = static_cast<float*>(result.data_);

    int64_t total = result.numel();
    for (int64_t linear = 0; linear < total; ++linear) {
        int64_t tmp = linear;
        int64_t aOffset = 0;
        int64_t bOffset = 0;
        for (int d = 0; d < ndR; ++d) {
            int64_t idx = tmp / rStrides[d];
            tmp = tmp % rStrides[d];
            int64_t aIdx = (aAligned[d] == 1) ? 0 : idx;
            int64_t bIdx = (bAligned[d] == 1) ? 0 : idx;
            aOffset += aIdx * aStrides[d];
            bOffset += bIdx * bStrides[d];
        }
        out[linear] = aData[aOffset] - bData[bOffset];
    }

    return result;
}

Tensor Tensor::mul(Context& /*ctx*/, const Tensor& other) const {
    if (dtype_ != DataType::FLOAT32 || other.dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("mul: only FLOAT32 supported in current implementation");
    }
    // Broadcasting logic similar to add
    const std::vector<int64_t>& aShape = shape_;
    const std::vector<int64_t>& bShape = other.shape_;
    int ndA = static_cast<int>(aShape.size());
    int ndB = static_cast<int>(bShape.size());
    int ndR = std::max(ndA, ndB);
    std::vector<int64_t> rShape(ndR, 1);
    for (int i = 0; i < ndR; ++i) {
        int aIdx = ndA - 1 - i;
        int bIdx = ndB - 1 - i;
        int64_t aDim = (aIdx >= 0 ? aShape[aIdx] : 1);
        int64_t bDim = (bIdx >= 0 ? bShape[bIdx] : 1);
        if (aDim != bDim && aDim != 1 && bDim != 1) {
            throw std::invalid_argument("mul: shapes not broadcastable");
        }
        rShape[ndR - 1 - i] = std::max(aDim, bDim);
    }

    Tensor result(rShape, dtype_);
    result.allocate();

    auto computeStrides = [](const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    };

    std::vector<int64_t> aAligned = aShape, bAligned = bShape;
    if (static_cast<int>(aAligned.size()) < ndR) aAligned.insert(aAligned.begin(), ndR - aAligned.size(), 1);
    if (static_cast<int>(bAligned.size()) < ndR) bAligned.insert(bAligned.begin(), ndR - bAligned.size(), 1);

    auto aStrides = computeStrides(aAligned);
    auto bStrides = computeStrides(bAligned);
    auto rStrides = computeStrides(rShape);

    const float* aData = static_cast<const float*>(data_);
    const float* bData = static_cast<const float*>(other.data_);
    float* out = static_cast<float*>(result.data_);

    int64_t total = result.numel();
    for (int64_t linear = 0; linear < total; ++linear) {
        int64_t tmp = linear;
        int64_t aOffset = 0;
        int64_t bOffset = 0;
        for (int d = 0; d < ndR; ++d) {
            int64_t idx = tmp / rStrides[d];
            tmp = tmp % rStrides[d];
            int64_t aIdx = (aAligned[d] == 1) ? 0 : idx;
            int64_t bIdx = (bAligned[d] == 1) ? 0 : idx;
            aOffset += aIdx * aStrides[d];
            bOffset += bIdx * bStrides[d];
        }
        out[linear] = aData[aOffset] * bData[bOffset];
    }

    return result;
}

Tensor Tensor::div(Context& /*ctx*/, const Tensor& other) const {
    if (dtype_ != DataType::FLOAT32 || other.dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("div: only FLOAT32 supported in current implementation");
    }
    // Broadcasting logic similar to add
    const std::vector<int64_t>& aShape = shape_;
    const std::vector<int64_t>& bShape = other.shape_;
    int ndA = static_cast<int>(aShape.size());
    int ndB = static_cast<int>(bShape.size());
    int ndR = std::max(ndA, ndB);
    std::vector<int64_t> rShape(ndR, 1);
    for (int i = 0; i < ndR; ++i) {
        int aIdx = ndA - 1 - i;
        int bIdx = ndB - 1 - i;
        int64_t aDim = (aIdx >= 0 ? aShape[aIdx] : 1);
        int64_t bDim = (bIdx >= 0 ? bShape[bIdx] : 1);
        if (aDim != bDim && aDim != 1 && bDim != 1) {
            throw std::invalid_argument("div: shapes not broadcastable");
        }
        rShape[ndR - 1 - i] = std::max(aDim, bDim);
    }

    Tensor result(rShape, dtype_);
    result.allocate();

    auto computeStrides = [](const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    };

    std::vector<int64_t> aAligned = aShape, bAligned = bShape;
    if (static_cast<int>(aAligned.size()) < ndR) aAligned.insert(aAligned.begin(), ndR - aAligned.size(), 1);
    if (static_cast<int>(bAligned.size()) < ndR) bAligned.insert(bAligned.begin(), ndR - bAligned.size(), 1);

    auto aStrides = computeStrides(aAligned);
    auto bStrides = computeStrides(bAligned);
    auto rStrides = computeStrides(rShape);

    const float* aData = static_cast<const float*>(data_);
    const float* bData = static_cast<const float*>(other.data_);
    float* out = static_cast<float*>(result.data_);

    int64_t total = result.numel();
    for (int64_t linear = 0; linear < total; ++linear) {
        int64_t tmp = linear;
        int64_t aOffset = 0;
        int64_t bOffset = 0;
        for (int d = 0; d < ndR; ++d) {
            int64_t idx = tmp / rStrides[d];
            tmp = tmp % rStrides[d];
            int64_t aIdx = (aAligned[d] == 1) ? 0 : idx;
            int64_t bIdx = (bAligned[d] == 1) ? 0 : idx;
            aOffset += aIdx * aStrides[d];
            bOffset += bIdx * bStrides[d];
        }
        out[linear] = aData[aOffset] / bData[bOffset];
    }

    return result;
}

Tensor Tensor::matmul(Context& /*ctx*/, const Tensor& other) const {
    if (dtype_ != DataType::FLOAT32 || other.dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("matmul: only FLOAT32 supported in current implementation");
    }
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("matmul dimension mismatch");
    }

    const int64_t M = shape_[0];
    const int64_t K = shape_[1];
    const int64_t N = other.shape_[1];

    // Ensure inputs are allocated
    if (!data_ || !other.data_) {
        throw std::runtime_error("matmul: input tensors must have allocated data");
    }

    Tensor result({M, N}, dtype_);
    // Keep allocation on the same backend as lhs when available
    result.setBackend(backend_);
    result.allocate();

    const float* A = static_cast<const float*>(data_);
    const float* B = static_cast<const float*>(other.data_);
    float* C = static_cast<float*>(result.data_);

    // Naive row-major GEMM: C[M,N] = A[M,K] x B[K,N]
    // Initialize output to 0
    std::fill(C, C + (M * N), 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        const float* Ai = A + i * K;
        float* Ci = C + i * N;
        for (int64_t k = 0; k < K; ++k) {
            const float a = Ai[k];
            const float* Bk = B + k * N;
            for (int64_t j = 0; j < N; ++j) {
                Ci[j] += a * Bk[j];
            }
        }
    }

    return result;
}

// Activation functions implementation
Tensor Tensor::relu(Context& /*ctx*/) const {
    Tensor result(shape_, dtype_);
    result.allocate();
    return result;
}

Tensor Tensor::sigmoid(Context& /*ctx*/) const {
    Tensor result(shape_, dtype_);
    result.allocate();
    return result;
}

Tensor Tensor::tanh(Context& /*ctx*/) const {
    Tensor result(shape_, dtype_);
    result.allocate();
    return result;
}

Tensor Tensor::softmax(Context& /*ctx*/, int /*dim*/) const {
    Tensor result(shape_, dtype_);
    result.allocate();
    return result;
}

// Shape operations implementation
Tensor Tensor::reshape(const std::vector<int64_t>& newShape) const {
    Tensor result(newShape, dtype_);
    result.data_ = data_;
    result.backend_ = backend_;
    result.ownsData_ = false;
    return result;
}

Tensor Tensor::view(const std::vector<int64_t>& newShape) const {
    return reshape(newShape);
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    // Normalize dimensions
    int nd = static_cast<int>(shape_.size());
    if (nd == 0) {
        return Tensor({}, dtype_);
    }
    if (dim0 < 0) dim0 += nd;
    if (dim1 < 0) dim1 += nd;
    if (dim0 < 0 || dim0 >= nd || dim1 < 0 || dim1 >= nd) {
        throw std::out_of_range("transpose: dimension out of range");
    }
    if (!data_) {
        throw std::runtime_error("transpose: input tensor has no data");
    }

    // New shape after swapping dim0 and dim1
    std::vector<int64_t> newShape = shape_;
    std::swap(newShape[dim0], newShape[dim1]);

    Tensor result(newShape, dtype_);
    result.setBackend(backend_);
    result.allocate();

    // Compute strides for original and result (row-major)
    auto computeStrides = [](const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    };

    const std::vector<int64_t> srcStrides = computeStrides(shape_);
    const std::vector<int64_t> dstStrides = computeStrides(newShape);

    // Copy data with index mapping
    const int64_t total = result.numel();
    const float* src = static_cast<const float*>(data_);
    float* dst = static_cast<float*>(result.data_);

    for (int64_t linear = 0; linear < total; ++linear) {
        // Decode destination indices
        int64_t tmp = linear;
        std::vector<int64_t> rIdx(nd, 0);
        for (int d = 0; d < nd; ++d) {
            rIdx[d] = tmp / dstStrides[d];
            tmp %= dstStrides[d];
        }
        // Map to source indices (swap dim0 and dim1 back)
        std::vector<int64_t> sIdx = rIdx;
        std::swap(sIdx[dim0], sIdx[dim1]);
        // Compute source offset
        int64_t sOff = 0;
        for (int d = 0; d < nd; ++d) {
            sOff += sIdx[d] * srcStrides[d];
        }
        dst[linear] = src[sOff];
    }

    return result;
}

Tensor Tensor::permute(const std::vector<int>& dims) const {
    int nd = static_cast<int>(shape_.size());
    if (dims.empty()) {
        throw std::invalid_argument("permute: dims must not be empty");
    }
    if (static_cast<int>(dims.size()) != nd) {
        throw std::invalid_argument("permute: dims size must equal tensor ndim");
    }
    // Validate dims are a permutation of [0..nd-1]
    std::vector<int> seen(nd, 0);
    for (int d : dims) {
        int dd = d < 0 ? d + nd : d;
        if (dd < 0 || dd >= nd || seen[dd]) {
            throw std::invalid_argument("permute: invalid dims permutation");
        }
        seen[dd] = 1;
    }
    if (!data_) {
        throw std::runtime_error("permute: input tensor has no data");
    }

    // Compute new shape
    std::vector<int64_t> newShape(nd);
    for (int i = 0; i < nd; ++i) {
        int srcDim = dims[i] < 0 ? dims[i] + nd : dims[i];
        newShape[i] = shape_[srcDim];
    }

    Tensor result(newShape, dtype_);
    result.setBackend(backend_);
    result.allocate();

    // Strides
    auto computeStrides = [](const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    };

    const std::vector<int64_t> srcStrides = computeStrides(shape_);
    const std::vector<int64_t> dstStrides = computeStrides(newShape);

    const int64_t total = result.numel();
    const float* src = static_cast<const float*>(data_);
    float* dst = static_cast<float*>(result.data_);

    for (int64_t linear = 0; linear < total; ++linear) {
        // Decode destination indices
        int64_t tmp = linear;
        std::vector<int64_t> rIdx(nd, 0);
        for (int d = 0; d < nd; ++d) {
            rIdx[d] = tmp / dstStrides[d];
            tmp %= dstStrides[d];
        }
        // Map to source indices: oldDim = dims[i], oldIdx[oldDim] = rIdx[i]
        std::vector<int64_t> sIdx(nd, 0);
        for (int i = 0; i < nd; ++i) {
            int oldDim = dims[i] < 0 ? dims[i] + nd : dims[i];
            sIdx[oldDim] = rIdx[i];
        }
        int64_t sOff = 0;
        for (int d = 0; d < nd; ++d) {
            sOff += sIdx[d] * srcStrides[d];
        }
        dst[linear] = src[sOff];
    }

    return result;
}

// Reduction operations implementation
Tensor Tensor::sum(Context& /*ctx*/, int dim, bool keepdim) const {
    if (dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("sum: only FLOAT32 supported in current implementation");
    }
    int nd = static_cast<int>(shape_.size());
    if (nd == 0) {
        Tensor result({1}, dtype_);
        result.allocate();
        float* out = static_cast<float*>(result.data_);
        out[0] = 0.0f;
        return result;
    }
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd) {
        throw std::invalid_argument("sum: dim out of range");
    }

    // Build result shape
    std::vector<int64_t> rShape = shape_;
    if (keepdim) {
        rShape[dim] = 1;
    } else {
        rShape.erase(rShape.begin() + dim);
    }
    Tensor result(rShape, dtype_);
    result.allocate();

    auto computeStrides = [](const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    };

    auto inStrides = computeStrides(shape_);
    auto rStrides = computeStrides(rShape);

    const float* in = static_cast<const float*>(data_);
    float* out = static_cast<float*>(result.data_);
    // Zero initialize output
    std::fill(out, out + result.numel(), 0.0f);

    // Iterate over all input elements and accumulate into reduced output
    int64_t inTotal = numel();
    for (int64_t linear = 0; linear < inTotal; ++linear) {
        int64_t tmp = linear;
        int64_t outOffset = 0;
        for (int d = 0, rd = 0; d < nd; ++d) {
            int64_t idx = tmp / inStrides[d];
            tmp = tmp % inStrides[d];
            if (d == dim) {
                continue; // reduced dimension
            }
            int64_t rIdx = keepdim ? (d < dim ? rd : rd + 1) : rd; // rd maps to result dims
            // Compute offset contribution for output
            outOffset += idx * rStrides[rIdx];
            rd++;
        }
        out[outOffset] += in[linear];
    }

    return result;
}

Tensor Tensor::mean(Context& ctx, int dim, bool keepdim) const {
    // mean = sum / size_along_dim
    Tensor s = sum(ctx, dim, keepdim);
    int nd = static_cast<int>(shape_.size());
    if (dim < 0) dim += nd;
    int64_t count = shape_[dim];
    if (s.dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("mean: only FLOAT32 supported in current implementation");
    }
    Tensor result(s.shape_, s.dtype_);
    result.allocate();
    const float* src = static_cast<const float*>(s.data_);
    float* dst = static_cast<float*>(result.data_);
    float inv = 1.0f / static_cast<float>(count);
    for (int64_t i = 0; i < s.numel(); ++i) {
        dst[i] = src[i] * inv;
    }
    return result;
}

Tensor Tensor::max(Context& ctx, int dim, bool keepdim) const {
    return sum(ctx, dim, keepdim);
}

Tensor Tensor::min(Context& ctx, int dim, bool keepdim) const {
    return sum(ctx, dim, keepdim);
}

// Indexing operations implementation
Tensor Tensor::slice(int /*dim*/, int64_t /*start*/, int64_t /*end*/, int64_t /*step*/) const {
    Tensor result(shape_, dtype_);
    result.allocate();
    return result;
}

Tensor Tensor::index(const std::vector<int64_t>& /*indices*/) const {
    Tensor result(shape_, dtype_);
    result.allocate();
    return result;
}

// String representation
std::string Tensor::toString() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], dtype=" << dataTypeToString(dtype_) << ")";
    return oss.str();
}

void Tensor::print() const {
    std::cout << toString() << std::endl;
}

// Helper methods implementation
void Tensor::validateShape(const std::vector<int64_t>& shape) const {
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("Shape dimensions must be non-negative");
        }
    }
}

size_t Tensor::getDataTypeSize(DataType dtype) const {
    switch (dtype) {
        case DataType::FLOAT32: return sizeof(float);
        case DataType::FLOAT16: return sizeof(uint16_t);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::INT16: return sizeof(int16_t);
        case DataType::INT8: return sizeof(int8_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BOOL: return sizeof(bool);
        default: return 0;
    }
}

// Utility functions implementation
std::string dataTypeToString(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "float32";
        case DataType::FLOAT16: return "float16";
        case DataType::INT32: return "int32";
        case DataType::INT16: return "int16";
        case DataType::INT8: return "int8";
        case DataType::UINT8: return "uint8";
        case DataType::BOOL: return "bool";
        default: return "unknown";
    }
}

DataType stringToDataType(const std::string& dtypeStr) {
    if (dtypeStr == "float32") return DataType::FLOAT32;
    if (dtypeStr == "float16") return DataType::FLOAT16;
    if (dtypeStr == "int32") return DataType::INT32;
    if (dtypeStr == "int16") return DataType::INT16;
    if (dtypeStr == "int8") return DataType::INT8;
    if (dtypeStr == "uint8") return DataType::UINT8;
    if (dtypeStr == "bool") return DataType::BOOL;
    return DataType::FLOAT32; // Default
}

} // namespace ml
} // namespace duorou