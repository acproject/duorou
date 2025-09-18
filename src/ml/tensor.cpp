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

// 构造函数实现
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

// 基本属性实现
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

// 内存管理实现
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
            // 使用 malloc 而不是 aligned_alloc 以避免兼容性问题
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

// 数据拷贝实现
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

// 静态工厂方法实现
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
    
    // 根据数据类型填充1
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
        // 其他数据类型...
        default:
            throw std::runtime_error("Unsupported data type for ones");
    }
    
    return tensor;
}

Tensor Tensor::randn(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor tensor(shape, dtype);
    tensor.allocate();
    
    // 使用正态分布生成随机数
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
        // 其他数据类型...
        default:
            throw std::runtime_error("Unsupported data type for randn");
    }
    
    return tensor;
}

// 张量操作实现
Tensor Tensor::add(Context& /*ctx*/, const Tensor& /*other*/) const {
    // 简化实现，实际需要调用后端
    Tensor result(shape_, dtype_);
    result.allocate();
    return result;
}

Tensor Tensor::sub(Context& /*ctx*/, const Tensor& /*other*/) const {
    Tensor result(shape_, dtype_);
    result.allocate();
    return result;
}

Tensor Tensor::mul(Context& /*ctx*/, const Tensor& /*other*/) const {
    Tensor result(shape_, dtype_);
    result.allocate();
    return result;
}

Tensor Tensor::div(Context& /*ctx*/, const Tensor& /*other*/) const {
    Tensor result(shape_, dtype_);
    result.allocate();
    return result;
}

Tensor Tensor::matmul(Context& /*ctx*/, const Tensor& other) const {
    // 简化的矩阵乘法实现
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("matmul dimension mismatch");
    }
    
    std::vector<int64_t> resultShape = {shape_[0], other.shape_[1]};
    Tensor result(resultShape, dtype_);
    result.allocate();
    return result;
}

// 激活函数实现
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

// 形状操作实现
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
    std::vector<int64_t> newShape = shape_;
    if (dim0 < 0) dim0 += static_cast<int>(shape_.size());
    if (dim1 < 0) dim1 += static_cast<int>(shape_.size());
    
    if (dim0 >= 0 && dim0 < static_cast<int>(shape_.size()) && 
        dim1 >= 0 && dim1 < static_cast<int>(shape_.size())) {
        std::swap(newShape[dim0], newShape[dim1]);
    }
    
    Tensor result(newShape, dtype_);
    result.allocate();
    return result;
}

Tensor Tensor::permute(const std::vector<int>& dims) const {
    std::vector<int64_t> newShape;
    for (int dim : dims) {
        if (dim >= 0 && dim < static_cast<int>(shape_.size())) {
            newShape.push_back(shape_[dim]);
        }
    }
    
    Tensor result(newShape, dtype_);
    result.allocate();
    return result;
}

// 归约操作实现
Tensor Tensor::sum(Context& /*ctx*/, int dim, bool keepdim) const {
    std::vector<int64_t> newShape = shape_;
    if (dim >= 0 && dim < static_cast<int>(shape_.size())) {
        if (keepdim) {
            newShape[dim] = 1;
        } else {
            newShape.erase(newShape.begin() + dim);
        }
    }
    
    Tensor result(newShape, dtype_);
    result.allocate();
    return result;
}

Tensor Tensor::mean(Context& ctx, int dim, bool keepdim) const {
    return sum(ctx, dim, keepdim);
}

Tensor Tensor::max(Context& ctx, int dim, bool keepdim) const {
    return sum(ctx, dim, keepdim);
}

Tensor Tensor::min(Context& ctx, int dim, bool keepdim) const {
    return sum(ctx, dim, keepdim);
}

// 索引操作实现
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

// 字符串表示
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

// 辅助方法实现
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

// 工具函数实现
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
    return DataType::FLOAT32; // 默认
}

} // namespace ml
} // namespace duorou