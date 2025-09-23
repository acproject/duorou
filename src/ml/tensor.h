#ifndef DUOROU_ML_TENSOR_H
#define DUOROU_ML_TENSOR_H

#include "context.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <vector>
#include <memory>
#include <string>
#include <initializer_list>

namespace duorou {
namespace ml {

// Forward declarations
class Backend;
class Context;

// Data types supported by tensors
enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT16,
    INT8,
    UINT8,
    BOOL
};

// Main tensor class
class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32);
    Tensor(std::initializer_list<int64_t> shape, DataType dtype = DataType::FLOAT32);
    
    // Copy and move semantics
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    ~Tensor();
    
    // Shape and type access
    const std::vector<int64_t>& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    int64_t size(int dim) const;
    int64_t dim(int index) const;
    void* data() const { return data_; }
    int64_t numel() const;
    size_t itemSize() const;
    size_t nbytes() const;
    
    // Data access
    template<typename T>
    T* data() const { return static_cast<T*>(data_); }
    
    template<typename T>
    T& at(const std::vector<int64_t>& indices);
    
    // Shape operations
    Tensor reshape(const std::vector<int64_t>& newShape) const;
    Tensor view(const std::vector<int64_t>& newShape) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor permute(const std::vector<int>& dims) const;
    
    // Mathematical operations
    Tensor add(Context& ctx, const Tensor& other) const;
    Tensor sub(Context& ctx, const Tensor& other) const;
    Tensor mul(Context& ctx, const Tensor& other) const;
    Tensor div(Context& ctx, const Tensor& other) const;
    Tensor matmul(Context& ctx, const Tensor& other) const;
    
    // Activation functions
    Tensor relu(Context& ctx) const;
    Tensor sigmoid(Context& ctx) const;
    Tensor tanh(Context& ctx) const;
    Tensor softmax(Context& ctx, int dim = -1) const;
    
    // Reduction operations
    Tensor sum(Context& ctx, int dim = -1, bool keepdim = false) const;
    Tensor mean(Context& ctx, int dim = -1, bool keepdim = false) const;
    Tensor (max)(Context& ctx, int dim = -1, bool keepdim = false) const;
    Tensor (min)(Context& ctx, int dim = -1, bool keepdim = false) const;
    
    // Indexing and slicing
    Tensor slice(int dim, int64_t start, int64_t end, int64_t step = 1) const;
    Tensor index(const std::vector<int64_t>& indices) const;
    
    // Memory management
    void allocate(Backend* backend = nullptr);
    void deallocate();
    
    // Backend access
    Backend* backend() const { return backend_; }
    void setBackend(Backend* backend) { backend_ = backend; }
    
    // Data copying
    void copyFrom(const Tensor& other);
    void copyTo(Tensor& other) const;
    void copyFromHost(const void* hostData, size_t bytes);
    void copyToHost(void* hostData, size_t bytes) const;
    
    // Validation and utilities
    bool isValid() const;
    bool isContiguous() const { return true; } // Simplified for now
    
    // String representation
    std::string toString() const;
    void print() const;
    
    // Static factory methods
    static Tensor zeros(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32);
    static Tensor ones(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32);
    static Tensor randn(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32);
    static Tensor arange(int64_t start, int64_t end, int64_t step = 1, DataType dtype = DataType::FLOAT32);
    
private:
    std::vector<int64_t> shape_;
    DataType dtype_;
    void* data_;
    Backend* backend_;
    bool ownsData_;
    
    // Helper methods
    int64_t calculateStride(int dim) const;
    void validateShape(const std::vector<int64_t>& shape) const;
    size_t getDataTypeSize(DataType dtype) const;
};

// Utility functions
std::string dataTypeToString(DataType dtype);
DataType stringToDataType(const std::string& dtypeStr);

} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_TENSOR_H