#ifndef DUOROU_ML_TENSOR_H
#define DUOROU_ML_TENSOR_H

#include <vector>
#include <memory>
#include <string>
#include <initializer_list>

namespace duorou {
namespace ml {

// 前向声明
class Backend;
class Context;

// 数据类型枚举
enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT16,
    INT8,
    UINT8,
    BOOL
};

// Tensor类
class Tensor {
public:
    // 构造函数
    Tensor();
    Tensor(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32);
    Tensor(std::initializer_list<int64_t> shape, DataType dtype = DataType::FLOAT32);
    
    // 拷贝和移动构造
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    ~Tensor();
    
    // 基本属性
    const std::vector<int64_t>& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    int64_t dim(int index) const;
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const;
    size_t itemSize() const;
    size_t nbytes() const;
    
    // 数据访问
    void* data() const { return data_; }
    template<typename T> T* data() const { return static_cast<T*>(data_); }
    
    // 形状操作
    Tensor reshape(const std::vector<int64_t>& newShape) const;
    Tensor view(const std::vector<int64_t>& newShape) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor permute(const std::vector<int>& dims) const;
    
    // 数学运算
    Tensor add(Context& ctx, const Tensor& other) const;
    Tensor sub(Context& ctx, const Tensor& other) const;
    Tensor mul(Context& ctx, const Tensor& other) const;
    Tensor div(Context& ctx, const Tensor& other) const;
    Tensor matmul(Context& ctx, const Tensor& other) const;
    
    // 激活函数
    Tensor relu(Context& ctx) const;
    Tensor sigmoid(Context& ctx) const;
    Tensor tanh(Context& ctx) const;
    Tensor softmax(Context& ctx, int dim = -1) const;
    
    // 归约操作
    Tensor sum(Context& ctx, int dim = -1, bool keepdim = false) const;
    Tensor mean(Context& ctx, int dim = -1, bool keepdim = false) const;
    Tensor max(Context& ctx, int dim = -1, bool keepdim = false) const;
    Tensor min(Context& ctx, int dim = -1, bool keepdim = false) const;
    
    // 索引和切片
    Tensor slice(int dim, int64_t start, int64_t end, int64_t step = 1) const;
    Tensor index(const std::vector<int64_t>& indices) const;
    
    // 内存管理
    void allocate(Backend* backend = nullptr);
    void deallocate();
    bool isAllocated() const { return data_ != nullptr; }
    
    // 数据拷贝
    void copyFrom(const Tensor& other);
    void copyTo(Tensor& other) const;
    void copyFromHost(const void* hostData, size_t bytes);
    void copyToHost(void* hostData, size_t bytes) const;
    
    // 设备管理
    void setBackend(Backend* backend) { backend_ = backend; }
    Backend* getBackend() const { return backend_; }
    
    // 调试和打印
    std::string toString() const;
    void print() const;
    
    // 静态工厂方法
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
    
    // 辅助方法
    int64_t calculateStride(int dim) const;
    void validateShape(const std::vector<int64_t>& shape) const;
    size_t getDataTypeSize(DataType dtype) const;
};

// 工具函数
std::string dataTypeToString(DataType dtype);
DataType stringToDataType(const std::string& dtypeStr);

} // namespace ml
} // namespace duorou

#endif // DUOROU_ML_TENSOR_H