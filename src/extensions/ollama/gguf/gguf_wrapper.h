#ifndef DUOROU_GGUF_WRAPPER_H
#define DUOROU_GGUF_WRAPPER_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <stdint.h>

namespace duorou {
namespace extensions {
namespace ollama {
namespace gguf {

// GGUF数据类型枚举
enum ValueType {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

// 张量类型枚举
enum TensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    BF16 = 21
};

// 简化的值类型，避免使用std::variant
class Value {
public:
    Value();
    Value(const std::string& str);
    Value(int32_t val);
    Value(uint32_t val);
    Value(int64_t val);
    Value(uint64_t val);
    Value(float val);
    Value(double val);
    Value(bool val);
    Value(const std::vector<std::string>& vals);
    Value(const std::vector<int64_t>& vals);
    
    ~Value();
    Value(const Value& other);
    Value& operator=(const Value& other);
    
    ValueType getType() const;
    
    int64_t asInt() const;
    uint64_t asUint() const;
    double asFloat() const;
    bool asBool() const;
    std::string asString() const;
    std::vector<int64_t> asInts() const;
    std::vector<std::string> asStrings() const;

private:
    ValueType type_;
    void* data_;
    
    void cleanup();
    void copyFrom(const Value& other);
};

// 键值对结构
struct KeyValue {
    std::string key;
    Value value;
    
    KeyValue();
    KeyValue(const std::string& k, const Value& v);
    
    bool valid() const;
    
    // 便捷访问方法
    int64_t asInt() const { return value.asInt(); }
    uint64_t asUint() const { return value.asUint(); }
    double asFloat() const { return value.asFloat(); }
    bool asBool() const { return value.asBool(); }
    std::string asString() const { return value.asString(); }
    std::vector<int64_t> asInts() const { return value.asInts(); }
    std::vector<std::string> asStrings() const { return value.asStrings(); }
};

// 张量信息结构
struct TensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    TensorType type;
    uint64_t offset;
    
    TensorInfo();
    
    bool valid() const;
    int64_t numValues() const;
    int64_t numBytes() const;
    double getBytesPerValue() const;
};

// 缓冲读取器
class BufferedReader {
public:
    explicit BufferedReader(const std::string& filename, size_t bufferSize = 8192);
    ~BufferedReader();
    
    // 禁用拷贝，允许移动
    BufferedReader(BufferedReader& other);
    BufferedReader& operator=(BufferedReader& other);
    BufferedReader(BufferedReader&& other);
    BufferedReader& operator=(BufferedReader&& other);
    
    bool isOpen() const;
    size_t read(void* buffer, size_t size);
    bool seek(int64_t offset);
    int64_t tell() const;
    int64_t size() const;

private:
    class Impl;
    Impl* impl_;
};

// 延迟加载模板类
template<typename T>
class Lazy {
public:
    typedef std::function<T()> LoadFunc;
    typedef std::function<void()> SuccessFunc;
    
    explicit Lazy(const LoadFunc& loadFunc) 
        : loadFunc_(loadFunc), loaded_(false), successFunc_() {}
    
    const T& get() {
        if (!loaded_) {
            data_ = loadFunc_();
            loaded_ = true;
            if (successFunc_) {
                successFunc_();
            }
        }
        return data_;
    }
    
    void setSuccessFunc(const SuccessFunc& func) {
        successFunc_ = func;
    }
    
    bool isLoaded() const { return loaded_; }
    void reset() { loaded_ = false; }

private:
    LoadFunc loadFunc_;
    SuccessFunc successFunc_;
    T data_;
    bool loaded_;
};

// GGUF文件类
class File {
public:
    File();
    ~File();
    
    // 禁用拷贝，允许移动
    File(File& other);
    File& operator=(const File&) = delete;
    File(File&& other);
    File& operator=(File&& other);
    
    bool open(const std::string& path);
    void close();
    bool isOpen() const;
    
    uint32_t getVersion() const;
    const char* getMagic() const;
    int64_t getOffset() const;
    
    const std::vector<KeyValue>& getKeyValues();
    KeyValue getKeyValue(const std::string& key);
    
    const std::vector<TensorInfo>& getTensors();
    TensorInfo getTensor(const std::string& name);
    
    bool readTensorData(const TensorInfo& tensor, void* buffer, size_t bufferSize);
    
    // 迭代器支持
    class KeyValueIterator {
    public:
        typedef KeyValue value_type;
        typedef KeyValue& reference;
        typedef KeyValue* pointer;
        typedef std::forward_iterator_tag iterator_category;
        typedef ptrdiff_t difference_type;
        
        KeyValueIterator(const std::vector<KeyValue>* kvs, size_t index);
        
        reference operator*() const;
        pointer operator->() const;
        KeyValueIterator& operator++();
        KeyValueIterator operator++(int);
        bool operator==(const KeyValueIterator& other) const;
        bool operator!=(const KeyValueIterator& other) const;
        
    private:
        const std::vector<KeyValue>* kvs_;
        size_t index_;
    };
    
    KeyValueIterator keyValueBegin();
    KeyValueIterator keyValueEnd();

private:
    class Impl;
    Impl* impl_;
};

// 工具函数
std::string tensorTypeToString(TensorType type);
TensorType parseTensorType(const std::string& str);
double getTensorTypeBytesPerValue(TensorType type);

// 错误类
class GGUFError {
public:
    explicit GGUFError(const std::string& message) : message_(message) {}
    const std::string& what() const { return message_; }

private:
    std::string message_;
};

} // namespace gguf
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_GGUF_WRAPPER_H