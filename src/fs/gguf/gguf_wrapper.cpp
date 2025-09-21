#include "gguf_wrapper.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace duorou {
namespace extensions {
namespace ollama {
namespace gguf {

// Value类实现
Value::Value() : type_(UINT32), data_(NULL) {}

Value::Value(const std::string& str) : type_(STRING) {
    data_ = new std::string(str);
}

Value::Value(int32_t val) : type_(INT32) {
    data_ = new int32_t(val);
}

Value::Value(uint32_t val) : type_(UINT32) {
    data_ = new uint32_t(val);
}

Value::Value(int64_t val) : type_(INT64) {
    data_ = new int64_t(val);
}

Value::Value(uint64_t val) : type_(UINT64) {
    data_ = new uint64_t(val);
}

Value::Value(float val) : type_(FLOAT32) {
    data_ = new float(val);
}

Value::Value(double val) : type_(FLOAT64) {
    data_ = new double(val);
}

Value::Value(bool val) : type_(BOOL) {
    data_ = new bool(val);
}

Value::Value(const std::vector<std::string>& vals) : type_(ARRAY) {
    data_ = new std::vector<std::string>(vals);
}

Value::Value(const std::vector<int64_t>& vals) : type_(ARRAY) {
    data_ = new std::vector<int64_t>(vals);
}

Value::~Value() {
    cleanup();
}

Value::Value(const Value& other) : type_(other.type_), data_(NULL) {
    copyFrom(other);
}

Value& Value::operator=(const Value& other) {
    if (this != &other) {
        cleanup();
        type_ = other.type_;
        copyFrom(other);
    }
    return *this;
}

void Value::cleanup() {
    if (data_) {
        switch (type_) {
            case STRING:
                delete static_cast<std::string*>(data_);
                break;
            case INT32:
                delete static_cast<int32_t*>(data_);
                break;
            case UINT32:
                delete static_cast<uint32_t*>(data_);
                break;
            case INT64:
                delete static_cast<int64_t*>(data_);
                break;
            case UINT64:
                delete static_cast<uint64_t*>(data_);
                break;
            case FLOAT32:
                delete static_cast<float*>(data_);
                break;
            case FLOAT64:
                delete static_cast<double*>(data_);
                break;
            case BOOL:
                delete static_cast<bool*>(data_);
                break;
            case ARRAY:
                // 这里需要根据实际存储的类型来删除
                // 简化处理，假设是字符串数组
                delete static_cast<std::vector<std::string>*>(data_);
                break;
            default:
                break;
        }
        data_ = NULL;
    }
}

void Value::copyFrom(const Value& other) {
    if (other.data_) {
        switch (other.type_) {
            case STRING:
                data_ = new std::string(*static_cast<std::string*>(other.data_));
                break;
            case INT32:
                data_ = new int32_t(*static_cast<int32_t*>(other.data_));
                break;
            case UINT32:
                data_ = new uint32_t(*static_cast<uint32_t*>(other.data_));
                break;
            case INT64:
                data_ = new int64_t(*static_cast<int64_t*>(other.data_));
                break;
            case UINT64:
                data_ = new uint64_t(*static_cast<uint64_t*>(other.data_));
                break;
            case FLOAT32:
                data_ = new float(*static_cast<float*>(other.data_));
                break;
            case FLOAT64:
                data_ = new double(*static_cast<double*>(other.data_));
                break;
            case BOOL:
                data_ = new bool(*static_cast<bool*>(other.data_));
                break;
            case ARRAY:
                data_ = new std::vector<std::string>(*static_cast<std::vector<std::string>*>(other.data_));
                break;
            default:
                data_ = NULL;
                break;
        }
    }
}

ValueType Value::getType() const {
    return type_;
}

int64_t Value::asInt() const {
    switch (type_) {
        case INT32:
            return static_cast<int64_t>(*static_cast<int32_t*>(data_));
        case UINT32:
            return static_cast<int64_t>(*static_cast<uint32_t*>(data_));
        case INT64:
            return *static_cast<int64_t*>(data_);
        case UINT64:
            return static_cast<int64_t>(*static_cast<uint64_t*>(data_));
        default:
            return 0;
    }
}

uint64_t Value::asUint() const {
    switch (type_) {
        case UINT32:
            return static_cast<uint64_t>(*static_cast<uint32_t*>(data_));
        case UINT64:
            return *static_cast<uint64_t*>(data_);
        case INT32:
            return static_cast<uint64_t>(*static_cast<int32_t*>(data_));
        case INT64:
            return static_cast<uint64_t>(*static_cast<int64_t*>(data_));
        default:
            return 0;
    }
}

double Value::asFloat() const {
    switch (type_) {
        case FLOAT32:
            return static_cast<double>(*static_cast<float*>(data_));
        case FLOAT64:
            return *static_cast<double*>(data_);
        default:
            return 0.0;
    }
}

bool Value::asBool() const {
    if (type_ == BOOL && data_) {
        return *static_cast<bool*>(data_);
    }
    return false;
}

std::string Value::asString() const {
    if (type_ == STRING && data_) {
        return *static_cast<std::string*>(data_);
    }
    return "";
}

std::vector<int64_t> Value::asInts() const {
    if (type_ == ARRAY && data_) {
        // 简化处理，假设是int64_t数组
        std::vector<int64_t>* ptr = static_cast<std::vector<int64_t>*>(data_);
        return *ptr;
    }
    return std::vector<int64_t>();
}

std::vector<std::string> Value::asStrings() const {
    if (type_ == ARRAY && data_) {
        std::vector<std::string>* ptr = static_cast<std::vector<std::string>*>(data_);
        return *ptr;
    }
    return std::vector<std::string>();
}

// KeyValue实现
KeyValue::KeyValue() {}

KeyValue::KeyValue(const std::string& k, const Value& v) : key(k), value(v) {}

bool KeyValue::valid() const {
    return !key.empty();
}

// TensorInfo实现
TensorInfo::TensorInfo() : type(F32), offset(0) {}

bool TensorInfo::valid() const {
    return !name.empty() && numBytes() > 0;
}

int64_t TensorInfo::numValues() const {
    int64_t numItems = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        numItems *= static_cast<int64_t>(shape[i]);
    }
    return numItems;
}

int64_t TensorInfo::numBytes() const {
    return static_cast<int64_t>(static_cast<double>(numValues()) * getBytesPerValue());
}

double TensorInfo::getBytesPerValue() const {
    return getTensorTypeBytesPerValue(type);
}

// BufferedReader::Impl实现
class BufferedReader::Impl {
public:
    std::ifstream file_;
    std::vector<char> buffer_;
    size_t bufferPos_;
    size_t bufferSize_;
    size_t bufferCapacity_;
    int64_t filePos_;
    int64_t fileSize_;
    
    explicit Impl(const std::string& filename, size_t bufferCapacity)
        : buffer_(bufferCapacity), bufferPos_(0), bufferSize_(0), 
          bufferCapacity_(bufferCapacity), filePos_(0), fileSize_(0) {
        file_.open(filename.c_str(), std::ios::binary);
        if (file_.is_open()) {
            file_.seekg(0, std::ios::end);
            fileSize_ = file_.tellg();
            file_.seekg(0, std::ios::beg);
        }
    }
    
    ~Impl() {
        if (file_.is_open()) {
            file_.close();
        }
    }
    
    bool fillBuffer() {
        if (!file_.is_open()) return false;
        
        file_.read(&buffer_[0], bufferCapacity_);
        bufferSize_ = file_.gcount();
        bufferPos_ = 0;
        return bufferSize_ > 0;
    }
    
    size_t read(void* dest, size_t size) {
        if (!file_.is_open()) return 0;
        
        char* destPtr = static_cast<char*>(dest);
        size_t totalRead = 0;
        
        while (size > 0) {
            if (bufferPos_ >= bufferSize_) {
                if (!fillBuffer()) break;
            }
            
            size_t available = bufferSize_ - bufferPos_;
            size_t toRead = std::min(size, available);
            
            std::memcpy(destPtr, &buffer_[bufferPos_], toRead);
            destPtr += toRead;
            bufferPos_ += toRead;
            size -= toRead;
            totalRead += toRead;
            filePos_ += toRead;
        }
        
        return totalRead;
    }
    
    bool seek(int64_t offset) {
        if (!file_.is_open()) return false;
        
        file_.clear();
        file_.seekg(offset, std::ios::beg);
        if (file_.fail()) return false;
        
        filePos_ = offset;
        bufferPos_ = 0;
        bufferSize_ = 0;
        return true;
    }
};

// BufferedReader实现
BufferedReader::BufferedReader(const std::string& filename, size_t bufferSize)
    : impl_(new Impl(filename, bufferSize)) {}

BufferedReader::~BufferedReader() {
    delete impl_;
}

BufferedReader::BufferedReader(BufferedReader& other) : impl_(other.impl_) {
    other.impl_ = NULL;
}

BufferedReader& BufferedReader::operator=(BufferedReader& other) {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = NULL;
    }
    return *this;
}

bool BufferedReader::isOpen() const {
    return impl_ && impl_->file_.is_open();
}

size_t BufferedReader::read(void* buffer, size_t size) {
    return impl_ ? impl_->read(buffer, size) : 0;
}

bool BufferedReader::seek(int64_t offset) {
    return impl_ ? impl_->seek(offset) : false;
}

int64_t BufferedReader::tell() const {
    return impl_ ? impl_->filePos_ : 0;
}

int64_t BufferedReader::size() const {
    return impl_ ? impl_->fileSize_ : 0;
}

// File::Impl实现
class File::Impl {
public:
    char magic_[4];
    uint32_t version_;
    int64_t offset_;
    bool isOpen_;
    
    BufferedReader* reader_;
    Lazy<std::vector<KeyValue> >* keyValues_;
    Lazy<std::vector<TensorInfo> >* tensors_;
    
    Impl() : version_(0), offset_(0), isOpen_(false), reader_(NULL), keyValues_(NULL), tensors_(NULL) {
        magic_[0] = magic_[1] = magic_[2] = magic_[3] = 0;
    }
    
    bool open(const std::string& path) {
        delete reader_;
        reader_ = new BufferedReader(path);
        if (!reader_->isOpen()) {
            return false;
        }
        
        // 读取魔数
        if (reader_->read(magic_, 4) != 4) {
            return false;
        }
        
        // 检查魔数
        if (std::string(magic_, 4) != "GGUF") {
            std::cerr << "Invalid magic number" << std::endl;
            return false;
        }
        
        // 读取版本
        if (reader_->read(&version_, sizeof(version_)) != sizeof(version_)) {
            return false;
        }
        
        if (version_ < 2) {
            std::cerr << "Unsupported version: " << version_ << std::endl;
            return false;
        }
        
        isOpen_ = true;
        return true;
    }
    
    void close() {
        delete reader_;
        reader_ = NULL;
        delete keyValues_;
        keyValues_ = NULL;
        delete tensors_;
        tensors_ = NULL;
        isOpen_ = false;
    }
};

// File实现
File::File() : impl_(new Impl()) {}

File::~File() {}

File::File(File& other) : impl_(other.impl_) {
    other.impl_ = NULL;
}

File& File::operator=(File&& other) {
    if (this != &other) {
        impl_ = other.impl_;
        other.impl_ = NULL;
    }
    return *this;
}

bool File::open(const std::string& path) {
    return impl_->open(path);
}

void File::close() {
    impl_->close();
}

bool File::isOpen() const {
    return impl_->isOpen_;
}

uint32_t File::getVersion() const {
    return impl_->version_;
}

const char* File::getMagic() const {
    return impl_->magic_;
}

int64_t File::getOffset() const {
    return impl_->offset_;
}

const std::vector<KeyValue>& File::getKeyValues() {
    static std::vector<KeyValue> empty;
    return empty; // 简化实现
}

KeyValue File::getKeyValue(const std::string& key) {
    return KeyValue();
}

const std::vector<TensorInfo>& File::getTensors() {
    static std::vector<TensorInfo> empty;
    return empty; // 简化实现
}

TensorInfo File::getTensor(const std::string& name) {
    return TensorInfo();
}

bool File::readTensorData(const TensorInfo& tensor, void* buffer, size_t bufferSize) {
    return false; // 简化实现
}

// KeyValueIterator实现
File::KeyValueIterator::KeyValueIterator(const std::vector<KeyValue>* kvs, size_t index)
    : kvs_(kvs), index_(index) {}

File::KeyValueIterator::reference File::KeyValueIterator::operator*() const {
    return const_cast<KeyValue&>((*kvs_)[index_]);
}

File::KeyValueIterator::pointer File::KeyValueIterator::operator->() const {
    return const_cast<KeyValue*>(&(*kvs_)[index_]);
}

File::KeyValueIterator& File::KeyValueIterator::operator++() {
    ++index_;
    return *this;
}

File::KeyValueIterator File::KeyValueIterator::operator++(int) {
    KeyValueIterator tmp = *this;
    ++index_;
    return tmp;
}

bool File::KeyValueIterator::operator==(const KeyValueIterator& other) const {
    return kvs_ == other.kvs_ && index_ == other.index_;
}

bool File::KeyValueIterator::operator!=(const KeyValueIterator& other) const {
    return !(*this == other);
}

File::KeyValueIterator File::keyValueBegin() {
    const std::vector<KeyValue>& kvs = getKeyValues();
    return KeyValueIterator(&kvs, 0);
}

File::KeyValueIterator File::keyValueEnd() {
    const std::vector<KeyValue>& kvs = getKeyValues();
    return KeyValueIterator(&kvs, kvs.size());
}

// 工具函数实现
std::string tensorTypeToString(TensorType type) {
    switch (type) {
        case F32: return "F32";
        case F16: return "F16";
        case Q4_0: return "Q4_0";
        case Q4_1: return "Q4_1";
        case Q5_0: return "Q5_0";
        case Q5_1: return "Q5_1";
        case Q8_0: return "Q8_0";
        case Q8_1: return "Q8_1";
        case Q2_K: return "Q2_K";
        case Q3_K: return "Q3_K";
        case Q4_K: return "Q4_K";
        case Q5_K: return "Q5_K";
        case Q6_K: return "Q6_K";
        case Q8_K: return "Q8_K";
        case BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

TensorType parseTensorType(const std::string& str) {
    if (str == "F32") return F32;
    if (str == "F16") return F16;
    if (str == "Q4_0") return Q4_0;
    if (str == "Q4_1") return Q4_1;
    if (str == "Q5_0") return Q5_0;
    if (str == "Q5_1") return Q5_1;
    if (str == "Q8_0") return Q8_0;
    if (str == "Q8_1") return Q8_1;
    if (str == "Q2_K") return Q2_K;
    if (str == "Q3_K") return Q3_K;
    if (str == "Q4_K") return Q4_K;
    if (str == "Q5_K") return Q5_K;
    if (str == "Q6_K") return Q6_K;
    if (str == "Q8_K") return Q8_K;
    if (str == "BF16") return BF16;
    return F32; // 默认值
}

double getTensorTypeBytesPerValue(TensorType type) {
    switch (type) {
        case F32: return 4.0;
        case F16: return 2.0;
        case BF16: return 2.0;
        case Q4_0: return 0.5 + 2.0/32.0;
        case Q4_1: return 0.5 + 4.0/32.0;
        case Q5_0: return 0.625 + 2.0/32.0;
        case Q5_1: return 0.625 + 4.0/32.0;
        case Q8_0: return 1.0 + 2.0/32.0;
        case Q8_1: return 1.0 + 4.0/32.0;
        case Q2_K: return 0.25 + 12.0/256.0;
        case Q3_K: return 0.375 + 12.0/256.0;
        case Q4_K: return 0.5 + 12.0/256.0;
        case Q5_K: return 0.625 + 12.0/256.0;
        case Q6_K: return 0.75 + 12.0/256.0;
        case Q8_K: return 1.0 + 12.0/256.0;
        case I8: return 1.0;
        case I16: return 2.0;
        case I32: return 4.0;
        case I64: return 8.0;
        case F64: return 8.0;
        default: return 4.0;
    }
}

} // namespace gguf
} // namespace ollama
} // namespace extensions
} // namespace duorou