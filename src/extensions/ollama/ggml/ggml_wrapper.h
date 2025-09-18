#ifndef GGML_WRAPPER_H
#define GGML_WRAPPER_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <variant>

namespace duorou {
namespace extensions {
namespace ollama {
namespace ggml {

// GGML文件类型枚举，对应Go版本的FileType
enum class FileType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    MXFP4 = 4,  // originally Q4_1_F16, unused by GGML
    Q4_2 = 5,   // unused by GGML
    Q4_3 = 6,   // unused by GGML
    Q8_0 = 7,
    Q5_0 = 8,
    Q5_1 = 9,
    Q2_K = 10,
    Q3_K_S = 11,
    Q3_K_M = 12,
    Q3_K_L = 13,
    Q4_K_S = 14,
    Q4_K_M = 15,
    Q5_K_S = 16,
    Q5_K_M = 17,
    Q6_K = 18,
    IQ2_XXS = 19,
    IQ2_XS = 20,
    Q2_K_S = 21,
    IQ3_XS = 22,
    IQ3_XXS = 23,
    IQ1_S = 24,
    IQ4_NL = 25,
    IQ3_S = 26,
    IQ3_M = 27,
    IQ2_S = 28,
    IQ2_M = 29,
    IQ4_XS = 30,
    IQ1_M = 31,
    BF16 = 32,
    Q4_0_4_4 = 33,  // unused by GGML
    Q4_0_4_8 = 34,  // unused by GGML
    Q4_0_8_8 = 35,  // unused by GGML
    TQ1_0 = 36,
    TQ2_0 = 37,
    UNKNOWN = 1024
};

// 键值对类型，支持多种数据类型
using KVValue = std::variant<
    std::string,
    int64_t,
    uint64_t,
    double,
    bool,
    std::vector<std::string>,
    std::vector<int64_t>,
    std::vector<uint64_t>,
    std::vector<double>,
    std::vector<bool>
>;

// 键值对映射
using KV = std::map<std::string, KVValue>;

// 张量信息结构
struct TensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type;
    uint64_t offset;
    
    bool valid() const;
    int64_t numValues() const;
    int64_t numBytes() const;
};

using Tensors = std::vector<TensorInfo>;

// 模型接口
class Model {
public:
    virtual ~Model() = default;
    virtual const KV& getKV() const = 0;
    virtual const Tensors& getTensors() const = 0;
};

// KV辅助方法类
class KVHelper {
public:
    explicit KVHelper(const KV& kv) : kv_(kv) {}
    
    std::string architecture() const;
    std::string kind() const;
    uint64_t parameterCount() const;
    FileType fileType() const;
    uint64_t blockCount() const;
    uint64_t embeddingLength() const;
    uint64_t headCountMax() const;
    uint64_t headCountMin() const;
    uint64_t headCountKVMax() const;
    uint64_t headCountKVMin() const;
    uint64_t embeddingHeadCountMax() const;
    uint64_t embeddingHeadCountK() const;
    uint64_t embeddingHeadCountV() const;
    uint64_t contextLength() const;
    std::string chatTemplate() const;
    
    // 通用访问方法
    std::string getString(const std::string& key, const std::string& defaultValue = "") const;
    uint32_t getUint(const std::string& key, uint32_t defaultValue = 0) const;
    uint64_t getUint64(const std::string& key, uint64_t defaultValue = 0) const;
    int64_t getInt64(const std::string& key, int64_t defaultValue = 0) const;
    double getFloat(const std::string& key, double defaultValue = 0.0) const;
    bool getBool(const std::string& key, bool defaultValue = false) const;
    
    // 数组访问方法
    std::vector<std::string> getStrings(const std::string& key) const;
    std::vector<int64_t> getInts(const std::string& key) const;
    std::vector<uint64_t> getUints(const std::string& key) const;
    std::vector<double> getFloats(const std::string& key) const;
    std::vector<bool> getBools(const std::string& key) const;
    
    // 数组最大/最小值方法
    uint32_t getUintOrMaxArrayValue(const std::string& key, uint32_t defaultValue = 1) const;
    uint32_t getUintOrMinArrayValue(const std::string& key, uint32_t defaultValue = 1) const;

private:
    const KV& kv_;
};

// GGML主类
class GGML {
public:
    GGML();
    ~GGML();
    
    // 禁用拷贝构造和赋值
    GGML(const GGML&) = delete;
    GGML& operator=(const GGML&) = delete;
    
    // 移动构造和赋值
    GGML(GGML&& other) noexcept;
    GGML& operator=(GGML&& other) noexcept;
    
    // 加载GGML文件
    bool load(const std::string& path);
    
    // 获取模型信息
    const Model* getModel() const;
    KVHelper getKVHelper() const;
    
    // 获取文件长度
    int64_t getLength() const;
    
    // 检查是否已加载
    bool isLoaded() const;
    
    // 卸载模型
    void unload();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// 文件类型解析和字符串转换
FileType parseFileType(const std::string& str);
std::string fileTypeToString(FileType type);

} // namespace ggml
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // GGML_WRAPPER_H