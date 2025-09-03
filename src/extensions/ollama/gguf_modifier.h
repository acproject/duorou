#ifndef DUOROU_EXTENSIONS_OLLAMA_GGUF_MODIFIER_H
#define DUOROU_EXTENSIONS_OLLAMA_GGUF_MODIFIER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <fstream>
#include <iostream>

namespace duorou {
namespace extensions {
namespace ollama {

// GGUF数据类型枚举
enum class GGUFType : uint32_t {
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

// GGUF键值对结构
struct GGUFKeyValue {
    std::string key;
    GGUFType type;
    std::vector<uint8_t> data;
    
    // 便捷构造函数
    static GGUFKeyValue createString(const std::string& key, const std::string& value);
    static GGUFKeyValue createInt32(const std::string& key, int32_t value);
    static GGUFKeyValue createFloat32(const std::string& key, float value);
    static GGUFKeyValue createBool(const std::string& key, bool value);
    
    // 数据提取函数
    std::string asString() const;
    int32_t asInt32() const;
    float asFloat32() const;
    bool asBool() const;
};

// GGUF文件头结构
struct GGUFHeader {
    uint32_t magic;           // GGUF magic number
    uint32_t version;         // GGUF version
    uint64_t tensor_count;    // 张量数量
    uint64_t metadata_kv_count; // 元数据键值对数量
};

// GGUF张量信息结构
struct GGUFTensorInfo {
    std::string name;         // 张量名称
    uint32_t n_dimensions;    // 维度数量
    std::vector<uint64_t> dimensions; // 各维度大小
    GGUFType type;           // 张量数据类型
    uint64_t offset;         // 张量数据偏移量
};

// 跳过的张量信息结构
struct SkippedTensorInfo {
    uint64_t offset;         // 原始偏移量
    size_t size;            // 张量数据大小
};

// 架构映射规则
struct ArchitectureMapping {
    std::string source_arch;      // 源架构名称
    std::string target_arch;      // 目标架构名称
    std::unordered_map<std::string, std::string> key_mappings;  // 键名映射
    std::unordered_map<std::string, GGUFKeyValue> additional_keys; // 额外添加的键值对
    std::vector<std::string> keys_to_remove;  // 需要移除的键
};

// GGUF文件修改器类
class GGUFModifier {
public:
    explicit GGUFModifier(bool verbose = false);
    ~GGUFModifier();

    /**
     * 加载GGUF文件
     * @param file_path GGUF文件路径
     * @return 成功返回true
     */
    bool loadFile(const std::string& file_path);
    
    /**
     * 保存修改后的GGUF文件
     * @param output_path 输出文件路径
     * @return 成功返回true
     */
    bool saveFile(const std::string& output_path);
    
    /**
     * 创建优化的临时文件，只修改header和metadata，tensor data使用符号链接或文件拼接
     * @param output_path 输出文件路径
     * @return 成功返回true，失败返回false
     */
    bool saveOptimizedFile(const std::string& output_path);
    
    /**
     * 应用架构映射
     * @param mapping 架构映射规则
     * @return 成功返回true
     */
    bool applyArchitectureMapping(const ArchitectureMapping& mapping);
    
    /**
     * 获取元数据键值对
     * @param key 键名
     * @return 键值对指针，不存在返回nullptr
     */
    const GGUFKeyValue* getMetadata(const std::string& key) const;
    
    /**
     * 设置元数据键值对
     * @param kv 键值对
     * @return 成功返回true
     */
    bool setMetadata(const GGUFKeyValue& kv);
    
    /**
     * 移除元数据键值对
     * @param key 键名
     * @return 成功返回true
     */
    bool removeMetadata(const std::string& key);
    
    /**
     * 获取当前架构名称
     * @return 架构名称
     */
    std::string getCurrentArchitecture() const;
    
    /**
     * 设置架构名称
     * @param arch_name 新的架构名称
     * @return 成功返回true
     */
    bool setArchitecture(const std::string& arch_name);
    
    /**
     * 列出所有元数据键
     * @return 键名列表
     */
    std::vector<std::string> listMetadataKeys() const;
    
    /**
     * 验证GGUF文件完整性
     * @return 验证通过返回true
     */
    bool validateFile() const;
    
    /**
     * 创建架构映射规则
     * @param source_arch 源架构
     * @param target_arch 目标架构
     * @return 架构映射规则
     */
    static ArchitectureMapping createArchitectureMapping(
        const std::string& source_arch,
        const std::string& target_arch);
    
    /**
     * 设置详细输出模式
     * @param verbose 是否启用详细输出
     */
    void setVerbose(bool verbose) { verbose_ = verbose; }

private:
    /**
     * 读取GGUF文件头
     * @param file 文件流
     * @return 成功返回true
     */
    bool readHeader(std::ifstream& file);
    
    /**
     * 读取元数据
     * @param file 文件流
     * @return 成功返回true
     */
    bool readMetadata(std::ifstream& file);
    
    /**
     * 读取张量信息
     * @param file 文件流
     * @return 成功返回true
     */
    bool readTensorInfo(std::ifstream& file);
    
    /**
     * 写入GGUF文件头
     * @param file 文件流
     * @return 成功返回true
     */
    bool writeHeader(std::ofstream& file) const;
    
    /**
     * 写入元数据
     * @param file 文件流
     * @return 成功返回true
     */
    bool writeMetadata(std::ofstream& file) const;
    
    /**
     * 写入张量信息
     * @param file 文件流
     * @return 成功返回true
     */
    bool writeTensorInfo(std::ofstream& file) const;
    
    /**
     * 复制张量数据
     * @param input 输入文件流
     * @param output 输出文件流
     * @return 成功返回true
     */
    bool copyTensorData(std::ifstream& input, std::ofstream& output) const;
    
    /**
     * 读取字符串
     * @param file 文件流
     * @return 字符串内容
     */
    std::string readString(std::ifstream& file);
    
    /**
     * 写入字符串
     * @param file 文件流
     * @param str 字符串内容
     */
    void writeString(std::ofstream& file, const std::string& str) const;
    
    /**
     * 读取键值对
     * @param file 文件流
     * @return 键值对
     */
    GGUFKeyValue readKeyValue(std::ifstream& file);
    
    /**
     * 写入键值对
     * @param file 文件流
     * @param kv 键值对
     */
    void writeKeyValue(std::ofstream& file, const GGUFKeyValue& kv) const;
    
    /**
     * 计算元数据大小
     * @return 元数据字节数
     */
    size_t calculateMetadataSize() const;
    
    /**
     * 日志输出
     * @param level 日志级别
     * @param message 日志消息
     */
    void log(const std::string& level, const std::string& message) const;

private:
    std::string file_path_;                                    // 当前文件路径
    GGUFHeader header_;                                        // GGUF文件头
    std::unordered_map<std::string, GGUFKeyValue> metadata_;   // 元数据映射
    std::vector<GGUFTensorInfo> tensor_infos_;                 // 张量信息列表
    std::vector<uint8_t> tensor_data_;                         // 张量数据
    std::vector<SkippedTensorInfo> skipped_tensors_;           // 跳过的张量信息
    size_t tensor_data_offset_;                                // 张量数据偏移量
    bool verbose_;                                             // 详细输出模式
    bool file_loaded_;                                         // 文件是否已加载
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_GGUF_MODIFIER_H