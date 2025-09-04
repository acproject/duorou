#ifndef DUOROU_EXTENSIONS_OLLAMA_GGUF_PARSER_H
#define DUOROU_EXTENSIONS_OLLAMA_GGUF_PARSER_H

#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#ifdef _WIN32
    #include <windows.h>
    #include <io.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

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

// GGML tensor types (for tensor data, not metadata)
enum class GGMLTensorType : uint32_t {
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
  Q8_K = 15
};

// GGUF键值对结构
struct GGUFKeyValue {
  std::string key;
  GGUFType type;
  std::vector<uint8_t> data;

  // 数据提取函数
  std::string asString() const;
  int32_t asInt32() const;
  int64_t asInt64() const;
  uint32_t asUInt32() const;
  uint64_t asUInt64() const;
  float asFloat32() const;
  double asFloat64() const;
  bool asBool() const;
  std::vector<int32_t> asInt32Array() const;
  std::vector<uint64_t> asUInt64Array() const;
  std::vector<std::string> asStringArray() const;
};

// GGUF文件头结构
struct GGUFHeader {
  uint32_t magic;             // GGUF magic number (0x46554747)
  uint32_t version;           // GGUF version
  uint64_t tensor_count;      // 张量数量
  uint64_t metadata_kv_count; // 元数据键值对数量
};

// GGUF张量信息结构
struct GGUFTensorInfo {
  std::string name;                 // 张量名称
  uint32_t n_dimensions;            // 维度数量
  std::vector<uint64_t> dimensions; // 各维度大小
  GGMLTensorType type;              // 张量数据类型
  uint64_t offset;                  // 张量数据偏移量
  uint64_t size;                    // 张量数据大小（字节）
};

// 模型架构信息
struct ModelArchitecture {
  std::string name;                    // 架构名称 (如 "qwen25vl")
  uint32_t context_length;             // 上下文长度
  uint32_t embedding_length;           // 嵌入维度
  uint32_t block_count;                // 层数
  uint32_t feed_forward_length;        // 前馈网络维度
  uint32_t attention_head_count;       // 注意力头数
  uint32_t attention_head_count_kv;    // KV注意力头数
  float layer_norm_rms_epsilon;        // RMS归一化epsilon
  uint32_t rope_dimension_count;       // RoPE维度数
  float rope_freq_base;                // RoPE频率基数
  std::vector<uint64_t> rope_dimension_sections; // RoPE维度分段
  
  // 视觉相关参数（用于多模态模型）
  bool has_vision = false;
  uint32_t vision_patch_size = 0;
  uint32_t vision_spatial_patch_size = 0;
  std::vector<uint64_t> vision_fullatt_block_indexes;
};

/**
 * @brief GGUF文件解析器
 * 
 * 直接解析GGUF格式文件，提取模型架构信息和张量数据，
 * 不依赖llama.cpp的架构映射机制
 */
class GGUFParser {
public:
  explicit GGUFParser(bool verbose = false);
  ~GGUFParser();
  
  // 内存映射相关
  bool useMmap() const { return use_mmap_; }
  void setUseMmap(bool use_mmap) { use_mmap_ = use_mmap; }

  /**
   * 解析GGUF文件
   * @param file_path GGUF文件路径
   * @return 成功返回true
   */
  bool parseFile(const std::string &file_path);

  /**
   * 获取模型架构信息
   * @return 模型架构信息
   */
  const ModelArchitecture& getArchitecture() const { return architecture_; }

  /**
   * 获取元数据键值对
   * @param key 键名
   * @return 键值对指针，不存在返回nullptr
   */
  const GGUFKeyValue* getMetadata(const std::string &key) const;

  /**
   * 获取张量信息
   * @param name 张量名称
   * @return 张量信息指针，不存在返回nullptr
   */
  const GGUFTensorInfo* getTensorInfo(const std::string &name) const;

  /**
   * 获取所有张量信息
   * @return 张量信息列表
   */
  const std::vector<GGUFTensorInfo>& getAllTensorInfos() const { return tensor_infos_; }

  /**
   * 获取文件头信息
   * @return 文件头
   */
  const GGUFHeader& getHeader() const { return header_; }

  /**
   * 获取张量数据偏移量（相对于文件开始）
   * @return 张量数据偏移量
   */
  uint64_t getTensorDataOffset() const { return tensor_data_offset_; }

  /**
   * 验证文件完整性
   * @return 验证通过返回true
   */
  bool validateFile() const;

  /**
   * 检查是否为支持的架构
   * @param arch_name 架构名称
   * @return 是否支持
   */
  static bool isSupportedArchitecture(const std::string& arch_name);

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
  bool readHeader(std::ifstream &file);

  /**
   * 读取元数据
   * @param file 文件流
   * @return 成功返回true
   */
  bool readMetadata(std::ifstream &file);

  /**
   * 读取张量信息
   * @param file 文件流
   * @return 成功返回true
   */
  bool readTensorInfo(std::ifstream &file);

  /**
   * 解析架构信息
   * @return 成功返回true
   */
  bool parseArchitecture();

  /**
   * 读取字符串
   * @param file 文件流
   * @return 字符串内容
   */
  std::string readString(std::ifstream &file);

  /**
   * 读取键值对
   * @param file 文件流
   * @return 键值对
   */
  GGUFKeyValue readKeyValue(std::ifstream &file);

  /**
   * 计算张量数据大小
   * @param tensor_info 张量信息
   * @return 数据大小（字节）
   */
  uint64_t calculateTensorSize(const GGUFTensorInfo& tensor_info) const;

  /**
   * 获取数据类型大小
   * @param type 数据类型
   * @return 类型大小（字节）
   */
  static uint32_t getTypeSize(GGUFType type);

  /**
   * 获取GGML张量类型大小
   * @param type GGML张量数据类型
   * @return 类型大小（字节）
   */
  static uint32_t getGGMLTypeSize(GGMLTensorType type);

  /**
   * 日志输出
   * @param level 日志级别
   * @param message 日志消息
   */
  void log(const std::string &level, const std::string &message) const;
  
  /**
   * 内存映射相关
   */
  bool initMmap(const std::string& file_path);
  void cleanupMmap();
  
  /**
   * 内存映射读取方法
   */
  bool readHeaderMmap();
   bool readMetadataMmap();
   bool readTensorInfoMmap();
   bool readKeyValueMmap(size_t& offset, GGUFKeyValue& kv);
   bool readFromMmap(void* buffer, size_t size);
    std::string readStringFromMmap();
    bool parseWithMmap();
    bool readHeaderFromMmap();
    bool readMetadataFromMmap();
    bool readTensorInfoFromMmap();
    GGUFKeyValue readKeyValueFromMmap();

private:
  std::string file_path_;                                  // 当前文件路径
  GGUFHeader header_;                                      // GGUF文件头
  std::unordered_map<std::string, GGUFKeyValue> metadata_; // 元数据映射
  std::vector<GGUFTensorInfo> tensor_infos_;               // 张量信息列表
  std::unordered_map<std::string, size_t> tensor_name_to_index_; // 张量名称到索引的映射
  ModelArchitecture architecture_;                         // 模型架构信息
  uint64_t tensor_data_offset_;                            // 张量数据偏移量
  bool verbose_;                                           // 详细输出模式
  bool file_parsed_;                                       // 文件是否已解析
  
  // 内存映射相关
  bool use_mmap_;
  int fd_;
  void* mapped_data_;
  size_t file_size_;
  size_t current_offset_;

  // 支持的架构列表
  static const std::vector<std::string> SUPPORTED_ARCHITECTURES;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_GGUF_PARSER_H