#ifndef DUOROU_EXTENSIONS_OLLAMA_GGUF_PARSER_H
#define DUOROU_EXTENSIONS_OLLAMA_GGUF_PARSER_H

#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#ifdef _WIN32
    #ifndef NOMINMAX
    #define NOMINMAX
    #endif
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

// GGUF data type enumeration
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

// GGUF key-value pair structure
struct GGUFKeyValue {
  std::string key;
  GGUFType type;
  std::vector<uint8_t> data;

  // Data extraction functions
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

// GGUF file header structure
struct GGUFHeader {
  uint32_t magic;             // GGUF magic number (0x46554747)
  uint32_t version;           // GGUF version
  uint64_t tensor_count;      // Number of tensors
  uint64_t metadata_kv_count; // Number of metadata key-value pairs
};

// GGUF tensor information structure
struct GGUFTensorInfo {
  std::string name;                 // Tensor name
  uint32_t n_dimensions;            // Number of dimensions
  std::vector<uint64_t> dimensions; // Size of each dimension
  GGMLTensorType type;              // Tensor data type
  uint64_t offset;                  // Tensor data offset
  uint64_t size;                    // Tensor data size (bytes)
};

// Model architecture information
struct ModelArchitecture {
  std::string name;                    // Architecture name (e.g. "qwen25vl")
  uint32_t context_length;             // Context length
  uint32_t embedding_length;           // Embedding dimension
  uint32_t block_count;                // Number of layers
  uint32_t feed_forward_length;        // Feed forward network dimension
  uint32_t attention_head_count;       // Number of attention heads
  uint32_t attention_head_count_kv;    // Number of KV attention heads
  float layer_norm_rms_epsilon;        // RMS normalization epsilon
  uint32_t rope_dimension_count;       // RoPE dimension count
  float rope_freq_base;                // RoPE frequency base
  std::vector<uint64_t> rope_dimension_sections; // RoPE dimension sections
  
  // Vision-related parameters (for multimodal models)
  bool has_vision = false;
  uint32_t vision_patch_size = 0;
  uint32_t vision_spatial_patch_size = 0;
  std::vector<uint64_t> vision_fullatt_block_indexes;
};

/**
 * @brief GGUF file parser
 * 
 * Directly parses GGUF format files, extracts model architecture information and tensor data,
 * without relying on llama.cpp's architecture mapping mechanism
 */
class GGUFParser {
public:
  explicit GGUFParser(bool verbose = false);
  ~GGUFParser();
  
  // Memory mapping related
  bool useMmap() const { return use_mmap_; }
  void setUseMmap(bool use_mmap) { use_mmap_ = use_mmap; }

  /**
   * Parse GGUF file
   * @param file_path GGUF file path
   * @return Returns true on success
   */
  bool parseFile(const std::string &file_path);

  /**
   * Get model architecture information
   * @return Model architecture information
   */
  const ModelArchitecture& getArchitecture() const { return architecture_; }

  /**
   * Get metadata key-value pair
   * @param key Key name
   * @return Key-value pair pointer, returns nullptr if not found
   */
  const GGUFKeyValue* getMetadata(const std::string &key) const;

  /**
   * Get tensor information
   * @param name Tensor name
   * @return Tensor information pointer, returns nullptr if not found
   */
  const GGUFTensorInfo* getTensorInfo(const std::string &name) const;

  /**
   * Get all tensor information
   * @return Tensor information list
   */
  const std::vector<GGUFTensorInfo>& getAllTensorInfos() const { return tensor_infos_; }

  /**
   * Get file header information
   * @return File header
   */
  const GGUFHeader& getHeader() const { return header_; }

  /**
   * Get tensor data offset (relative to file start)
   * @return Tensor data offset
   */
  uint64_t getTensorDataOffset() const { return tensor_data_offset_; }

  /**
   * Validate file integrity
   * @return Returns true if validation passes
   */
  bool validateFile() const;

  /**
   * Check if architecture is supported
   * @param arch_name Architecture name
   * @return Whether it is supported
   */
  static bool isSupportedArchitecture(const std::string& arch_name);

  /**
   * Set verbose output mode
   * @param verbose Whether to enable verbose output
   */
  void setVerbose(bool verbose) { verbose_ = verbose; }

private:
  /**
   * Read GGUF file header
   * @param file File stream
   * @return Returns true on success
   */
  bool readHeader(std::ifstream &file);

  /**
   * Read metadata
   * @param file File stream
   * @return Returns true on success
   */
  bool readMetadata(std::ifstream &file);

  /**
   * Read tensor information
   * @param file File stream
   * @return Returns true on success
   */
  bool readTensorInfo(std::ifstream &file);

  /**
   * Parse architecture information
   * @return Returns true on success
   */
  bool parseArchitecture();

  /**
   * Read string
   * @param file File stream
   * @return String content
   */
  std::string readString(std::ifstream &file);

  /**
   * Read key-value pair
   * @param file File stream
   * @return Key-value pair
   */
  GGUFKeyValue readKeyValue(std::ifstream &file);

  /**
   * Calculate tensor data size
   * @param tensor_info Tensor information
   * @return Data size (bytes)
   */
  uint64_t calculateTensorSize(const GGUFTensorInfo& tensor_info) const;

  /**
   * Get data type size
   * @param type Data type
   * @return Type size (bytes)
   */
  static uint32_t getTypeSize(GGUFType type);

  /**
   * Get GGML tensor type size
   * @param type GGML tensor data type
   * @return Type size (bytes)
   */
  static uint32_t getGGMLTypeSize(GGMLTensorType type);

  /**
   * Log output
   * @param level Log level
   * @param message Log message
   */
  void log(const std::string &level, const std::string &message) const;
  
  /**
   * Memory mapping related
   */
  bool initMmap(const std::string& file_path);
  void cleanupMmap();
  
  /**
   * Memory mapping read methods
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
  std::string file_path_;                                  // Current file path
  GGUFHeader header_;                                      // GGUF file header
  std::unordered_map<std::string, GGUFKeyValue> metadata_; // Metadata mapping
  std::vector<GGUFTensorInfo> tensor_infos_;               // Tensor information list
  std::unordered_map<std::string, size_t> tensor_name_to_index_; // Tensor name to index mapping
  ModelArchitecture architecture_;                         // Model architecture information
  uint64_t tensor_data_offset_;                            // Tensor data offset
  bool verbose_;                                           // Verbose output mode
  bool file_parsed_;                                       // Whether file has been parsed
  
  // Memory mapping related
  bool use_mmap_;
  int fd_;
  void* mapped_data_;
  size_t file_size_;
  size_t current_offset_;

#ifdef _WIN32
  // Windows specific handles
  HANDLE file_handle_;
  HANDLE mapping_handle_;
#endif

  // List of supported architectures
  static const std::vector<std::string> SUPPORTED_ARCHITECTURES;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // DUOROU_EXTENSIONS_OLLAMA_GGUF_PARSER_H