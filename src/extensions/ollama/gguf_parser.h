#ifndef DUOROU_EXTENSIONS_OLLAMA_GGUF_PARSER_H
#define DUOROU_EXTENSIONS_OLLAMA_GGUF_PARSER_H

#ifdef __cplusplus

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
#include <cstring>
#include <algorithm>

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
  Q8_K = 15,
  // Align with upstream GGML: BF16 code is 30
  BF16 = 30
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
  uint32_t attention_head_dim;         // Per-head dimension for Q
  uint32_t attention_head_dim_k;       // Per-head dimension for K/V (GQA)
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
  // Enumerate all metadata keys for diagnostics
  std::vector<std::string> listMetadataKeys() const;

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

  // ===== New tensor data access APIs =====
  /**
   * Get a direct pointer to the tensor data if mmap is enabled.
   * Returns nullptr if mmap is disabled or tensor is not found.
   */
  const uint8_t* getTensorDataPtr(const std::string& name) const;

  /**
   * Read tensor data into destination buffer. Supports both mmap and stream IO.
   * @param name Tensor name
   * @param dst Destination buffer
   * @param bytes Number of bytes to read (clamped to tensor size - offset)
   * @param offset Byte offset within the tensor data
   * @return true on success
   */
  bool readTensorData(const std::string& name, void* dst, size_t bytes, size_t offset = 0) const;

  /**
   * Overload: read by tensor info
   */
  bool readTensorData(const GGUFTensorInfo& info, void* dst, size_t bytes, size_t offset = 0) const;

  /**
   * Get tensor data size by name, returns 0 if not found
   */
  size_t getTensorSize(const std::string& name) const;

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

  // Parse and helpers
  bool parseArchitecture();
  std::string readString(std::ifstream &file);
  GGUFKeyValue readKeyValue(std::ifstream &file);
  uint64_t calculateTensorSize(const GGUFTensorInfo& tensor_info) const;
  static uint32_t getTypeSize(GGUFType type);
  static uint32_t getGGMLTypeSize(GGMLTensorType type);
  void log(const std::string &level, const std::string &message) const;

  bool initMmap(const std::string& file_path);
  void cleanupMmap();
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
  
  // mmap fields
  bool use_mmap_;
  int fd_;
  void* mapped_data_;
  size_t file_size_;
  size_t current_offset_;
  
#ifdef _WIN32
  // Windows 特定的句柄
  HANDLE file_handle_;
  HANDLE mapping_handle_;
#endif

#ifdef _WIN32
  HANDLE file_handle_;
  HANDLE mapping_handle_;
#endif

  static const std::vector<std::string> SUPPORTED_ARCHITECTURES;
};

} // namespace ollama
} // namespace extensions
} // namespace duorou

namespace duorou {
namespace extensions {
namespace ollama {

inline const uint8_t* GGUFParser::getTensorDataPtr(const std::string& name) const {
  if (!file_parsed_ || !use_mmap_ || mapped_data_ == nullptr) {
    return nullptr;
  }
  const GGUFTensorInfo* info = getTensorInfo(name);
  if (!info) return nullptr;
  const size_t base = static_cast<size_t>(tensor_data_offset_) + static_cast<size_t>(info->offset);
  if (base >= file_size_) return nullptr;
  return static_cast<const uint8_t*>(mapped_data_) + base;
}

inline bool GGUFParser::readTensorData(const std::string& name, void* dst, size_t bytes, size_t offset) const {
  const GGUFTensorInfo* info = getTensorInfo(name);
  if (!info || dst == nullptr) return false;
  return readTensorData(*info, dst, bytes, offset);
}

inline bool GGUFParser::readTensorData(const GGUFTensorInfo& info, void* dst, size_t bytes, size_t offset) const {
  if (dst == nullptr) return false;
  if (!file_parsed_) return false;
  if (offset > info.size) return false;
  const size_t max_bytes = static_cast<size_t>(info.size) - static_cast<size_t>(offset);
  const size_t to_read = std::min(bytes, max_bytes);
  if (to_read == 0) return true;

  const size_t base = static_cast<size_t>(tensor_data_offset_) + static_cast<size_t>(info.offset) + static_cast<size_t>(offset);

  if (use_mmap_) {
    if (mapped_data_ == nullptr) return false;
    if (base + to_read > file_size_) return false;
    std::memcpy(dst, static_cast<const uint8_t*>(mapped_data_) + base, to_read);
    return true;
  } else {
    std::ifstream file(file_path_, std::ios::binary);
    if (!file.is_open()) return false;
    file.seekg(static_cast<std::streamoff>(base), std::ios::beg);
    if (!file.good()) return false;
    file.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(to_read));
    return static_cast<size_t>(file.gcount()) == to_read;
  }
}

inline size_t GGUFParser::getTensorSize(const std::string& name) const {
  const GGUFTensorInfo* info = getTensorInfo(name);
  return info ? static_cast<size_t>(info->size) : 0ULL;
}

} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // __cplusplus

#endif // DUOROU_EXTENSIONS_OLLAMA_GGUF_PARSER_H