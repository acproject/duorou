#include "gguf_parser.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace duorou {
namespace extensions {
namespace ollama {

// GGUF magic number
static const uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
static const uint32_t GGUF_VERSION = 3;

// 支持的架构列表
const std::vector<std::string> GGUFParser::SUPPORTED_ARCHITECTURES = {
    "qwen25vl", "qwen2.5vl", "qwen-2.5vl",
    "qwen2vl", "qwen2", "llama", "mistral"
};

// GGUFKeyValue 数据提取函数实现
std::string GGUFKeyValue::asString() const {
  if (type != GGUFType::STRING || data.size() < 8) {
    return "";
  }

  uint64_t len;
  std::memcpy(&len, data.data(), 8);

  if (data.size() < 8 + len) {
    return "";
  }

  return std::string(reinterpret_cast<const char *>(data.data() + 8), len);
}

int32_t GGUFKeyValue::asInt32() const {
  if (type != GGUFType::INT32 || data.size() < 4) {
    return 0;
  }

  int32_t value;
  std::memcpy(&value, data.data(), 4);
  return value;
}

int64_t GGUFKeyValue::asInt64() const {
  if (type != GGUFType::INT64 || data.size() < 8) {
    return 0;
  }

  int64_t value;
  std::memcpy(&value, data.data(), 8);
  return value;
}

uint32_t GGUFKeyValue::asUInt32() const {
  if (type != GGUFType::UINT32 || data.size() < 4) {
    return 0;
  }

  uint32_t value;
  std::memcpy(&value, data.data(), 4);
  return value;
}

uint64_t GGUFKeyValue::asUInt64() const {
  if (type != GGUFType::UINT64 || data.size() < 8) {
    return 0;
  }

  uint64_t value;
  std::memcpy(&value, data.data(), 8);
  return value;
}

float GGUFKeyValue::asFloat32() const {
  if (type != GGUFType::FLOAT32 || data.size() < 4) {
    return 0.0f;
  }

  float value;
  std::memcpy(&value, data.data(), 4);
  return value;
}

double GGUFKeyValue::asFloat64() const {
  if (type != GGUFType::FLOAT64 || data.size() < 8) {
    return 0.0;
  }

  double value;
  std::memcpy(&value, data.data(), 8);
  return value;
}

bool GGUFKeyValue::asBool() const {
  if (type != GGUFType::BOOL || data.empty()) {
    return false;
  }

  return data[0] != 0;
}

std::vector<int32_t> GGUFKeyValue::asInt32Array() const {
  std::vector<int32_t> result;
  
  if (type != GGUFType::ARRAY || data.size() < 12) {
    return result;
  }

  // 读取数组类型
  uint32_t array_type;
  std::memcpy(&array_type, data.data(), 4);
  
  if (array_type != static_cast<uint32_t>(GGUFType::INT32)) {
    return result;
  }

  // 读取数组长度
  uint64_t array_length;
  std::memcpy(&array_length, data.data() + 4, 8);

  if (data.size() < 12 + array_length * 4) {
    return result;
  }

  result.reserve(array_length);
  const uint8_t* ptr = data.data() + 12;
  
  for (uint64_t i = 0; i < array_length; ++i) {
    int32_t value;
    std::memcpy(&value, ptr, 4);
    result.push_back(value);
    ptr += 4;
  }

  return result;
}

std::vector<uint64_t> GGUFKeyValue::asUInt64Array() const {
  std::vector<uint64_t> result;
  
  if (type != GGUFType::ARRAY || data.size() < 12) {
    return result;
  }

  // 读取数组类型
  uint32_t array_type;
  std::memcpy(&array_type, data.data(), 4);
  
  if (array_type != static_cast<uint32_t>(GGUFType::UINT64)) {
    return result;
  }

  // 读取数组长度
  uint64_t array_length;
  std::memcpy(&array_length, data.data() + 4, 8);

  // 添加安全检查：限制最大数组长度以防止内存耗尽
  const uint64_t MAX_ARRAY_LENGTH = 1000000;  // 最大100万个元素
  if (array_length > MAX_ARRAY_LENGTH) {
    // 记录错误但不抛出异常，返回空数组
    return result;
  }

  if (data.size() < 12 + array_length * 8) {
    return result;
  }

  // 使用try-catch保护内存分配
  try {
    result.reserve(array_length);
    const uint8_t* ptr = data.data() + 12;
    
    for (uint64_t i = 0; i < array_length; ++i) {
      uint64_t value;
      std::memcpy(&value, ptr, 8);
      result.push_back(value);
      ptr += 8;
    }
  } catch (const std::bad_alloc&) {
    // 内存分配失败，返回空数组
    result.clear();
  }

  return result;
}

std::vector<std::string> GGUFKeyValue::asStringArray() const {
  std::vector<std::string> result;
  
  if (type != GGUFType::ARRAY || data.size() < 12) {
    return result;
  }

  // 读取数组类型
  uint32_t array_type;
  std::memcpy(&array_type, data.data(), 4);
  
  if (array_type != static_cast<uint32_t>(GGUFType::STRING)) {
    return result;
  }

  // 读取数组长度
  uint64_t array_length;
  std::memcpy(&array_length, data.data() + 4, 8);

  // 添加安全检查：限制最大数组长度以防止内存耗尽
  const uint64_t MAX_ARRAY_LENGTH = 200000;  // 增加限制以支持大词汇表
  if (array_length > MAX_ARRAY_LENGTH) {
    // 记录错误但不抛出异常，返回空数组
    std::cout << "[DEBUG] Array length " << array_length << " exceeds maximum " << MAX_ARRAY_LENGTH << std::endl;
    return result;
  }

  // 使用try-catch保护内存分配
  try {
    result.reserve(array_length);
    const uint8_t* ptr = data.data() + 12;
    const uint8_t* end = data.data() + data.size();
    
    for (uint64_t i = 0; i < array_length; ++i) {
      if (ptr + 8 > end) {
        break; // 数据不足
      }
      
      // 读取字符串长度
      uint64_t str_length;
      std::memcpy(&str_length, ptr, 8);
      ptr += 8;
      
      // 检查单个字符串长度是否合理
      const uint64_t MAX_STRING_LENGTH = 1000000;  // 单个字符串最大1MB
      if (str_length > MAX_STRING_LENGTH || ptr + str_length > end) {
        break; // 字符串过长或数据不足
      }
      
      // 读取字符串内容
      std::string str(reinterpret_cast<const char*>(ptr), str_length);
      result.push_back(str);
      ptr += str_length;
    }
  } catch (const std::bad_alloc&) {
    // 内存分配失败，返回空数组
    result.clear();
  }

  return result;
}

// GGUFParser 实现
GGUFParser::GGUFParser(bool verbose)
    : verbose_(verbose), file_parsed_(false), tensor_data_offset_(0),
      use_mmap_(true), fd_(-1), mapped_data_(nullptr), file_size_(0), current_offset_(0) 
#ifdef _WIN32
    , file_handle_(INVALID_HANDLE_VALUE), mapping_handle_(nullptr)
#endif
{
  std::memset(&header_, 0, sizeof(header_));
  std::memset(&architecture_, 0, sizeof(architecture_));
  log("DEBUG", "GGUFParser initialized with verbose=" + std::to_string(verbose) + ", mmap enabled");
}

GGUFParser::~GGUFParser() {
  cleanupMmap();
  log("DEBUG", "GGUFParser destroyed");
}

bool GGUFParser::parseFile(const std::string &file_path) {
  file_path_ = file_path;
  file_parsed_ = false;
  metadata_.clear();
  tensor_infos_.clear();
  tensor_name_to_index_.clear();

  log("INFO", "Starting to parse GGUF file: " + file_path + (use_mmap_ ? " (using mmap)" : " (using ifstream)"));

  if (use_mmap_) {
    // 使用内存映射模式
    if (!initMmap(file_path)) {
      log("WARNING", "Failed to initialize mmap, falling back to ifstream");
      use_mmap_ = false;
    } else {
      return parseWithMmap();
    }
  }

  // 传统ifstream模式
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    log("ERROR", "Failed to open file: " + file_path);
    return false;
  }

  // 读取文件头
  if (!readHeader(file)) {
    log("ERROR", "Failed to read GGUF header");
    return false;
  }

  // 读取元数据
  try {
    if (!readMetadata(file)) {
      log("ERROR", "Failed to read GGUF metadata");
      return false;
    }
    log("DEBUG", "Metadata read successfully");
  } catch (const std::exception& e) {
    log("ERROR", "Exception in readMetadata: " + std::string(e.what()));
    return false;
  }

  // 读取张量信息
  try {
    if (!readTensorInfo(file)) {
      log("ERROR", "Failed to read GGUF tensor info");
      return false;
    }
    log("DEBUG", "Tensor info read successfully");
  } catch (const std::exception& e) {
    log("ERROR", "Exception in readTensorInfo: " + std::string(e.what()));
    return false;
  }

  // 记录张量数据偏移量
  tensor_data_offset_ = file.tellg();

  // 解析架构信息
  try {
    if (!parseArchitecture()) {
      log("ERROR", "Failed to parse architecture information");
      return false;
    }
    log("DEBUG", "Architecture parsed successfully");
  } catch (const std::exception& e) {
    log("ERROR", "Exception in parseArchitecture: " + std::string(e.what()));
    return false;
  }

  file_parsed_ = true;
  log("INFO", "Successfully parsed GGUF file: " + file_path);
  log("INFO", "Architecture: " + architecture_.name);
  log("INFO", "Metadata keys: " + std::to_string(metadata_.size()));
  log("INFO", "Tensor count: " + std::to_string(header_.tensor_count));

  return true;
}

const GGUFKeyValue* GGUFParser::getMetadata(const std::string &key) const {
  auto it = metadata_.find(key);
  return (it != metadata_.end()) ? &it->second : nullptr;
}

const GGUFTensorInfo* GGUFParser::getTensorInfo(const std::string &name) const {
  auto it = tensor_name_to_index_.find(name);
  if (it != tensor_name_to_index_.end() && it->second < tensor_infos_.size()) {
    return &tensor_infos_[it->second];
  }
  return nullptr;
}

bool GGUFParser::validateFile() const {
  if (!file_parsed_) {
    return false;
  }

  // 检查magic number
  if (header_.magic != GGUF_MAGIC) {
    log("ERROR", "Invalid GGUF magic number");
    return false;
  }

  // 检查版本
  if (header_.version != GGUF_VERSION) {
    log("WARNING", "GGUF version mismatch: expected " +
                       std::to_string(GGUF_VERSION) + ", got " +
                       std::to_string(header_.version));
  }

  // 打印所有可用的元数据键用于调试
  std::cout << "[DEBUG] GGUFParser: Available metadata keys:" << std::endl;
  for (const auto& pair : metadata_) {
    std::cout << "[DEBUG]   - " << pair.first << std::endl;
  }
  
  // 检查必需的元数据键
  const std::vector<std::string> required_keys = {"general.architecture"};

  for (const std::string &key : required_keys) {
    if (metadata_.find(key) == metadata_.end()) {
      log("ERROR", "Missing required metadata key: " + key);
      return false;
    }
  }

  return true;
}

bool GGUFParser::isSupportedArchitecture(const std::string& arch_name) {
  return std::find(SUPPORTED_ARCHITECTURES.begin(), 
                   SUPPORTED_ARCHITECTURES.end(), 
                   arch_name) != SUPPORTED_ARCHITECTURES.end();
}

bool GGUFParser::readHeader(std::ifstream &file) {
  file.read(reinterpret_cast<char *>(&header_), sizeof(header_));
  if (file.gcount() != sizeof(header_)) {
    log("ERROR", "Failed to read complete GGUF header");
    return false;
  }

  std::string debug_msg = "GGUF Header: magic=0x" + 
      std::to_string(header_.magic) + 
      ", version=" + std::to_string(header_.version) +
      ", tensor_count=" + std::to_string(header_.tensor_count) +
      ", metadata_kv_count=" + std::to_string(header_.metadata_kv_count);
  log("DEBUG", debug_msg);

  return true;
}

bool GGUFParser::readMetadata(std::ifstream &file) {
  for (uint64_t i = 0; i < header_.metadata_kv_count; ++i) {
    log("DEBUG", "Reading metadata key-value pair " + std::to_string(i + 1) + "/" + std::to_string(header_.metadata_kv_count));
    GGUFKeyValue kv = readKeyValue(file);
    if (kv.key.empty()) {
      log("ERROR", "Failed to read metadata key-value pair " + std::to_string(i));
      return false;
    }
    log("DEBUG", "Successfully read metadata key: " + kv.key + ", data size: " + std::to_string(kv.data.size()) + " bytes");
    metadata_[kv.key] = std::move(kv);
  }

  log("DEBUG", "Read " + std::to_string(metadata_.size()) + " metadata entries");
  return true;
}

bool GGUFParser::readTensorInfo(std::ifstream &file) {
  log("DEBUG", "Starting to read tensor info, tensor count: " + std::to_string(header_.tensor_count));
  
  try {
    tensor_infos_.reserve(header_.tensor_count);
    tensor_name_to_index_.reserve(header_.tensor_count);
    log("DEBUG", "Successfully reserved memory for " + std::to_string(header_.tensor_count) + " tensors");
  } catch (const std::exception& e) {
    log("ERROR", "Failed to reserve memory for tensors: " + std::string(e.what()));
    return false;
  }

  for (uint64_t i = 0; i < header_.tensor_count; ++i) {
    log("DEBUG", "Reading tensor " + std::to_string(i + 1) + "/" + std::to_string(header_.tensor_count));
    GGUFTensorInfo tensor_info;

    // 读取张量名称
    tensor_info.name = readString(file);
    log("DEBUG", "Read tensor name: " + tensor_info.name);
    if (tensor_info.name.empty()) {
      log("ERROR", "Failed to read tensor name for tensor " + std::to_string(i));
      return false;
    }

    // 读取维度数量
    file.read(reinterpret_cast<char *>(&tensor_info.n_dimensions), 4);
    if (file.gcount() != 4) {
      log("ERROR", "Failed to read n_dimensions for tensor " + tensor_info.name);
      return false;
    }

    // 读取各维度大小
    tensor_info.dimensions.resize(tensor_info.n_dimensions);
    for (uint32_t j = 0; j < tensor_info.n_dimensions; ++j) {
      file.read(reinterpret_cast<char *>(&tensor_info.dimensions[j]), 8);
      if (file.gcount() != 8) {
        log("ERROR", "Failed to read dimension " + std::to_string(j) + 
            " for tensor " + tensor_info.name);
        return false;
      }
    }

    // 读取数据类型
    uint32_t type_value;
    file.read(reinterpret_cast<char *>(&type_value), 4);
    if (file.gcount() != 4) {
      log("ERROR", "Failed to read type for tensor " + tensor_info.name);
      return false;
    }
    tensor_info.type = static_cast<GGMLTensorType>(type_value);

    // 读取偏移量
    file.read(reinterpret_cast<char *>(&tensor_info.offset), 8);
    if (file.gcount() != 8) {
      log("ERROR", "Failed to read offset for tensor " + tensor_info.name);
      return false;
    }

    // 计算张量大小
    tensor_info.size = calculateTensorSize(tensor_info);

    // 添加到列表和索引
    tensor_name_to_index_[tensor_info.name] = tensor_infos_.size();
    tensor_infos_.push_back(std::move(tensor_info));
  }

  log("DEBUG", "Read " + std::to_string(tensor_infos_.size()) + " tensor infos");
  return true;
}

bool GGUFParser::parseArchitecture() {
  // 获取架构名称
  const auto* arch_kv = getMetadata("general.architecture");
  if (!arch_kv) {
    log("ERROR", "Missing general.architecture metadata");
    return false;
  }
  
  architecture_.name = arch_kv->asString();
  
  if (!isSupportedArchitecture(architecture_.name)) {
    log("WARNING", "Unsupported architecture: " + architecture_.name);
  }

  // 根据架构类型解析参数
  std::string arch_prefix = architecture_.name;
  
  // 对于qwen25vl，使用qwen25vl前缀
  if (architecture_.name == "qwen25vl" || architecture_.name == "qwen2.5vl" || architecture_.name == "qwen-2.5vl") {
    arch_prefix = "qwen25vl";
  }

  // 解析基本架构参数
  if (const auto* kv = getMetadata(arch_prefix + ".context_length")) {
    architecture_.context_length = kv->asUInt32();
  }
  
  if (const auto* kv = getMetadata(arch_prefix + ".embedding_length")) {
    architecture_.embedding_length = kv->asUInt32();
  }
  
  if (const auto* kv = getMetadata(arch_prefix + ".block_count")) {
    architecture_.block_count = kv->asUInt32();
  }
  
  if (const auto* kv = getMetadata(arch_prefix + ".feed_forward_length")) {
    architecture_.feed_forward_length = kv->asUInt32();
  }
  
  if (const auto* kv = getMetadata(arch_prefix + ".attention.head_count")) {
    architecture_.attention_head_count = kv->asUInt32();
  }
  
  if (const auto* kv = getMetadata(arch_prefix + ".attention.head_count_kv")) {
    architecture_.attention_head_count_kv = kv->asUInt32();
  }
  
  if (const auto* kv = getMetadata(arch_prefix + ".attention.layer_norm_rms_epsilon")) {
    architecture_.layer_norm_rms_epsilon = kv->asFloat32();
  }
  
  if (const auto* kv = getMetadata(arch_prefix + ".rope.dimension_count")) {
    architecture_.rope_dimension_count = kv->asUInt32();
  }
  
  if (const auto* kv = getMetadata(arch_prefix + ".rope.freq_base")) {
    architecture_.rope_freq_base = kv->asFloat32();
  }
  
  // 解析RoPE维度分段（数组类型）
  if (const auto* kv = getMetadata(arch_prefix + ".rope.mrope_section")) {
    log("DEBUG", "Found rope.mrope_section metadata, data size: " + std::to_string(kv->data.size()) + " bytes");
    
    // 检查数组长度以避免内存问题
    if (kv->type == GGUFType::ARRAY && kv->data.size() >= 12) {
      uint64_t array_length;
      std::memcpy(&array_length, kv->data.data() + 4, 8);
      log("DEBUG", "rope.mrope_section array length: " + std::to_string(array_length));
      
      if (array_length > 1000000) {  // 限制数组长度避免内存问题
        log("ERROR", "rope.mrope_section array too large: " + std::to_string(array_length) + " elements");
      } else {
        auto sections = kv->asUInt64Array();
        architecture_.rope_dimension_sections = sections;
        log("DEBUG", "Successfully parsed rope.mrope_section with " + std::to_string(sections.size()) + " elements");
      }
    }
  }

  // 解析视觉相关参数（用于多模态模型）
  if (const auto* kv = getMetadata(arch_prefix + ".vision.patch_size")) {
    architecture_.has_vision = true;
    architecture_.vision_patch_size = kv->asUInt32();
  }
  
  if (const auto* kv = getMetadata(arch_prefix + ".vision.spatial_patch_size")) {
    architecture_.vision_spatial_patch_size = kv->asUInt32();
  }
  
  if (const auto* kv = getMetadata(arch_prefix + ".vision.fullatt_block_indexes")) {
    log("DEBUG", "Found vision.fullatt_block_indexes metadata, data size: " + std::to_string(kv->data.size()) + " bytes");
    
    // 检查数组长度以避免内存问题
    if (kv->type == GGUFType::ARRAY && kv->data.size() >= 12) {
      uint64_t array_length;
      std::memcpy(&array_length, kv->data.data() + 4, 8);
      log("DEBUG", "vision.fullatt_block_indexes array length: " + std::to_string(array_length));
      
      if (array_length > 1000000) {  // 限制数组长度避免内存问题
        log("ERROR", "vision.fullatt_block_indexes array too large: " + std::to_string(array_length) + " elements");
      } else {
        auto indexes = kv->asUInt64Array();
        architecture_.vision_fullatt_block_indexes = indexes;
        log("DEBUG", "Successfully parsed vision.fullatt_block_indexes with " + std::to_string(indexes.size()) + " elements");
      }
    }
  }

  log("INFO", "Parsed architecture: " + architecture_.name);
  log("INFO", "  Context length: " + std::to_string(architecture_.context_length));
  log("INFO", "  Embedding length: " + std::to_string(architecture_.embedding_length));
  log("INFO", "  Block count: " + std::to_string(architecture_.block_count));
  log("INFO", std::string("  Has vision: ") + (architecture_.has_vision ? "Yes" : "No"));

  return true;
}

std::string GGUFParser::readString(std::ifstream &file) {
  uint64_t len;
  file.read(reinterpret_cast<char *>(&len), 8);
  if (file.gcount() != 8) {
    return "";
  }

  if (len == 0) {
    return "";
  }

  std::string str(len, '\0');
  file.read(&str[0], len);
  if (static_cast<uint64_t>(file.gcount()) != len) {
    return "";
  }

  return str;
}

GGUFKeyValue GGUFParser::readKeyValue(std::ifstream &file) {
  GGUFKeyValue kv;

  // 读取键名
  kv.key = readString(file);
  if (kv.key.empty()) {
    return kv;
  }

  // 读取类型
  uint32_t type_value;
  file.read(reinterpret_cast<char *>(&type_value), 4);
  if (file.gcount() != 4) {
    kv.key.clear();
    return kv;
  }
  kv.type = static_cast<GGUFType>(type_value);

  // 根据类型读取数据
  switch (kv.type) {
    case GGUFType::UINT8:
    case GGUFType::INT8:
    case GGUFType::BOOL:
      kv.data.resize(1);
      file.read(reinterpret_cast<char *>(kv.data.data()), 1);
      break;

    case GGUFType::UINT16:
    case GGUFType::INT16:
      kv.data.resize(2);
      file.read(reinterpret_cast<char *>(kv.data.data()), 2);
      break;

    case GGUFType::UINT32:
    case GGUFType::INT32:
    case GGUFType::FLOAT32:
      kv.data.resize(4);
      file.read(reinterpret_cast<char *>(kv.data.data()), 4);
      break;

    case GGUFType::UINT64:
    case GGUFType::INT64:
    case GGUFType::FLOAT64:
      kv.data.resize(8);
      file.read(reinterpret_cast<char *>(kv.data.data()), 8);
      break;

    case GGUFType::STRING: {
      std::string str = readString(file);
      uint64_t len = str.length();
      kv.data.resize(8 + len);
      std::memcpy(kv.data.data(), &len, 8);
      std::memcpy(kv.data.data() + 8, str.c_str(), len);
      break;
    }

    case GGUFType::ARRAY: {
      // 读取数组类型
      uint32_t array_type;
      file.read(reinterpret_cast<char *>(&array_type), 4);
      if (file.gcount() != 4) {
        log("ERROR", "Failed to read array type for key: " + kv.key);
        kv.key.clear();
        return kv;
      }
      
      // 读取数组长度
      uint64_t array_length;
      file.read(reinterpret_cast<char *>(&array_length), 8);
      if (file.gcount() != 8) {
        log("ERROR", "Failed to read array length for key: " + kv.key);
        kv.key.clear();
        return kv;
      }
      
      // 验证数组长度合理性
      log("DEBUG", "Reading array for key: " + kv.key + ", type: " + std::to_string(array_type) + ", length: " + std::to_string(array_length));
      if (array_length > 1000000) { // 限制数组长度
        log("ERROR", "Array length too large: " + std::to_string(array_length) + " for key: " + kv.key);
        kv.key.clear();
        return kv;
      }
      
      // 处理不同类型的数组
      if (array_type == static_cast<uint32_t>(GGUFType::STRING)) {
        // STRING数组需要特殊处理，因为每个字符串都有长度前缀
        std::vector<uint8_t> temp_data;
        temp_data.resize(4 + 8); // 类型和长度
        std::memcpy(temp_data.data(), &array_type, 4);
        std::memcpy(temp_data.data() + 4, &array_length, 8);
        
        for (uint64_t i = 0; i < array_length; i++) {
          uint64_t str_length;
          file.read(reinterpret_cast<char *>(&str_length), 8);
          if (file.gcount() != 8) {
            log("ERROR", "Failed to read string length in array for key: " + kv.key);
            kv.key.clear();
            return kv;
          }
          
          // 添加字符串长度到数据中
          size_t old_size = temp_data.size();
          temp_data.resize(old_size + 8 + str_length);
          std::memcpy(temp_data.data() + old_size, &str_length, 8);
          
          // 读取字符串内容
          if (str_length > 0) {
            file.read(reinterpret_cast<char *>(temp_data.data() + old_size + 8), str_length);
            if (static_cast<uint64_t>(file.gcount()) != str_length) {
              log("ERROR", "Failed to read string data in array for key: " + kv.key);
              kv.key.clear();
              return kv;
            }
          }
        }
        
        kv.data = std::move(temp_data);
      } else {
        // 处理固定大小的数组元素
        uint32_t element_size = getTypeSize(static_cast<GGUFType>(array_type));
        if (element_size == 0) {
          log("ERROR", "Unsupported array element type: " + std::to_string(array_type) + " for key: " + kv.key);
          kv.key.clear();
          return kv;
        }
        
        // 分配数据空间
        size_t total_size = 4 + 8 + array_length * element_size;
        try {
          kv.data.resize(total_size);
        } catch (const std::exception& e) {
          log("ERROR", "Failed to allocate memory for array data: " + std::string(e.what()) + " for key: " + kv.key);
          kv.key.clear();
          return kv;
        }
        
        // 写入数组类型和长度
        std::memcpy(kv.data.data(), &array_type, 4);
        std::memcpy(kv.data.data() + 4, &array_length, 8);
        
        // 读取数组数据
        if (array_length > 0) {
          file.read(reinterpret_cast<char *>(kv.data.data() + 12), 
                    array_length * element_size);
          if (static_cast<uint64_t>(file.gcount()) != array_length * element_size) {
            log("ERROR", "Failed to read array data for key: " + kv.key);
            kv.key.clear();
            return kv;
          }
        }
      }
      break;
    }

    default:
      log("ERROR", "Unsupported GGUF type: " + std::to_string(static_cast<uint32_t>(kv.type)));
      kv.key.clear();
      return kv;
  }

  return kv;
}

uint64_t GGUFParser::calculateTensorSize(const GGUFTensorInfo& tensor_info) const {
  uint64_t total_elements = 1;
  for (uint64_t dim : tensor_info.dimensions) {
    total_elements *= dim;
  }
  
  uint32_t type_size = getGGMLTypeSize(tensor_info.type);
  uint64_t calculated_size = total_elements * type_size;
  
  // 对于量化类型，限制计算出的大小以避免内存问题
  if (tensor_info.type != GGMLTensorType::F32 && tensor_info.type != GGMLTensorType::F16) {
    // 量化tensor的实际大小通常比计算值小得多
    // 这里使用一个保守的估计来避免过度分配
    uint64_t max_size = 100 * 1024 * 1024; // 100MB限制
    if (calculated_size > max_size) {
      calculated_size = max_size;
    }
  }
  
  return calculated_size;
}

uint32_t GGUFParser::getTypeSize(GGUFType type) {
  switch (type) {
    case GGUFType::UINT8:
    case GGUFType::INT8:
    case GGUFType::BOOL:
      return 1;
    case GGUFType::UINT16:
    case GGUFType::INT16:
      return 2;
    case GGUFType::UINT32:
    case GGUFType::INT32:
    case GGUFType::FLOAT32:
      return 4;
    case GGUFType::UINT64:
    case GGUFType::INT64:
    case GGUFType::FLOAT64:
      return 8;
    default:
      return 0;
  }
}

uint32_t GGUFParser::getGGMLTypeSize(GGMLTensorType type) {
  switch (type) {
    case GGMLTensorType::F32:
      return 4;
    case GGMLTensorType::F16:
      return 2;
    case GGMLTensorType::Q4_0:
    case GGMLTensorType::Q4_1:
      return 1; // 约0.5 bytes per element (block-based)
    case GGMLTensorType::Q5_0:
    case GGMLTensorType::Q5_1:
      return 1; // 约0.625 bytes per element (block-based)
    case GGMLTensorType::Q8_0:
    case GGMLTensorType::Q8_1:
      return 1; // 约1.0625 bytes per element (block-based)
    case GGMLTensorType::Q2_K:
      return 1; // 约0.3125 bytes per element (block-based)
    case GGMLTensorType::Q3_K:
      return 1; // 约0.4375 bytes per element (block-based)
    case GGMLTensorType::Q4_K:
      return 1; // 约0.5625 bytes per element (block-based)
    case GGMLTensorType::Q5_K:
      return 1; // 约0.6875 bytes per element (block-based)
    case GGMLTensorType::Q6_K:
      return 1; // 约0.8125 bytes per element (block-based)
    case GGMLTensorType::Q8_K:
      return 1; // 约1.0 bytes per element (block-based)
    default:
      return 0;
  }
}

bool GGUFParser::initMmap(const std::string& file_path) {
#ifdef _WIN32
  // Windows implementation
  HANDLE hFile = CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ, 
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (hFile == INVALID_HANDLE_VALUE) {
    log("ERROR", "Failed to open file for mmap: " + file_path);
    return false;
  }

  LARGE_INTEGER fileSize;
  if (!GetFileSizeEx(hFile, &fileSize)) {
    log("ERROR", "Failed to get file size for mmap");
    CloseHandle(hFile);
    return false;
  }
  file_size_ = fileSize.QuadPart;

  HANDLE hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (hMapping == nullptr) {
    log("ERROR", "Failed to create file mapping");
    CloseHandle(hFile);
    return false;
  }

  mapped_data_ = static_cast<uint8_t*>(MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0));
  if (mapped_data_ == nullptr) {
    log("ERROR", "Failed to map view of file");
    CloseHandle(hMapping);
    CloseHandle(hFile);
    return false;
  }

  // Store handles for cleanup
  file_handle_ = hFile;
  mapping_handle_ = hMapping;
#else
  // Unix/Linux implementation
  fd_ = open(file_path.c_str(), O_RDONLY);
  if (fd_ == -1) {
    log("ERROR", "Failed to open file for mmap: " + file_path + ", error: " + strerror(errno));
    return false;
  }

  struct stat st;
  if (fstat(fd_, &st) == -1) {
    log("ERROR", "Failed to get file size for mmap: " + std::string(strerror(errno)));
    close(fd_);
    fd_ = -1;
    return false;
  }
  file_size_ = st.st_size;

  mapped_data_ = static_cast<uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
  if (mapped_data_ == MAP_FAILED) {
    log("ERROR", "Failed to mmap file: " + std::string(strerror(errno)));
    close(fd_);
    fd_ = -1;
    mapped_data_ = nullptr;
    return false;
  }
#endif

  current_offset_ = 0;
  log("DEBUG", "Successfully mapped file: " + file_path + ", size: " + std::to_string(file_size_) + " bytes");
  return true;
}

void GGUFParser::cleanupMmap() {
#ifdef _WIN32
  if (mapped_data_ != nullptr) {
    UnmapViewOfFile(mapped_data_);
    mapped_data_ = nullptr;
  }
  if (mapping_handle_ != nullptr) {
    CloseHandle(mapping_handle_);
    mapping_handle_ = nullptr;
  }
  if (file_handle_ != INVALID_HANDLE_VALUE) {
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
  }
#else
  if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
    munmap(mapped_data_, file_size_);
    mapped_data_ = nullptr;
  }
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
#endif
  file_size_ = 0;
  current_offset_ = 0;
}

bool GGUFParser::readFromMmap(void* buffer, size_t size) {
   if (!mapped_data_ || current_offset_ + size > file_size_) {
     log("ERROR", "Read beyond file boundary or invalid mmap data");
     return false;
   }
 
   std::memcpy(buffer, static_cast<uint8_t*>(mapped_data_) + current_offset_, size);
   current_offset_ += size;
   return true;
 }

std::string GGUFParser::readStringFromMmap() {
  uint64_t len;
  if (!readFromMmap(&len, 8)) {
    return "";
  }

  if (len == 0) {
    return "";
  }

  if (current_offset_ + len > file_size_) {
    log("ERROR", "String length exceeds file boundary");
    return "";
  }

  std::string str(reinterpret_cast<const char*>(static_cast<uint8_t*>(mapped_data_) + current_offset_), len);
   current_offset_ += len;
  return str;
}

bool GGUFParser::parseWithMmap() {
  current_offset_ = 0;

  // 读取文件头
  if (!readHeaderFromMmap()) {
    log("ERROR", "Failed to read GGUF header from mmap");
    return false;
  }

  // 读取元数据
  try {
    if (!readMetadataFromMmap()) {
      log("ERROR", "Failed to read GGUF metadata from mmap");
      return false;
    }
    log("DEBUG", "Metadata read successfully from mmap");
  } catch (const std::exception& e) {
    log("ERROR", "Exception in readMetadataFromMmap: " + std::string(e.what()));
    return false;
  }

  // 读取张量信息
  try {
    if (!readTensorInfoFromMmap()) {
      log("ERROR", "Failed to read GGUF tensor info from mmap");
      return false;
    }
    log("DEBUG", "Tensor info read successfully from mmap");
  } catch (const std::exception& e) {
    log("ERROR", "Exception in readTensorInfoFromMmap: " + std::string(e.what()));
    return false;
  }

  // 记录张量数据偏移量
  tensor_data_offset_ = current_offset_;

  // 解析架构信息
  try {
    if (!parseArchitecture()) {
      log("ERROR", "Failed to parse architecture information");
      return false;
    }
    log("DEBUG", "Architecture parsed successfully");
  } catch (const std::exception& e) {
    log("ERROR", "Exception in parseArchitecture: " + std::string(e.what()));
    return false;
  }

  file_parsed_ = true;
  log("INFO", "Successfully parsed GGUF file with mmap: " + file_path_);
  log("INFO", "Architecture: " + architecture_.name);
  log("INFO", "Metadata keys: " + std::to_string(metadata_.size()));
  log("INFO", "Tensor count: " + std::to_string(header_.tensor_count));

  return true;
}

bool GGUFParser::readHeaderFromMmap() {
  if (!readFromMmap(&header_, sizeof(header_))) {
    log("ERROR", "Failed to read complete GGUF header from mmap");
    return false;
  }

  std::string debug_msg = "GGUF Header: magic=0x" + 
      std::to_string(header_.magic) + 
      ", version=" + std::to_string(header_.version) +
      ", tensor_count=" + std::to_string(header_.tensor_count) +
      ", metadata_kv_count=" + std::to_string(header_.metadata_kv_count);
  log("DEBUG", debug_msg);

  return true;
}

bool GGUFParser::readMetadataFromMmap() {
  for (uint64_t i = 0; i < header_.metadata_kv_count; ++i) {
    log("DEBUG", "Reading metadata key-value pair " + std::to_string(i + 1) + "/" + std::to_string(header_.metadata_kv_count));
    GGUFKeyValue kv = readKeyValueFromMmap();
    if (kv.key.empty()) {
      log("ERROR", "Failed to read metadata key-value pair " + std::to_string(i));
      return false;
    }
    log("DEBUG", "Successfully read metadata key: " + kv.key + ", data size: " + std::to_string(kv.data.size()) + " bytes");
    metadata_[kv.key] = std::move(kv);
  }

  log("DEBUG", "Read " + std::to_string(metadata_.size()) + " metadata entries from mmap");
  return true;
}

bool GGUFParser::readTensorInfoFromMmap() {
  log("DEBUG", "Starting to read tensor info from mmap, tensor count: " + std::to_string(header_.tensor_count));
  
  try {
    tensor_infos_.reserve(header_.tensor_count);
    tensor_name_to_index_.reserve(header_.tensor_count);
    log("DEBUG", "Successfully reserved memory for " + std::to_string(header_.tensor_count) + " tensors");
  } catch (const std::exception& e) {
    log("ERROR", "Failed to reserve memory for tensors: " + std::string(e.what()));
    return false;
  }

  for (uint64_t i = 0; i < header_.tensor_count; ++i) {
    log("DEBUG", "Reading tensor " + std::to_string(i + 1) + "/" + std::to_string(header_.tensor_count));
    GGUFTensorInfo tensor_info;

    // 读取张量名称
    tensor_info.name = readStringFromMmap();
    log("DEBUG", "Read tensor name: " + tensor_info.name);
    if (tensor_info.name.empty()) {
      log("ERROR", "Failed to read tensor name for tensor " + std::to_string(i));
      return false;
    }

    // 读取维度数量
    if (!readFromMmap(&tensor_info.n_dimensions, 4)) {
      log("ERROR", "Failed to read n_dimensions for tensor " + tensor_info.name);
      return false;
    }

    // 读取各维度大小
    tensor_info.dimensions.resize(tensor_info.n_dimensions);
    for (uint32_t j = 0; j < tensor_info.n_dimensions; ++j) {
      if (!readFromMmap(&tensor_info.dimensions[j], 8)) {
        log("ERROR", "Failed to read dimension " + std::to_string(j) + 
            " for tensor " + tensor_info.name);
        return false;
      }
    }

    // 读取数据类型
    uint32_t type_value;
    if (!readFromMmap(&type_value, 4)) {
      log("ERROR", "Failed to read type for tensor " + tensor_info.name);
      return false;
    }
    tensor_info.type = static_cast<GGMLTensorType>(type_value);

    // 读取偏移量
    if (!readFromMmap(&tensor_info.offset, 8)) {
      log("ERROR", "Failed to read offset for tensor " + tensor_info.name);
      return false;
    }

    // 计算张量大小
    tensor_info.size = calculateTensorSize(tensor_info);

    // 添加到列表和索引
    tensor_name_to_index_[tensor_info.name] = tensor_infos_.size();
    tensor_infos_.push_back(std::move(tensor_info));
  }

  log("DEBUG", "Read " + std::to_string(tensor_infos_.size()) + " tensor infos from mmap");
  return true;
}

GGUFKeyValue GGUFParser::readKeyValueFromMmap() {
  GGUFKeyValue kv;

  // 读取键名
  kv.key = readStringFromMmap();
  if (kv.key.empty()) {
    return kv;
  }

  // 读取类型
  uint32_t type_value;
  if (!readFromMmap(&type_value, 4)) {
    kv.key.clear();
    return kv;
  }
  kv.type = static_cast<GGUFType>(type_value);

  // 根据类型读取数据
  switch (kv.type) {
    case GGUFType::UINT8:
    case GGUFType::INT8:
    case GGUFType::BOOL:
      kv.data.resize(1);
      if (!readFromMmap(kv.data.data(), 1)) {
        kv.key.clear();
        return kv;
      }
      break;

    case GGUFType::UINT16:
    case GGUFType::INT16:
      kv.data.resize(2);
      if (!readFromMmap(kv.data.data(), 2)) {
        kv.key.clear();
        return kv;
      }
      break;

    case GGUFType::UINT32:
    case GGUFType::INT32:
    case GGUFType::FLOAT32:
      kv.data.resize(4);
      if (!readFromMmap(kv.data.data(), 4)) {
        kv.key.clear();
        return kv;
      }
      break;

    case GGUFType::UINT64:
    case GGUFType::INT64:
    case GGUFType::FLOAT64:
      kv.data.resize(8);
      if (!readFromMmap(kv.data.data(), 8)) {
        kv.key.clear();
        return kv;
      }
      break;

    case GGUFType::STRING: {
      std::string str = readStringFromMmap();
      uint64_t len = str.length();
      kv.data.resize(8 + len);
      std::memcpy(kv.data.data(), &len, 8);
      std::memcpy(kv.data.data() + 8, str.c_str(), len);
      break;
    }

    case GGUFType::ARRAY: {
      // 读取数组类型
      uint32_t array_type;
      if (!readFromMmap(&array_type, 4)) {
        log("ERROR", "Failed to read array type for key: " + kv.key);
        kv.key.clear();
        return kv;
      }
      
      // 读取数组长度
      uint64_t array_length;
      if (!readFromMmap(&array_length, 8)) {
        log("ERROR", "Failed to read array length for key: " + kv.key);
        kv.key.clear();
        return kv;
      }
      
      // 验证数组长度合理性
      log("DEBUG", "Reading array for key: " + kv.key + ", type: " + std::to_string(array_type) + ", length: " + std::to_string(array_length));
      if (array_length > 1000000) { // 限制数组长度
        log("ERROR", "Array length too large: " + std::to_string(array_length) + " for key: " + kv.key);
        kv.key.clear();
        return kv;
      }
      
      // 处理不同类型的数组
      if (array_type == static_cast<uint32_t>(GGUFType::STRING)) {
        // STRING数组需要特殊处理，因为每个字符串都有长度前缀
        std::vector<uint8_t> temp_data;
        temp_data.resize(4 + 8); // 类型和长度
        std::memcpy(temp_data.data(), &array_type, 4);
        std::memcpy(temp_data.data() + 4, &array_length, 8);
        
        for (uint64_t i = 0; i < array_length; i++) {
          uint64_t str_length;
          if (!readFromMmap(&str_length, 8)) {
            log("ERROR", "Failed to read string length in array for key: " + kv.key);
            kv.key.clear();
            return kv;
          }
          
          // 添加字符串长度到数据中
          size_t old_size = temp_data.size();
          temp_data.resize(old_size + 8 + str_length);
          std::memcpy(temp_data.data() + old_size, &str_length, 8);
          
          // 读取字符串内容
          if (str_length > 0) {
            if (!readFromMmap(temp_data.data() + old_size + 8, str_length)) {
              log("ERROR", "Failed to read string data in array for key: " + kv.key);
              kv.key.clear();
              return kv;
            }
          }
        }
        
        kv.data = std::move(temp_data);
      } else {
        // 处理固定大小的数组元素
        uint32_t element_size = getTypeSize(static_cast<GGUFType>(array_type));
        if (element_size == 0) {
          log("ERROR", "Unsupported array element type: " + std::to_string(array_type) + " for key: " + kv.key);
          kv.key.clear();
          return kv;
        }
        
        // 分配数据空间
        size_t total_size = 4 + 8 + array_length * element_size;
        try {
          kv.data.resize(total_size);
        } catch (const std::exception& e) {
          log("ERROR", "Failed to allocate memory for array data: " + std::string(e.what()) + " for key: " + kv.key);
          kv.key.clear();
          return kv;
        }
        
        // 写入数组类型和长度
        std::memcpy(kv.data.data(), &array_type, 4);
        std::memcpy(kv.data.data() + 4, &array_length, 8);
        
        // 读取数组数据
        if (array_length > 0) {
          if (!readFromMmap(kv.data.data() + 12, array_length * element_size)) {
            log("ERROR", "Failed to read array data for key: " + kv.key);
            kv.key.clear();
            return kv;
          }
        }
      }
      break;
    }

    default:
      log("ERROR", "Unsupported GGUF type: " + std::to_string(static_cast<uint32_t>(kv.type)));
      kv.key.clear();
      return kv;
  }

  return kv;
}

void GGUFParser::log(const std::string &level, const std::string &message) const {
  if (verbose_ || level == "ERROR") {
    std::cout << "[" << level << "] GGUFParser: " << message << std::endl;
  }
}

} // namespace ollama
} // namespace extensions
} // namespace duorou