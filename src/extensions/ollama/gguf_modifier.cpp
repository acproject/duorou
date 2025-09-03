#include "gguf_modifier.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <iomanip>

namespace duorou {
namespace extensions {
namespace ollama {

// GGUF magic number
static const uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
static const uint32_t GGUF_VERSION = 3;

// GGUFKeyValue 静态方法实现
GGUFKeyValue GGUFKeyValue::createString(const std::string& key, const std::string& value) {
    GGUFKeyValue kv;
    kv.key = key;
    kv.type = GGUFType::STRING;
    
    // 字符串格式：长度(8字节) + 内容
    uint64_t len = value.length();
    kv.data.resize(8 + len);
    std::memcpy(kv.data.data(), &len, 8);
    std::memcpy(kv.data.data() + 8, value.c_str(), len);
    
    return kv;
}

GGUFKeyValue GGUFKeyValue::createInt32(const std::string& key, int32_t value) {
    GGUFKeyValue kv;
    kv.key = key;
    kv.type = GGUFType::INT32;
    kv.data.resize(4);
    std::memcpy(kv.data.data(), &value, 4);
    return kv;
}

GGUFKeyValue GGUFKeyValue::createFloat32(const std::string& key, float value) {
    GGUFKeyValue kv;
    kv.key = key;
    kv.type = GGUFType::FLOAT32;
    kv.data.resize(4);
    std::memcpy(kv.data.data(), &value, 4);
    return kv;
}

GGUFKeyValue GGUFKeyValue::createBool(const std::string& key, bool value) {
    GGUFKeyValue kv;
    kv.key = key;
    kv.type = GGUFType::BOOL;
    kv.data.resize(1);
    kv.data[0] = value ? 1 : 0;
    return kv;
}

std::string GGUFKeyValue::asString() const {
    if (type != GGUFType::STRING || data.size() < 8) {
        return "";
    }
    
    uint64_t len;
    std::memcpy(&len, data.data(), 8);
    
    if (data.size() < 8 + len) {
        return "";
    }
    
    return std::string(reinterpret_cast<const char*>(data.data() + 8), len);
}

int32_t GGUFKeyValue::asInt32() const {
    if (type != GGUFType::INT32 || data.size() < 4) {
        return 0;
    }
    
    int32_t value;
    std::memcpy(&value, data.data(), 4);
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

bool GGUFKeyValue::asBool() const {
    if (type != GGUFType::BOOL || data.empty()) {
        return false;
    }
    
    return data[0] != 0;
}

// GGUFModifier 实现
GGUFModifier::GGUFModifier(bool verbose)
    : verbose_(verbose), file_loaded_(false), tensor_data_offset_(0) {
    std::memset(&header_, 0, sizeof(header_));
}

GGUFModifier::~GGUFModifier() = default;

bool GGUFModifier::loadFile(const std::string& file_path) {
    file_path_ = file_path;
    file_loaded_ = false;
    metadata_.clear();
    tensor_data_.clear();
    
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
    if (!readMetadata(file)) {
        log("ERROR", "Failed to read GGUF metadata");
        return false;
    }
    
    // 记录张量数据偏移量
    tensor_data_offset_ = file.tellg();
    
    // 读取剩余的张量数据
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(tensor_data_offset_);
    
    size_t tensor_data_size = file_size - tensor_data_offset_;
    tensor_data_.resize(tensor_data_size);
    file.read(reinterpret_cast<char*>(tensor_data_.data()), tensor_data_size);
    
    file_loaded_ = true;
    log("INFO", "Successfully loaded GGUF file: " + file_path);
    log("INFO", "Metadata keys: " + std::to_string(metadata_.size()));
    log("INFO", "Tensor count: " + std::to_string(header_.tensor_count));
    
    return true;
}

bool GGUFModifier::saveFile(const std::string& output_path) {
    if (!file_loaded_) {
        log("ERROR", "No file loaded");
        return false;
    }
    
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        log("ERROR", "Failed to create output file: " + output_path);
        return false;
    }
    
    // 更新元数据数量
    header_.metadata_kv_count = metadata_.size();
    
    // 写入文件头
    if (!writeHeader(file)) {
        log("ERROR", "Failed to write GGUF header");
        return false;
    }
    
    // 写入元数据
    if (!writeMetadata(file)) {
        log("ERROR", "Failed to write GGUF metadata");
        return false;
    }
    
    // 写入张量数据
    file.write(reinterpret_cast<const char*>(tensor_data_.data()), tensor_data_.size());
    
    log("INFO", "Successfully saved GGUF file: " + output_path);
    return true;
}

bool GGUFModifier::applyArchitectureMapping(const ArchitectureMapping& mapping) {
    if (!file_loaded_) {
        log("ERROR", "No file loaded");
        return false;
    }
    
    std::string current_arch = getCurrentArchitecture();
    if (current_arch != mapping.source_arch) {
        log("WARNING", "Current architecture (" + current_arch + 
            ") does not match mapping source (" + mapping.source_arch + ")");
    }
    
    log("INFO", "Applying architecture mapping: " + mapping.source_arch + " -> " + mapping.target_arch);
    
    // 应用键名映射
    std::unordered_map<std::string, GGUFKeyValue> new_metadata;
    for (const auto& [key, kv] : metadata_) {
        auto it = mapping.key_mappings.find(key);
        if (it != mapping.key_mappings.end()) {
            // 重命名键
            GGUFKeyValue new_kv = kv;
            new_kv.key = it->second;
            new_metadata[it->second] = new_kv;
            log("INFO", "Mapped key: " + key + " -> " + it->second);
        } else {
            // 保持原键名
            new_metadata[key] = kv;
        }
    }
    
    // 移除指定的键
    for (const std::string& key : mapping.keys_to_remove) {
        auto it = new_metadata.find(key);
        if (it != new_metadata.end()) {
            new_metadata.erase(it);
            log("INFO", "Removed key: " + key);
        }
    }
    
    // 添加额外的键值对
    for (const auto& [key, kv] : mapping.additional_keys) {
        new_metadata[key] = kv;
        log("INFO", "Added key: " + key);
    }
    
    // 更新架构名称
    auto arch_kv = GGUFKeyValue::createString("general.architecture", mapping.target_arch);
    new_metadata["general.architecture"] = arch_kv;
    
    metadata_ = std::move(new_metadata);
    
    log("INFO", "Architecture mapping applied successfully");
    return true;
}

const GGUFKeyValue* GGUFModifier::getMetadata(const std::string& key) const {
    auto it = metadata_.find(key);
    return (it != metadata_.end()) ? &it->second : nullptr;
}

bool GGUFModifier::setMetadata(const GGUFKeyValue& kv) {
    metadata_[kv.key] = kv;
    log("INFO", "Set metadata key: " + kv.key);
    return true;
}

bool GGUFModifier::removeMetadata(const std::string& key) {
    auto it = metadata_.find(key);
    if (it != metadata_.end()) {
        metadata_.erase(it);
        log("INFO", "Removed metadata key: " + key);
        return true;
    }
    return false;
}

std::string GGUFModifier::getCurrentArchitecture() const {
    const auto* kv = getMetadata("general.architecture");
    return kv ? kv->asString() : "unknown";
}

bool GGUFModifier::setArchitecture(const std::string& arch_name) {
    auto kv = GGUFKeyValue::createString("general.architecture", arch_name);
    return setMetadata(kv);
}

std::vector<std::string> GGUFModifier::listMetadataKeys() const {
    std::vector<std::string> keys;
    keys.reserve(metadata_.size());
    
    for (const auto& [key, _] : metadata_) {
        keys.push_back(key);
    }
    
    std::sort(keys.begin(), keys.end());
    return keys;
}

bool GGUFModifier::validateFile() const {
    if (!file_loaded_) {
        return false;
    }
    
    // 检查magic number
    if (header_.magic != GGUF_MAGIC) {
        log("ERROR", "Invalid GGUF magic number");
        return false;
    }
    
    // 检查版本
    if (header_.version != GGUF_VERSION) {
        log("WARNING", "GGUF version mismatch: expected " + std::to_string(GGUF_VERSION) + 
            ", got " + std::to_string(header_.version));
    }
    
    // 检查必需的元数据键
    const std::vector<std::string> required_keys = {
        "general.architecture",
        "general.name"
    };
    
    for (const std::string& key : required_keys) {
        if (metadata_.find(key) == metadata_.end()) {
            log("ERROR", "Missing required metadata key: " + key);
            return false;
        }
    }
    
    return true;
}

ArchitectureMapping GGUFModifier::createArchitectureMapping(
    const std::string& source_arch,
    const std::string& target_arch) {
    
    ArchitectureMapping mapping;
    mapping.source_arch = source_arch;
    mapping.target_arch = target_arch;
    
    // 基础架构映射规则
    if (source_arch == "llama" && target_arch == "mistral") {
        // Llama到Mistral的映射
        mapping.key_mappings["llama.attention.head_count"] = "mistral.attention.head_count";
        mapping.key_mappings["llama.attention.head_count_kv"] = "mistral.attention.head_count_kv";
        mapping.key_mappings["llama.embedding_length"] = "mistral.embedding_length";
        mapping.key_mappings["llama.feed_forward_length"] = "mistral.feed_forward_length";
        mapping.key_mappings["llama.block_count"] = "mistral.block_count";
        mapping.key_mappings["llama.rope.dimension_count"] = "mistral.rope.dimension_count";
        mapping.key_mappings["llama.rope.freq_base"] = "mistral.rope.freq_base";
        mapping.key_mappings["llama.attention.layer_norm_rms_epsilon"] = "mistral.attention.layer_norm_rms_epsilon";
    } else if (source_arch == "mistral" && target_arch == "llama") {
        // Mistral到Llama的映射
        mapping.key_mappings["mistral.attention.head_count"] = "llama.attention.head_count";
        mapping.key_mappings["mistral.attention.head_count_kv"] = "llama.attention.head_count_kv";
        mapping.key_mappings["mistral.embedding_length"] = "llama.embedding_length";
        mapping.key_mappings["mistral.feed_forward_length"] = "llama.feed_forward_length";
        mapping.key_mappings["mistral.block_count"] = "llama.block_count";
        mapping.key_mappings["mistral.rope.dimension_count"] = "llama.rope.dimension_count";
        mapping.key_mappings["mistral.rope.freq_base"] = "llama.rope.freq_base";
        mapping.key_mappings["mistral.attention.layer_norm_rms_epsilon"] = "llama.attention.layer_norm_rms_epsilon";
    } else if (source_arch == "qwen25vl" && target_arch == "qwen2vl") {
        // Qwen2.5VL到Qwen2VL的映射
        mapping.key_mappings["qwen25vl.context_length"] = "qwen2vl.context_length";
        mapping.key_mappings["qwen25vl.embedding_length"] = "qwen2vl.embedding_length";
        mapping.key_mappings["qwen25vl.block_count"] = "qwen2vl.block_count";
        mapping.key_mappings["qwen25vl.feed_forward_length"] = "qwen2vl.feed_forward_length";
        mapping.key_mappings["qwen25vl.attention.head_count"] = "qwen2vl.attention.head_count";
        mapping.key_mappings["qwen25vl.attention.head_count_kv"] = "qwen2vl.attention.head_count_kv";
        mapping.key_mappings["qwen25vl.attention.layer_norm_rms_epsilon"] = "qwen2vl.attention.layer_norm_rms_epsilon";
        mapping.key_mappings["qwen25vl.rope.dimension_count"] = "qwen2vl.rope.dimension_count";
        mapping.key_mappings["qwen25vl.rope.freq_base"] = "qwen2vl.rope.freq_base";
        mapping.key_mappings["qwen25vl.rope.mrope_section"] = "qwen2vl.rope.mrope_section";
        mapping.key_mappings["qwen25vl.rope.dimension_sections"] = "qwen2vl.rope.dimension_sections";
        mapping.key_mappings["qwen25vl.vision.patch_size"] = "qwen2vl.vision.patch_size";
        mapping.key_mappings["qwen25vl.vision.spatial_patch_size"] = "qwen2vl.vision.spatial_patch_size";
        mapping.key_mappings["qwen25vl.vision.fullatt_block_indexes"] = "qwen2vl.vision.fullatt_block_indexes";
        
        // 添加llama.cpp期望但原始文件中可能缺失的键
        mapping.additional_keys["qwen2vl.rope.dimension_sections"] = GGUFKeyValue::createInt32("qwen2vl.rope.dimension_sections", 128);
    }
    
    return mapping;
}

// 私有方法实现
bool GGUFModifier::readHeader(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&header_), sizeof(header_));
    
    if (file.gcount() != sizeof(header_)) {
        return false;
    }
    
    if (header_.magic != GGUF_MAGIC) {
        log("ERROR", "Invalid GGUF magic number: 0x" + 
            std::to_string(header_.magic));
        return false;
    }
    
    log("INFO", "GGUF version: " + std::to_string(header_.version));
    log("INFO", "Tensor count: " + std::to_string(header_.tensor_count));
    log("INFO", "Metadata KV count: " + std::to_string(header_.metadata_kv_count));
    
    return true;
}

bool GGUFModifier::readMetadata(std::ifstream& file) {
    metadata_.clear();
    
    for (uint64_t i = 0; i < header_.metadata_kv_count; ++i) {
        try {
            GGUFKeyValue kv = readKeyValue(file);
            metadata_[kv.key] = std::move(kv);
        } catch (const std::exception& e) {
            log("ERROR", "Failed to read metadata entry " + std::to_string(i) + ": " + e.what());
            return false;
        }
    }
    
    return true;
}

bool GGUFModifier::writeHeader(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&header_), sizeof(header_));
    return file.good();
}

bool GGUFModifier::writeMetadata(std::ofstream& file) const {
    for (const auto& [key, kv] : metadata_) {
        try {
            writeKeyValue(file, kv);
        } catch (const std::exception& e) {
            log("ERROR", "Failed to write metadata entry " + key + ": " + e.what());
            return false;
        }
    }
    
    return true;
}

std::string GGUFModifier::readString(std::ifstream& file) {
    uint64_t len;
    file.read(reinterpret_cast<char*>(&len), 8);
    
    if (len > 1024 * 1024) { // 1MB limit for safety
        throw std::runtime_error("String too long: " + std::to_string(len));
    }
    
    std::string str(len, '\0');
    file.read(&str[0], len);
    
    return str;
}

void GGUFModifier::writeString(std::ofstream& file, const std::string& str) const {
    uint64_t len = str.length();
    file.write(reinterpret_cast<const char*>(&len), 8);
    file.write(str.c_str(), len);
}

GGUFKeyValue GGUFModifier::readKeyValue(std::ifstream& file) {
    GGUFKeyValue kv;
    
    // 读取键名
    kv.key = readString(file);
    
    // 读取类型
    uint32_t type_value;
    file.read(reinterpret_cast<char*>(&type_value), 4);
    kv.type = static_cast<GGUFType>(type_value);
    
    // 根据类型读取数据
    switch (kv.type) {
        case GGUFType::UINT8:
        case GGUFType::INT8:
        case GGUFType::BOOL:
            kv.data.resize(1);
            file.read(reinterpret_cast<char*>(kv.data.data()), 1);
            break;
            
        case GGUFType::UINT16:
        case GGUFType::INT16:
            kv.data.resize(2);
            file.read(reinterpret_cast<char*>(kv.data.data()), 2);
            break;
            
        case GGUFType::UINT32:
        case GGUFType::INT32:
        case GGUFType::FLOAT32:
            kv.data.resize(4);
            file.read(reinterpret_cast<char*>(kv.data.data()), 4);
            break;
            
        case GGUFType::UINT64:
        case GGUFType::INT64:
        case GGUFType::FLOAT64:
            kv.data.resize(8);
            file.read(reinterpret_cast<char*>(kv.data.data()), 8);
            break;
            
        case GGUFType::STRING: {
            std::string value = readString(file);
            uint64_t len = value.length();
            kv.data.resize(8 + len);
            std::memcpy(kv.data.data(), &len, 8);
            std::memcpy(kv.data.data() + 8, value.c_str(), len);
            break;
        }
        
        case GGUFType::ARRAY: {
            // 读取数组元素类型
            uint32_t array_type;
            file.read(reinterpret_cast<char*>(&array_type), 4);
            
            // 读取数组长度
            uint64_t array_length;
            file.read(reinterpret_cast<char*>(&array_length), 8);
            
            log("DEBUG", "Skipping array with type " + std::to_string(array_type) + " and length " + std::to_string(array_length));
            
            // 跳过数组数据
            for (uint64_t i = 0; i < array_length; ++i) {
                switch (array_type) {
                    case 0: case 1: case 7: // UINT8, INT8, BOOL
                        file.seekg(1, std::ios::cur);
                        break;
                    case 2: case 3: // UINT16, INT16
                        file.seekg(2, std::ios::cur);
                        break;
                    case 4: case 5: case 6: // UINT32, INT32, FLOAT32
                        file.seekg(4, std::ios::cur);
                        break;
                    case 10: case 11: case 12: // UINT64, INT64, FLOAT64
                        file.seekg(8, std::ios::cur);
                        break;
                    case 8: { // STRING
                        std::string str = readString(file);
                        break;
                    }
                    default:
                        log("WARNING", "Unknown array element type: " + std::to_string(array_type));
                        // 尝试跳过，假设是8字节
                        file.seekg(8, std::ios::cur);
                        break;
                }
            }
            
            // 为数组创建一个占位符数据
            kv.data.resize(12); // 4字节类型 + 8字节长度
            std::memcpy(kv.data.data(), &array_type, 4);
            std::memcpy(kv.data.data() + 4, &array_length, 8);
            break;
        }
            
        default:
            throw std::runtime_error("Unknown GGUF type: " + std::to_string(static_cast<uint32_t>(kv.type)));
    }
    
    return kv;
}

void GGUFModifier::writeKeyValue(std::ofstream& file, const GGUFKeyValue& kv) const {
    // 写入键名
    writeString(file, kv.key);
    
    // 写入类型
    uint32_t type_value = static_cast<uint32_t>(kv.type);
    file.write(reinterpret_cast<const char*>(&type_value), 4);
    
    // 对于数组类型，需要特殊处理
    if (kv.type == GGUFType::ARRAY) {
        // 数组数据格式：4字节元素类型 + 8字节数组长度 + 实际数据
        // 但我们只保存了占位符数据，所以直接写入0长度数组
        if (kv.data.size() >= 12) {
            // 写入元素类型和长度（从占位符数据中读取）
            uint32_t array_type;
            uint64_t array_length;
            std::memcpy(&array_type, kv.data.data(), 4);
            std::memcpy(&array_length, kv.data.data() + 4, 8);
            
            file.write(reinterpret_cast<const char*>(&array_type), 4);
            
            // 写入0长度，因为我们跳过了原始数组数据
            uint64_t zero_length = 0;
            file.write(reinterpret_cast<const char*>(&zero_length), 8);
            
            log("DEBUG", "Writing empty array for key: " + kv.key + ", original length: " + std::to_string(array_length));
        } else {
            log("ERROR", "Invalid array data size for key: " + kv.key);
            // 写入默认的空数组
            uint32_t default_type = 8; // STRING
            uint64_t zero_length = 0;
            file.write(reinterpret_cast<const char*>(&default_type), 4);
            file.write(reinterpret_cast<const char*>(&zero_length), 8);
        }
    } else {
        // 写入数据
        file.write(reinterpret_cast<const char*>(kv.data.data()), kv.data.size());
    }
}

size_t GGUFModifier::calculateMetadataSize() const {
    size_t size = 0;
    
    for (const auto& [key, kv] : metadata_) {
        size += 8; // key length
        size += key.length(); // key content
        size += 4; // type
        size += kv.data.size(); // data
    }
    
    return size;
}

void GGUFModifier::log(const std::string& level, const std::string& message) const {
    if (verbose_ || level == "ERROR") {
        std::cout << "[" << level << "] GGUFModifier: " << message << std::endl;
    }
}

} // namespace ollama
} // namespace extensions
} // namespace duorou