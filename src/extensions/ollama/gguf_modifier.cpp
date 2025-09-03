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
    skipped_tensors_.clear();
}

GGUFModifier::~GGUFModifier() = default;

bool GGUFModifier::loadFile(const std::string& file_path) {
    file_path_ = file_path;
    file_loaded_ = false;
    metadata_.clear();
    tensor_infos_.clear();
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
    
    // 读取张量信息
    if (!readTensorInfo(file)) {
        log("ERROR", "Failed to read GGUF tensor info");
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
    log("INFO", "Tensor info count: " + std::to_string(tensor_infos_.size()));
    
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
    
    // 更新header中的metadata计数
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
    
    // 计算新的tensor data偏移量
    size_t new_tensor_data_offset = file.tellp();
    
    // 计算tensor info部分的大小
    size_t tensor_info_size = 0;
    for (const auto& tensor_info : tensor_infos_) {
        tensor_info_size += 8; // name length
        tensor_info_size += tensor_info.name.length(); // name content
        tensor_info_size += 4; // n_dimensions
        tensor_info_size += tensor_info.n_dimensions * 8; // dimensions
        tensor_info_size += 4; // type
        tensor_info_size += 8; // offset
    }
    
    // 计算对齐后的tensor data开始位置
    const size_t GGUF_DEFAULT_ALIGNMENT = 32;
    uint64_t tensor_info_end = new_tensor_data_offset + tensor_info_size;
    size_t padding_needed = (GGUF_DEFAULT_ALIGNMENT - (tensor_info_end % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT;
    uint64_t aligned_tensor_data_start = tensor_info_end + padding_needed;
    
    // 更新所有tensor的偏移量
    for (auto& tensor_info : tensor_infos_) {
        // 计算tensor相对于原始tensor data开始位置的偏移量
        uint64_t relative_offset = tensor_info.offset - tensor_data_offset_;
        // 设置新的绝对偏移量（相对于对齐后的tensor data开始位置）
        tensor_info.offset = relative_offset;
    }
    
    log("INFO", "Updated tensor offsets: old_tensor_data_start=" + std::to_string(tensor_data_offset_) + 
        ", aligned_tensor_data_start=" + std::to_string(aligned_tensor_data_start) + 
        ", padding_needed=" + std::to_string(padding_needed));
    
    // 写入张量信息
    if (!writeTensorInfo(file)) {
        log("ERROR", "Failed to write GGUF tensor info");
        return false;
    }
    
    // 确保tensor data按照32字节对齐
    size_t current_pos = file.tellp();
    size_t actual_padding_needed = (GGUF_DEFAULT_ALIGNMENT - (current_pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT;
    
    if (actual_padding_needed > 0) {
        std::vector<char> padding(actual_padding_needed, 0);
        file.write(padding.data(), actual_padding_needed);
        log("DEBUG", "Added " + std::to_string(actual_padding_needed) + " bytes of padding for alignment");
    }
    
    // 记录tensor data开始位置
    size_t tensor_data_start_pos = file.tellp();
    
    // 写入张量数据
    file.write(reinterpret_cast<const char*>(tensor_data_.data()), tensor_data_.size());
    
    log("DEBUG", "Tensor data written at position: " + std::to_string(tensor_data_start_pos));
    
    log("INFO", "Successfully saved GGUF file: " + output_path);
    return true;
}

bool GGUFModifier::saveOptimizedFile(const std::string& output_path) {
    if (!file_loaded_) {
        log("ERROR", "No file loaded");
        return false;
    }
    
    log("INFO", "Creating optimized GGUF file: " + output_path);
    
    // 创建输出文件
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        log("ERROR", "Failed to create output file: " + output_path);
        return false;
    }
    
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
    
    // 计算新的tensor data偏移量
    size_t tensor_info_size = 0;
    for (const auto& tensor_info : tensor_infos_) {
        tensor_info_size += 8; // name length (uint64_t)
        tensor_info_size += tensor_info.name.size(); // name string
        tensor_info_size += 4; // n_dimensions (uint32_t)
        tensor_info_size += tensor_info.n_dimensions * 8; // dimensions (uint64_t each)
        tensor_info_size += 4; // type (uint32_t)
        tensor_info_size += 8; // offset (uint64_t)
    }
    
    uint64_t new_tensor_data_offset = static_cast<uint64_t>(file.tellp()) + tensor_info_size;
    
    // 计算对齐后的tensor data开始位置
    const size_t GGUF_DEFAULT_ALIGNMENT = 32;
    uint64_t tensor_info_end = new_tensor_data_offset;
    size_t padding_needed = (GGUF_DEFAULT_ALIGNMENT - (tensor_info_end % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT;
    uint64_t aligned_tensor_data_start = tensor_info_end + padding_needed;
    
    // 更新所有tensor的偏移量（相对于新的tensor data开始位置）
    for (auto& tensor_info : tensor_infos_) {
        // 计算tensor相对于原始tensor data开始位置的偏移量
        uint64_t relative_offset = tensor_info.offset - tensor_data_offset_;
        // 设置新的绝对偏移量（相对于对齐后的tensor data开始位置）
        tensor_info.offset = aligned_tensor_data_start + relative_offset;
    }
    
    log("INFO", "Updated tensor offsets: old_tensor_data_start=" + std::to_string(tensor_data_offset_) + 
        ", aligned_tensor_data_start=" + std::to_string(aligned_tensor_data_start) + 
        ", padding_needed=" + std::to_string(padding_needed));
    
    // 写入张量信息
    if (!writeTensorInfo(file)) {
        log("ERROR", "Failed to write GGUF tensor info");
        return false;
    }
    
    // 确保tensor data按照32字节对齐
    size_t current_pos = file.tellp();
    size_t actual_padding_needed = (GGUF_DEFAULT_ALIGNMENT - (current_pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT;
    
    if (actual_padding_needed > 0) {
        std::vector<char> padding(actual_padding_needed, 0);
        file.write(padding.data(), actual_padding_needed);
        log("DEBUG", "Added " + std::to_string(actual_padding_needed) + " bytes of padding for alignment");
    }
    
    // 记录当前位置（应该等于aligned_tensor_data_start）
    size_t tensor_data_start_pos = file.tellp();
    file.close();
    
    // 现在使用文件拼接的方式添加原始的tensor data
    // 打开原始文件读取tensor data
    std::ifstream original_file(file_path_, std::ios::binary);
    if (!original_file.is_open()) {
        log("ERROR", "Failed to open original file for tensor data copy: " + file_path_);
        return false;
    }
    
    // 重新打开输出文件进行追加
    std::ofstream output_file(output_path, std::ios::binary | std::ios::app);
    if (!output_file.is_open()) {
        log("ERROR", "Failed to reopen output file for tensor data append: " + output_path);
        return false;
    }
    
    // 定位到原始文件的tensor data开始位置
    original_file.seekg(tensor_data_offset_);
    
    size_t total_copied = 0;
    
    if (skipped_tensors_.empty()) {
        // 没有跳过的tensor，直接复制所有tensor data
        const size_t buffer_size = 1024 * 1024; // 1MB buffer
        std::vector<char> buffer(buffer_size);
        
        while (original_file && total_copied < tensor_data_.size()) {
            size_t to_read = std::min(buffer_size, tensor_data_.size() - total_copied);
            original_file.read(buffer.data(), to_read);
            size_t actually_read = original_file.gcount();
            
            if (actually_read > 0) {
                output_file.write(buffer.data(), actually_read);
                total_copied += actually_read;
            }
            
            if (actually_read < to_read) {
                break; // EOF or error
            }
        }
    } else {
        // 有跳过的tensor，需要选择性复制
        log("INFO", "Copying tensor data while skipping " + std::to_string(skipped_tensors_.size()) + " tensors");
        
        // 创建跳过区间的排序列表
        std::vector<std::pair<uint64_t, uint64_t>> skip_ranges;
        for (const auto& skipped : skipped_tensors_) {
            uint64_t start = skipped.offset - tensor_data_offset_;
            uint64_t end = start + skipped.size;
            skip_ranges.push_back({start, end});
        }
        
        // 按起始位置排序
        std::sort(skip_ranges.begin(), skip_ranges.end());
        
        const size_t buffer_size = 1024 * 1024; // 1MB buffer
         std::vector<char> buffer(buffer_size);
         size_t current_pos = 0;
        
        for (const auto& skip_range : skip_ranges) {
            // 复制跳过区间之前的数据
            if (current_pos < skip_range.first) {
                size_t copy_size = skip_range.first - current_pos;
                original_file.seekg(tensor_data_offset_ + current_pos);
                
                while (copy_size > 0) {
                    size_t to_read = std::min(buffer_size, copy_size);
                    original_file.read(buffer.data(), to_read);
                    size_t actually_read = original_file.gcount();
                    
                    if (actually_read > 0) {
                        output_file.write(buffer.data(), actually_read);
                        total_copied += actually_read;
                        copy_size -= actually_read;
                    } else {
                        break;
                    }
                }
            }
            
            // 跳过当前区间
            current_pos = skip_range.second;
            log("DEBUG", "Skipped tensor data from " + std::to_string(skip_range.first) + 
                " to " + std::to_string(skip_range.second));
        }
        
        // 复制最后一个跳过区间之后的数据
        if (current_pos < tensor_data_.size()) {
            size_t copy_size = tensor_data_.size() - current_pos;
            original_file.seekg(tensor_data_offset_ + current_pos);
            
            while (copy_size > 0) {
                size_t to_read = std::min(buffer_size, copy_size);
                original_file.read(buffer.data(), to_read);
                size_t actually_read = original_file.gcount();
                
                if (actually_read > 0) {
                    output_file.write(buffer.data(), actually_read);
                    total_copied += actually_read;
                    copy_size -= actually_read;
                } else {
                    break;
                }
            }
        }
    }
    
    original_file.close();
    output_file.close();
    
    log("DEBUG", "Tensor data copied: " + std::to_string(total_copied) + " bytes");
    log("DEBUG", "Tensor data written at position: " + std::to_string(tensor_data_start_pos));
    
    log("INFO", "Successfully created optimized GGUF file: " + output_path);
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
    
    // 调试特定键
    if (kv.key == "tokenizer.ggml.pre") {
        log("DEBUG", "Reading tokenizer.ggml.pre with type: " + std::to_string(static_cast<uint32_t>(kv.type)) + " (STRING=8, ARRAY=9)");
    }
    
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
            
            // 检查是否是关键数组，需要保留的数组
            // 注意：tokenizer.ggml.pre是STRING类型，不是数组，所以要排除
            bool is_important_array = ((kv.key.find("tokenizer.ggml") != std::string::npos && kv.key != "tokenizer.ggml.pre") ||
                                    (kv.key.find("general.architecture") != std::string::npos) ||
                                    (kv.key.find(".attention.head_count") != std::string::npos) ||
                                    (kv.key.find(".embedding_length") != std::string::npos) ||
                                    (kv.key.find(".feed_forward_length") != std::string::npos) ||
                                    (kv.key.find(".block_count") != std::string::npos) ||
                                    (kv.key.find(".rope") != std::string::npos) ||
                                    (kv.key.find(".vision") != std::string::npos));
            
            if (is_important_array) {
                log("DEBUG", "Preserving important array: " + kv.key + " with type " + std::to_string(array_type) + " and length " + std::to_string(array_length));
                
                // 计算数组数据的总大小
                size_t array_data_size = 0;
                std::streampos start_pos = file.tellg();
                
                // 先计算数据大小
                for (uint64_t i = 0; i < array_length; ++i) {
                    switch (array_type) {
                        case 0: case 1: case 7: // UINT8, INT8, BOOL
                            array_data_size += 1;
                            file.seekg(1, std::ios::cur);
                            break;
                        case 2: case 3: // UINT16, INT16
                            array_data_size += 2;
                            file.seekg(2, std::ios::cur);
                            break;
                        case 4: case 5: case 6: // UINT32, INT32, FLOAT32
                            array_data_size += 4;
                            file.seekg(4, std::ios::cur);
                            break;
                        case 10: case 11: case 12: // UINT64, INT64, FLOAT64
                            array_data_size += 8;
                            file.seekg(8, std::ios::cur);
                            break;
                        case 8: { // STRING
                            std::string str = readString(file);
                            array_data_size += 8 + str.length(); // 8字节长度 + 字符串内容
                            break;
                        }
                        default:
                            log("WARNING", "Unknown array element type: " + std::to_string(array_type));
                            array_data_size += 8;
                            file.seekg(8, std::ios::cur);
                            break;
                    }
                }
                
                // 回到数组数据开始位置
                file.seekg(start_pos);
                
                // 读取完整的数组数据
                kv.data.resize(12 + array_data_size); // 4字节类型 + 8字节长度 + 数组数据
                std::memcpy(kv.data.data(), &array_type, 4);
                std::memcpy(kv.data.data() + 4, &array_length, 8);
                file.read(reinterpret_cast<char*>(kv.data.data() + 12), array_data_size);
            } else {
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
            }
            break;
        }
            
        default:
            throw std::runtime_error("Unknown GGUF type: " + std::to_string(static_cast<uint32_t>(kv.type)));
    }
    
    return kv;
}

void GGUFModifier::writeKeyValue(std::ofstream& file, const GGUFKeyValue& kv) const {
    // 创建一个可修改的副本以进行类型修正
    GGUFKeyValue corrected_kv = kv;
    
    // 调试特定键
    if (kv.key == "tokenizer.ggml.pre") {
        log("DEBUG", "Writing tokenizer.ggml.pre with type: " + std::to_string(static_cast<uint32_t>(kv.type)) + ", data size: " + std::to_string(kv.data.size()));
    }
    
    // 写入键名
    writeString(file, corrected_kv.key);
    
    // 写入类型
    uint32_t type_value = static_cast<uint32_t>(corrected_kv.type);
    file.write(reinterpret_cast<const char*>(&type_value), 4);
    
    // 对于数组类型，需要特殊处理
    if (corrected_kv.type == GGUFType::ARRAY) {
        if (corrected_kv.data.size() >= 12) {
            // 读取元素类型和长度
            uint32_t array_type;
            uint64_t array_length;
            std::memcpy(&array_type, corrected_kv.data.data(), 4);
            std::memcpy(&array_length, corrected_kv.data.data() + 4, 8);
            
            file.write(reinterpret_cast<const char*>(&array_type), 4);
            
            // 检查是否是重要数组
            // 注意：tokenizer.ggml.pre是STRING类型，不是数组，所以要排除
            bool is_important_array = ((corrected_kv.key.find("tokenizer.ggml") != std::string::npos && corrected_kv.key != "tokenizer.ggml.pre") ||
                                    (corrected_kv.key.find("general.architecture") != std::string::npos) ||
                                    (corrected_kv.key.find(".attention.head_count") != std::string::npos) ||
                                    (corrected_kv.key.find(".embedding_length") != std::string::npos) ||
                                    (corrected_kv.key.find(".feed_forward_length") != std::string::npos) ||
                                    (corrected_kv.key.find(".block_count") != std::string::npos) ||
                                    (corrected_kv.key.find(".rope") != std::string::npos) ||
                                    (corrected_kv.key.find(".vision") != std::string::npos));
            
            if (is_important_array && corrected_kv.data.size() > 12) {
                // 写入原始长度和数据
                file.write(reinterpret_cast<const char*>(&array_length), 8);
                file.write(reinterpret_cast<const char*>(corrected_kv.data.data() + 12), corrected_kv.data.size() - 12);
                log("DEBUG", "Writing preserved important array for key: " + corrected_kv.key + ", length: " + std::to_string(array_length));
            } else {
                // 写入0长度，因为我们跳过了原始数组数据
                uint64_t zero_length = 0;
                file.write(reinterpret_cast<const char*>(&zero_length), 8);
                log("DEBUG", "Writing empty array for key: " + corrected_kv.key + ", original length: " + std::to_string(array_length));
            }
        } else {
            log("ERROR", "Invalid array data size for key: " + corrected_kv.key);
            // 写入默认的空数组
            uint32_t default_type = 8; // STRING
            uint64_t zero_length = 0;
            file.write(reinterpret_cast<const char*>(&default_type), 4);
            file.write(reinterpret_cast<const char*>(&zero_length), 8);
        }
    } else {
        // 写入数据
        file.write(reinterpret_cast<const char*>(corrected_kv.data.data()), corrected_kv.data.size());
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

bool GGUFModifier::readTensorInfo(std::ifstream& file) {
    tensor_infos_.clear();
    tensor_infos_.reserve(header_.tensor_count);
    
    for (uint64_t i = 0; i < header_.tensor_count; ++i) {
        GGUFTensorInfo tensor_info;
        
        // 读取张量名称
        try {
            tensor_info.name = readString(file);
            if (verbose_ && i < 5) {
                log("DEBUG", "Read tensor " + std::to_string(i) + " name: '" + tensor_info.name + "'");
            }
        } catch (const std::exception& e) {
            log("ERROR", "Exception reading tensor name for tensor " + std::to_string(i) + ": " + e.what());
            return false;
        }
        
        if (tensor_info.name.empty()) {
            log("ERROR", "Empty tensor name for tensor " + std::to_string(i));
            return false;
        }
        
        // 读取维度数量
        file.read(reinterpret_cast<char*>(&tensor_info.n_dimensions), sizeof(uint32_t));
        if (file.fail()) {
            log("ERROR", "Failed to read n_dimensions for tensor " + tensor_info.name);
            return false;
        }
        
        // 读取各维度大小
        tensor_info.dimensions.resize(tensor_info.n_dimensions);
        for (uint32_t j = 0; j < tensor_info.n_dimensions; ++j) {
            file.read(reinterpret_cast<char*>(&tensor_info.dimensions[j]), sizeof(uint64_t));
            if (file.fail()) {
                log("ERROR", "Failed to read dimension " + std::to_string(j) + " for tensor " + tensor_info.name);
                return false;
            }
        }
        
        // 读取张量数据类型
        uint32_t type_value;
        file.read(reinterpret_cast<char*>(&type_value), sizeof(uint32_t));
        if (file.fail()) {
            log("ERROR", "Failed to read type for tensor " + tensor_info.name);
            return false;
        }
        tensor_info.type = static_cast<GGUFType>(type_value);
        
        // 读取张量数据偏移量
        file.read(reinterpret_cast<char*>(&tensor_info.offset), sizeof(uint64_t));
        if (file.fail()) {
            log("ERROR", "Failed to read offset for tensor " + tensor_info.name);
            return false;
        }
        
        // 检查是否是错误存储为tensor的tokenizer.ggml.pre
        if (tensor_info.name == "tokenizer.ggml.pre") {
            log("INFO", "Found tokenizer.ggml.pre incorrectly stored as tensor with " + 
                std::to_string(tensor_info.n_dimensions) + " dimensions, converting to metadata");
            
            // 计算这个tensor占用的数据大小
            uint64_t tensor_size = 1;
            for (uint32_t j = 0; j < tensor_info.n_dimensions; ++j) {
                tensor_size *= tensor_info.dimensions[j];
            }
            
            // 根据tensor类型计算字节大小
            size_t element_size = 1; // 默认为1字节
            switch (tensor_info.type) {
                case GGUFType::UINT8:
                case GGUFType::INT8:
                    element_size = 1;
                    break;
                case GGUFType::UINT16:
                case GGUFType::INT16:
                    element_size = 2;
                    break;
                case GGUFType::UINT32:
                case GGUFType::INT32:
                case GGUFType::FLOAT32:
                    element_size = 4;
                    break;
                case GGUFType::UINT64:
                case GGUFType::INT64:
                case GGUFType::FLOAT64:
                    element_size = 8;
                    break;
                default:
                    element_size = 1;
                    break;
            }
            
            size_t skipped_tensor_size = tensor_size * element_size;
            log("INFO", "Skipping tokenizer.ggml.pre tensor data of size: " + std::to_string(skipped_tensor_size) + " bytes");
            
            // 调整后续所有tensor的偏移量
            for (uint64_t j = i + 1; j < header_.tensor_count; ++j) {
                // 我们需要预读后续tensor的信息来调整它们的偏移量
                // 但这会使逻辑变得复杂，所以我们采用另一种方法：
                // 记录需要跳过的tensor，在后续处理中调整偏移量
            }
            
            // 创建正确的metadata条目
            auto tokenizer_pre_kv = GGUFKeyValue::createString("tokenizer.ggml.pre", "qwen2");
            metadata_["tokenizer.ggml.pre"] = tokenizer_pre_kv;
            log("INFO", "Added tokenizer.ggml.pre to metadata as STRING type");
            
            // 记录跳过的tensor信息，用于后续调整偏移量
            skipped_tensors_.push_back({tensor_info.offset, skipped_tensor_size});
            
            // 不将其添加到tensor_infos_中
            continue;
        }
        
        tensor_infos_.push_back(std::move(tensor_info));
        
        if (verbose_) {
            log("DEBUG", "Read tensor info: " + tensor_info.name + 
                ", dimensions: " + std::to_string(tensor_info.n_dimensions) +
                ", type: " + std::to_string(static_cast<uint32_t>(tensor_info.type)) +
                ", offset: " + std::to_string(tensor_info.offset));
        }
    }
    
    log("INFO", "Successfully read " + std::to_string(tensor_infos_.size()) + " tensor infos");
    
    // 调整所有tensor的偏移量，考虑跳过的tensor
    if (!skipped_tensors_.empty()) {
        log("INFO", "Adjusting tensor offsets for " + std::to_string(skipped_tensors_.size()) + " skipped tensors");
        
        for (auto& tensor_info : tensor_infos_) {
            size_t total_skipped_size = 0;
            
            // 计算在当前tensor之前跳过的所有tensor的总大小
            for (const auto& skipped : skipped_tensors_) {
                if (skipped.offset < tensor_info.offset) {
                    total_skipped_size += skipped.size;
                }
            }
            
            // 调整偏移量
            if (total_skipped_size > 0) {
                uint64_t old_offset = tensor_info.offset;
                tensor_info.offset -= total_skipped_size;
                log("DEBUG", "Adjusted tensor " + tensor_info.name + " offset from " + 
                    std::to_string(old_offset) + " to " + std::to_string(tensor_info.offset));
            }
        }
    }
    
    // 更新header中的tensor count以反映实际的tensor数量
    if (header_.tensor_count != tensor_infos_.size()) {
        log("INFO", "Updating header tensor count from " + std::to_string(header_.tensor_count) + 
            " to " + std::to_string(tensor_infos_.size()));
        header_.tensor_count = tensor_infos_.size();
    }
    return true;
}

bool GGUFModifier::writeTensorInfo(std::ofstream& file) const {
    for (const auto& tensor_info : tensor_infos_) {
        // 写入张量名称
        writeString(file, tensor_info.name);
        
        // 限制维度数量不超过4（llama.cpp的限制）
        uint32_t limited_dimensions = std::min(tensor_info.n_dimensions, static_cast<uint32_t>(4));
        if (tensor_info.n_dimensions > 4) {
            log("WARNING", "Tensor " + tensor_info.name + " has " + std::to_string(tensor_info.n_dimensions) + 
                " dimensions, limiting to 4 for llama.cpp compatibility");
        }
        
        // 写入维度数量
        file.write(reinterpret_cast<const char*>(&limited_dimensions), sizeof(uint32_t));
        
        // 写入各维度大小（只写入前4个维度）
        for (uint32_t i = 0; i < limited_dimensions; ++i) {
            file.write(reinterpret_cast<const char*>(&tensor_info.dimensions[i]), sizeof(uint64_t));
        }
        
        // 写入张量数据类型
        uint32_t type_value = static_cast<uint32_t>(tensor_info.type);
        file.write(reinterpret_cast<const char*>(&type_value), sizeof(uint32_t));
        
        // 写入张量数据偏移量
        file.write(reinterpret_cast<const char*>(&tensor_info.offset), sizeof(uint64_t));
        
        if (file.fail()) {
            log("ERROR", "Failed to write tensor info for: " + tensor_info.name);
            return false;
        }
        
        if (verbose_) {
            log("DEBUG", "Wrote tensor info: " + tensor_info.name + 
                ", dimensions: " + std::to_string(tensor_info.n_dimensions) +
                ", type: " + std::to_string(static_cast<uint32_t>(tensor_info.type)) +
                ", offset: " + std::to_string(tensor_info.offset));
        }
    }
    
    log("INFO", "Successfully wrote " + std::to_string(tensor_infos_.size()) + " tensor infos");
    return true;
}

void GGUFModifier::log(const std::string& level, const std::string& message) const {
    if (verbose_ || level == "ERROR") {
        std::cout << "[" << level << "] GGUFModifier: " << message << std::endl;
    }
}

} // namespace ollama
} // namespace extensions
} // namespace duorou