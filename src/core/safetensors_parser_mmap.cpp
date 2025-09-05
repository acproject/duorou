#include "safetensors_parser_mmap.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <set>
#include <cerrno>

namespace duorou {

// Simple JSON parser for SafeTensors header
class SimpleJsonParser {
public:
    static std::unordered_map<std::string, std::string> parseWeightMap(const std::string& json) {
        std::unordered_map<std::string, std::string> result;
        
        // Find "weight_map" section
        size_t weight_map_pos = json.find("\"weight_map\"");
        if (weight_map_pos == std::string::npos) {
            return result;
        }
        
        // Find opening brace after "weight_map"
        size_t brace_start = json.find("{", weight_map_pos);
        if (brace_start == std::string::npos) {
            return result;
        }
        
        // Find matching closing brace
        int brace_count = 1;
        size_t pos = brace_start + 1;
        size_t brace_end = std::string::npos;
        
        while (pos < json.length() && brace_count > 0) {
            if (json[pos] == '{') {
                brace_count++;
            } else if (json[pos] == '}') {
                brace_count--;
                if (brace_count == 0) {
                    brace_end = pos;
                    break;
                }
            }
            pos++;
        }
        
        if (brace_end == std::string::npos) {
            return result;
        }
        
        // Parse key-value pairs
        std::string weight_map_content = json.substr(brace_start + 1, brace_end - brace_start - 1);
        
        size_t start = 0;
        while (start < weight_map_content.length()) {
            // Find key
            size_t key_start = weight_map_content.find('\"', start);
            if (key_start == std::string::npos) break;
            
            size_t key_end = weight_map_content.find('\"', key_start + 1);
            if (key_end == std::string::npos) break;
            
            std::string key = weight_map_content.substr(key_start + 1, key_end - key_start - 1);
            
            // Find value
            size_t colon_pos = weight_map_content.find(':', key_end);
            if (colon_pos == std::string::npos) break;
            
            size_t value_start = weight_map_content.find('\"', colon_pos);
            if (value_start == std::string::npos) break;
            
            size_t value_end = weight_map_content.find('\"', value_start + 1);
            if (value_end == std::string::npos) break;
            
            std::string value = weight_map_content.substr(value_start + 1, value_end - value_start - 1);
            
            result[key] = value;
            start = value_end + 1;
        }
        
        return result;
    }
};

SafeTensorsParserMmap::SafeTensorsParserMmap(bool verbose) 
    : header_size_(0), file_size_(0), verbose_(verbose), use_mmap_(true), 
      fd_(-1), mapped_data_(nullptr)
#ifdef _WIN32
    , file_handle_(INVALID_HANDLE_VALUE), mapping_handle_(nullptr)
#endif
{
}

SafeTensorsParserMmap::~SafeTensorsParserMmap() {
    cleanupMmap();
}

bool SafeTensorsParserMmap::initMmap(const std::string& file_path) {
#ifdef _WIN32
    // Windows implementation
    file_handle_ = CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        log("ERROR", "Failed to open file for mmap: " + file_path);
        return false;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(file_handle_, &fileSize)) {
        log("ERROR", "Failed to get file size for mmap");
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }
    file_size_ = fileSize.QuadPart;

    mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_handle_ == nullptr) {
        log("ERROR", "Failed to create file mapping");
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    mapped_data_ = static_cast<uint8_t*>(MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
    if (mapped_data_ == nullptr) {
        log("ERROR", "Failed to map view of file");
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        mapping_handle_ = nullptr;
        file_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }
#else
    // Unix/Linux implementation
    fd_ = open(file_path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        log("ERROR", "Failed to open file for mmap: " + file_path + 
                     ", error: " + strerror(errno));
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

    mapped_data_ = static_cast<uint8_t*>(
        mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (mapped_data_ == MAP_FAILED) {
        log("ERROR", "Failed to mmap file: " + std::string(strerror(errno)));
        close(fd_);
        fd_ = -1;
        mapped_data_ = nullptr;
        return false;
    }
#endif

    log("DEBUG", "Successfully mapped SafeTensors file: " + file_path + 
                 ", size: " + std::to_string(file_size_) + " bytes");
    return true;
}

void SafeTensorsParserMmap::cleanupMmap() {
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
}

bool SafeTensorsParserMmap::loadFile(const std::string& filepath) {
    filepath_ = filepath;
    
    // Initialize memory mapping
    if (!initMmap(filepath)) {
        return false;
    }
    
    if (file_size_ < 8) {
        log("ERROR", "File too small to contain SafeTensors header");
        return false;
    }
    
    // Read header size (first 8 bytes, little-endian) from mapped memory
    uint64_t header_size_raw;
    std::memcpy(&header_size_raw, mapped_data_, 8);
    header_size_ = header_size_raw;
    
    if (header_size_ > file_size_ - 8) {
        log("ERROR", "Invalid header size: " + std::to_string(header_size_));
        return false;
    }
    
    // Read header JSON from mapped memory
    std::string header_json(reinterpret_cast<const char*>(mapped_data_ + 8), header_size_);
    
    return parseHeader(header_json);
}

bool SafeTensorsParserMmap::parseHeader(const std::string& json_str) {
    // Simple JSON parsing for SafeTensors format
    // Expected format: {"tensor_name": {"dtype": "F32", "shape": [dim1, dim2, ...], "data_offsets": [start, end]}, ...}
    
    tensors_.clear();
    
    // This is a simplified parser - in production, use a proper JSON library
    size_t pos = 0;
    size_t data_offset = 8 + header_size_;  // Start after header
    
    while (pos < json_str.length()) {
        // Find tensor name
        size_t name_start = json_str.find('\"', pos);
        if (name_start == std::string::npos) break;
        
        size_t name_end = json_str.find('\"', name_start + 1);
        if (name_end == std::string::npos) break;
        
        std::string tensor_name = json_str.substr(name_start + 1, name_end - name_start - 1);
        
        // Skip if this is metadata
        if (tensor_name == "__metadata__") {
            pos = name_end + 1;
            continue;
        }
        
        TensorInfo info;
        info.name = tensor_name;
        
        // Find dtype
        size_t dtype_pos = json_str.find("\"dtype\"", name_end);
        if (dtype_pos != std::string::npos) {
            size_t dtype_start = json_str.find('\"', dtype_pos + 7);
            size_t dtype_end = json_str.find('\"', dtype_start + 1);
            if (dtype_start != std::string::npos && dtype_end != std::string::npos) {
                info.dtype = json_str.substr(dtype_start + 1, dtype_end - dtype_start - 1);
            }
        }
        
        // Find shape
        size_t shape_pos = json_str.find("\"shape\"", name_end);
        if (shape_pos != std::string::npos) {
            size_t shape_start = json_str.find('[', shape_pos);
            size_t shape_end = json_str.find(']', shape_start);
            if (shape_start != std::string::npos && shape_end != std::string::npos) {
                std::string shape_str = json_str.substr(shape_start + 1, shape_end - shape_start - 1);
                std::istringstream iss(shape_str);
                std::string dim;
                while (std::getline(iss, dim, ',')) {
                    // Remove whitespace
                    dim.erase(std::remove_if(dim.begin(), dim.end(), ::isspace), dim.end());
                    if (!dim.empty()) {
                        info.shape.push_back(std::stoull(dim));
                    }
                }
            }
        }
        
        // Find data_offsets
        size_t offsets_pos = json_str.find("\"data_offsets\"", name_end);
        if (offsets_pos != std::string::npos) {
            size_t offsets_start = json_str.find('[', offsets_pos);
            size_t offsets_end = json_str.find(']', offsets_start);
            if (offsets_start != std::string::npos && offsets_end != std::string::npos) {
                std::string offsets_str = json_str.substr(offsets_start + 1, offsets_end - offsets_start - 1);
                std::istringstream iss(offsets_str);
                std::string offset;
                std::vector<size_t> offsets;
                while (std::getline(iss, offset, ',')) {
                    // Remove whitespace
                    offset.erase(std::remove_if(offset.begin(), offset.end(), ::isspace), offset.end());
                    if (!offset.empty()) {
                        offsets.push_back(std::stoull(offset));
                    }
                }
                if (offsets.size() >= 2) {
                    info.data_offset = data_offset + offsets[0];
                    info.data_size = offsets[1] - offsets[0];
                }
            }
        }
        
        tensors_[tensor_name] = info;
        
        // Move to next tensor
        size_t next_brace = json_str.find('}', name_end);
        if (next_brace == std::string::npos) break;
        pos = next_brace + 1;
    }
    
    log("INFO", "Parsed " + std::to_string(tensors_.size()) + " tensors from SafeTensors file");
    return !tensors_.empty();
}

size_t SafeTensorsParserMmap::getDataTypeSize(const std::string& dtype) const {
    if (dtype == "F32") return 4;
    if (dtype == "F16" || dtype == "BF16") return 2;
    if (dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "I64" || dtype == "U64") return 8;
    return 4; // Default to F32
}

const TensorInfo* SafeTensorsParserMmap::getTensorInfo(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

std::vector<std::string> SafeTensorsParserMmap::getTensorNames() const {
    std::vector<std::string> names;
    for (const auto& pair : tensors_) {
        names.push_back(pair.first);
    }
    return names;
}

bool SafeTensorsParserMmap::hasTensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const void* SafeTensorsParserMmap::getTensorDataPtr(const std::string& name) const {
    const TensorInfo* info = getTensorInfo(name);
    if (!info || !mapped_data_) {
        return nullptr;
    }
    
    if (info->data_offset >= file_size_ || 
        info->data_offset + info->data_size > file_size_) {
        log("ERROR", "Tensor data offset out of bounds for: " + name);
        return nullptr;
    }
    
    return mapped_data_ + info->data_offset;
}

std::vector<float> SafeTensorsParserMmap::getTensorAsFloat(const std::string& name) const {
    const TensorInfo* info = getTensorInfo(name);
    if (!info) {
        return {};
    }
    
    size_t element_count = 1;
    for (size_t dim : info->shape) {
        element_count *= dim;
    }
    
    const void* raw_data = getTensorDataPtr(name);
    if (!raw_data) {
        return {};
    }
    
    std::vector<float> float_data(element_count);
    convertToFloat(raw_data, float_data.data(), element_count, info->dtype);
    
    return float_data;
}

void SafeTensorsParserMmap::convertToFloat(const void* src_data, float* dst_data, 
                                         size_t element_count, const std::string& dtype) const {
    if (dtype == "F32") {
        std::memcpy(dst_data, src_data, element_count * sizeof(float));
    } else if (dtype == "F16") {
        // Convert F16 to F32 (simplified)
        const uint16_t* src = static_cast<const uint16_t*>(src_data);
        for (size_t i = 0; i < element_count; ++i) {
            // Simple F16 to F32 conversion (not IEEE compliant)
            uint32_t f32_bits = (static_cast<uint32_t>(src[i]) << 16);
            dst_data[i] = *reinterpret_cast<const float*>(&f32_bits);
        }
    } else if (dtype == "BF16") {
        // Convert BF16 to F32
        const uint16_t* src = static_cast<const uint16_t*>(src_data);
        for (size_t i = 0; i < element_count; ++i) {
            // BF16 to F32: BF16 is the upper 16 bits of F32
            uint32_t f32_bits = (static_cast<uint32_t>(src[i]) << 16);
            dst_data[i] = *reinterpret_cast<const float*>(&f32_bits);
        }
    } else {
        // Default: treat as F32
        std::memcpy(dst_data, src_data, element_count * sizeof(float));
    }
}

void SafeTensorsParserMmap::log(const std::string& level, const std::string& message) const {
    if (verbose_ || level == "ERROR") {
        std::cout << "[SafeTensorsParserMmap " << level << "] " << message << std::endl;
    }
}

// SafeTensorsModelLoaderMmap implementation
SafeTensorsModelLoaderMmap::SafeTensorsModelLoaderMmap(bool verbose) : verbose_(verbose) {}

SafeTensorsModelLoaderMmap::~SafeTensorsModelLoaderMmap() {}

bool SafeTensorsModelLoaderMmap::loadModel(const std::string& model_dir) {
    model_dir_ = model_dir;
    
    // Try to load weight map from model.safetensors.index.json
    std::string index_file = model_dir + "/model.safetensors.index.json";
    if (!loadWeightMap(index_file)) {
        // If no index file, try to load single model.safetensors file
        std::string single_file = model_dir + "/model.safetensors";
        auto parser = std::make_unique<SafeTensorsParserMmap>(verbose_);
        if (parser->loadFile(single_file)) {
            parsers_["model.safetensors"] = std::move(parser);
            // Add all tensors to weight map
            for (const std::string& tensor_name : parsers_["model.safetensors"]->getTensorNames()) {
                weight_map_[tensor_name] = "model.safetensors";
            }
            log("INFO", "Loaded single SafeTensors file: " + single_file);
            return true;
        }
        log("ERROR", "Failed to load SafeTensors model from: " + model_dir);
        return false;
    }
    
    // Load all referenced files
    std::set<std::string> unique_files;
    for (const auto& pair : weight_map_) {
        unique_files.insert(pair.second);
    }
    
    for (const std::string& filename : unique_files) {
        std::string filepath = model_dir + "/" + filename;
        auto parser = std::make_unique<SafeTensorsParserMmap>(verbose_);
        if (parser->loadFile(filepath)) {
            parsers_[filename] = std::move(parser);
            log("INFO", "Loaded SafeTensors file: " + filepath);
        } else {
            log("ERROR", "Failed to load SafeTensors file: " + filepath);
            return false;
        }
    }
    
    log("INFO", "Successfully loaded SafeTensors model with " + 
                 std::to_string(weight_map_.size()) + " tensors from " + 
                 std::to_string(parsers_.size()) + " files");
    return true;
}

bool SafeTensorsModelLoaderMmap::loadWeightMap(const std::string& index_file) {
    std::ifstream file(index_file);
    if (!file.is_open()) {
        return false;
    }
    
    std::string json_content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    file.close();
    
    weight_map_ = SimpleJsonParser::parseWeightMap(json_content);
    return !weight_map_.empty();
}

const void* SafeTensorsModelLoaderMmap::getTensorDataPtr(const std::string& name) const {
    auto it = weight_map_.find(name);
    if (it == weight_map_.end()) {
        return nullptr;
    }
    
    auto parser_it = parsers_.find(it->second);
    if (parser_it == parsers_.end()) {
        return nullptr;
    }
    
    return parser_it->second->getTensorDataPtr(name);
}

std::vector<float> SafeTensorsModelLoaderMmap::getTensorAsFloat(const std::string& name) const {
    auto it = weight_map_.find(name);
    if (it == weight_map_.end()) {
        return {};
    }
    
    auto parser_it = parsers_.find(it->second);
    if (parser_it == parsers_.end()) {
        return {};
    }
    
    return parser_it->second->getTensorAsFloat(name);
}

bool SafeTensorsModelLoaderMmap::hasTensor(const std::string& name) const {
    return weight_map_.find(name) != weight_map_.end();
}

const TensorInfo* SafeTensorsModelLoaderMmap::getTensorInfo(const std::string& name) const {
    auto it = weight_map_.find(name);
    if (it == weight_map_.end()) {
        return nullptr;
    }
    
    auto parser_it = parsers_.find(it->second);
    if (parser_it == parsers_.end()) {
        return nullptr;
    }
    
    return parser_it->second->getTensorInfo(name);
}

std::vector<std::string> SafeTensorsModelLoaderMmap::getAllTensorNames() const {
    std::vector<std::string> names;
    for (const auto& pair : weight_map_) {
        names.push_back(pair.first);
    }
    return names;
}

void SafeTensorsModelLoaderMmap::log(const std::string& level, const std::string& message) const {
    if (verbose_ || level == "ERROR") {
        std::cout << "[SafeTensorsModelLoaderMmap " << level << "] " << message << std::endl;
    }
}

} // namespace duorou