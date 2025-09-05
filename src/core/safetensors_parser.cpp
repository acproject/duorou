#include "safetensors_parser.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <set>

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

SafeTensorsParser::SafeTensorsParser() : header_size_(0), file_size_(0) {}

SafeTensorsParser::~SafeTensorsParser() {}

bool SafeTensorsParser::loadFile(const std::string& filepath) {
    filepath_ = filepath;
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    file_size_ = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read header size (first 8 bytes, little-endian)
    uint64_t header_size_raw;
    file.read(reinterpret_cast<char*>(&header_size_raw), 8);
    header_size_ = header_size_raw;
    
    if (header_size_ > file_size_ - 8) {
        std::cerr << "Invalid header size: " << header_size_ << std::endl;
        return false;
    }
    
    // Read header JSON
    std::vector<char> header_data(header_size_);
    file.read(header_data.data(), header_size_);
    std::string header_json(header_data.begin(), header_data.end());
    
    file.close();
    
    return parseHeader(header_json);
}

bool SafeTensorsParser::parseHeader(const std::string& json_str) {
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
        info.data_offset = data_offset;
        
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
            size_t bracket_start = json_str.find('[', shape_pos);
            size_t bracket_end = json_str.find(']', bracket_start);
            if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
                std::string shape_str = json_str.substr(bracket_start + 1, bracket_end - bracket_start - 1);
                
                // Parse shape dimensions
                std::stringstream ss(shape_str);
                std::string dim;
                while (std::getline(ss, dim, ',')) {
                    // Remove whitespace
                    dim.erase(std::remove_if(dim.begin(), dim.end(), ::isspace), dim.end());
                    if (!dim.empty()) {
                        try {
                            info.shape.push_back(std::stoull(dim));
                        } catch (const std::exception& e) {
                            std::cerr << "Error parsing dimension '" << dim << "' for tensor '" << tensor_name << "': " << e.what() << std::endl;
                            return false;
                        }
                    }
                }
            }
        }
        
        // Find data_offsets
        size_t offsets_pos = json_str.find("\"data_offsets\"", name_end);
        if (offsets_pos != std::string::npos) {
            size_t bracket_start = json_str.find('[', offsets_pos);
            size_t bracket_end = json_str.find(']', bracket_start);
            if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
                std::string offsets_str = json_str.substr(bracket_start + 1, bracket_end - bracket_start - 1);
                
                // Parse start and end offsets
                size_t comma_pos = offsets_str.find(',');
                if (comma_pos != std::string::npos) {
                    std::string start_str = offsets_str.substr(0, comma_pos);
                    std::string end_str = offsets_str.substr(comma_pos + 1);
                    
                    // Remove whitespace
                    start_str.erase(std::remove_if(start_str.begin(), start_str.end(), ::isspace), start_str.end());
                    end_str.erase(std::remove_if(end_str.begin(), end_str.end(), ::isspace), end_str.end());
                    
                    try {
                        size_t start_offset = std::stoull(start_str);
                        size_t end_offset = std::stoull(end_str);
                        
                        info.data_offset = 8 + header_size_ + start_offset;  // Add header size
                        info.data_size = end_offset - start_offset;
                    } catch (const std::exception& e) {
                        std::cerr << "Error parsing offsets for tensor '" << tensor_name << "': " << e.what() << std::endl;
                        return false;
                    }
                }
            }
        }
        
        // Fallback: calculate data size if not set from offsets
        if (info.data_size == 0) {
            size_t element_count = 1;
            for (size_t dim : info.shape) {
                element_count *= dim;
            }
            info.data_size = element_count * getDataTypeSize(info.dtype);
        }
        
        tensors_[tensor_name] = info;
        
        pos = name_end + 1;
    }
    
    return !tensors_.empty();
}

size_t SafeTensorsParser::getDataTypeSize(const std::string& dtype) const {
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
    if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
    return 4;  // Default to 4 bytes
}

const TensorInfo* SafeTensorsParser::getTensorInfo(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

std::vector<std::string> SafeTensorsParser::getTensorNames() const {
    std::vector<std::string> names;
    for (const auto& pair : tensors_) {
        names.push_back(pair.first);
    }
    return names;
}

bool SafeTensorsParser::hasTensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

bool SafeTensorsParser::readTensorData(const std::string& name, void* buffer, size_t buffer_size) const {
    const TensorInfo* info = getTensorInfo(name);
    if (!info) {
        return false;
    }
    
    if (buffer_size < info->data_size) {
        return false;
    }
    
    std::ifstream file(filepath_, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    file.seekg(info->data_offset);
    file.read(static_cast<char*>(buffer), info->data_size);
    
    return file.good();
}

std::vector<float> SafeTensorsParser::getTensorAsFloat(const std::string& name) const {
    const TensorInfo* info = getTensorInfo(name);
    if (!info) {
        return {};
    }
    
    size_t element_count = 1;
    for (size_t dim : info->shape) {
        element_count *= dim;
    }
    
    std::vector<char> raw_data(info->data_size);
    if (!readTensorData(name, raw_data.data(), raw_data.size())) {
        return {};
    }
    
    std::vector<float> float_data(element_count);
    convertToFloat(raw_data.data(), float_data.data(), element_count, info->dtype);
    
    return float_data;
}

void SafeTensorsParser::convertToFloat(const void* src_data, float* dst_data, 
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

// SafeTensorsModelLoader implementation
SafeTensorsModelLoader::SafeTensorsModelLoader() {}

SafeTensorsModelLoader::~SafeTensorsModelLoader() {}

bool SafeTensorsModelLoader::loadModel(const std::string& model_dir) {
    model_dir_ = model_dir;
    
    // Load weight map from index file
    std::string index_file = model_dir + "/model.safetensors.index.json";
    std::cout << "Loading weight map from: " << index_file << std::endl;
    if (!loadWeightMap(index_file)) {
        std::cerr << "Failed to load weight map" << std::endl;
        return false;
    }
    
    std::cout << "Loaded " << weight_map_.size() << " weight mappings" << std::endl;
    
    // Load all safetensors files
    std::set<std::string> unique_files;
    for (const auto& pair : weight_map_) {
        unique_files.insert(pair.second);
    }
    
    std::cout << "Found " << unique_files.size() << " unique safetensors files" << std::endl;
    
    for (const std::string& filename : unique_files) {
        std::string filepath = model_dir + "/" + filename;
        std::cout << "Loading file: " << filepath << std::endl;
        auto parser = std::make_unique<SafeTensorsParser>();
        if (parser->loadFile(filepath)) {
            parsers_[filename] = std::move(parser);
            std::cout << "Successfully loaded: " << filename << std::endl;
        } else {
            std::cerr << "Failed to load " << filepath << std::endl;
            return false;
        }
    }
    
    std::cout << "Loaded " << parsers_.size() << " safetensors files" << std::endl;
    return !parsers_.empty();
}

bool SafeTensorsModelLoader::loadWeightMap(const std::string& index_file) {
    std::ifstream file(index_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open index file: " << index_file << std::endl;
        return false;
    }
    
    std::string json_content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    file.close();
    
    weight_map_ = SimpleJsonParser::parseWeightMap(json_content);
    
    return !weight_map_.empty();
}

std::vector<float> SafeTensorsModelLoader::getTensorAsFloat(const std::string& name) const {
    auto it = weight_map_.find(name);
    if (it == weight_map_.end()) {
        return {};
    }
    
    const std::string& filename = it->second;
    auto parser_it = parsers_.find(filename);
    if (parser_it == parsers_.end()) {
        return {};
    }
    
    return parser_it->second->getTensorAsFloat(name);
}

bool SafeTensorsModelLoader::hasTensor(const std::string& name) const {
    return weight_map_.find(name) != weight_map_.end();
}

std::vector<std::string> SafeTensorsModelLoader::getAllTensorNames() const {
    std::vector<std::string> names;
    for (const auto& pair : weight_map_) {
        names.push_back(pair.first);
    }
    return names;
}

} // namespace duorou