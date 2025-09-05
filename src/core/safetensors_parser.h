#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <cstdint>
#include <iostream>

namespace duorou {

// SafeTensors tensor metadata
struct TensorInfo {
    std::string name;
    std::string dtype;  // "F32", "F16", "BF16", "I32", etc.
    std::vector<size_t> shape;
    size_t data_offset;
    size_t data_size;
};

// SafeTensors file parser
class SafeTensorsParser {
public:
    SafeTensorsParser();
    ~SafeTensorsParser();
    
    // Load safetensors file and parse header
    bool loadFile(const std::string& filepath);
    
    // Get tensor info by name
    const TensorInfo* getTensorInfo(const std::string& name) const;
    
    // Get all tensor names
    std::vector<std::string> getTensorNames() const;
    
    // Read tensor data
    bool readTensorData(const std::string& name, void* buffer, size_t buffer_size) const;
    
    // Get tensor data as float vector (with type conversion)
    std::vector<float> getTensorAsFloat(const std::string& name) const;
    
    // Check if tensor exists
    bool hasTensor(const std::string& name) const;
    
    // Get file size
    size_t getFileSize() const { return file_size_; }
    
private:
    std::string filepath_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    size_t header_size_;
    size_t file_size_;
    
    // Parse JSON header
    bool parseHeader(const std::string& json_str);
    
    // Convert data type string to size
    size_t getDataTypeSize(const std::string& dtype) const;
    
    // Convert tensor data to float
    void convertToFloat(const void* src_data, float* dst_data, 
                       size_t element_count, const std::string& dtype) const;
};

// SafeTensors model loader for multiple files
class SafeTensorsModelLoader {
public:
    SafeTensorsModelLoader();
    ~SafeTensorsModelLoader();
    
    // Load model from directory with index file
    bool loadModel(const std::string& model_dir);
    
    // Get tensor data from any file
    std::vector<float> getTensorAsFloat(const std::string& name) const;
    
    // Check if tensor exists
    bool hasTensor(const std::string& name) const;
    
    // Get all tensor names
    std::vector<std::string> getAllTensorNames() const;
    
private:
    std::string model_dir_;
    std::unordered_map<std::string, std::string> weight_map_;  // tensor_name -> file_name
    std::unordered_map<std::string, std::unique_ptr<SafeTensorsParser>> parsers_;
    
    // Load weight map from index file
    bool loadWeightMap(const std::string& index_file);
};

} // namespace duorou