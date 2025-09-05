#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <iostream>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace duorou {

// SafeTensors tensor metadata
struct TensorInfo {
    std::string name;
    std::string dtype;  // "F32", "F16", "BF16", "I32", etc.
    std::vector<size_t> shape;
    size_t data_offset;
    size_t data_size;
};

// Memory-mapped SafeTensors file parser
class SafeTensorsParserMmap {
public:
    SafeTensorsParserMmap(bool verbose = false);
    ~SafeTensorsParserMmap();
    
    // Load safetensors file and parse header using mmap
    bool loadFile(const std::string& filepath);
    
    // Get tensor info by name
    const TensorInfo* getTensorInfo(const std::string& name) const;
    
    // Get all tensor names
    std::vector<std::string> getTensorNames() const;
    
    // Read tensor data directly from mapped memory (zero-copy)
    const void* getTensorDataPtr(const std::string& name) const;
    
    // Get tensor data as float vector (with type conversion)
    std::vector<float> getTensorAsFloat(const std::string& name) const;
    
    // Check if tensor exists
    bool hasTensor(const std::string& name) const;
    
    // Get file size
    size_t getFileSize() const { return file_size_; }
    
    // Get mapped data pointer (for advanced usage)
    const uint8_t* getMappedData() const { return mapped_data_; }
    
private:
    std::string filepath_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    size_t header_size_;
    size_t file_size_;
    bool verbose_;
    
    // Memory mapping related
    bool use_mmap_;
    int fd_;
    uint8_t* mapped_data_;
    
#ifdef _WIN32
    HANDLE file_handle_;
    HANDLE mapping_handle_;
#endif
    
    // Initialize memory mapping
    bool initMmap(const std::string& file_path);
    void cleanupMmap();
    
    // Parse JSON header from mapped memory
    bool parseHeader(const std::string& json_str);
    
    // Convert data type string to size
    size_t getDataTypeSize(const std::string& dtype) const;
    
    // Convert tensor data to float
    void convertToFloat(const void* src_data, float* dst_data, 
                       size_t element_count, const std::string& dtype) const;
    
    // Logging function
    void log(const std::string& level, const std::string& message) const;
};

// Memory-mapped SafeTensors model loader for multiple files
class SafeTensorsModelLoaderMmap {
public:
    SafeTensorsModelLoaderMmap(bool verbose = false);
    ~SafeTensorsModelLoaderMmap();
    
    // Load model from directory with index file
    bool loadModel(const std::string& model_dir);
    
    // Get tensor data pointer from any file (zero-copy)
    const void* getTensorDataPtr(const std::string& name) const;
    
    // Get tensor data from any file
    std::vector<float> getTensorAsFloat(const std::string& name) const;
    
    // Check if tensor exists
    bool hasTensor(const std::string& name) const;
    
    // Get all tensor names
    std::vector<std::string> getAllTensorNames() const;
    
    // Get tensor info
    const TensorInfo* getTensorInfo(const std::string& name) const;
    
private:
    std::string model_dir_;
    std::unordered_map<std::string, std::string> weight_map_;  // tensor_name -> file_name
    std::unordered_map<std::string, std::unique_ptr<SafeTensorsParserMmap>> parsers_;
    bool verbose_;
    
    // Load weight map from index file
    bool loadWeightMap(const std::string& index_file);
    
    // Logging function
    void log(const std::string& level, const std::string& message) const;
};

} // namespace duorou