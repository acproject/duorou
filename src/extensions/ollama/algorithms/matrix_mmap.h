#ifndef MATRIX_MMAP_H
#define MATRIX_MMAP_H

#include "base_algorithm.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cerrno>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// 内存映射文件类 - 模仿llama.cpp的实现
class MatrixFile {
public:
    MatrixFile(const char* fname, const char* mode);
    ~MatrixFile();
    
    size_t tell() const;
    size_t size() const;
    int file_id() const;
    
    void seek(size_t offset, int whence) const;
    void read_raw(void* ptr, size_t len) const;
    uint32_t read_u32() const;
    
    void write_raw(const void* ptr, size_t len) const;
    void write_u32(uint32_t val) const;
    
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// 内存映射类 - 模仿llama.cpp的实现
class MatrixMmap {
public:
    MatrixMmap(const MatrixMmap&) = delete;
    MatrixMmap(MatrixFile* file, size_t prefetch = (size_t)-1, bool numa = false);
    ~MatrixMmap();
    
    size_t size() const;
    void* addr() const;
    
    void unmap_fragment(size_t first, size_t last);
    
    static const bool SUPPORTED;
    
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// 内存锁定类 - 模仿llama.cpp的实现
class MatrixMlock {
public:
    MatrixMlock();
    ~MatrixMlock();
    
    void init(void* ptr);
    void grow_to(size_t target_size);
    
    static const bool SUPPORTED;
    
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// 内存映射矩阵数据结构
struct MmapMatrixData {
    void* data_ptr;          // 映射的数据指针
    size_t rows;
    size_t cols;
    size_t element_size;     // 元素大小（字节）
    size_t total_size;       // 总大小（字节）
    std::string dtype;       // 数据类型
    
    MmapMatrixData() : data_ptr(nullptr), rows(0), cols(0), 
                       element_size(0), total_size(0) {}
};

// 内存映射矩阵操作类
class MmapMatrixOperations : public IMatrixAlgorithm {
public:
    MmapMatrixOperations(bool verbose = false);
    virtual ~MmapMatrixOperations();
    
    // IMatrixAlgorithm接口实现
    bool initialize(const ModelConfig& config, const AlgorithmContext& context) override;
    std::string getName() const override { return "MmapMatrixOperations"; }
    std::string getVersion() const override { return "1.0.0"; }
    bool validateInput(const Tensor& input) const override;
    
    void multiply(const float* a, const float* b, float* c,
                 size_t m, size_t n, size_t k) override;
    void vectorAdd(const float* a, const float* b, float* result, size_t size) override;
    void vectorMul(const float* a, const float* b, float* result, size_t size) override;
    
    // 内存映射特有方法
    bool loadMatrixFromFile(const std::string& filepath, const std::string& matrix_name);
    bool saveMatrixToFile(const std::string& filepath, const std::string& matrix_name,
                         const float* data, size_t rows, size_t cols);
    
    // 获取映射的矩阵数据
    const MmapMatrixData* getMappedMatrix(const std::string& name) const;
    
    // 零拷贝矩阵乘法（直接使用映射内存）
    void multiplyMapped(const std::string& matrix_a_name,
                       const std::string& matrix_b_name,
                       float* result, size_t result_rows, size_t result_cols);
    
    // 内存预取优化
    void prefetchMatrix(const std::string& matrix_name, size_t size = 0);
    
    // 内存锁定（防止交换到磁盘）
    bool lockMatrix(const std::string& matrix_name);
    void unlockMatrix(const std::string& matrix_name);
    
    // 内存使用统计
    size_t getTotalMappedSize() const;
    size_t getLockedSize() const;
    
    // 清理映射
    void unmapMatrix(const std::string& matrix_name);
    void unmapAll();
    
private:
    struct MatrixMapping {
        std::unique_ptr<MatrixFile> file;
        std::unique_ptr<MatrixMmap> mmap;
        std::unique_ptr<MatrixMlock> mlock;
        MmapMatrixData data;
        bool is_locked;
        
        MatrixMapping() : is_locked(false) {}
    };
    
    std::unordered_map<std::string, std::unique_ptr<MatrixMapping>> mappings_;
    AlgorithmContext context_;
    bool verbose_;
    size_t total_mapped_size_;
    size_t total_locked_size_;
    
    // 内部辅助方法
    bool parseMatrixHeader(MatrixFile* file, MmapMatrixData& data);
    void writeMatrixHeader(MatrixFile* file, const MmapMatrixData& data);
    
    // 数据类型转换
    void convertToFloat(const void* src, float* dst, size_t count, const std::string& dtype);
    void convertFromFloat(const float* src, void* dst, size_t count, const std::string& dtype);
    
    // 日志输出
    void log(const std::string& level, const std::string& message) const;
    
    // 内存对齐优化
    static size_t alignSize(size_t size, size_t alignment = 64);
    
    // NUMA优化提示
    void optimizeForNuma(void* ptr, size_t size);
};

// 工厂函数
std::unique_ptr<MmapMatrixOperations> createMmapMatrixOperations(bool verbose = false);

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // MATRIX_MMAP_H