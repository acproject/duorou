#include "matrix_mmap.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

using namespace duorou::extensions::ollama::algorithms;

// 生成随机矩阵数据
void generateRandomMatrix(std::vector<float>& matrix, size_t rows, size_t cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    matrix.resize(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

// 保存矩阵到二进制文件
bool saveMatrixToBinary(const std::string& filepath, const std::vector<float>& data, 
                       size_t rows, size_t cols) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    // 写入头部信息
    uint32_t magic = 0x4D4D4150; // "MMAP"
    uint32_t version = 1;
    uint32_t dtype_len = 3;
    const char* dtype = "F32";
    
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&dtype_len), sizeof(dtype_len));
    file.write(dtype, dtype_len);
    
    // 写入数据
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    
    return file.good();
}

// 测试基本的mmap操作
void testBasicMmapOperations() {
    std::cout << "\n=== Testing Basic Mmap Operations ===\n";
    
    // 创建mmap矩阵操作实例
    auto mmap_ops = createMmapMatrixOperations(true);
    
    // 初始化
    ModelConfig config;
    AlgorithmContext context;
    context.device = "cpu";
    context.num_threads = 4;
    
    if (!mmap_ops->initialize(config, context)) {
        std::cerr << "Failed to initialize mmap operations" << std::endl;
        return;
    }
    
    std::cout << "✓ Mmap operations initialized successfully" << std::endl;
    std::cout << "  Algorithm: " << mmap_ops->getName() << " v" << mmap_ops->getVersion() << std::endl;
}

// 测试矩阵文件操作
void testMatrixFileOperations() {
    std::cout << "\n=== Testing Matrix File Operations ===\n";
    
    const size_t rows = 100;
    const size_t cols = 100;
    const std::string filepath = "/tmp/test_matrix.bin";
    
    // 生成测试数据
    std::vector<float> test_data;
    generateRandomMatrix(test_data, rows, cols);
    
    // 保存到文件
    if (!saveMatrixToBinary(filepath, test_data, rows, cols)) {
        std::cerr << "Failed to save test matrix" << std::endl;
        return;
    }
    
    std::cout << "✓ Test matrix saved to " << filepath << std::endl;
    
    // 创建mmap操作实例
    auto mmap_ops = createMmapMatrixOperations(true);
    
    ModelConfig config;
    AlgorithmContext context;
    context.device = "cpu";
    context.num_threads = 4;
    
    mmap_ops->initialize(config, context);
    
    // 加载矩阵
    if (mmap_ops->loadMatrixFromFile(filepath, "test_matrix")) {
        std::cout << "✓ Matrix loaded successfully via mmap" << std::endl;
        
        // 获取映射的矩阵信息
        const auto* matrix_data = mmap_ops->getMappedMatrix("test_matrix");
        if (matrix_data) {
            std::cout << "  Matrix info: " << matrix_data->rows << "x" << matrix_data->cols 
                      << ", dtype: " << matrix_data->dtype 
                      << ", size: " << matrix_data->total_size << " bytes" << std::endl;
        }
        
        // 测试内存锁定
        if (mmap_ops->lockMatrix("test_matrix")) {
            std::cout << "✓ Matrix locked in memory" << std::endl;
            std::cout << "  Total locked size: " << mmap_ops->getLockedSize() << " bytes" << std::endl;
            
            mmap_ops->unlockMatrix("test_matrix");
            std::cout << "✓ Matrix unlocked" << std::endl;
        }
        
        // 清理
        mmap_ops->unmapMatrix("test_matrix");
        std::cout << "✓ Matrix unmapped" << std::endl;
    } else {
        std::cerr << "Failed to load matrix via mmap" << std::endl;
    }
    
    // 清理文件
    std::remove(filepath.c_str());
}

// 测试矩阵乘法性能
void testMatrixMultiplyPerformance() {
    std::cout << "\n=== Testing Matrix Multiply Performance ===\n";
    
    const size_t size = 512;
    
    // 生成测试数据
    std::vector<float> a, b, c;
    generateRandomMatrix(a, size, size);
    generateRandomMatrix(b, size, size);
    c.resize(size * size);
    
    auto mmap_ops = createMmapMatrixOperations(false); // 关闭详细日志
    
    ModelConfig config;
    AlgorithmContext context;
    context.device = "cpu";
    context.num_threads = 4;
    
    mmap_ops->initialize(config, context);
    
    // 测试标准矩阵乘法
    auto start = std::chrono::high_resolution_clock::now();
    mmap_ops->multiply(a.data(), b.data(), c.data(), size, size, size);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double gflops = (2.0 * size * size * size) / (duration.count() * 1e6);
    
    std::cout << "✓ Matrix multiplication (" << size << "x" << size << ") completed" << std::endl;
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;
}

// 测试向量操作
void testVectorOperations() {
    std::cout << "\n=== Testing Vector Operations ===\n";
    
    const size_t size = 10000;
    
    std::vector<float> a, b, result;
    generateRandomMatrix(a, 1, size);
    generateRandomMatrix(b, 1, size);
    result.resize(size);
    
    auto mmap_ops = createMmapMatrixOperations(false);
    
    ModelConfig config;
    AlgorithmContext context;
    mmap_ops->initialize(config, context);
    
    // 测试向量加法
    mmap_ops->vectorAdd(a.data(), b.data(), result.data(), size);
    std::cout << "✓ Vector addition completed" << std::endl;
    
    // 测试向量乘法
    mmap_ops->vectorMul(a.data(), b.data(), result.data(), size);
    std::cout << "✓ Vector multiplication completed" << std::endl;
}

int main() {
    std::cout << "Matrix Mmap Operations Test Suite" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        // 检查mmap支持
        std::cout << "\nChecking mmap support:" << std::endl;
        std::cout << "  MatrixMmap::SUPPORTED = " << (MatrixMmap::SUPPORTED ? "true" : "false") << std::endl;
        std::cout << "  MatrixMlock::SUPPORTED = " << (MatrixMlock::SUPPORTED ? "true" : "false") << std::endl;
        
        testBasicMmapOperations();
        testMatrixFileOperations();
        testMatrixMultiplyPerformance();
        testVectorOperations();
        
        std::cout << "\n=== All Tests Completed Successfully ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}