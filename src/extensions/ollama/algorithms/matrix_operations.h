#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include "base_algorithm.h"
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <future>
#include <vector>

// SIMD支持检测
#if defined(__AVX512F__)
#include <immintrin.h>
#define SIMD_WIDTH 16
#define USE_AVX512
#elif defined(__AVX2__)
#include <immintrin.h>
#define SIMD_WIDTH 8
#define USE_AVX2
#elif defined(__AVX__)
#include <immintrin.h>
#define SIMD_WIDTH 8
#define USE_AVX
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_WIDTH 4
#define USE_NEON
#else
#define SIMD_WIDTH 1
#endif

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// 基础矩阵运算类
class MatrixOperations : public IMatrixAlgorithm {
public:
  MatrixOperations() : memory_pool_(nullptr) {}
  virtual ~MatrixOperations() = default;

  bool initialize(const ModelConfig& config, const AlgorithmContext& context) override {
    context_ = context;
    // 获取内存池引用
    if (context.memory_pool) {
      memory_pool_ = context.memory_pool;
    }
    return true;
  }

  std::string getName() const override {
    return "MatrixOperations";
  }

  std::string getVersion() const override {
    return "1.0.0";
  }

  bool validateInput(const Tensor& input) const override {
    return !input.data.empty() && input.size > 0;
  }

  void multiply(const float* a, const float* b, float* c,
               size_t m, size_t n, size_t k) override {
    std::cerr << "[DEBUG] MatrixOperations::multiply called with dimensions: "
              << "m=" << m << ", n=" << n << ", k=" << k << std::endl;
    
    // 添加空指针检查
    if (!a || !b || !c) {
      std::cerr << "[ERROR] Null pointer in matrix multiply: a=" << (void*)a 
                << ", b=" << (void*)b << ", c=" << (void*)c << std::endl;
      throw std::invalid_argument("Null pointer in matrix multiply");
    }
    
    // 添加维度检查
    if (m == 0 || n == 0 || k == 0) {
      std::cerr << "[ERROR] Invalid matrix dimensions: m=" << m 
                << ", n=" << n << ", k=" << k << std::endl;
      throw std::invalid_argument("Invalid matrix dimensions");
    }
    
    multiplyStandard(a, b, c, m, n, k);
    std::cerr << "[DEBUG] MatrixOperations::multiply completed successfully" << std::endl;
  }

  void vectorAdd(const float* a, const float* b, float* result, size_t size) override {
    for (size_t i = 0; i < size; ++i) {
      result[i] = a[i] + b[i];
    }
  }

  void vectorMul(const float* a, const float* b, float* result, size_t size) override {
    for (size_t i = 0; i < size; ++i) {
      result[i] = a[i] * b[i];
    }
  }

  void transpose(const float* input, float* output, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        output[j * rows + i] = input[i * cols + j];
      }
    }
  }

  void scale(float* data, float factor, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      data[i] *= factor;
    }
  }

protected:
  AlgorithmContext context_;
  MemoryPool* memory_pool_; // 内存池指针

  void multiplyStandard(const float* a, const float* b, float* c,
                       size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (size_t l = 0; l < k; ++l) {
          sum += a[i * k + l] * b[l * n + j];
        }
        c[i * n + j] = sum;
      }
    }
  }
};

// 优化矩阵运算类（基于llama.cpp的优化策略）
class OptimizedMatrixOperations : public MatrixOperations {
public:
  OptimizedMatrixOperations() {
    // 获取硬件线程数
    num_threads_ = std::max(1u, std::thread::hardware_concurrency());
    std::cerr << "[INFO] OptimizedMatrixOperations initialized with " 
              << num_threads_ << " threads, SIMD width: " << SIMD_WIDTH << std::endl;
  }
  virtual ~OptimizedMatrixOperations() = default;

  std::string getName() const override {
    return "OptimizedMatrixOperations";
  }

  void multiply(const float* a, const float* b, float* c,
               size_t m, size_t n, size_t k) override {
    std::cerr << "[DEBUG] OptimizedMatrixOperations::multiply called with dimensions: "
              << "m=" << m << ", n=" << n << ", k=" << k << std::endl;
    
    // 添加空指针检查
    if (!a || !b || !c) {
      std::cerr << "[ERROR] Null pointer in matrix multiply" << std::endl;
      throw std::invalid_argument("Null pointer in matrix multiply");
    }
    
    // 添加维度检查
    if (m == 0 || n == 0 || k == 0) {
      std::cerr << "[ERROR] Invalid matrix dimensions" << std::endl;
      throw std::invalid_argument("Invalid matrix dimensions");
    }
    
    // 选择最优算法
    const size_t total_ops = m * n * k;
    const size_t min_parallel_ops = 1024 * 1024; // 1M ops threshold
    
    if (total_ops > min_parallel_ops && num_threads_ > 1) {
      multiplyParallel(a, b, c, m, n, k);
    } else if (canUseSIMD(m, n, k)) {
      multiplySIMD(a, b, c, m, n, k);
    } else {
      multiplyOptimized(a, b, c, m, n, k);
    }
    
    std::cerr << "[DEBUG] OptimizedMatrixOperations::multiply completed" << std::endl;
  }

private:
  unsigned int num_threads_;
  
  bool canUseSIMD(size_t m, size_t n, size_t k) const {
#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_AVX) || defined(USE_NEON)
    return k >= SIMD_WIDTH && n >= SIMD_WIDTH;
#else
    return false;
#endif
  }
  
  void multiplyOptimized(const float* a, const float* b, float* c,
                        size_t m, size_t n, size_t k) {
    // 优化的标量实现，基于llama.cpp的缓存友好策略
    const size_t block_size = 64; // L1缓存友好的块大小
    
    // 初始化输出矩阵
    std::memset(c, 0, m * n * sizeof(float));
    
    for (size_t i = 0; i < m; i += block_size) {
      for (size_t j = 0; j < n; j += block_size) {
        for (size_t l = 0; l < k; l += block_size) {
          size_t block_m = std::min(block_size, m - i);
          size_t block_n = std::min(block_size, n - j);
          size_t block_k = std::min(block_size, k - l);
          
          multiplyBlockOptimized(a + i * k + l, b + l * n + j, c + i * n + j,
                               block_m, block_n, block_k, k, n, n);
        }
      }
    }
  }
  
  void multiplyBlockOptimized(const float* a, const float* b, float* c,
                            size_t m, size_t n, size_t k,
                            size_t lda, size_t ldb, size_t ldc) {
    // 内核循环优化，减少内存访问
    for (size_t i = 0; i < m; ++i) {
      const float* a_row = a + i * lda;
      float* c_row = c + i * ldc;
      
      for (size_t l = 0; l < k; ++l) {
        const float a_val = a_row[l];
        const float* b_row = b + l * ldb;
        
        // 向量化友好的内层循环
        for (size_t j = 0; j < n; ++j) {
          c_row[j] += a_val * b_row[j];
        }
      }
    }
  }
  
  void multiplySIMD(const float* a, const float* b, float* c,
                   size_t m, size_t n, size_t k) {
#if defined(USE_AVX2) || defined(USE_AVX)
    multiplySIMD_AVX(a, b, c, m, n, k);
#elif defined(USE_NEON)
    multiplySIMD_NEON(a, b, c, m, n, k);
#else
    multiplyOptimized(a, b, c, m, n, k);
#endif
  }
  
#if defined(USE_AVX2) || defined(USE_AVX)
  void multiplySIMD_AVX(const float* a, const float* b, float* c,
                       size_t m, size_t n, size_t k) {
    // 初始化输出矩阵
    std::memset(c, 0, m * n * sizeof(float));
    
    const size_t n_simd = (n / SIMD_WIDTH) * SIMD_WIDTH;
    
    for (size_t i = 0; i < m; ++i) {
      for (size_t l = 0; l < k; ++l) {
        const __m256 a_broadcast = _mm256_broadcast_ss(&a[i * k + l]);
        const float* b_row = &b[l * n];
        float* c_row = &c[i * n];
        
        // SIMD向量化计算
        for (size_t j = 0; j < n_simd; j += SIMD_WIDTH) {
          __m256 b_vec = _mm256_loadu_ps(&b_row[j]);
          __m256 c_vec = _mm256_loadu_ps(&c_row[j]);
          __m256 result = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
          _mm256_storeu_ps(&c_row[j], result);
        }
        
        // 处理剩余元素
        for (size_t j = n_simd; j < n; ++j) {
          c_row[j] += a[i * k + l] * b_row[j];
        }
      }
    }
  }
#endif
  
#if defined(USE_NEON)
  void multiplySIMD_NEON(const float* a, const float* b, float* c,
                        size_t m, size_t n, size_t k) {
    // 初始化输出矩阵
    std::memset(c, 0, m * n * sizeof(float));
    
    const size_t n_simd = (n / SIMD_WIDTH) * SIMD_WIDTH;
    
    for (size_t i = 0; i < m; ++i) {
      for (size_t l = 0; l < k; ++l) {
        const float32x4_t a_broadcast = vdupq_n_f32(a[i * k + l]);
        const float* b_row = &b[l * n];
        float* c_row = &c[i * n];
        
        // NEON向量化计算
        for (size_t j = 0; j < n_simd; j += SIMD_WIDTH) {
          float32x4_t b_vec = vld1q_f32(&b_row[j]);
          float32x4_t c_vec = vld1q_f32(&c_row[j]);
          float32x4_t result = vmlaq_f32(c_vec, a_broadcast, b_vec);
          vst1q_f32(&c_row[j], result);
        }
        
        // 处理剩余元素
        for (size_t j = n_simd; j < n; ++j) {
          c_row[j] += a[i * k + l] * b_row[j];
        }
      }
    }
  }
#endif
  
  void multiplyParallel(const float* a, const float* b, float* c,
                       size_t m, size_t n, size_t k) {
    // 初始化输出矩阵
    std::memset(c, 0, m * n * sizeof(float));
    
    // 计算每个线程处理的行数
    const size_t rows_per_thread = std::max(size_t(1), m / num_threads_);
    std::vector<std::future<void>> futures;
    
    // 为每个线程预分配临时缓冲区（如果使用内存池）
    std::vector<std::vector<float>*> temp_buffers;
    if (memory_pool_) {
      temp_buffers.reserve(num_threads_);
      for (unsigned int t = 0; t < num_threads_; ++t) {
        size_t start_row = t * rows_per_thread;
        size_t end_row = (t == num_threads_ - 1) ? m : std::min(m, (t + 1) * rows_per_thread);
        if (start_row < end_row) {
          size_t temp_size = (end_row - start_row) * k; // 临时缓冲区大小
          temp_buffers.push_back(memory_pool_->getBuffer(temp_size));
        }
      }
    }
    
    size_t buffer_idx = 0;
    for (unsigned int t = 0; t < num_threads_; ++t) {
      size_t start_row = t * rows_per_thread;
      size_t end_row = (t == num_threads_ - 1) ? m : std::min(m, (t + 1) * rows_per_thread);
      
      if (start_row >= end_row) break;
      
      std::vector<float>* temp_buffer = memory_pool_ ? temp_buffers[buffer_idx++] : nullptr;
      
      futures.emplace_back(std::async(std::launch::async, [=]() {
        if (canUseSIMD(end_row - start_row, n, k)) {
          multiplySIMD(a + start_row * k, b, c + start_row * n,
                      end_row - start_row, n, k);
        } else {
          multiplyOptimized(a + start_row * k, b, c + start_row * n,
                          end_row - start_row, n, k);
        }
      }));
    }
    
    // 等待所有线程完成
    for (auto& future : futures) {
      future.wait();
    }
    
    // 归还临时缓冲区到内存池
    if (memory_pool_) {
      buffer_idx = 0;
      for (unsigned int t = 0; t < num_threads_; ++t) {
        size_t start_row = t * rows_per_thread;
        size_t end_row = (t == num_threads_ - 1) ? m : std::min(m, (t + 1) * rows_per_thread);
        if (start_row < end_row) {
          size_t temp_size = (end_row - start_row) * k;
          memory_pool_->returnBuffer(temp_buffers[buffer_idx++], temp_size);
        }
      }
    }
  }
};

// 分块矩阵运算类
class BlockMatrixOperations : public MatrixOperations {
public:
  explicit BlockMatrixOperations(size_t block_size = 64) : block_size_(block_size) {}
  virtual ~BlockMatrixOperations() = default;

  std::string getName() const override {
    return "BlockMatrixOperations";
  }

  void multiply(const float* a, const float* b, float* c,
               size_t m, size_t n, size_t k) override {
    if (m > block_size_ || n > block_size_ || k > block_size_) {
      multiplyBlocked(a, b, c, m, n, k);
    } else {
      multiplyStandard(a, b, c, m, n, k);
    }
  }

private:
  size_t block_size_;

  void multiplyBlocked(const float* a, const float* b, float* c,
                      size_t m, size_t n, size_t k) {
    // 简化的分块实现
    for (size_t i = 0; i < m; i += block_size_) {
      for (size_t j = 0; j < n; j += block_size_) {
        for (size_t l = 0; l < k; l += block_size_) {
          size_t block_m = std::min(block_size_, m - i);
          size_t block_n = std::min(block_size_, n - j);
          size_t block_k = std::min(block_size_, k - l);
          
          multiplyBlock(a + i * k + l, b + l * n + j, c + i * n + j,
                       block_m, block_n, block_k, k, n, n);
        }
      }
    }
  }

  void multiplyBlock(const float* a, const float* b, float* c,
                    size_t m, size_t n, size_t k,
                    size_t lda, size_t ldb, size_t ldc) {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (size_t l = 0; l < k; ++l) {
          sum += a[i * lda + l] * b[l * ldb + j];
        }
        c[i * ldc + j] += sum;
      }
    }
  }
};

} // namespace algorithms
} // namespace ollama
} // namespace extensions
} // namespace duorou

#endif // MATRIX_OPERATIONS_H