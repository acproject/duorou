#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include "base_algorithm.h"
#include <cstring>
#include <immintrin.h> // for SIMD

namespace duorou {
namespace extensions {
namespace ollama {
namespace algorithms {

// 基础矩阵运算类
class MatrixOperations : public IMatrixAlgorithm {
public:
  MatrixOperations() = default;
  virtual ~MatrixOperations() = default;

  bool initialize(const ModelConfig& config, const AlgorithmContext& context) override {
    context_ = context;
    return true;
  }

  std::string getName() const override {
    return "MatrixOperations";
  }

  std::string getVersion() const override {
    return "1.0.0";
  }

  bool validateInput(const Tensor& input) const override {
    return !input.data.empty() && input.shape.size() >= 2;
  }

  void multiply(const float* a, const float* b, float* c,
               size_t m, size_t n, size_t k) override {
    if (context_.use_simd) {
      multiplySIMD(a, b, c, m, n, k);
    } else {
      multiplyStandard(a, b, c, m, n, k);
    }
  }

  void vectorAdd(const float* a, const float* b, float* result, size_t size) override {
    if (context_.use_simd && size >= 8) {
      vectorAddSIMD(a, b, result, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
      }
    }
  }

  void vectorMul(const float* a, const float* b, float* result, size_t size) override {
    if (context_.use_simd && size >= 8) {
      vectorMulSIMD(a, b, result, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
      }
    }
  }

  // 额外的矩阵运算方法
  void transpose(const float* input, float* output, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        output[j * rows + i] = input[i * cols + j];
      }
    }
  }

  void scale(float* data, float factor, size_t size) {
    if (context_.use_simd && size >= 8) {
      scaleSIMD(data, factor, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        data[i] *= factor;
      }
    }
  }

private:
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

  void multiplySIMD(const float* a, const float* b, float* c,
                   size_t m, size_t n, size_t k) {
#ifdef __AVX2__
    const size_t simd_width = 8;
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; j += simd_width) {
        __m256 sum = _mm256_setzero_ps();
        for (size_t l = 0; l < k; ++l) {
          __m256 a_vec = _mm256_broadcast_ss(&a[i * k + l]);
          __m256 b_vec = _mm256_loadu_ps(&b[l * n + j]);
          sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }
        _mm256_storeu_ps(&c[i * n + j], sum);
      }
    }
#else
    multiplyStandard(a, b, c, m, n, k);
#endif
  }

  void vectorAddSIMD(const float* a, const float* b, float* result, size_t size) {
#ifdef __AVX2__
    const size_t simd_width = 8;
    size_t simd_end = (size / simd_width) * simd_width;
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
      __m256 a_vec = _mm256_loadu_ps(&a[i]);
      __m256 b_vec = _mm256_loadu_ps(&b[i]);
      __m256 result_vec = _mm256_add_ps(a_vec, b_vec);
      _mm256_storeu_ps(&result[i], result_vec);
    }
    
    // 处理剩余元素
    for (size_t i = simd_end; i < size; ++i) {
      result[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < size; ++i) {
      result[i] = a[i] + b[i];
    }
#endif
  }

  void vectorMulSIMD(const float* a, const float* b, float* result, size_t size) {
#ifdef __AVX2__
    const size_t simd_width = 8;
    size_t simd_end = (size / simd_width) * simd_width;
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
      __m256 a_vec = _mm256_loadu_ps(&a[i]);
      __m256 b_vec = _mm256_loadu_ps(&b[i]);
      __m256 result_vec = _mm256_mul_ps(a_vec, b_vec);
      _mm256_storeu_ps(&result[i], result_vec);
    }
    
    // 处理剩余元素
    for (size_t i = simd_end; i < size; ++i) {
      result[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < size; ++i) {
      result[i] = a[i] * b[i];
    }
#endif
  }

  void scaleSIMD(float* data, float factor, size_t size) {
#ifdef __AVX2__
    const size_t simd_width = 8;
    size_t simd_end = (size / simd_width) * simd_width;
    __m256 factor_vec = _mm256_set1_ps(factor);
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
      __m256 data_vec = _mm256_loadu_ps(&data[i]);
      __m256 result_vec = _mm256_mul_ps(data_vec, factor_vec);
      _mm256_storeu_ps(&data[i], result_vec);
    }
    
    // 处理剩余元素
    for (size_t i = simd_end; i < size; ++i) {
      data[i] *= factor;
    }
#else
    for (size_t i = 0; i < size; ++i) {
      data[i] *= factor;
    }
#endif
  }
};

// 优化的矩阵运算类（使用BLAS）
class OptimizedMatrixOperations : public MatrixOperations {
public:
  OptimizedMatrixOperations() = default;
  virtual ~OptimizedMatrixOperations() = default;

  std::string getName() const override {
    return "OptimizedMatrixOperations";
  }

  void multiply(const float* a, const float* b, float* c,
               size_t m, size_t n, size_t k) override {
    if (context_.use_blas) {
      multiplyBLAS(a, b, c, m, n, k);
    } else {
      MatrixOperations::multiply(a, b, c, m, n, k);
    }
  }

private:
  void multiplyBLAS(const float* a, const float* b, float* c,
                   size_t m, size_t n, size_t k) {
    // 这里可以调用BLAS库的sgemm函数
    // 为了简化，这里使用标准实现
    MatrixOperations::multiply(a, b, c, m, n, k);
  }
};

// 块状矩阵运算类（适用于大矩阵）
class BlockMatrixOperations : public MatrixOperations {
public:
  BlockMatrixOperations(size_t block_size = 64) : block_size_(block_size) {}
  virtual ~BlockMatrixOperations() = default;

  std::string getName() const override {
    return "BlockMatrixOperations";
  }

  void multiply(const float* a, const float* b, float* c,
               size_t m, size_t n, size_t k) override {
    if (m > block_size_ || n > block_size_ || k > block_size_) {
      multiplyBlocked(a, b, c, m, n, k);
    } else {
      MatrixOperations::multiply(a, b, c, m, n, k);
    }
  }

private:
  size_t block_size_;

  void multiplyBlocked(const float* a, const float* b, float* c,
                      size_t m, size_t n, size_t k) {
    // 初始化结果矩阵
    std::memset(c, 0, m * n * sizeof(float));
    
    for (size_t i = 0; i < m; i += block_size_) {
      for (size_t j = 0; j < n; j += block_size_) {
        for (size_t l = 0; l < k; l += block_size_) {
          size_t block_m = std::min(block_size_, m - i);
          size_t block_n = std::min(block_size_, n - j);
          size_t block_k = std::min(block_size_, k - l);
          
          multiplyBlock(&a[i * k + l], &b[l * n + j], &c[i * n + j],
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