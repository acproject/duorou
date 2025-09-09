#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include "base_algorithm.h"
#include <cstring>

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
    return !input.data.empty() && input.size > 0;
  }

  void multiply(const float* a, const float* b, float* c,
               size_t m, size_t n, size_t k) override {
    multiplyStandard(a, b, c, m, n, k);
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

// 优化矩阵运算类（暂时使用标准实现）
class OptimizedMatrixOperations : public MatrixOperations {
public:
  OptimizedMatrixOperations() = default;
  virtual ~OptimizedMatrixOperations() = default;

  std::string getName() const override {
    return "OptimizedMatrixOperations";
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