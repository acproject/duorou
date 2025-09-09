#pragma once

#include "qwen25vl_inference_engine.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <limits>
#include <functional>
#include <map>
#include <string>

namespace duorou {
namespace extensions {
namespace ollama {

// 重构后的推理引擎类（使用组合而不是继承）
class Qwen25VLRefactoredEngine {
public:
  Qwen25VLRefactoredEngine();
  explicit Qwen25VLRefactoredEngine(bool verbose);
  virtual ~Qwen25VLRefactoredEngine();

  // 模型加载管理
  bool loadModel(const std::string &model_path);
  bool unloadModel();
  bool isModelLoaded() const;

  // 文本生成接口
  std::string generateText(const std::string &prompt, int max_tokens = 100);
  std::string generateTextWithImage(const std::string &prompt,
                                   const std::string &image_path,
                                   int max_tokens = 100);

  // 算法配置接口
  void setAlgorithmConfig(const std::string& config_name);
  std::string getAlgorithmConfig() const;

  // 算法统计接口
  std::map<std::string, double> getAlgorithmStatistics() const;
  void resetAlgorithmStatistics();

  // 算法切换接口
  bool switchAttentionAlgorithm(const std::string& algorithm_type);
  bool switchFeedForwardAlgorithm(const std::string& algorithm_type);
  bool switchPositionalEncodingAlgorithm(const std::string& algorithm_type);

  // 获取支持的算法类型
  std::vector<std::string> getSupportedAttentionTypes() const;
  std::vector<std::string> getSupportedFeedForwardTypes() const;
  std::vector<std::string> getSupportedPositionalEncodingTypes() const;

  // 模块化推理接口
  Tensor computeAttention(const Tensor &input, const TransformerLayer &layer, uint32_t layer_idx);
  Tensor computeFeedForward(const Tensor &input, const TransformerLayer &layer);
  Tensor computeRoPE(const Tensor &input, uint32_t position);

  // 矩阵运算接口
  void performMatrixMultiply(const float *a, const float *b, float *c, size_t m, size_t n, size_t k);
  void performVectorAdd(const float *a, const float *b, float *result, size_t size);
  void performVectorMul(const float *a, const float *b, float *result, size_t size);

private:
  // 原始推理引擎实例
  std::unique_ptr<Qwen25VLInferenceEngine> original_engine_;

  // 算法配置
  std::string current_algorithm_config_;
  
  // 性能统计
  std::map<std::string, double> algorithm_statistics_;
  
  // 内部辅助方法
  void initializeAlgorithms();
  void updateAlgorithmStatistics(const std::string& algorithm_name, double execution_time);
  bool validateAlgorithmCompatibility() const;
};

// 工厂函数
std::unique_ptr<Qwen25VLRefactoredEngine> createRefactoredEngine(
  bool verbose = false
);

std::unique_ptr<Qwen25VLRefactoredEngine> createOptimizedEngine(
  const ModelConfig& model_config,
  bool auto_benchmark = true
);

} // namespace ollama
} // namespace extensions
} // namespace duorou