#pragma once

#include "../ml/tensor.h"
#include "../ml/context.h"
#include "../ml/nn/attention.h"
#include <memory>
#include <string>
#include <vector>

namespace duorou {

// Forward declarations
namespace kvcache {
    class CacheWrapper;
}

namespace extensions {
namespace ollama {
namespace gguf {
    class File;
}
}
}

namespace model {

/**
 * 集成模型示例 - 展示如何串联所有模块
 * 这个类展示了model、ml、kvcache、gguf模块的完整集成
 */
class IntegratedModelExample {
public:
    IntegratedModelExample();
    ~IntegratedModelExample() = default;
    
    // 初始化所有组件
    bool initialize();
    
    // 从GGUF文件加载模型
    bool loadFromGGUF(const std::string& modelPath);
    
    // 使用集成的ML框架进行推理
    ml::Tensor forward(const ml::Tensor& input);
    
    // 带缓存的推理
    ml::Tensor forwardWithCache(const ml::Tensor& input, 
                               const std::string& cacheKey = "default");
    
    // 多模态推理示例
    ml::Tensor multimodalForward(const ml::Tensor& textInput,
                                const ml::Tensor& imageInput);
    
    // 获取组件状态
    bool isInitialized() const { return initialized_; }
    const ml::Context& getMLContext() const { return *mlContext_; }
    
private:
    bool initialized_ = false;
    
    // ML框架组件
    std::unique_ptr<ml::Context> mlContext_;
    std::unique_ptr<ml::nn::MultiHeadAttention> attention_;
    
    // KV缓存组件
    std::unique_ptr<kvcache::CacheWrapper> kvCacheWrapper_;
    
    // GGUF模型加载器
    std::unique_ptr<extensions::ollama::gguf::File> ggufFile_;
    
    // 模型参数
    ml::Tensor embeddings_;
    ml::Tensor weights_;
    
    // 辅助方法
    bool initializeMLComponents();
    bool initializeKVCache();
    bool loadModelWeights();
    
    // 数据转换
    ml::Tensor preprocessInput(const ml::Tensor& input);
    ml::Tensor postprocessOutput(const ml::Tensor& output);
};

/**
 * 工厂函数 - 创建集成模型实例
 */
std::unique_ptr<IntegratedModelExample> createIntegratedModel();

/**
 * 模块集成工具函数
 */
namespace IntegrationUtils {
    // 检查所有模块是否可用
    bool checkModuleAvailability();
    
    // 创建统一的数据流
    struct DataFlow {
        ml::Tensor input;
        ml::Tensor processed;
        ml::Tensor output;
        std::string cacheKey;
    };
    
    // 处理数据流
    DataFlow processDataFlow(const ml::Tensor& input, 
                           IntegratedModelExample& model);
}

} // namespace model
} // namespace duorou