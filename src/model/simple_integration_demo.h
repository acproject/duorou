#pragma once

#include "../ml/tensor.h"
#include "../ml/context.h"
#include "../ml/nn/attention.h"
#include "../kvcache/wrapper.h"
#include <memory>
#include <string>
#include <vector>

namespace duorou {
namespace model {

/**
 * 简化的模块集成演示
 * 展示如何将model、ml、kvcache模块串联起来
 */
class SimpleIntegrationDemo {
public:
    SimpleIntegrationDemo();
    ~SimpleIntegrationDemo() = default;
    
    // 初始化所有组件
    bool initialize();
    
    // 基本的前向传播，展示模块串联
    ml::Tensor forward(const ml::Tensor& input);
    
    // 带缓存的前向传播
    ml::Tensor forwardWithCache(const ml::Tensor& input, const std::string& cacheKey = "default");
    
    // 多模态处理示例
    ml::Tensor processMultimodal(const ml::Tensor& textInput, const ml::Tensor& imageInput);
    
    // 获取状态
    bool isInitialized() const { return initialized_; }
    
private:
    bool initialized_ = false;
    
    // ML框架组件
    std::unique_ptr<ml::Context> mlContext_;
    std::unique_ptr<ml::nn::MultiHeadAttention> attention_;
    
    // KV缓存组件
    std::unique_ptr<kvcache::CacheWrapper> kvCache_;
    
    // 模型参数（简化版本）
    ml::Tensor embeddings_;
    ml::Tensor weights_;
    
    // 辅助方法
    bool initializeMLComponents();
    bool initializeKVCache();
    ml::Tensor preprocessInput(const ml::Tensor& input);
    ml::Tensor postprocessOutput(const ml::Tensor& output);
};

/**
 * 模块集成工具类
 */
class ModuleIntegrator {
public:
    // 检查所有模块是否可用
    static bool checkModuleCompatibility();
    
    // 创建统一的数据流处理器
    static std::unique_ptr<SimpleIntegrationDemo> createIntegratedModel();
    
    // 测试模块串联
    static bool testModuleChaining();
};

} // namespace model
} // namespace duorou