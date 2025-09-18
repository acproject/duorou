#include "simple_integration_demo.h"
#include "../kvcache/causal.h"
#include <iostream>
#include <stdexcept>

namespace duorou {
namespace model {

SimpleIntegrationDemo::SimpleIntegrationDemo() {
    // 构造函数
}

bool SimpleIntegrationDemo::initialize() {
    try {
        std::cout << "开始初始化集成模型..." << std::endl;
        
        // 1. 初始化ML框架组件
        if (!initializeMLComponents()) {
            std::cerr << "ML组件初始化失败" << std::endl;
            return false;
        }
        std::cout << "✓ ML组件初始化成功" << std::endl;
        
        // 2. 初始化KV缓存
        if (!initializeKVCache()) {
            std::cerr << "KV缓存初始化失败" << std::endl;
            return false;
        }
        std::cout << "✓ KV缓存初始化成功" << std::endl;
        
        initialized_ = true;
        std::cout << "✓ 集成模型初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "初始化异常: " << e.what() << std::endl;
        return false;
    }
}

ml::Tensor SimpleIntegrationDemo::forward(const ml::Tensor& input) {
    if (!initialized_) {
        throw std::runtime_error("模型未初始化");
    }
    
    std::cout << "执行前向传播..." << std::endl;
    
    // 1. 预处理输入
    ml::Tensor processed = preprocessInput(input);
    std::cout << "  ✓ 输入预处理完成" << std::endl;
    
    // 2. 使用ML框架的注意力机制（自注意力：Q=K=V）
    ml::Tensor attended = attention_->forward(*mlContext_, processed, processed, processed);
    std::cout << "  ✓ 注意力计算完成" << std::endl;
    
    // 3. 后处理输出
    ml::Tensor output = postprocessOutput(attended);
    std::cout << "  ✓ 输出后处理完成" << std::endl;
    
    return output;
}

ml::Tensor SimpleIntegrationDemo::forwardWithCache(const ml::Tensor& input, const std::string& cacheKey) {
    if (!initialized_) {
        throw std::runtime_error("模型未初始化");
    }
    
    std::cout << "执行带缓存的前向传播 (缓存键: " << cacheKey << ")..." << std::endl;
    
    // 1. 预处理输入
    ml::Tensor processed = preprocessInput(input);
    std::cout << "  ✓ 输入预处理完成" << std::endl;
    
    // 2. 使用带缓存的注意力机制
    // 这里展示了如何将KV缓存与ML框架集成
    kvcache::Cache* cache = kvCache_->getCache();
    ml::Tensor attended = attention_->forward(*mlContext_, processed, processed, processed, cache);
    std::cout << "  ✓ 带缓存的注意力计算完成" << std::endl;
    
    // 3. 后处理输出
    ml::Tensor output = postprocessOutput(attended);
    std::cout << "  ✓ 输出后处理完成" << std::endl;
    
    return output;
}

ml::Tensor SimpleIntegrationDemo::processMultimodal(const ml::Tensor& textInput, const ml::Tensor& imageInput) {
    if (!initialized_) {
        throw std::runtime_error("模型未初始化");
    }
    
    std::cout << "执行多模态处理..." << std::endl;
    
    // 1. 分别处理文本和图像输入
    ml::Tensor processedText = preprocessInput(textInput);
    ml::Tensor processedImage = preprocessInput(imageInput);
    std::cout << "  ✓ 多模态输入预处理完成" << std::endl;
    
    // 2. 融合多模态特征
    ml::Tensor fused = processedText.add(*mlContext_, processedImage);
    std::cout << "  ✓ 多模态特征融合完成" << std::endl;
    
    // 3. 使用注意力机制处理融合特征 (自注意力: Q=K=V)
    ml::Tensor attended = attention_->forward(*mlContext_, fused, fused, fused);
    std::cout << "  ✓ 多模态注意力计算完成" << std::endl;
    
    // 4. 后处理输出
    ml::Tensor output = postprocessOutput(attended);
    std::cout << "  ✓ 多模态输出后处理完成" << std::endl;
    
    return output;
}

bool SimpleIntegrationDemo::initializeMLComponents() {
    try {
        // 创建ML上下文
        mlContext_ = std::make_unique<ml::Context>();
        std::cout << "    - ML上下文创建成功" << std::endl;
        
        // 创建多头注意力层
        // 参数：embed_dim=512, num_heads=8, kv_heads=8, bias=true, dropout=0.1
        attention_ = std::make_unique<ml::nn::MultiHeadAttention>(
            512, 8, 8, true, 0.1f
        );
        std::cout << "    - 多头注意力层创建成功" << std::endl;
        
        // 初始化权重
        attention_->initializeWeights(*mlContext_, "xavier_uniform");
        std::cout << "    - 注意力权重初始化成功" << std::endl;
        
        // 创建示例张量
        embeddings_ = ml::Tensor::randn({10000, 512});  // 词嵌入
        weights_ = ml::Tensor::randn({512, 512});       // 线性层权重
        std::cout << "    - 模型参数张量创建成功" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ML组件初始化失败: " << e.what() << std::endl;
        return false;
    }
}

bool SimpleIntegrationDemo::initializeKVCache() {
    try {
        // 创建KV缓存包装器，使用因果缓存
        kvCache_ = std::make_unique<kvcache::CacheWrapper>(kvcache::CacheType::CAUSAL);
        std::cout << "    - KV缓存包装器创建成功" << std::endl;
        
        // 注意：这里简化了初始化过程，实际使用时需要提供具体的Backend实现
        std::cout << "    - KV缓存初始化成功（简化版本）" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "KV缓存初始化失败: " << e.what() << std::endl;
        return false;
    }
}

ml::Tensor SimpleIntegrationDemo::preprocessInput(const ml::Tensor& input) {
    // 输入预处理：归一化、形状调整等
    // 这里是简化实现
    return input;
}

ml::Tensor SimpleIntegrationDemo::postprocessOutput(const ml::Tensor& output) {
    // 输出后处理：softmax、形状调整等
    return output.softmax(*mlContext_, -1);
}

// ModuleIntegrator 实现
bool ModuleIntegrator::checkModuleCompatibility() {
    std::cout << "检查模块兼容性..." << std::endl;
    
    try {
        // 检查ML模块
        ml::Context testContext;
        std::cout << "  ✓ ML模块可用" << std::endl;
        
        // 检查KV缓存模块
        kvcache::CacheWrapper testCache(kvcache::CacheType::CAUSAL);
        std::cout << "  ✓ KV缓存模块可用" << std::endl;
        
        std::cout << "✓ 所有模块兼容性检查通过" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "模块兼容性检查失败: " << e.what() << std::endl;
        return false;
    }
}

std::unique_ptr<SimpleIntegrationDemo> ModuleIntegrator::createIntegratedModel() {
    std::cout << "创建集成模型..." << std::endl;
    
    auto model = std::make_unique<SimpleIntegrationDemo>();
    if (model->initialize()) {
        std::cout << "✓ 集成模型创建成功" << std::endl;
        return model;
    }
    
    std::cerr << "✗ 集成模型创建失败" << std::endl;
    return nullptr;
}

bool ModuleIntegrator::testModuleChaining() {
    std::cout << "\n=== 开始模块串联测试 ===" << std::endl;
    
    // 1. 检查模块兼容性
    if (!checkModuleCompatibility()) {
        return false;
    }
    
    // 2. 创建集成模型
    auto model = createIntegratedModel();
    if (!model) {
        return false;
    }
    
    // 3. 测试基本前向传播
    try {
        std::cout << "\n--- 测试基本前向传播 ---" << std::endl;
        ml::Tensor testInput = ml::Tensor::randn({1, 10, 512});
        ml::Tensor output1 = model->forward(testInput);
        std::cout << "✓ 基本前向传播测试通过" << std::endl;
        
        // 4. 测试带缓存的前向传播
        std::cout << "\n--- 测试带缓存的前向传播 ---" << std::endl;
        ml::Tensor output2 = model->forwardWithCache(testInput, "test_cache");
        std::cout << "✓ 带缓存的前向传播测试通过" << std::endl;
        
        // 5. 测试多模态处理
        std::cout << "\n--- 测试多模态处理 ---" << std::endl;
        ml::Tensor textInput = ml::Tensor::randn({1, 5, 512});
        ml::Tensor imageInput = ml::Tensor::randn({1, 5, 512});
        ml::Tensor output3 = model->processMultimodal(textInput, imageInput);
        std::cout << "✓ 多模态处理测试通过" << std::endl;
        
        std::cout << "\n✓ 所有模块串联测试通过！" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "模块串联测试失败: " << e.what() << std::endl;
        return false;
    }
}

} // namespace model
} // namespace duorou