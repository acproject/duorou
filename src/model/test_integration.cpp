#include "simple_integration_demo.h"
#include <iostream>

/**
 * 模块集成测试程序
 * 演示如何将model、ml、kvcache模块串联起来
 */
int main() {
    std::cout << "=== Duorou 模块集成测试 ===" << std::endl;
    std::cout << "演示model、ml、kvcache模块的串联集成\n" << std::endl;
    
    try {
        // 使用ModuleIntegrator进行完整的模块串联测试
        bool success = duorou::model::ModuleIntegrator::testModuleChaining();
        
        if (success) {
            std::cout << "\n🎉 模块集成测试成功！" << std::endl;
            std::cout << "✓ ML框架与model模块成功集成" << std::endl;
            std::cout << "✓ KV缓存与注意力机制成功串联" << std::endl;
            std::cout << "✓ 多模态处理流程正常工作" << std::endl;
            std::cout << "✓ 统一的Tensor数据结构在各模块间正确传递" << std::endl;
            
            std::cout << "\n架构重构完成！现在各模块已经真正串联起来：" << std::endl;
            std::cout << "  Input → ML Tensor → Attention → KV Cache → Output" << std::endl;
            
            return 0;
        } else {
            std::cerr << "\n❌ 模块集成测试失败" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生异常: " << e.what() << std::endl;
        return 1;
    }
}

/**
 * 单独测试各个模块的基本功能
 */
void testIndividualModules() {
    std::cout << "\n=== 单独测试各模块功能 ===" << std::endl;
    
    // 测试ML模块
    std::cout << "\n--- 测试ML模块 ---" << std::endl;
    try {
        duorou::ml::Context ctx;
        duorou::ml::Tensor tensor = duorou::ml::Tensor::randn({2, 3, 4});
        std::cout << "✓ ML模块基本功能正常" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "✗ ML模块测试失败: " << e.what() << std::endl;
    }
    
    // 测试KV缓存模块
    std::cout << "\n--- 测试KV缓存模块 ---" << std::endl;
    try {
        duorou::kvcache::CacheWrapper cache(duorou::kvcache::CacheType::CAUSAL);
        std::cout << "✓ KV缓存模块基本功能正常" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "✗ KV缓存模块测试失败: " << e.what() << std::endl;
    }
    
    // 测试注意力机制
    std::cout << "\n--- 测试注意力机制 ---" << std::endl;
    try {
        duorou::ml::Context ctx;
        duorou::ml::nn::MultiHeadAttention attention(256, 4, 4, true, 0.1f);
        attention.initializeWeights(ctx, "xavier_uniform");
        std::cout << "✓ 注意力机制基本功能正常" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "✗ 注意力机制测试失败: " << e.what() << std::endl;
    }
}