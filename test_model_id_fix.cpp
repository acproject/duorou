#include <iostream>
#include <memory>
#include "src/core/text_generator.h"
#include "src/extensions/ollama/ollama_model_manager.h"

int main() {
    std::cout << "Testing Model ID Normalization Fix..." << std::endl;
    
    // 创建OllamaModelManager
    auto model_manager = std::make_shared<duorou::extensions::ollama::OllamaModelManager>(true);
    
    // 测试原始模型名称
    std::string original_model_name = "registry.ollama.ai/library/qwen2.5vl:7b";
    std::cout << "Original model name: " << original_model_name << std::endl;
    
    // 创建TextGenerator，它会自动归一化模型ID
    duorou::core::TextGenerator text_generator(model_manager, original_model_name);
    
    // 测试OllamaModelManager的归一化
    std::string normalized_by_manager = model_manager->normalizeModelId(original_model_name);
    std::cout << "Normalized by OllamaModelManager: " << normalized_by_manager << std::endl;
    
    // 注册模型（这会使用归一化的ID）
    std::cout << "\nTesting model registration..." << std::endl;
    bool registered = model_manager->registerModelByName(original_model_name);
    std::cout << "Model registration result: " << (registered ? "SUCCESS" : "FAILED") << std::endl;
    
    // 获取注册的模型列表
    auto registered_models = model_manager->getRegisteredModels();
    std::cout << "\nRegistered models:" << std::endl;
    for (const auto& model : registered_models) {
        std::cout << "  - " << model << std::endl;
    }
    
    // 测试模型查找
    std::cout << "\nTesting model lookup..." << std::endl;
    const auto* model_info = model_manager->getModelInfo(original_model_name);
    std::cout << "Model info lookup result: " << (model_info ? "FOUND" : "NOT FOUND") << std::endl;
    
    // 测试归一化ID查找
    const auto* model_info_normalized = model_manager->getModelInfo(normalized_by_manager);
    std::cout << "Normalized ID lookup result: " << (model_info_normalized ? "FOUND" : "NOT FOUND") << std::endl;
    
    std::cout << "\nTest completed!" << std::endl;
    return 0;
}