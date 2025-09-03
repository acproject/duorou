#include "ollama_model_loader.h"
#include "model_path_manager.h"
#include "modelfile_parser.h"
#include "compatibility_checker.h"
#include "gguf_modifier.h"
#include "config_manager.h"
#include <iostream>
#include <memory>

using namespace duorou::extensions::ollama;

int main() {
    std::cout << "=== Duorou Ollama Extension Test ===" << std::endl;
    
    try {
        // 测试路径管理器
        std::cout << "\n1. Testing ModelPathManager..." << std::endl;
        ModelPathManager pathManager("/tmp/test_ollama");
        std::cout << "   Models directory: " << pathManager.getModelsDirectory() << std::endl;
        
        ModelPath testPath;
        testPath.parseFromString("llama3.2:latest");
        std::cout << "   Test model path: " << testPath.toString() << std::endl;
        
        // 测试Modelfile解析器
        std::cout << "\n2. Testing ModelfileParser..." << std::endl;
        ModelfileParser parser;
        std::string testModelfile = "FROM llama2\nPARAMETER temperature 0.7\nSYSTEM You are a helpful assistant.";
        ParsedModelfile modelfile;
        bool parseSuccess = parser.parseFromString(testModelfile, modelfile);
        if (parseSuccess) {
            std::cout << "   Parsed modelfile successfully" << std::endl;
            std::cout << "   Base model: " << modelfile.from_model << std::endl;
            std::cout << "   Parameters count: " << modelfile.parameters.size() << std::endl;
        }
        
        // 测试兼容性检查器
        std::cout << "\n3. Testing CompatibilityChecker..." << std::endl;
        CompatibilityChecker checker;
        ModelArchitecture arch = checker.detectArchitecture("llama");
        std::cout << "   Detected architecture: " << architectureToString(arch) << std::endl;
        std::string mapped = checker.mapToLlamaCppArchitecture("llama");
        std::cout << "   Mapped to llama.cpp: " << mapped << std::endl;
        
        // 测试配置管理器
        std::cout << "\n4. Testing ConfigManager..." << std::endl;
        ConfigManager configManager;
        auto standardConfig = configManager.createStandardConfig("llama");
        configManager.registerArchitecture(standardConfig);
        std::cout << "   Registered llama architecture" << std::endl;
        std::cout << "   Config keys count: " << configManager.getConfigKeys("llama").size() << std::endl;
        
        // 测试GGUF修改器
        std::cout << "\n5. Testing GGUFModifier..." << std::endl;
        GGUFModifier modifier;
        std::cout << "   GGUF modifier initialized" << std::endl;
        
        // 测试模型加载器
        std::cout << "\n6. Testing OllamaModelLoader..." << std::endl;
        auto pathManagerPtr = std::make_shared<ModelPathManager>("/tmp/test_ollama");
        OllamaModelLoader loader(pathManagerPtr);
        std::cout << "   Model loader initialized" << std::endl;
        auto supportedArchs = loader.getSupportedArchitectures();
        std::cout << "   Supported architectures count: " << supportedArchs.size() << std::endl;
        
        std::cout << "\n=== All tests completed successfully! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}