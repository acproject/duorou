#include "../src/extensions/ollama/ollama_model_manager.h"
#include <iostream>
#include <string>

using namespace duorou::extensions::ollama;

int main() {
  std::cout << "Testing OllamaModelManager text generation error handling..."
            << std::endl;

  // 创建模型管理器
  auto manager = std::make_unique<OllamaModelManager>(true);

  // 测试1: 在没有初始化text processor的情况下进行tokenization
  std::cout << "\nTest 1: Tokenization without text processor initialization"
            << std::endl;
  auto tokens = manager->tokenize("你好");
  std::cout << "Tokenization result: " << tokens.size() << " tokens"
            << std::endl;

  // 测试2: 模拟InferenceRequest
  std::cout << "\nTest 2: Simulating text generation request" << std::endl;
  InferenceRequest request;
  request.model_id = "test_model";
  request.prompt = "你好";
  request.max_tokens = 100;

  auto response = manager->generateText(request);
  std::cout << "Generation success: " << response.success << std::endl;
  std::cout << "Error message: " << response.error_message << std::endl;

  // 测试3: 空prompt处理
  std::cout << "\nTest 3: Empty prompt handling" << std::endl;
  request.prompt = "";
  response = manager->generateText(request);
  std::cout << "Generation success: " << response.success << std::endl;
  std::cout << "Error message: " << response.error_message << std::endl;

  std::cout << "\nText generation error handling test completed." << std::endl;
  return 0;
}