#include "third_party/llama.cpp/src/llama-arch.h"
#include <iostream>
#include <string>

int main() {
    std::cout << "Testing qwen25vl architecture support..." << std::endl;
    
    // Test if qwen25vl is recognized
    for (int i = 0; i < LLM_ARCH_COUNT; i++) {
        if (LLM_ARCH_NAMES[i] == std::string("qwen25vl")) {
            std::cout << "SUCCESS: qwen25vl architecture found at index " << i << std::endl;
            std::cout << "Architecture enum value: " << i << std::endl;
            return 0;
        }
    }
    
    std::cout << "ERROR: qwen25vl architecture not found" << std::endl;
    std::cout << "Available architectures:" << std::endl;
    for (int i = 0; i < LLM_ARCH_COUNT; i++) {
        std::cout << "  " << i << ": " << LLM_ARCH_NAMES[i] << std::endl;
    }
    
    return 1;
}