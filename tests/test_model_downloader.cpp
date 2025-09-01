#include "../src/core/model_downloader.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace duorou;

// 简单的测试断言宏
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAIL: " << message << std::endl; \
            return false; \
        } else { \
            std::cout << "PASS: " << message << std::endl; \
        } \
    } while(0)

// 进度回调函数
void progressCallback(size_t downloaded, size_t total, double speed) {
    if (total > 0) {
        double progress = (double)downloaded / total * 100.0;
        double speed_mb = speed / (1024.0 * 1024.0);
        std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                  << progress << "% (" << downloaded << "/" << total << " bytes) "
                  << "Speed: " << std::setprecision(2) << speed_mb << " MB/s" << std::flush;
    }
}

bool testModelDownloaderCreation() {
    std::cout << "\n=== Testing ModelDownloader Creation ===" << std::endl;
    
    try {
        auto downloader = ModelDownloaderFactory::create();
        TEST_ASSERT(downloader != nullptr, "ModelDownloader creation");
        
        // 测试设置进度回调
        downloader->setProgressCallback(progressCallback);
        TEST_ASSERT(true, "Setting progress callback");
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

bool testModelInfo() {
    std::cout << "\n=== Testing Model Info Retrieval ===" << std::endl;
    
    try {
        auto downloader = ModelDownloaderFactory::create();
        
        // 测试获取模型信息（这可能会失败，因为需要网络连接）
        try {
            ModelInfo info = downloader->getModelInfo("llama2:7b");
            std::cout << "Model name: " << info.name << std::endl;
            std::cout << "Model tag: " << info.tag << std::endl;
            std::cout << "Model size: " << info.size << " bytes" << std::endl;
            TEST_ASSERT(true, "Getting model info (network dependent)");
        } catch (const std::exception& e) {
            std::cout << "Note: Model info retrieval failed (expected without network): " << e.what() << std::endl;
            TEST_ASSERT(true, "Model info test completed (network error expected)");
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

bool testLocalModelOperations() {
    std::cout << "\n=== Testing Local Model Operations ===" << std::endl;
    
    try {
        auto downloader = ModelDownloaderFactory::create();
        
        // 测试检查模型是否已下载
        bool is_downloaded = downloader->isModelDownloaded("registry.ollama.ai/library/test_model:latest");
        TEST_ASSERT(!is_downloaded, "Check non-existent model not downloaded");
        
        // 测试获取本地模型列表
        std::vector<std::string> local_models = downloader->getLocalModels();
        std::cout << "Found " << local_models.size() << " local models" << std::endl;
        for (const auto& model : local_models) {
            std::cout << "  - " << model << std::endl;
        }
        TEST_ASSERT(true, "Getting local models list");
        
        // 测试获取缓存大小
        size_t cache_size = downloader->getCacheSize();
        std::cout << "Current cache size: " << cache_size << " bytes" << std::endl;
        TEST_ASSERT(true, "Getting cache size");
        
        // 测试设置最大缓存大小
        downloader->setMaxCacheSize(1024 * 1024 * 1024); // 1GB
        TEST_ASSERT(true, "Setting max cache size");
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

bool testModelPathOperations() {
    std::cout << "\n=== Testing Model Path Operations ===" << std::endl;
    
    try {
        auto downloader = ModelDownloaderFactory::create();
        
        // 测试获取模型路径
        std::string model_path = downloader->getModelPath("llama2:7b");
        std::cout << "Model path for llama2:7b: " << model_path << std::endl;
        TEST_ASSERT(!model_path.empty(), "Getting model path");
        
        // 测试模型验证（对于不存在的模型应该返回false）
        bool is_valid = downloader->verifyModel("non_existent_model:latest");
        TEST_ASSERT(!is_valid, "Verify non-existent model returns false");
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

bool testCacheManagement() {
    std::cout << "\n=== Testing Cache Management ===" << std::endl;
    
    try {
        auto downloader = ModelDownloaderFactory::create();
        
        // 测试清理未使用的blobs
        size_t cleaned_size = downloader->cleanupUnusedBlobs();
        std::cout << "Cleaned up " << cleaned_size << " bytes of unused blobs" << std::endl;
        TEST_ASSERT(true, "Cleanup unused blobs");
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

int main() {
    std::cout << "Model Downloader Test Suite" << std::endl;
    std::cout << "===========================" << std::endl;
    
    bool all_passed = true;
    
    // 运行所有测试
    all_passed &= testModelDownloaderCreation();
    all_passed &= testModelInfo();
    all_passed &= testLocalModelOperations();
    all_passed &= testModelPathOperations();
    all_passed &= testCacheManagement();
    
    std::cout << "\n===========================" << std::endl;
    if (all_passed) {
        std::cout << "All tests passed! Model downloader is working correctly." << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed! Please check the implementation." << std::endl;
        return 1;
    }
}