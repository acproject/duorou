#include "../src/core/workflow_engine.h"
#include "../src/core/model_switch_task.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <memory>

using namespace duorou::core;

void testBasicResourceLocking() {
    std::cout << "\n=== Testing Basic Resource Locking ===\n" << std::endl;
    
    WorkflowEngine engine;
    engine.initialize(2);
    engine.start();
    
    // 启用模型切换优化
    engine.optimizeModelSwitching(true);
    
    // 创建需要相同资源的任务
    auto task1 = std::make_shared<TextGenerationTask>("text_1", "Hello, how are you?");
    auto task2 = std::make_shared<TextGenerationTask>("text_2", "What is the weather like?");
    auto task3 = std::make_shared<ImageGenerationTask>("image_1", "A beautiful sunset");
    
    task1->setSimulatedDuration(std::chrono::milliseconds(1500));
    task2->setSimulatedDuration(std::chrono::milliseconds(1000));
    task3->setSimulatedDuration(std::chrono::milliseconds(2000));
    
    // 提交需要资源的任务
    std::vector<std::string> llama_resources = {"llama_model", "gpu_memory"};
    std::vector<std::string> sd_resources = {"stable_diffusion_model", "gpu_memory"};
    
    bool success1 = engine.submitTaskWithResources(task1, llama_resources);
    bool success2 = engine.submitTaskWithResources(task2, llama_resources);
    bool success3 = engine.submitTaskWithResources(task3, sd_resources);
    
    std::cout << "Task submission results: " << success1 << ", " << success2 << ", " << success3 << std::endl;
    
    // 等待任务完成
    auto result1 = engine.waitForTask("text_1", 10000);
    auto result2 = engine.waitForTask("text_2", 10000);
    auto result3 = engine.waitForTask("image_1", 15000);
    
    std::cout << "\nTask Results:" << std::endl;
    std::cout << "Text 1: " << (result1.success ? "SUCCESS" : "FAILED") << " - " << result1.message << std::endl;
    std::cout << "Text 2: " << (result2.success ? "SUCCESS" : "FAILED") << " - " << result2.message << std::endl;
    std::cout << "Image 1: " << (result3.success ? "SUCCESS" : "FAILED") << " - " << result3.message << std::endl;
    
    engine.stop();
}

void testModelSwitchingOptimization() {
    std::cout << "\n=== Testing Model Switching Optimization ===\n" << std::endl;
    
    WorkflowEngine engine;
    engine.initialize(1); // 单线程以便观察模型切换
    engine.start();
    
    // 启用模型切换优化
    engine.optimizeModelSwitching(true);
    
    // 创建需要不同模型的任务序列
    auto text_task1 = std::make_shared<TextGenerationTask>("text_seq_1", "First text task");
    auto image_task1 = std::make_shared<ImageGenerationTask>("image_seq_1", "First image task");
    auto text_task2 = std::make_shared<TextGenerationTask>("text_seq_2", "Second text task");
    auto image_task2 = std::make_shared<ImageGenerationTask>("image_seq_2", "Second image task");
    
    // 设置较短的执行时间以便快速观察切换
    text_task1->setSimulatedDuration(std::chrono::milliseconds(800));
    image_task1->setSimulatedDuration(std::chrono::milliseconds(800));
    text_task2->setSimulatedDuration(std::chrono::milliseconds(800));
    image_task2->setSimulatedDuration(std::chrono::milliseconds(800));
    
    // 按顺序提交任务
    engine.submitTask(text_task1);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    engine.submitTask(image_task1);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    engine.submitTask(text_task2);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    engine.submitTask(image_task2);
    
    // 等待所有任务完成
    auto result1 = engine.waitForTask("text_seq_1", 5000);
    auto result2 = engine.waitForTask("image_seq_1", 5000);
    auto result3 = engine.waitForTask("text_seq_2", 5000);
    auto result4 = engine.waitForTask("image_seq_2", 5000);
    
    std::cout << "\nSequential Task Results:" << std::endl;
    std::cout << "Text 1: " << (result1.success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Image 1: " << (result2.success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Text 2: " << (result3.success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Image 2: " << (result4.success ? "SUCCESS" : "FAILED") << std::endl;
    
    engine.stop();
}

void testResourceManagerFeatures() {
    std::cout << "\n=== Testing Resource Manager Features ===\n" << std::endl;
    
    WorkflowEngine engine;
    engine.initialize(2);
    engine.start();
    
    auto& resource_manager = engine.getResourceManager();
    
    // 注册额外的资源
    ResourceInfo custom_model_info;
    custom_model_info.id = "custom_model";
    custom_model_info.type = ResourceType::MODEL;
    custom_model_info.name = "Custom Model";
    custom_model_info.capacity = 1;
    custom_model_info.used = 0;
    custom_model_info.available = true;
    custom_model_info.last_accessed = std::chrono::system_clock::now();
    resource_manager.registerResource(custom_model_info);
    
    ResourceInfo network_info;
    network_info.id = "network_bandwidth";
    network_info.type = ResourceType::NETWORK;
    network_info.name = "Network Bandwidth";
    network_info.capacity = 100;
    network_info.used = 0;
    network_info.available = true;
    network_info.last_accessed = std::chrono::system_clock::now();
    resource_manager.registerResource(network_info);
    
    // 获取资源信息
    auto resources = resource_manager.getResourceList();
    std::cout << "\nRegistered Resources:" << std::endl;
    for (const auto& resource_id : resources) {
        auto info = resource_manager.getResourceInfo(resource_id);
        std::cout << "- " << resource_id << ": capacity=" << info.capacity 
                  << ", used=" << info.used << ", available=" << (info.available ? "YES" : "NO") << std::endl;
    }
    
    // 测试资源预留
    std::cout << "\nTesting resource reservation..." << std::endl;
    bool reserved = resource_manager.reserveResource("custom_model", "test_holder", 1);
    std::cout << "Resource reservation: " << (reserved ? "SUCCESS" : "FAILED") << std::endl;
    
    // 检查资源可用性
    bool available = resource_manager.isResourceAvailable("custom_model", LockMode::SHARED);
    std::cout << "Resource availability after reservation: " << (available ? "AVAILABLE" : "NOT AVAILABLE") << std::endl;

    // 释放资源
    resource_manager.releaseReservation("custom_model", "test_holder");
    available = resource_manager.isResourceAvailable("custom_model", LockMode::SHARED);
    std::cout << "Resource availability after release: " << (available ? "AVAILABLE" : "NOT AVAILABLE") << std::endl;
    
    // 获取统计信息
    auto stats = resource_manager.getResourceStatistics();
    std::cout << "\nResource Statistics:" << std::endl;
    for (const auto& [resource_id, count] : stats) {
        std::cout << "- " << resource_id << ": " << count << std::endl;
    }
    
    engine.stop();
}

void testTaskCancellation() {
    std::cout << "\n=== Testing Task Cancellation with Resources ===\n" << std::endl;
    
    WorkflowEngine engine;
    engine.initialize(1);
    engine.start();
    
    // 创建长时间运行的任务
    auto long_task = std::make_shared<ImageGenerationTask>("long_image", "A very detailed image");
    long_task->setSimulatedDuration(std::chrono::milliseconds(5000));
    
    std::vector<std::string> resources = {"stable_diffusion_model", "gpu_memory"};
    
    // 提交任务
    bool submitted = engine.submitTaskWithResources(long_task, resources);
    std::cout << "Long task submitted: " << (submitted ? "SUCCESS" : "FAILED") << std::endl;
    
    // 等待一段时间后取消
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    bool cancelled = engine.cancelTask("long_image");
    std::cout << "Task cancellation: " << (cancelled ? "SUCCESS" : "FAILED") << std::endl;
    
    // 等待任务结果
    auto result = engine.waitForTask("long_image", 3000);
    std::cout << "Cancelled task result: " << (result.success ? "SUCCESS" : "FAILED") 
              << " - " << result.message << std::endl;
    
    engine.stop();
}

int main() {
    std::cout << "Enhanced Workflow Engine Test Suite" << std::endl;
    std::cout << "===================================" << std::endl;
    
    try {
        testBasicResourceLocking();
        testModelSwitchingOptimization();
        testResourceManagerFeatures();
        testTaskCancellation();
        
        std::cout << "\n=== All Tests Completed ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
    
    return 0;
}