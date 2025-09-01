#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <cassert>
#include "../src/core/workflow_engine.h"
#include "../src/core/resource_manager.h"

using namespace duorou::core;

// 简单的测试任务类
class SimpleTestTask : public BaseTask {
public:
    SimpleTestTask(const std::string& id, const std::string& name, int sleep_ms = 100)
        : BaseTask(id, name, TaskPriority::NORMAL)
        , sleep_ms_(sleep_ms)
        , executed_(false) {
    }

    TaskResult execute() override {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms_));
        executed_ = true;
        result_ = "Task " + getName() + " completed";
        
        TaskResult task_result;
        task_result.success = true;
        task_result.message = result_;
        task_result.output_data = result_;
        task_result.duration = std::chrono::milliseconds(sleep_ms_);
        return task_result;
    }

    std::string getRequiredModel() const override {
        return "test_model";
    }

    bool wasExecuted() const { return executed_; }
    const std::string& getResult() const { return result_; }

private:
    int sleep_ms_;
    bool executed_;
    std::string result_;
};

// 简单的测试断言宏
#define TEST_ASSERT(condition, message) \
    if (!(condition)) { \
        std::cerr << "FAILED: " << message << std::endl; \
        return false; \
    } else { \
        std::cout << "PASSED: " << message << std::endl; \
    }

// 测试基本的工作流引擎功能
bool testBasicTaskExecution() {
    std::cout << "\n=== Testing Basic Task Execution ===" << std::endl;
    
    WorkflowEngine engine;
    TEST_ASSERT(engine.initialize(2), "Engine initialization");
    TEST_ASSERT(engine.start(), "Engine start");

    auto task = std::make_shared<SimpleTestTask>("test1", "TestTask1", 50);
    TEST_ASSERT(engine.submitTask(task), "Task submission");

    // 等待任务完成
    auto result = engine.waitForTask("test1", 5000);
    TEST_ASSERT(result.success, "Task completion");
    TEST_ASSERT(task->wasExecuted(), "Task execution");

    engine.stop();
    return true;
}

// 测试资源锁定功能
bool testResourceLocking() {
    std::cout << "\n=== Testing Resource Locking ===" << std::endl;
    
    WorkflowEngine engine;
    TEST_ASSERT(engine.initialize(2), "Engine initialization");
    TEST_ASSERT(engine.start(), "Engine start");

    // 注册测试资源
    auto& resource_manager = engine.getResourceManager();
    ResourceInfo test_resource;
    test_resource.id = "test_resource";
    test_resource.type = ResourceType::COMPUTE_UNIT;
    test_resource.name = "Test Resource";
    test_resource.capacity = 1;
    test_resource.used = 0;
    test_resource.available = true;
    TEST_ASSERT(resource_manager.registerResource(test_resource), "Resource registration");

    auto task1 = std::make_shared<SimpleTestTask>("task1", "Task1", 200);
    auto task2 = std::make_shared<SimpleTestTask>("task2", "Task2", 100);

    // 提交第一个需要资源的任务
    std::vector<std::string> resources = {"test_resource"};
    TEST_ASSERT(engine.submitTaskWithResources(task1, resources, LockMode::EXCLUSIVE), "Task1 submission with resources");

    // 等待第一个任务完成
    auto result1 = engine.waitForTask("task1", 5000);
    TEST_ASSERT(result1.success, "Task1 completion");

    // 第一个任务完成后，提交第二个任务
    TEST_ASSERT(engine.submitTaskWithResources(task2, resources, LockMode::EXCLUSIVE), "Task2 submission with resources");

    // 等待第二个任务完成
    auto result2 = engine.waitForTask("task2", 5000);
    TEST_ASSERT(result2.success, "Task2 completion");

    engine.stop();
    return true;
}

// 测试任务取消功能
bool testTaskCancellation() {
    std::cout << "\n=== Testing Task Cancellation ===" << std::endl;
    
    WorkflowEngine engine;
    TEST_ASSERT(engine.initialize(1), "Engine initialization");
    TEST_ASSERT(engine.start(), "Engine start");

    auto task = std::make_shared<SimpleTestTask>("cancel_test", "CancelTask", 1000);
    TEST_ASSERT(engine.submitTask(task), "Task submission");

    // 等待一小段时间后取消任务
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    TEST_ASSERT(engine.cancelTask("cancel_test"), "Task cancellation");

    // 检查任务状态
    auto status = engine.getTaskStatus("cancel_test");
    TEST_ASSERT(status == TaskStatus::CANCELLED, "Task status after cancellation");

    engine.stop();
    return true;
}

// 测试模型切换优化
bool testModelSwitchingOptimization() {
    std::cout << "\n=== Testing Model Switching Optimization ===" << std::endl;
    
    WorkflowEngine engine;
    TEST_ASSERT(engine.initialize(1), "Engine initialization");
    
    // 启用模型切换优化
    engine.optimizeModelSwitching(true);
    TEST_ASSERT(engine.isModelSwitchingOptimized(), "Model switching optimization enabled");
    
    TEST_ASSERT(engine.start(), "Engine start");

    auto task1 = std::make_shared<SimpleTestTask>("model_test1", "ModelTask1", 50);
    auto task2 = std::make_shared<SimpleTestTask>("model_test2", "ModelTask2", 50);

    TEST_ASSERT(engine.submitTask(task1), "Task1 submission");
    TEST_ASSERT(engine.submitTask(task2), "Task2 submission");

    // 等待任务完成
    auto result1 = engine.waitForTask("model_test1", 5000);
    auto result2 = engine.waitForTask("model_test2", 5000);
    
    TEST_ASSERT(result1.success, "Task1 completion");
    TEST_ASSERT(result2.success, "Task2 completion");

    engine.stop();
    return true;
}

int main() {
    std::cout << "Starting Workflow Engine Core Tests..." << std::endl;
    
    bool all_passed = true;
    
    all_passed &= testBasicTaskExecution();
    all_passed &= testResourceLocking();
    all_passed &= testTaskCancellation();
    all_passed &= testModelSwitchingOptimization();
    
    if (all_passed) {
        std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== SOME TESTS FAILED ===" << std::endl;
        return 1;
    }
}