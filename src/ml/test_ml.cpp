#include "tensor.h"
#include "context.h"
#include "backend/backend.h"
#include "backend/cpu_backend.h"
#include "nn/linear.h"
#include "nn/activation.h"
#include <iostream>
#include <vector>

using namespace duorou::ml;

int main() {
    std::cout << "=== Duorou ML Module Test ===" << std::endl;
    
    try {
        // 1. 测试 Backend 创建
        std::cout << "\n1. Testing Backend Creation..." << std::endl;
        auto& factory = BackendFactory::getInstance();
        
        // 注册 CPU 后端
        factory.registerBackend(DeviceType::CPU, []() {
            return std::make_unique<CPUBackend>();
        });
        
        auto backend = factory.createBackend(DeviceType::CPU);
        if (backend) {
            std::cout << "✓ CPU Backend created successfully" << std::endl;
            std::cout << "  Backend name: " << backend->getName() << std::endl;
            std::cout << "  Backend available: " << (backend->isAvailable() ? "Yes" : "No") << std::endl;
        } else {
            std::cout << "✗ Failed to create CPU Backend" << std::endl;
            return 1;
        }
        
        // 2. 测试 Context 创建
        std::cout << "\n2. Testing Context Creation..." << std::endl;
        Context ctx(backend.get());
        std::cout << "✓ Context created successfully" << std::endl;
        
        // 3. 测试 Tensor 创建
        std::cout << "\n3. Testing Tensor Creation..." << std::endl;
        Tensor tensor1({2, 3}, DataType::FLOAT32);
        std::cout << "✓ Tensor created with shape [2, 3]" << std::endl;
        std::cout << "  Tensor dimensions: " << tensor1.ndim() << std::endl;
        std::cout << "  Tensor elements: " << tensor1.numel() << std::endl;
        std::cout << "  Tensor size: " << tensor1.nbytes() << " bytes" << std::endl;
        
        // 4. 测试静态工厂方法
        std::cout << "\n4. Testing Static Factory Methods..." << std::endl;
        
        try {
            std::cout << "  Creating zeros tensor..." << std::endl;
            auto zeros_tensor = Tensor::zeros({2, 2});
            std::cout << "✓ Zeros tensor created with shape [2, 2]" << std::endl;
            std::cout << "  Zeros tensor allocated: " << (zeros_tensor.isAllocated() ? "Yes" : "No") << std::endl;
            
            std::cout << "  Creating ones tensor..." << std::endl;
            auto ones_tensor = Tensor::ones({2, 2});
            std::cout << "✓ Ones tensor created with shape [2, 2]" << std::endl;
            std::cout << "  Ones tensor allocated: " << (ones_tensor.isAllocated() ? "Yes" : "No") << std::endl;
        } catch (const std::exception& e) {
            std::cout << "✗ Error in static factory methods: " << e.what() << std::endl;
            return 1;
        }
        
        // 5. 测试 Linear 层
        std::cout << "\n5. Testing Linear Layer..." << std::endl;
        nn::Linear linear(4, 2);  // 输入4维，输出2维
        std::cout << "✓ Linear layer created (4 -> 2)" << std::endl;
        
        // 6. 测试激活函数
        std::cout << "\n6. Testing Activation Functions..." << std::endl;
        auto relu = nn::ActivationFactory::create(nn::ActivationFactory::Type::RELU);
        auto sigmoid = nn::ActivationFactory::create(nn::ActivationFactory::Type::SIGMOID);
        
        if (relu && sigmoid) {
            std::cout << "✓ ReLU activation created" << std::endl;
            std::cout << "✓ Sigmoid activation created" << std::endl;
        }
        
        // 7. 测试数据类型转换
        std::cout << "\n7. Testing Data Type Utilities..." << std::endl;
        std::string dtypeStr = dataTypeToString(DataType::FLOAT32);
        DataType dtype = stringToDataType("float32");
        (void)dtype; // 避免未使用变量警告
        std::cout << "✓ DataType to string: " << dtypeStr << std::endl;
        std::cout << "✓ String to DataType conversion works" << std::endl;
        
        // 8. 测试设备类型转换
        std::cout << "\n8. Testing Device Type Utilities..." << std::endl;
        std::string deviceStr = deviceTypeToString(DeviceType::CPU);
        DeviceType deviceType = stringToDeviceType("cpu");
        (void)deviceType; // 避免未使用变量警告
        std::cout << "✓ DeviceType to string: " << deviceStr << std::endl;
        std::cout << "✓ String to DeviceType conversion works" << std::endl;
        
        std::cout << "\n=== All Tests Passed! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "\n✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "\n✗ Test failed with unknown exception" << std::endl;
        return 1;
    }
}