#include "core/application.h"
#include <iostream>
#include <memory>
#include <exception>

int main(int argc, char* argv[]) {
    try {
        // 创建应用程序实例
        auto app = std::make_unique<duorou::core::Application>(argc, argv);
        
        // 初始化应用程序
        if (!app->initialize()) {
            std::cerr << "Failed to initialize application" << std::endl;
            return 1;
        }
        
        std::cout << "Duorou application initialized successfully" << std::endl;
        
        // 运行应用程序
        int result = app->run();
        
        std::cout << "Duorou application exited with code: " << result << std::endl;
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return 1;
    }
}