#include "core/application.h"
#include "core/mtmd_demo.h"
#include <iostream>
#include <memory>
#include <exception>

int main(int argc, char* argv[]) {
    try {
        // 如果设置了演示标志，则运行多模态演示并退出
        if (const char* demo = std::getenv("DUOROU_RUN_MTMD_DEMO")) {
            std::string val(demo);
            if (val == "1" || val == "true" || val == "TRUE") {
                return run_mtmd_demo();
            }
        }

        // Create application instance
        auto app = std::make_unique<duorou::core::Application>(argc, argv);
        
        // Initialize application
        if (!app->initialize()) {
            std::cerr << "Failed to initialize application" << std::endl;
            return 1;
        }
        
        std::cout << "Duorou application initialized successfully" << std::endl;
        
        // Run application
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