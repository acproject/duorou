#include "core/application.h"
#include <iostream>
#include <memory>
#include <exception>

int main(int argc, char* argv[]) {
    try {
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