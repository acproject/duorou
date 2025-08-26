#include "application.h"
#include "config_manager.h"
#include "logger.h"
#include "model_manager.h"
#include "workflow_engine.h"
#include "../api/api_server.h"
#include "../gui/system_tray.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <unistd.h>  // for getpid

// 版本定义
#ifndef DUOROU_VERSION
#define DUOROU_VERSION "1.0.0"
#endif

// 条件编译GTK支持
#ifdef HAVE_GTK
#include <gtk/gtk.h>
#else
// GTK类型的占位符定义
typedef void* GApplication;
#define G_APPLICATION(x) ((GApplication)(x))
#define G_APPLICATION_FLAGS_NONE 0
#define G_CALLBACK(x) ((void(*)(void))(x))
#define g_application_run(app, argc, argv) 0
#define g_application_quit(app) do {} while(0)
#define g_object_unref(obj) do {} while(0)
#define g_signal_connect(obj, signal, callback, data) do {} while(0)
#define gtk_application_new(id, flags) nullptr
#endif

namespace duorou {
namespace core {

// 静态成员初始化
Application* Application::instance_ = nullptr;

Application::Application(int argc, char* argv[])
    : app_name_("Duorou")
    , version_(DUOROU_VERSION)
    , status_(Status::NotInitialized)
    , service_mode_(false)
    , gtk_app_(nullptr) {
    
    // 保存命令行参数
    for (int i = 0; i < argc; ++i) {
        args_.emplace_back(argv[i]);
    }
    
    // 检查是否有服务模式参数
    for (const auto& arg : args_) {
        if (arg == "--service" || arg == "-s" || arg == "--mode=server") {
            service_mode_ = true;
            break;
        }
    }
    
    // 设置静态实例指针
    instance_ = this;
    
    // 注册信号处理函数
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
}

Application::~Application() {
    cleanup();
    instance_ = nullptr;
}

void Application::setServiceMode(bool service_mode) {
    service_mode_ = service_mode;
}

bool Application::initialize() {
    if (status_ != Status::NotInitialized) {
        std::cerr << "Application already initialized" << std::endl;
        return false;
    }
    
    status_ = Status::Initializing;
    
    // 在非服务模式下初始化GTK
    if (!service_mode_) {
        if (!initializeGtk()) {
            std::cerr << "Failed to initialize GTK" << std::endl;
            status_ = Status::NotInitialized;
            return false;
        }
    } else {
        std::cout << "Running in service mode (no GUI)" << std::endl;
    }
    
    // 初始化核心组件
    if (!initializeComponents()) {
        std::cerr << "Failed to initialize core components" << std::endl;
        status_ = Status::NotInitialized;
        return false;
    }
    
    status_ = Status::Running;
    return true;
}

int Application::run() {
    if (status_ != Status::Running) {
        std::cerr << "Application not properly initialized" << std::endl;
        return -1;
    }
    
    if (service_mode_) {
        // 服务模式：启动API服务器
        std::cout << "Service mode started. Press Ctrl+C to stop." << std::endl;
        
        // 创建并启动API服务器
        std::cout << "Creating API server..." << std::endl;
        api_server_ = std::make_unique<::duorou::ApiServer>(
            std::shared_ptr<core::ModelManager>(model_manager_.get(), [](core::ModelManager*){}),
            std::shared_ptr<core::Logger>(logger_.get(), [](core::Logger*){})
        );
        
        if (!api_server_->start()) {
            std::cerr << "Failed to start API server" << std::endl;
            return -1;
        }
        
        std::cout << "API server started on port " << api_server_->getPort() << std::endl;
        
        // 记录启动信息
        if (logger_) {
            logger_->info("Application started in service mode");
            logger_->info("Version: " + version_);
            logger_->info("API server started on port " + std::to_string(api_server_->getPort()));
        }
        
        // 简单的服务循环
        while (status_ == Status::Running) {
            // 这里可以添加定期任务
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            // 定期记录状态
            static int counter = 0;
            if (++counter % 60 == 0 && logger_) {  // 每分钟记录一次
                logger_->debug("Service running, uptime: " + std::to_string(counter) + " seconds");
            }
        }
        
        if (logger_) {
            logger_->info("Service mode stopped");
        }
        
        return 0;
    }
    
    // 运行GTK主循环
    int result = g_application_run(G_APPLICATION(gtk_app_), 
                                   static_cast<int>(args_.size()), 
                                   const_cast<char**>(reinterpret_cast<const char* const*>(args_.data())));
    
    return result;
}

void Application::stop() {
    if (status_ == Status::Running) {
        status_ = Status::Stopping;
        
        // 调用退出回调函数
        for (auto& callback : exit_callbacks_) {
            try {
                callback();
            } catch (const std::exception& e) {
                std::cerr << "Exception in exit callback: " << e.what() << std::endl;
            }
        }
        
        // 停止GTK应用程序
        if (gtk_app_) {
            g_application_quit(G_APPLICATION(gtk_app_));
        }
        
        status_ = Status::Stopped;
    }
}

void Application::registerExitCallback(std::function<void()> callback) {
    exit_callbacks_.push_back(std::move(callback));
}

bool Application::initializeGtk() {
    // 创建GTK应用程序
    gtk_app_ = gtk_application_new("com.duorou.app", G_APPLICATION_FLAGS_NONE);
    if (!gtk_app_) {
        return false;
    }
    
    // 连接信号
    g_signal_connect(gtk_app_, "activate", G_CALLBACK(onActivate), this);
    g_signal_connect(gtk_app_, "shutdown", G_CALLBACK(onShutdown), this);
    
    return true;
}

bool Application::initializeComponents() {
    try {
        std::cout << "Initializing core components..." << std::endl;
        
        // 初始化日志系统
        std::cout << "Creating logger..." << std::endl;
        logger_ = std::make_unique<Logger>();
        std::cout << "Initializing logger..." << std::endl;
        if (!logger_->initialize()) {
            std::cerr << "Failed to initialize logger" << std::endl;
            return false;
        }
        std::cout << "Logger initialized successfully" << std::endl;
        
        // 初始化配置管理器
        std::cout << "Creating config manager..." << std::endl;
        config_manager_ = std::make_unique<ConfigManager>();
        std::cout << "Initializing config manager..." << std::endl;
        if (!config_manager_->initialize()) {
            logger_->error("Failed to initialize config manager");
            return false;
        }
        std::cout << "Config manager initialized successfully" << std::endl;
        
        // 根据配置设置Logger的文件输出
        bool file_output = config_manager_->getBool("log.file_output", true);
        if (file_output) {
            std::string log_path = logger_->getDefaultLogPath();
            if (logger_->setLogFile(log_path)) {
                logger_->info("Log file output enabled: " + log_path);
            } else {
                logger_->warning("Failed to enable log file output");
            }
        }
        
        // 初始化模型管理器
        std::cout << "Creating model manager..." << std::endl;
        model_manager_ = std::make_unique<ModelManager>();
        std::cout << "Initializing model manager..." << std::endl;
        if (!model_manager_->initialize()) {
            logger_->error("Failed to initialize model manager");
            return false;
        }
        std::cout << "Model manager initialized successfully" << std::endl;
        
        // 初始化工作流引擎
        std::cout << "Creating workflow engine..." << std::endl;
        workflow_engine_ = std::make_unique<WorkflowEngine>();
        std::cout << "Initializing workflow engine..." << std::endl;
        if (!workflow_engine_->initialize()) {
            logger_->error("Failed to initialize workflow engine");
            return false;
        }
        std::cout << "Workflow engine initialized successfully" << std::endl;
        
        // 在服务模式下初始化API服务器
        if (service_mode_) {
            std::cout << "API server will be started in service mode" << std::endl;
        }
        
        // 初始化系统托盘（仅在GUI模式下）
        if (!service_mode_ && !initializeSystemTray()) {
            logger_->info("Failed to initialize system tray, continuing without it");
        }
        
        logger_->info("All core components initialized successfully");
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during component initialization: " << e.what() << std::endl;
        return false;
    }
}

void Application::cleanup() {
    if (status_ == Status::Stopped || status_ == Status::NotInitialized) {
        return;
    }
    
    // 停止API服务器
    if (api_server_) {
        api_server_->stop();
        api_server_.reset();
    }
    
    // 清理核心组件（按相反顺序）
    workflow_engine_.reset();
    model_manager_.reset();
    config_manager_.reset();
    logger_.reset();
    
    // 清理GTK资源
    if (gtk_app_) {
        g_object_unref(gtk_app_);
        gtk_app_ = nullptr;
    }
    
    status_ = Status::Stopped;
}

void Application::onActivate(GtkApplication* app, gpointer user_data) {
    Application* application = static_cast<Application*>(user_data);
    if (application && application->logger_) {
        application->logger_->info("Application activated");
    }
    
    // TODO: 创建主窗口
    // 这里将在GUI模块开发时实现
}

void Application::onShutdown(GtkApplication* app, gpointer user_data) {
    Application* application = static_cast<Application*>(user_data);
    if (application) {
        if (application->logger_) {
            application->logger_->info("Application shutting down");
        }
        application->cleanup();
    }
}

bool Application::initializeSystemTray() {
    try {
        system_tray_ = std::make_unique<::duorou::gui::SystemTray>();
        
        // 初始化系统托盘
        if (!system_tray_->initialize("Duorou AI Assistant")) {
            logger_->error("Failed to initialize system tray");
            return false;
        }
        
        // 设置系统托盘属性
        system_tray_->setTooltip("Duorou AI Assistant");
        system_tray_->setStatus(::duorou::gui::TrayStatus::Idle);
        
        // 设置回调函数
        system_tray_->setLeftClickCallback([this]() {
            showMainWindow();
        });
        
        // 添加菜单项
        std::vector<::duorou::gui::TrayMenuItem> menu_items;
        
        ::duorou::gui::TrayMenuItem show_item;
        show_item.id = "show";
        show_item.label = "显示主窗口";
        show_item.callback = [this]() { showMainWindow(); };
        menu_items.push_back(show_item);
        
        ::duorou::gui::TrayMenuItem separator;
        separator.separator = true;
        menu_items.push_back(separator);
        
        ::duorou::gui::TrayMenuItem exit_item;
        exit_item.id = "exit";
        exit_item.label = "退出";
        exit_item.callback = [this]() { stop(); };
        menu_items.push_back(exit_item);
        
        system_tray_->setMenu(menu_items);
        
        // 显示系统托盘
        system_tray_->show();
        
        logger_->info("System tray initialized successfully");
        return true;
    } catch (const std::exception& e) {
        logger_->error("Exception during system tray initialization: " + std::string(e.what()));
        return false;
    }
}

void Application::showMainWindow() {
    // TODO: 实现显示主窗口的逻辑
    if (logger_) {
        logger_->info("Show main window requested");
    }
}

void Application::hideMainWindow() {
    // TODO: 实现隐藏主窗口的逻辑
    if (logger_) {
        logger_->info("Hide main window requested");
    }
}

void Application::toggleMainWindow() {
    // TODO: 实现切换主窗口显示状态的逻辑
    if (logger_) {
        logger_->info("Toggle main window requested");
    }
}

void Application::signalHandler(int signal) {
    if (instance_) {
        std::cout << "\nReceived signal " << signal << ", shutting down gracefully..." << std::endl;
        instance_->stop();
    }
}

} // namespace core
} // namespace duorou