#include "application.h"
#include "../api/api_server.h"
#include "../gui/main_window.h"
#include "../gui/system_tray.h"
#include "config_manager.h"
#include "logger.h"
#include "model_manager.h"
#include "workflow_engine.h"
#ifdef __APPLE__
#include "../media/macos_screen_capture.h"
#endif

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <unistd.h> // for getpid

// 版本定义
#ifndef DUOROU_VERSION
#define DUOROU_VERSION "1.0.0"
#endif

// 条件编译GTK支持
#ifdef HAVE_GTK
#include <gtk/gtk.h>
#else
// GTK类型的占位符定义
typedef void *GApplication;
#define G_APPLICATION(x) ((GApplication)(x))
#define G_APPLICATION_FLAGS_NONE 0
#define G_CALLBACK(x) ((void (*)(void))(x))
#define g_application_run(app, argc, argv) 0
#define g_application_quit(app)                                                \
  do {                                                                         \
  } while (0)
#define g_object_unref(obj)                                                    \
  do {                                                                         \
  } while (0)
#define g_signal_connect(obj, signal, callback, data)                          \
  do {                                                                         \
  } while (0)
#define gtk_application_new(id, flags) nullptr
#endif

namespace duorou {
namespace core {

// 静态成员初始化
Application *Application::instance_ = nullptr;

Application::Application(int argc, char *argv[])
    : app_name_("Duorou"), version_(DUOROU_VERSION),
      status_(Status::NotInitialized), service_mode_(false), gtk_app_(nullptr),
      minimemory_running_(false) {

  // 保存命令行参数
  for (int i = 0; i < argc; ++i) {
    args_.emplace_back(argv[i]);
  }

  // 检查是否有服务模式参数
  for (const auto &arg : args_) {
    if (arg == "--service" || arg == "-s" || arg == "--mode=server") {
      service_mode_ = true;
      break;
    }
  }

  // 设置MiniMemory可执行文件路径（相对于duorou的build目录）
  minimemory_executable_path_ =
      "../third_party/MiniMemory/build/bin/mini_cache_server";

  // 设置静态实例指针
  instance_ = this;

  // 注册信号处理函数
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);
}

Application::~Application() {
  // 确保MiniMemory服务器被停止
  stopMiniMemoryServer();
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
        std::shared_ptr<core::ModelManager>(model_manager_.get(),
                                            [](core::ModelManager *) {}),
        std::shared_ptr<core::Logger>(logger_.get(), [](core::Logger *) {}));

    if (!api_server_->start()) {
      std::cerr << "Failed to start API server" << std::endl;
      return -1;
    }

    std::cout << "API server started on port " << api_server_->getPort()
              << std::endl;

    // 记录启动信息
    if (logger_) {
      logger_->info("Application started in service mode");
      logger_->info("Version: " + version_);
      logger_->info(
          "Service mode started (API server disabled for development)");
    }

    // 简单的服务循环
    while (status_ == Status::Running) {
      // 这里可以添加定期任务
      std::this_thread::sleep_for(std::chrono::seconds(1));

      // 定期记录状态
      static int counter = 0;
      if (++counter % 60 == 0 && logger_) { // 每分钟记录一次
        logger_->debug("Service running, uptime: " + std::to_string(counter) +
                       " seconds");
      }
    }

    if (logger_) {
      logger_->info("Service mode stopped");
    }

    return 0;
  }

  // 运行GTK主循环
  // 将std::vector<std::string>转换为char**
  std::vector<char *> argv_ptrs;
  for (auto &arg : args_) {
    argv_ptrs.push_back(const_cast<char *>(arg.c_str()));
  }

  int result =
      g_application_run(G_APPLICATION(gtk_app_),
                        static_cast<int>(argv_ptrs.size()), argv_ptrs.data());

  return result;
}

void Application::stop() {
  if (status_ == Status::Running) {
    status_ = Status::Stopping;

    // 调用退出回调函数
    for (auto &callback : exit_callbacks_) {
      try {
        callback();
      } catch (const std::exception &e) {
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

    // 初始化系统托盘（仅在非服务模式下）
    // 暂时禁用系统托盘以调试其他组件的段错误问题
    if (!service_mode_) {
      logger_->info(
          "System tray initialization temporarily disabled for debugging");
      // if (!initializeSystemTray()) {
      //     logger_->warning("Failed to initialize system tray, continuing
      //     without it");
      // }
    }

    // 启动MiniMemory服务器
    std::cout << "Starting MiniMemory server..." << std::endl;
    if (!startMiniMemoryServer()) {
      logger_->error("Failed to start MiniMemory server");
      return false;
    }
    std::cout << "MiniMemory server started successfully" << std::endl;

    logger_->info("All core components initialized successfully");
    return true;

  } catch (const std::exception &e) {
    std::cerr << "Exception during component initialization: " << e.what()
              << std::endl;
    return false;
  }
}

void Application::cleanup() {
  if (status_ == Status::Stopped || status_ == Status::NotInitialized) {
    return;
  }

  // 停止MiniMemory服务器
  stopMiniMemoryServer();

  // 停止API服务器
  if (api_server_) {
    api_server_->stop();
    api_server_.reset();
  }

#ifdef __APPLE__
  // 清理 ScreenCaptureKit 资源
  duorou::media::cleanup_macos_screen_capture();
#endif

  // 清理GUI组件
  main_window_.reset();
  system_tray_.reset();

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

void Application::onActivate(GtkApplication *app, gpointer user_data) {
  Application *application = static_cast<Application *>(user_data);
  if (application && application->logger_) {
    application->logger_->info("Application activated");
  }

  // 创建并显示主窗口
  if (application && !application->main_window_) {
    application->main_window_ =
        std::make_unique<::duorou::gui::MainWindow>(application);
    if (application->main_window_->initialize()) {
      application->main_window_->show();
      if (application->logger_) {
        application->logger_->info("Main window created and shown");
      }
      // 保持应用程序活动状态
      g_application_hold(G_APPLICATION(app));
    } else {
      if (application->logger_) {
        application->logger_->error("Failed to initialize main window");
      }
    }
  }
}

void Application::onShutdown(GtkApplication *app, gpointer user_data) {
  Application *application = static_cast<Application *>(user_data);
  if (application) {
    if (application->logger_) {
      application->logger_->info("Application shutting down");
    }

    // 释放应用程序保持状态
    g_application_release(G_APPLICATION(app));

    // 停止MiniMemory服务器
    application->stopMiniMemoryServer();

#ifdef __APPLE__
    // 清理 ScreenCaptureKit 资源
    duorou::media::cleanup_macos_screen_capture();
#endif

    // 只清理非GTK资源，GTK资源由GTK自动管理
    if (application->api_server_) {
      application->api_server_->stop();
      application->api_server_.reset();
    }
    application->main_window_.reset();
    application->system_tray_.reset();
    application->workflow_engine_.reset();
    application->model_manager_.reset();
    application->config_manager_.reset();
    application->logger_.reset();
    application->status_ = Status::Stopped;
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
    system_tray_->setLeftClickCallback([this]() { showMainWindow(); });

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
  } catch (const std::exception &e) {
    logger_->error("Exception during system tray initialization: " +
                   std::string(e.what()));
    return false;
  }
}

void Application::showMainWindow() {
  if (main_window_) {
    main_window_->show();
    if (logger_) {
      logger_->info("Main window shown");
    }
  } else {
    if (logger_) {
      logger_->warning("Main window not initialized");
    }
  }
}

void Application::hideMainWindow() {
  if (main_window_) {
    main_window_->hide();
    if (logger_) {
      logger_->info("Main window hidden");
    }
  } else {
    if (logger_) {
      logger_->warning("Main window not initialized");
    }
  }
}

void Application::toggleMainWindow() {
  if (main_window_) {
    // 这里需要检查窗口当前状态，暂时先实现简单的显示逻辑
    main_window_->show();
    if (logger_) {
      logger_->info("Main window toggled");
    }
  } else {
    if (logger_) {
      logger_->warning("Main window not initialized");
    }
  }
}

void Application::signalHandler(int signal) {
  if (instance_) {
    std::cout << "\nReceived signal " << signal << ", shutting down..."
              << std::endl;
    instance_->stop();
  }
}

bool Application::startMiniMemoryServer() {
  if (minimemory_running_.load()) {
    if (logger_) {
      logger_->warning("MiniMemory server is already running");
    }
    return true;
  }

  try {
    minimemory_running_.store(true);

    // 创建线程来运行MiniMemory服务器
    minimemory_thread_ = std::make_unique<std::thread>([this]() {
      // 切换到MiniMemory构建目录（相对于duorou的build目录）
      std::string command =
          "cd ../third_party/MiniMemory/build && ./bin/mini_cache_server";

      if (logger_) {
        logger_->info("Starting MiniMemory server with command: " + command);
      }

      // 使用system()执行命令
      int result = std::system(command.c_str());

      if (logger_) {
        if (result == 0) {
          logger_->info("MiniMemory server exited normally");
        } else {
          logger_->error("MiniMemory server exited with code: " +
                         std::to_string(result));
        }
      }

      minimemory_running_.store(false);
    });

    // 等待一小段时间确保服务器启动
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    if (logger_) {
      logger_->info("MiniMemory server started successfully");
    }

    return true;

  } catch (const std::exception &e) {
    minimemory_running_.store(false);
    if (logger_) {
      logger_->error("Failed to start MiniMemory server: " +
                     std::string(e.what()));
    }
    return false;
  }
}

void Application::stopMiniMemoryServer() {
  if (!minimemory_running_.load()) {
    return;
  }

  if (logger_) {
    logger_->info("Stopping MiniMemory server...");
  }

  // 设置停止标志
  minimemory_running_.store(false);

  // 发送SIGTERM信号给MiniMemory进程
  std::system("pkill -f mini_cache_server");

  // 等待进程真正停止
  int attempts = 0;
  const int max_attempts = 50; // 最多等待5秒
  while (attempts < max_attempts) {
    int result = std::system("pgrep -f mini_cache_server > /dev/null 2>&1");
    if (result != 0) {
      // 进程已停止
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    attempts++;
  }

  // 如果进程仍在运行，强制杀死
  if (attempts >= max_attempts) {
    std::system("pkill -9 -f mini_cache_server");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  // 等待线程结束
  if (minimemory_thread_ && minimemory_thread_->joinable()) {
    try {
      minimemory_thread_->join();
    } catch (const std::exception &e) {
      // 如果join失败，使用detach
      if (minimemory_thread_->joinable()) {
        minimemory_thread_->detach();
      }
    }
    minimemory_thread_.reset();
  }

  if (logger_) {
    logger_->info("MiniMemory server stopped");
  }
}

} // namespace core
} // namespace duorou