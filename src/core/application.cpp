#include "application.h"
// #include "../api/api_server.h"  // 注释：暂时禁用API服务器
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
#include <sys/wait.h> // for waitpid

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
      minimemory_running_(false), is_destructing_(false) {

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
  // 设置析构标志
  is_destructing_.store(true);
  
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
    // 服务模式：运行服务循环
    std::cout << "Service mode started. Press Ctrl+C to stop." << std::endl;

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

    // 启动API服务器（无论是否服务模式）
    // 注释：暂时禁用API服务器以解决调试问题
    /*
    std::cout << "Creating API server..." << std::endl;
    api_server_ = std::make_unique<::duorou::ApiServer>(
        std::shared_ptr<core::ModelManager>(model_manager_.get(),
                                            [](core::ModelManager *) {}),
        std::shared_ptr<core::Logger>(logger_.get(), [](core::Logger *) {}));

    if (!api_server_->start()) {
      logger_->error("Failed to start API server");
      return false;
    }
    
    std::cout << "API server started on port " << api_server_->getPort() << std::endl;
    logger_->info("API server started on port " + std::to_string(api_server_->getPort()));
    
    if (service_mode_) {
      std::cout << "Running in service mode" << std::endl;
    } else {
      std::cout << "API server started for GUI mode" << std::endl;
    }
    */

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
  // 注释：API服务器已被禁用
  /*
  if (api_server_) {
    api_server_->stop();
    api_server_.reset();
  }
  */

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
    if (application->logger_) {
      application->logger_->info("正在退出MiniMemory服务...");
    }
    application->stopMiniMemoryServer();

#ifdef __APPLE__
    // 清理 ScreenCaptureKit 资源
    if (application->logger_) {
      application->logger_->info("Cleaning up ScreenCaptureKit resources...");
    }
    duorou::media::cleanup_macos_screen_capture();
#endif

    // 只清理非GTK资源，GTK资源由GTK自动管理
    if (application->logger_) {
      application->logger_->info("Cleaning up application resources...");
    }
    
    // 注释：API服务器已被禁用
    /*
    if (application->api_server_) {
      application->api_server_->stop();
      application->api_server_.reset();
    }
    */
    application->main_window_.reset();
    application->system_tray_.reset();
    application->workflow_engine_.reset();
    application->model_manager_.reset();
    application->config_manager_.reset();
    
    if (application->logger_) {
      application->logger_->info("退出Duorou应用程序");
      application->logger_.reset();
    }
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

  // 避免使用system调用，直接设置停止标志让线程自然退出
  if (logger_) {
    logger_->info("Signaling MiniMemory thread to stop...");
  }
  
  // 等待一段时间让线程自然结束
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  
  // 如果需要强制终止进程，使用更安全的方式
  if (logger_) {
    logger_->info("Attempting to terminate MiniMemory process...");
  }
  
  // 首先检查进程是否存在
  pid_t check_pid = fork();
  if (check_pid == 0) {
    // 子进程：检查进程是否存在
    execl("/usr/bin/pgrep", "pgrep", "-f", "mini_cache_server", (char*)NULL);
    _exit(1);
  } else if (check_pid > 0) {
    int status;
    waitpid(check_pid, &status, 0);
    
    if (WEXITSTATUS(status) == 0) {
      // 进程存在，尝试发送SIGINT信号
      if (logger_) {
        logger_->info("Found MiniMemory process, sending SIGINT...");
      }
      
      pid_t kill_pid = fork();
      if (kill_pid == 0) {
        execl("/usr/bin/pkill", "pkill", "-INT", "-f", "mini_cache_server", (char*)NULL);
        _exit(1);
      } else if (kill_pid > 0) {
        waitpid(kill_pid, &status, 0);
        
        // 等待2秒让进程优雅退出
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // 再次检查进程是否还存在
        pid_t recheck_pid = fork();
        if (recheck_pid == 0) {
          execl("/usr/bin/pgrep", "pgrep", "-f", "mini_cache_server", (char*)NULL);
          _exit(1);
        } else if (recheck_pid > 0) {
          waitpid(recheck_pid, &status, 0);
          
          if (WEXITSTATUS(status) == 0) {
            // 进程仍然存在，使用SIGKILL强制终止
            if (logger_) {
              logger_->warning("MiniMemory process still running, sending SIGKILL...");
            }
            
            pid_t force_kill_pid = fork();
            if (force_kill_pid == 0) {
              execl("/usr/bin/pkill", "pkill", "-9", "-f", "mini_cache_server", (char*)NULL);
              _exit(1);
            } else if (force_kill_pid > 0) {
              waitpid(force_kill_pid, &status, 0);
              
              // 最后再次确认
              std::this_thread::sleep_for(std::chrono::milliseconds(500));
              pid_t final_check_pid = fork();
              if (final_check_pid == 0) {
                execl("/usr/bin/pgrep", "pgrep", "-f", "mini_cache_server", (char*)NULL);
                _exit(1);
              } else if (final_check_pid > 0) {
                waitpid(final_check_pid, &status, 0);
                if (WEXITSTATUS(status) == 0) {
                  if (logger_) {
                    logger_->error("Failed to terminate MiniMemory process");
                  }
                } else {
                  if (logger_) {
                    logger_->info("MiniMemory process terminated successfully");
                  }
                }
              }
            }
          } else {
            if (logger_) {
              logger_->info("MiniMemory process exited gracefully");
            }
          }
        }
      }
    } else {
      if (logger_) {
        logger_->info("MiniMemory process not found");
      }
    }
  }

  // 安全地处理线程结束
  if (minimemory_thread_) {
    if (logger_) {
      logger_->info("Waiting for MiniMemory thread to finish...");
    }
    
    if (is_destructing_.load()) {
      // 在析构过程中，直接detach避免线程异常
      if (logger_) {
        logger_->info("Detaching MiniMemory thread during destruction");
      }
      if (minimemory_thread_->joinable()) {
        minimemory_thread_->detach();
      }
      // 立即释放线程对象
      minimemory_thread_.reset();
    } else {
      // 正常停止时，尝试join线程，但设置超时
      if (minimemory_thread_->joinable()) {
        std::atomic<bool> join_completed{false};
        std::thread join_thread([this, &join_completed]() {
          try {
            minimemory_thread_->join();
            join_completed.store(true);
          } catch (const std::exception &e) {
            // join失败
            join_completed.store(true);
          }
        });
        
        // 等待最多3秒
        auto start_time = std::chrono::steady_clock::now();
        while (!join_completed.load() && 
               std::chrono::steady_clock::now() - start_time < std::chrono::seconds(3)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (!join_completed.load()) {
          // 超时，强制detach
          if (logger_) {
            logger_->warning("MiniMemory thread join timeout, detaching...");
          }
          if (minimemory_thread_->joinable()) {
            minimemory_thread_->detach();
          }
          join_thread.detach();
        } else {
          join_thread.join();
          if (logger_) {
            logger_->info("MiniMemory thread joined successfully");
          }
        }
      }
      minimemory_thread_.reset();
    }
  }

  if (logger_) {
    logger_->info("MiniMemory server stopped");
  }
}

} // namespace core
} // namespace duorou