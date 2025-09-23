#include "application.h"
// #include "../api/api_server.h"  // Comment: Temporarily disable API server
#include "../gui/main_window.h"
#include "../gui/system_tray.h"
#include "config_manager.h"
#include "logger.h"
#include "model_manager.h"
#include "workflow_engine.h"
#include "../extensions/ollama/ollama_model_manager.h"
#ifdef __APPLE__
#include "../media/macos_screen_capture.h"
#endif

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <thread>

#ifdef _WIN32
#include <process.h> // for _getpid on Windows
#define getpid _getpid
#else
#include <unistd.h> // for getpid on Unix/Linux
#include <sys/wait.h> // for waitpid on Unix/Linux
#endif

// Version definition
#ifndef DUOROU_VERSION
#define DUOROU_VERSION "1.0.0"
#endif

// Conditional compilation for GTK support
#ifdef HAVE_GTK
#include <gtk/gtk.h>
#else
// Placeholder definitions for GTK types
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

// Static member initialization
Application *Application::instance_ = nullptr;

Application::Application(int argc, char *argv[])
    : app_name_("Duorou"), version_(DUOROU_VERSION),
      status_(Status::NotInitialized), service_mode_(false), gtk_app_(nullptr),
      minimemory_running_(false), is_destructing_(false) {

  // Save command line arguments
  for (int i = 0; i < argc; ++i) {
    args_.emplace_back(argv[i]);
  }

  // Check for service mode parameters
  for (const auto &arg : args_) {
    if (arg == "--service" || arg == "-s" || arg == "--mode=server") {
      service_mode_ = true;
      break;
    }
  }

  // Set MiniMemory executable path (relative to duorou's build directory)
  minimemory_executable_path_ =
      "../third_party/MiniMemory/build/bin/mini_cache_server";

  // Set static instance pointer
  instance_ = this;

  // Register signal handlers
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);
}

Application::~Application() {
  // Set destruction flag
  is_destructing_.store(true);
  
  // Ensure MiniMemory server is stopped
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

  // Initialize GTK in non-service mode
  if (!service_mode_) {
    if (!initializeGtk()) {
      std::cerr << "Failed to initialize GTK" << std::endl;
      status_ = Status::NotInitialized;
      return false;
    }
  } else {
    std::cout << "Running in service mode (no GUI)" << std::endl;
  }

  // Initialize core components
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
    // Service mode: run service loop
    std::cout << "Service mode started. Press Ctrl+C to stop." << std::endl;

    // Log startup information
    if (logger_) {
      logger_->info("Application started in service mode");
      logger_->info("Version: " + version_);
      logger_->info(
          "Service mode started (API server disabled for development)");
    }

    // Simple service loop
    while (status_ == Status::Running) {
      // Periodic tasks can be added here
      std::this_thread::sleep_for(std::chrono::seconds(1));

      // Periodically log status
      static int counter = 0;
      if (++counter % 60 == 0 && logger_) { // Log every minute
        logger_->debug("Service running, uptime: " + std::to_string(counter) +
                       " seconds");
      }
    }

    if (logger_) {
      logger_->info("Service mode stopped");
    }

    return 0;
  }

  // Run GTK main loop
  // Convert std::vector<std::string> to char**
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

    // Call exit callback functions
    for (auto &callback : exit_callbacks_) {
      try {
        callback();
      } catch (const std::exception &e) {
        std::cerr << "Exception in exit callback: " << e.what() << std::endl;
      }
    }

    // Stop GTK application
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
  // Create GTK application
  gtk_app_ = gtk_application_new("com.duorou.app", G_APPLICATION_FLAGS_NONE);
  if (!gtk_app_) {
    return false;
  }

  // Connect signals
  g_signal_connect(gtk_app_, "activate", G_CALLBACK(onActivate), this);
  g_signal_connect(gtk_app_, "shutdown", G_CALLBACK(onShutdown), this);

  return true;
}

bool Application::initializeComponents() {
  try {
    std::cout << "Initializing core components..." << std::endl;

    // Initialize logging system
    std::cout << "Creating logger..." << std::endl;
    logger_ = std::make_unique<Logger>();
    std::cout << "Initializing logger..." << std::endl;
    if (!logger_->initialize()) {
      std::cerr << "Failed to initialize logger" << std::endl;
      return false;
    }
    std::cout << "Logger initialized successfully" << std::endl;

    // Initialize configuration manager
    std::cout << "Creating config manager..." << std::endl;
    config_manager_ = std::make_unique<ConfigManager>();
    std::cout << "Initializing config manager..." << std::endl;
    if (!config_manager_->initialize()) {
      logger_->error("Failed to initialize config manager");
      return false;
    }
    std::cout << "Config manager initialized successfully" << std::endl;

    // Set Logger file output based on configuration
    bool file_output = config_manager_->getBool("log.file_output", true);
    if (file_output) {
      std::string log_path = logger_->getDefaultLogPath();
      if (logger_->setLogFile(log_path)) {
        logger_->info("Log file output enabled: " + log_path);
      } else {
        logger_->warning("Failed to enable log file output");
      }
    }

    // Initialize model manager
    std::cout << "Creating model manager..." << std::endl;
    model_manager_ = std::make_unique<ModelManager>();
    std::cout << "Initializing model manager..." << std::endl;
    if (!model_manager_->initialize()) {
      logger_->error("Failed to initialize model manager");
      return false;
    }
    std::cout << "Model manager initialized successfully" << std::endl;

    // Initialize global model manager
    std::cout << "Initializing global model manager..." << std::endl;
    try {
      extensions::ollama::GlobalModelManager::initialize();
      std::cout << "Global model manager initialized successfully" << std::endl;
    } catch (const std::exception& e) {
      logger_->error("Failed to initialize global model manager: " + std::string(e.what()));
      return false;
    }

    // Initialize workflow engine
    std::cout << "Creating workflow engine..." << std::endl;
    workflow_engine_ = std::make_unique<WorkflowEngine>();
    std::cout << "Initializing workflow engine..." << std::endl;
    if (!workflow_engine_->initialize()) {
      logger_->error("Failed to initialize workflow engine");
      return false;
    }
    std::cout << "Workflow engine initialized successfully" << std::endl;

    // Start API server (regardless of service mode)
    // Comment: Temporarily disable API server to resolve debugging issues
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

    // Initialize system tray (only in non-service mode)
    // Temporarily disable system tray to debug segfault issues in other components
    if (!service_mode_) {
      logger_->info(
          "System tray initialization temporarily disabled for debugging");
      // if (!initializeSystemTray()) {
      //     logger_->warning("Failed to initialize system tray, continuing
      //     without it");
      // }
    }

    // Start MiniMemory server
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

  // Stop MiniMemory server
  stopMiniMemoryServer();

  // Stop API server
  // Comment: API server has been disabled
  /*
  if (api_server_) {
    api_server_->stop();
    api_server_.reset();
  }
  */

#ifdef __APPLE__
  // Clean up ScreenCaptureKit resources
  duorou::media::cleanup_macos_screen_capture();
#endif

  // Clean up GUI components
  main_window_.reset();
  system_tray_.reset();

  // Clean up core components (in reverse order)
  workflow_engine_.reset();
  
  // Shutdown global model manager
  try {
    extensions::ollama::GlobalModelManager::shutdown();
  } catch (const std::exception& e) {
    // Continue cleanup even if errors occur during shutdown
    std::cerr << "Warning: Failed to shutdown global model manager: " << e.what() << std::endl;
  }
  
  model_manager_.reset();
  config_manager_.reset();
  logger_.reset();

  // Clean up GTK resources
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

  // Create and show main window
  if (application && !application->main_window_) {
    application->main_window_ =
        std::make_unique<::duorou::gui::MainWindow>(application);
    if (application->main_window_->initialize()) {
      application->main_window_->show();
      if (application->logger_) {
        application->logger_->info("Main window created and shown");
      }
      // Keep application active
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

    // Release application hold state
    g_application_release(G_APPLICATION(app));

    // Stop MiniMemory server
    if (application->logger_) {
      application->logger_->info("Exiting MiniMemory service...");
    }
    application->stopMiniMemoryServer();

#ifdef __APPLE__
    // Clean up ScreenCaptureKit resources
    if (application->logger_) {
      application->logger_->info("Cleaning up ScreenCaptureKit resources...");
    }
    duorou::media::cleanup_macos_screen_capture();
#endif

    // Only clean up non-GTK resources, GTK resources are managed automatically by GTK
    if (application->logger_) {
      application->logger_->info("Cleaning up application resources...");
    }
    
    // Comment: API server has been disabled
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
      application->logger_->info("Exiting Duorou application");
      application->logger_.reset();
    }
    application->status_ = Status::Stopped;
  }
}

bool Application::initializeSystemTray() {
  try {
    system_tray_ = std::make_unique<::duorou::gui::SystemTray>();

    // Initialize system tray
    if (!system_tray_->initialize("Duorou AI Assistant")) {
      logger_->error("Failed to initialize system tray");
      return false;
    }

    // Set system tray properties
    system_tray_->setTooltip("Duorou AI Assistant");
    system_tray_->setStatus(::duorou::gui::TrayStatus::Idle);

    // Set callback functions
    system_tray_->setLeftClickCallback([this]() { showMainWindow(); });

    // Add menu items
    std::vector<::duorou::gui::TrayMenuItem> menu_items;

    ::duorou::gui::TrayMenuItem show_item;
    show_item.id = "show";
    show_item.label = "show main windows";
    show_item.callback = [this]() { showMainWindow(); };
    menu_items.push_back(show_item);

    ::duorou::gui::TrayMenuItem separator;
    separator.separator = true;
    menu_items.push_back(separator);

    ::duorou::gui::TrayMenuItem exit_item;
    exit_item.id = "exit";
    exit_item.label = "exit";
    exit_item.callback = [this]() { stop(); };
    menu_items.push_back(exit_item);

    system_tray_->setMenu(menu_items);

    // Show system tray
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
    // Need to check current window state, implement simple show logic for now
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

    // Create thread to run MiniMemory server
    minimemory_thread_ = std::make_unique<std::thread>([this]() {
      // Switch to MiniMemory build directory (relative to duorou's build directory)
      std::string command =
          "cd ../third_party/MiniMemory/build && ./bin/mini_cache_server";

      if (logger_) {
        logger_->info("Starting MiniMemory server with command: " + command);
      }

      // Execute command using system()
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

    // Wait a short time to ensure server startup
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

  // Set stop flag
  minimemory_running_.store(false);

  // Avoid using system calls, directly set stop flag to let thread exit naturally
  if (logger_) {
    logger_->info("Signaling MiniMemory thread to stop...");
  }
  
  // Wait for a while to let thread end naturally
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  
  // If need to forcefully terminate process, use safer method
  if (logger_) {
    logger_->info("Attempting to terminate MiniMemory process...");
  }
  
#ifdef _WIN32
  // Windows platform: Use tasklist and taskkill commands
  if (logger_) {
    logger_->info("Checking for MiniMemory process on Windows...");
  }
  
  // Check if process exists using tasklist
  int check_result = std::system("tasklist /FI \"IMAGENAME eq mini_cache_server.exe\" 2>nul | find /I \"mini_cache_server.exe\" >nul");
  
  if (check_result == 0) {
    // Process exists, try to terminate gracefully
    if (logger_) {
      logger_->info("Found MiniMemory process, attempting graceful termination...");
    }
    
    // Try graceful termination first
    std::system("taskkill /IM mini_cache_server.exe /T >nul 2>&1");
    
    // Wait 2 seconds for graceful process exit
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Check again if process still exists
    int recheck_result = std::system("tasklist /FI \"IMAGENAME eq mini_cache_server.exe\" 2>nul | find /I \"mini_cache_server.exe\" >nul");
    
    if (recheck_result == 0) {
      // Process still exists, force terminate
      if (logger_) {
        logger_->warning("MiniMemory process still running, forcing termination...");
      }
      
      std::system("taskkill /F /IM mini_cache_server.exe /T >nul 2>&1");
      
      // Final confirmation
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      int final_check = std::system("tasklist /FI \"IMAGENAME eq mini_cache_server.exe\" 2>nul | find /I \"mini_cache_server.exe\" >nul");
      
      if (final_check == 0) {
        if (logger_) {
          logger_->error("Failed to terminate MiniMemory process");
        }
      } else {
        if (logger_) {
          logger_->info("MiniMemory process terminated successfully");
        }
      }
    } else {
      if (logger_) {
        logger_->info("MiniMemory process terminated gracefully");
      }
    }
  } else {
    if (logger_) {
      logger_->info("No MiniMemory process found");
    }
  }
#else
  // Unix/Linux platform: Use original fork/exec approach
  // First check if process exists
  pid_t check_pid = fork();
  if (check_pid == 0) {
    // Child process: check if process exists
    execl("/usr/bin/pgrep", "pgrep", "-f", "mini_cache_server", (char*)NULL);
    _exit(1);
  } else if (check_pid > 0) {
    int status;
    waitpid(check_pid, &status, 0);
    
    if (WEXITSTATUS(status) == 0) {
      // Process exists, try sending SIGINT signal
      if (logger_) {
        logger_->info("Found MiniMemory process, sending SIGINT...");
      }
      
      pid_t kill_pid = fork();
      if (kill_pid == 0) {
        execl("/usr/bin/pkill", "pkill", "-INT", "-f", "mini_cache_server", (char*)NULL);
        _exit(1);
      } else if (kill_pid > 0) {
        waitpid(kill_pid, &status, 0);
        
        // Wait 2 seconds for graceful process exit
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Check again if process still exists
        pid_t recheck_pid = fork();
        if (recheck_pid == 0) {
          execl("/usr/bin/pgrep", "pgrep", "-f", "mini_cache_server", (char*)NULL);
          _exit(1);
        } else if (recheck_pid > 0) {
          waitpid(recheck_pid, &status, 0);
          
          if (WEXITSTATUS(status) == 0) {
            // Process still exists, use SIGKILL to force terminate
            if (logger_) {
              logger_->warning("MiniMemory process still running, sending SIGKILL...");
            }
            
            pid_t force_kill_pid = fork();
            if (force_kill_pid == 0) {
              execl("/usr/bin/pkill", "pkill", "-9", "-f", "mini_cache_server", (char*)NULL);
              _exit(1);
            } else if (force_kill_pid > 0) {
              waitpid(force_kill_pid, &status, 0);
              
              // Final confirmation
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
#endif

  // Safely handle thread termination
  if (minimemory_thread_) {
    if (logger_) {
      logger_->info("Waiting for MiniMemory thread to finish...");
    }
    
    if (is_destructing_.load()) {
      // During destruction, directly detach to avoid thread exceptions
      if (logger_) {
        logger_->info("Detaching MiniMemory thread during destruction");
      }
      if (minimemory_thread_->joinable()) {
        minimemory_thread_->detach();
      }
      // Immediately release thread object
      minimemory_thread_.reset();
    } else {
      // During normal stop, try to join thread but set timeout
      if (minimemory_thread_->joinable()) {
        std::atomic<bool> join_completed{false};
        std::thread join_thread([this, &join_completed]() {
          try {
            minimemory_thread_->join();
            join_completed.store(true);
          } catch (const std::exception &e) {
            // join failed
            join_completed.store(true);
          }
        });
        
        // Wait for at most 3 seconds
        auto start_time = std::chrono::steady_clock::now();
        while (!join_completed.load() && 
               std::chrono::steady_clock::now() - start_time < std::chrono::seconds(3)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (!join_completed.load()) {
          // Timeout, force detach
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