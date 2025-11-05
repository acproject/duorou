#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Forward declarations for GTK types to avoid including GTK headers directly
typedef struct _GtkApplication GtkApplication;
typedef void *gpointer;

// API-related forward declarations - in duorou namespace
// Note: API server temporarily disabled
/*
namespace duorou {
class ApiServer;
}
*/

namespace duorou {
namespace core {

// Forward declarations
class ConfigManager;
class Logger;
class ModelManager;
class WorkflowEngine;

} // namespace core

// GUI-related forward declarations
namespace gui {
class SystemTray;
class MainWindow;
} // namespace gui

namespace core {

/**
 * @brief Main application class
 *
 * Responsible for application initialization, running, and cleanup
 * Manages the lifecycle of all core components
 */
class Application {
public:
  /**
   * @brief Application running status
   */
  enum class Status {
    NotInitialized, ///< Not initialized
    Initializing,   ///< Initializing
    Running,        ///< Running
    Stopping,       ///< Stopping
    Stopped         ///< Stopped
  };

  /**
   * @brief Constructor
   * @param argc Number of command line arguments
   * @param argv Command line argument array
   */
  Application(int argc, char *argv[]);

  /**
   * @brief Set running mode
   * @param service_mode Whether to run in service mode (no GUI)
   */
  void setServiceMode(bool service_mode);

  /**
   * @brief Destructor
   */
  ~Application();

  // Disable copy constructor and assignment
  Application(const Application &) = delete;
  Application &operator=(const Application &) = delete;

  /**
   * @brief Initialize application
   * @return Whether initialization was successful
   */
  bool initialize();

  /**
   * @brief Run application main loop
   * @return Application exit code
   */
  int run();

  /**
   * @brief Stop application
   */
  void stop();

  /**
   * @brief Get application status
   * @return Current status
   */
  Status getStatus() const { return status_; }

  /**
   * @brief Get application name
   * @return Application name
   */
  const std::string &getName() const { return app_name_; }

  /**
   * @brief Get application version
   * @return Version string
   */
  const std::string &getVersion() const { return version_; }

  /**
   * @brief Show main window
   */
  void showMainWindow();

  /**
   * @brief Hide main window
   */
  void hideMainWindow();

  /**
   * @brief Toggle main window display status
   */
  void toggleMainWindow();

  /**
   * @brief Get system tray instance
   * @return System tray pointer
   */
  ::duorou::gui::SystemTray *getSystemTray() const {
    return system_tray_.get();
  }

  /**
   * @brief Get configuration manager
   * @return Configuration manager pointer
   */
  ConfigManager *getConfigManager() const { return config_manager_.get(); }

  /**
   * @brief Get logging system
   * @return Logging system pointer
   */
  Logger *getLogger() const { return logger_.get(); }

  /**
   * @brief Get model manager
   * @return Model manager pointer
   */
  ModelManager *getModelManager() const { return model_manager_.get(); }

  /**
   * @brief Get workflow engine
   * @return Workflow engine pointer
   */
  WorkflowEngine *getWorkflowEngine() const { return workflow_engine_.get(); }

  /**
   * @brief Get GTK application instance
   * @return GTK application pointer
   */
  GtkApplication *getGtkApp() const { return gtk_app_; }

  /**
   * @brief Register exit callback function
   * @param callback Callback function to call on exit
   */
  void registerExitCallback(std::function<void()> callback);

private:
  /**
   * @brief Initialize GTK application
   * @return Whether initialization was successful
   */
  bool initializeGtk();

  /**
   * @brief Initialize components
   * @return Whether initialization was successful
   */
  bool initializeComponents();

  /**
   * @brief Initialize system tray
   * @return Whether initialization was successful
   */
  bool initializeSystemTray();

  /**
   * @brief Start MiniMemory server
   * @return Whether startup was successful
   */
  bool startMiniMemoryServer();

  /**
   * @brief Stop MiniMemory server
   */
  void stopMiniMemoryServer();

  /**
   * @brief Clean up resources
   */
  void cleanup();

  /**
   * @brief GTK application activation callback
   * @param app GTK application instance
   * @param user_data User data
   */
  static void onActivate(GtkApplication *app, gpointer user_data);

  /**
   * @brief GTK application shutdown callback
   * @param app GTK application instance
   * @param user_data User data
   */
  static void onShutdown(GtkApplication *app, gpointer user_data);

  /**
   * @brief Signal handler function
   * @param signal Signal number
   */
  static void signalHandler(int signal);

private:
  // Application information
  std::string app_name_;
  std::string version_;
  std::vector<std::string> args_;

  // Runtime status
  Status status_;
  bool service_mode_; ///< Service mode flag

  // GTK application instance
  GtkApplication *gtk_app_;

  // Core components
  std::unique_ptr<ConfigManager> config_manager_;
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<ModelManager> model_manager_;
  std::unique_ptr<WorkflowEngine> workflow_engine_;

  // API server
  // Note: API server temporarily disabled
  // std::unique_ptr<::duorou::ApiServer> api_server_;

  // GUI components
  std::unique_ptr<::duorou::gui::SystemTray> system_tray_;
  std::unique_ptr<::duorou::gui::MainWindow> main_window_;

  // Exit callback function list
  std::vector<std::function<void()>> exit_callbacks_;

  // MiniMemory server management
  std::unique_ptr<std::thread> minimemory_thread_;
  std::atomic<bool> minimemory_running_;
  std::atomic<bool> is_destructing_; ///< Flag indicating if destructing
  std::string minimemory_executable_path_;

  // Static instance pointer (for signal handling)
  static Application *instance_;
};

} // namespace core
} // namespace duorou