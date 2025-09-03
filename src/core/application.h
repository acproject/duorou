#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// 前向声明GTK类型，避免直接包含GTK头文件
typedef struct _GtkApplication GtkApplication;
typedef void *gpointer;

// API相关前向声明 - 在duorou命名空间中
// 注释：暂时禁用API服务器
/*
namespace duorou {
class ApiServer;
}
*/

namespace duorou {
namespace core {

// 前向声明
class ConfigManager;
class Logger;
class ModelManager;
class WorkflowEngine;

} // namespace core

// GUI相关前向声明
namespace gui {
class SystemTray;
class MainWindow;
} // namespace gui

namespace core {

/**
 * @brief 应用程序主类
 *
 * 负责应用程序的初始化、运行和清理工作
 * 管理所有核心组件的生命周期
 */
class Application {
public:
  /**
   * @brief 应用程序运行状态
   */
  enum class Status {
    NotInitialized, ///< 未初始化
    Initializing,   ///< 初始化中
    Running,        ///< 运行中
    Stopping,       ///< 停止中
    Stopped         ///< 已停止
  };

  /**
   * @brief 构造函数
   * @param argc 命令行参数数量
   * @param argv 命令行参数数组
   */
  Application(int argc, char *argv[]);

  /**
   * @brief 设置运行模式
   * @param service_mode 是否以服务模式运行（无GUI）
   */
  void setServiceMode(bool service_mode);

  /**
   * @brief 析构函数
   */
  ~Application();

  // 禁用拷贝构造和赋值
  Application(const Application &) = delete;
  Application &operator=(const Application &) = delete;

  /**
   * @brief 初始化应用程序
   * @return 初始化是否成功
   */
  bool initialize();

  /**
   * @brief 运行应用程序主循环
   * @return 应用程序退出码
   */
  int run();

  /**
   * @brief 停止应用程序
   */
  void stop();

  /**
   * @brief 获取应用程序状态
   * @return 当前状态
   */
  Status getStatus() const { return status_; }

  /**
   * @brief 获取应用程序名称
   * @return 应用程序名称
   */
  const std::string &getName() const { return app_name_; }

  /**
   * @brief 获取应用程序版本
   * @return 版本字符串
   */
  const std::string &getVersion() const { return version_; }

  /**
   * @brief 显示主窗口
   */
  void showMainWindow();

  /**
   * @brief 隐藏主窗口
   */
  void hideMainWindow();

  /**
   * @brief 切换主窗口显示状态
   */
  void toggleMainWindow();

  /**
   * @brief 获取系统托盘实例
   * @return 系统托盘指针
   */
  ::duorou::gui::SystemTray *getSystemTray() const {
    return system_tray_.get();
  }

  /**
   * @brief 获取配置管理器
   * @return 配置管理器指针
   */
  ConfigManager *getConfigManager() const { return config_manager_.get(); }

  /**
   * @brief 获取日志系统
   * @return 日志系统指针
   */
  Logger *getLogger() const { return logger_.get(); }

  /**
   * @brief 获取模型管理器
   * @return 模型管理器指针
   */
  ModelManager *getModelManager() const { return model_manager_.get(); }

  /**
   * @brief 获取工作流引擎
   * @return 工作流引擎指针
   */
  WorkflowEngine *getWorkflowEngine() const { return workflow_engine_.get(); }

  /**
   * @brief 获取GTK应用程序实例
   * @return GTK应用程序指针
   */
  GtkApplication *getGtkApp() const { return gtk_app_; }

  /**
   * @brief 注册退出回调函数
   * @param callback 退出时调用的回调函数
   */
  void registerExitCallback(std::function<void()> callback);

private:
  /**
   * @brief 初始化GTK应用程序
   * @return 初始化是否成功
   */
  bool initializeGtk();

  /**
   * @brief 初始化组件
   * @return 初始化是否成功
   */
  bool initializeComponents();

  /**
   * @brief 初始化系统托盘
   * @return 初始化是否成功
   */
  bool initializeSystemTray();

  /**
   * @brief 启动MiniMemory服务器
   * @return 启动是否成功
   */
  bool startMiniMemoryServer();

  /**
   * @brief 停止MiniMemory服务器
   */
  void stopMiniMemoryServer();

  /**
   * @brief 清理资源
   */
  void cleanup();

  /**
   * @brief GTK应用程序激活回调
   * @param app GTK应用程序实例
   * @param user_data 用户数据
   */
  static void onActivate(GtkApplication *app, gpointer user_data);

  /**
   * @brief GTK应用程序关闭回调
   * @param app GTK应用程序实例
   * @param user_data 用户数据
   */
  static void onShutdown(GtkApplication *app, gpointer user_data);

  /**
   * @brief 信号处理函数
   * @param signal 信号编号
   */
  static void signalHandler(int signal);

private:
  // 应用程序信息
  std::string app_name_;
  std::string version_;
  std::vector<std::string> args_;

  // 运行状态
  Status status_;
  bool service_mode_; ///< 服务模式标志

  // GTK应用程序实例
  GtkApplication *gtk_app_;

  // 核心组件
  std::unique_ptr<ConfigManager> config_manager_;
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<ModelManager> model_manager_;
  std::unique_ptr<WorkflowEngine> workflow_engine_;

  // API服务器
  // 注释：暂时禁用API服务器
  // std::unique_ptr<::duorou::ApiServer> api_server_;

  // GUI组件
  std::unique_ptr<::duorou::gui::SystemTray> system_tray_;
  std::unique_ptr<::duorou::gui::MainWindow> main_window_;

  // 退出回调函数列表
  std::vector<std::function<void()>> exit_callbacks_;

  // MiniMemory服务器管理
  std::unique_ptr<std::thread> minimemory_thread_;
  std::atomic<bool> minimemory_running_;
  std::atomic<bool> is_destructing_; ///< 标记是否正在析构
  std::string minimemory_executable_path_;

  // 静态实例指针（用于信号处理）
  static Application *instance_;
};

} // namespace core
} // namespace duorou