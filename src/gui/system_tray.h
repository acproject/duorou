#pragma once

#include <memory>
#include <functional>
#include <string>
#include <vector>

#ifdef HAVE_GTK
#include <gtk/gtk.h>
#else
// GTK类型的占位符定义
typedef void* GtkWidget;
typedef void* GtkMenu;
typedef void* GtkPopoverMenu;
typedef void* GdkPixbuf;
typedef void* gpointer;
typedef unsigned int guint;
typedef int gboolean;
#define TRUE 1
#define FALSE 0
#define G_CALLBACK(f) ((void(*)(void))(f))
#define g_signal_connect(instance, detailed_signal, c_handler, data) \
    ((void)0)
#endif

namespace duorou {
namespace gui {

/**
 * 系统托盘状态枚举
 */
enum class TrayStatus {
    Idle,           // 空闲状态
    Working,        // 工作中
    Error,          // 错误状态
    Generating,     // 生成中
    Loading         // 加载中
};

/**
 * 托盘菜单项结构
 */
struct TrayMenuItem {
    std::string id;                           // 菜单项ID
    std::string label;                        // 显示文本
    std::string icon_name;                    // 图标名称（可选）
    bool enabled = true;                      // 是否启用
    bool separator = false;                   // 是否为分隔符
    std::function<void()> callback;           // 点击回调
    std::vector<TrayMenuItem> submenu;        // 子菜单
};

/**
 * 系统托盘类
 * 提供系统托盘图标、菜单和状态指示功能
 */
class SystemTray {
public:
    SystemTray();
    ~SystemTray();

    // 禁用拷贝构造和赋值
    SystemTray(const SystemTray&) = delete;
    SystemTray& operator=(const SystemTray&) = delete;

    /**
     * 初始化系统托盘
     * @param app_name 应用程序名称
     * @param icon_path 托盘图标路径
     * @return 是否初始化成功
     */
    bool initialize(const std::string& app_name, const std::string& icon_path = "");

    /**
     * 显示系统托盘图标
     */
    void show();

    /**
     * 隐藏系统托盘图标
     */
    void hide();

    /**
     * 检查托盘是否可见
     * @return 是否可见
     */
    bool isVisible() const;

    /**
     * 设置托盘图标
     * @param icon_path 图标文件路径
     * @return 是否设置成功
     */
    bool setIcon(const std::string& icon_path);

    /**
     * 设置托盘图标（从资源）
     * @param icon_name 图标名称
     * @return 是否设置成功
     */
    bool setIconFromTheme(const std::string& icon_name);

    /**
     * 设置托盘提示文本
     * @param tooltip 提示文本
     */
    void setTooltip(const std::string& tooltip);

    /**
     * 设置托盘状态
     * @param status 状态
     */
    void setStatus(TrayStatus status);

    /**
     * 获取当前状态
     * @return 当前状态
     */
    TrayStatus getStatus() const;

    /**
     * 设置托盘菜单
     * @param menu_items 菜单项列表
     */
    void setMenu(const std::vector<TrayMenuItem>& menu_items);

    /**
     * 添加菜单项
     * @param item 菜单项
     */
    void addMenuItem(const TrayMenuItem& item);

    /**
     * 移除菜单项
     * @param item_id 菜单项ID
     */
    void removeMenuItem(const std::string& item_id);

    /**
     * 启用/禁用菜单项
     * @param item_id 菜单项ID
     * @param enabled 是否启用
     */
    void setMenuItemEnabled(const std::string& item_id, bool enabled);

    /**
     * 显示通知消息
     * @param title 标题
     * @param message 消息内容
     * @param icon_name 图标名称（可选）
     * @param timeout_ms 超时时间（毫秒，0表示不自动关闭）
     */
    void showNotification(const std::string& title, 
                         const std::string& message,
                         const std::string& icon_name = "",
                         int timeout_ms = 5000);

    /**
     * 设置左键点击回调
     * @param callback 回调函数
     */
    void setLeftClickCallback(std::function<void()> callback);

    /**
     * 设置右键点击回调
     * @param callback 回调函数
     */
    void setRightClickCallback(std::function<void()> callback);

    /**
     * 设置双击回调
     * @param callback 回调函数
     */
    void setDoubleClickCallback(std::function<void()> callback);

    /**
     * 设置状态变化回调
     * @param callback 回调函数
     */
    void setStatusChangeCallback(std::function<void(TrayStatus)> callback);

    /**
     * 更新进度显示
     * @param progress 进度值（0.0-1.0）
     * @param text 进度文本
     */
    void updateProgress(double progress, const std::string& text = "");

    /**
     * 清除进度显示
     */
    void clearProgress();

    /**
     * 检查系统是否支持托盘
     * @return 是否支持
     */
    static bool isSystemTraySupported();

    /**
     * 获取默认图标路径
     * @param status 状态
     * @return 图标路径
     */
    static std::string getDefaultIconPath(TrayStatus status);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;

    // GTK回调函数
    static void onActivate(GtkWidget* widget, gpointer user_data);
    static void onPopupMenu(GtkWidget* widget, gpointer user_data);
    static void onMenuItemActivated(GtkWidget* menu_item, gpointer user_data);
    static gboolean onButtonPress(GtkWidget* widget, gpointer user_data);
};

/**
 * 系统托盘管理器
 * 单例模式，管理全局系统托盘实例
 */
class SystemTrayManager {
public:
    /**
     * 获取单例实例
     * @return 系统托盘管理器实例
     */
    static SystemTrayManager& getInstance();

    /**
     * 初始化系统托盘
     * @param app_name 应用程序名称
     * @param icon_path 图标路径
     * @return 是否初始化成功
     */
    bool initialize(const std::string& app_name, const std::string& icon_path = "");

    /**
     * 获取系统托盘实例
     * @return 系统托盘指针
     */
    SystemTray* getTray();

    /**
     * 关闭系统托盘
     */
    void shutdown();

private:
    SystemTrayManager() = default;
    ~SystemTrayManager() = default;
    SystemTrayManager(const SystemTrayManager&) = delete;
    SystemTrayManager& operator=(const SystemTrayManager&) = delete;

    std::unique_ptr<SystemTray> tray_;
};

} // namespace gui
} // namespace duorou