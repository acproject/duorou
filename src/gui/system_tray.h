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
 * 系统托盘图标类型
 */
enum class TrayIconType {
    SYSTEM,     // 系统图标
    CUSTOM,     // 自定义图标
    TEXT        // 文本图标
};

/**
 * 托盘菜单项结构
 */
struct TrayMenuItem {
        std::string id;                           // 菜单项ID
        std::string label;                        // 显示文本
        std::string icon;                         // 图标路径或名称
        bool enabled = true;                      // 是否启用
        bool separator = false;                   // 是否为分隔符
        bool visible = true;                      // 是否可见
        bool checked = false;                     // 是否选中
        std::string badge;                        // 徽章文本
        std::string tooltip;                      // 提示文本
        std::string shortcut;                     // 快捷键 (如 "Ctrl+N", "Cmd+Q")
        int priority = 0;                         // 菜单项优先级 (用于排序)
        std::function<void()> callback;           // 点击回调
         std::function<void(bool)> toggle_callback; // 切换回调 (用于复选框菜单项)
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
     * 查找菜单项
     * @param item_id 菜单项ID
     * @return 菜单项指针，未找到返回nullptr
     */
    TrayMenuItem* findMenuItem(const std::string& item_id);

    /**
     * 更新菜单项标签
     * @param item_id 菜单项ID
     * @param label 新标签
     * @return 是否更新成功
     */
    bool updateMenuItemLabel(const std::string& item_id, const std::string& label);

    /**
     * 更新菜单项图标
     * @param item_id 菜单项ID
     * @param icon_name 图标名称
     * @return 是否更新成功
     */
    bool updateMenuItemIcon(const std::string& item_id, const std::string& icon_name);

    /**
     * 更新菜单项回调
     * @param item_id 菜单项ID
     * @param callback 新回调函数
     * @return 是否更新成功
     */
    bool updateMenuItemCallback(const std::string& item_id, std::function<void()> callback);

    /**
     * 批量添加菜单项
     * @param items 菜单项列表
     */
    void addMenuItems(const std::vector<TrayMenuItem>& items);

    /**
     * 批量移除菜单项
     * @param item_ids 菜单项ID列表
     */
    void removeMenuItems(const std::vector<std::string>& item_ids);

    /**
     * 清空所有菜单项
     */
    void clearMenu();

    /**
     * 获取所有菜单项
     * @return 菜单项列表的常量引用
     */
    const std::vector<TrayMenuItem>& getMenuItems() const;

    /**
     * 检查菜单项是否存在
     * @param item_id 菜单项ID
     * @return 是否存在
     */
    bool hasMenuItem(const std::string& item_id) const;

    /**
     * 添加子菜单项
     * @param parent_id 父菜单项ID
     * @param item 子菜单项
     * @return 是否添加成功
     */
    bool addSubMenuItem(const std::string& parent_id, const TrayMenuItem& item);

    /**
     * 移除子菜单项
     * @param parent_id 父菜单项ID
     * @param item_id 子菜单项ID
     * @return 是否移除成功
     */
    bool removeSubMenuItem(const std::string& parent_id, const std::string& item_id);

    /**
     * 查找子菜单项
     * @param parent_id 父菜单项ID
     * @param item_id 子菜单项ID
     * @return 子菜单项指针，未找到返回nullptr
     */
    TrayMenuItem* findSubMenuItem(const std::string& parent_id, const std::string& item_id);

    /**
     * 设置子菜单
     * @param parent_id 父菜单项ID
     * @param submenu_items 子菜单项列表
     * @return 是否设置成功
     */
    bool setSubMenu(const std::string& parent_id, const std::vector<TrayMenuItem>& submenu_items);

    /**
     * 清空子菜单
     * @param parent_id 父菜单项ID
     * @return 是否清空成功
     */
    bool clearSubMenu(const std::string& parent_id);

    /**
     * 设置菜单项可见性
     * @param item_id 菜单项ID
     * @param visible 是否可见
     */
    void setMenuItemVisible(const std::string& item_id, bool visible);

    /**
     * 检查菜单项是否可见
     * @param item_id 菜单项ID
     * @return 是否可见
     */
    bool isMenuItemVisible(const std::string& item_id) const;

    /**
     * 设置菜单项选中状态
     * @param item_id 菜单项ID
     * @param checked 是否选中
     */
    void setMenuItemChecked(const std::string& item_id, bool checked);

    /**
     * 检查菜单项是否选中
     * @param item_id 菜单项ID
     * @return 是否选中
     */
    bool isMenuItemChecked(const std::string& item_id) const;

    /**
     * 设置菜单项徽章
     * @param item_id 菜单项ID
     * @param badge 徽章文本
     */
    void setMenuItemBadge(const std::string& item_id, const std::string& badge);

    /**
     * 获取菜单项徽章
     * @param item_id 菜单项ID
     * @return 徽章文本
     */
    std::string getMenuItemBadge(const std::string& item_id) const;

    /**
     * 设置菜单项提示文本
     * @param item_id 菜单项ID
     * @param tooltip 提示文本
     */
    void setMenuItemTooltip(const std::string& item_id, const std::string& tooltip);

    /**
     * 获取菜单项提示文本
     * @param item_id 菜单项ID
     * @return 提示文本
     */
    std::string getMenuItemTooltip(const std::string& item_id) const;

     /**
      * 设置菜单项快捷键
      * @param item_id 菜单项ID
      * @param shortcut 快捷键字符串 (如 "Ctrl+N", "Cmd+Q")
      */
     void setMenuItemShortcut(const std::string& item_id, const std::string& shortcut);

     /**
      * 获取菜单项快捷键
      * @param item_id 菜单项ID
      * @return 快捷键字符串
      */
     std::string getMenuItemShortcut(const std::string& item_id) const;

     /**
      * 设置菜单项优先级
      * @param item_id 菜单项ID
      * @param priority 优先级值
      */
     void setMenuItemPriority(const std::string& item_id, int priority);

     /**
      * 获取菜单项优先级
      * @param item_id 菜单项ID
      * @return 优先级值
      */
     int getMenuItemPriority(const std::string& item_id) const;

     /**
      * 设置菜单项切换回调
      * @param item_id 菜单项ID
      * @param callback 切换回调函数
      */
     void setMenuItemToggleCallback(const std::string& item_id, std::function<void(bool)> callback);

     /**
      * 按优先级排序菜单项
      */
     void sortMenuItemsByPriority();

     /**
      * 批量更新菜单项 (优化性能)
      * @param updates 更新操作的函数
      */
     void batchUpdateMenuItems(std::function<void()> updates);

     /**
      * 强制重建菜单
      */
     void forceRebuildMenu();

     /**
      * 检查菜单是否需要重建
      * @return 是否需要重建
      */
     bool needsMenuRebuild() const;

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
     * @param callback 状态变化时的回调函数
     */
    void setStatusChangeCallback(std::function<void(TrayStatus)> callback);
    
    /**
     * 设置退出回调
     * @param callback 退出时的回调函数
     */
    void setQuitCallback(std::function<void()> callback);
    
    /**
     * 根据窗口状态更新菜单
     * @param window_visible 窗口是否可见
     */
    void updateWindowStateMenu(bool window_visible);

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