#include "system_tray.h"
#include "../core/logger.h"

#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <chrono>

#ifdef HAVE_GTK
#include <gtk/gtk.h>
#include <gdk/gdk.h>
#else
// 占位符定义，当没有GTK时使用
typedef void* gpointer;
typedef unsigned int guint;
typedef int gboolean;
#define TRUE 1
#define FALSE 0
#define GTK_MENU_SHELL(obj) ((void*)(obj))
#define GTK_MENU(obj) ((void*)(obj))
#define G_OBJECT(obj) ((void*)(obj))
#define G_CALLBACK(f) ((void(*)(void))(f))

// GTK函数占位符
#define gtk_application_new(app_id, flags) nullptr
#define gtk_widget_new(type, first_property_name) nullptr
#define gtk_widget_set_tooltip_text(widget, text) do {} while(0)
#define gtk_widget_set_visible(widget, visible) do {} while(0)
#define gtk_widget_get_visible(widget) false
#define gtk_popover_menu_new_from_model(model) nullptr
#define gtk_menu_button_new() nullptr
#define gtk_menu_button_set_popover(button, popover) do {} while(0)
#define gtk_widget_show(widget) do {} while(0)
#define gtk_widget_set_sensitive(widget, sensitive) do {} while(0)
#define g_signal_connect(obj, signal, callback, data) 0
#define g_object_unref(obj) do {} while(0)
#define g_object_set_data(obj, key, data) do {} while(0)
#define g_object_get_data(obj, key) nullptr
// 移除重复的GdkEventButton定义，使用头文件中的定义
#endif

namespace duorou {
namespace gui {

/**
 * SystemTray私有实现类
 */
class SystemTray::Impl {
public:
    Impl() : tray_widget_(nullptr), menu_(nullptr), status_(TrayStatus::Idle),
             visible_(false), progress_(0.0) {}
    
    ~Impl() {
        cleanup();
    }

    bool initialize(const std::string& app_name, const std::string& icon_path) {
        app_name_ = app_name;
        
        // 检查系统托盘支持
        if (!SystemTray::isSystemTraySupported()) {
            std::cerr << "System tray is not supported on this system" << std::endl;
            return false;
        }

        // GTK4: 创建托盘widget (简化实现)
        tray_widget_ = gtk_menu_button_new();
        if (!tray_widget_) {
            std::cerr << "Failed to create system tray widget" << std::endl;
            return false;
        }

        // 设置初始提示文本
        setTooltip(app_name_);

        // 连接信号
        g_signal_connect(tray_widget_, "clicked", G_CALLBACK(onActivateStatic), this);

        // 创建默认菜单
        createDefaultMenu();

        return true;
    }

    void show() {
        if (tray_widget_) {
            gtk_widget_set_visible(tray_widget_, TRUE);
            visible_ = true;
        }
    }

    void hide() {
        if (tray_widget_) {
            gtk_widget_set_visible(tray_widget_, FALSE);
            visible_ = false;
        }
    }

    bool isVisible() const {
        if (tray_widget_) {
            return gtk_widget_get_visible(tray_widget_);
        }
        return false;
    }

    bool setIcon(const std::string& icon_path) {
        if (!tray_widget_) return false;
        
        // GTK4: 简化图标设置
        return true;
    }

    bool setIconFromTheme(const std::string& icon_name) {
        if (!tray_widget_) return false;
        
        // GTK4: 简化图标设置
        return true;
    }

    void setTooltip(const std::string& tooltip) {
        if (tray_widget_) {
            std::string full_tooltip = tooltip;
            if (progress_ > 0.0) {
                full_tooltip += " (" + std::to_string(static_cast<int>(progress_ * 100)) + "%)";
            }
            gtk_widget_set_tooltip_text(tray_widget_, full_tooltip.c_str());
        }
    }

    void setStatus(TrayStatus status) {
        if (status_ == status) return;
        
        TrayStatus old_status = status_;
        status_ = status;
        
        // 更新图标
        updateStatusIcon();
        
        // 触发状态变化回调
        if (status_change_callback_) {
            status_change_callback_(status);
        }
    }

    TrayStatus getStatus() const {
        return status_;
    }

    void setMenu(const std::vector<TrayMenuItem>& menu_items) {
        menu_items_ = menu_items;
        markMenuForRebuild();
    }

    void addMenuItem(const TrayMenuItem& item) {
        menu_items_.push_back(item);
        markMenuForRebuild();
    }

    void removeMenuItem(const std::string& item_id) {
        menu_items_.erase(
            std::remove_if(menu_items_.begin(), menu_items_.end(),
                          [&item_id](const TrayMenuItem& item) {
                              return item.id == item_id;
                          }),
            menu_items_.end());
        markMenuForRebuild();
    }

    void setMenuItemEnabled(const std::string& item_id, bool enabled) {
        for (auto& item : menu_items_) {
            if (item.id == item_id) {
                item.enabled = enabled;
                break;
            }
        }
        markMenuForRebuild();
    }

    TrayMenuItem* findMenuItem(const std::string& item_id) {
        for (auto& item : menu_items_) {
            if (item.id == item_id) {
                return &item;
            }
        }
        return nullptr;
    }

    bool updateMenuItemLabel(const std::string& item_id, const std::string& label) {
        auto* item = findMenuItem(item_id);
        if (item) {
            item->label = label;
            markMenuForRebuild();
            return true;
        }
        return false;
    }

    bool updateMenuItemIcon(const std::string& item_id, const std::string& icon_name) {
        auto* item = findMenuItem(item_id);
        if (item) {
            item->icon = icon_name;
            markMenuForRebuild();
            return true;
        }
        return false;
    }

    bool updateMenuItemCallback(const std::string& item_id, std::function<void()> callback) {
        auto* item = findMenuItem(item_id);
        if (item) {
            item->callback = callback;
            markMenuForRebuild();
            return true;
        }
        return false;
    }

    void addMenuItems(const std::vector<TrayMenuItem>& items) {
        menu_items_.insert(menu_items_.end(), items.begin(), items.end());
        markMenuForRebuild();
    }

    void removeMenuItems(const std::vector<std::string>& item_ids) {
        for (const auto& item_id : item_ids) {
            menu_items_.erase(
                std::remove_if(menu_items_.begin(), menu_items_.end(),
                              [&item_id](const TrayMenuItem& item) {
                                  return item.id == item_id;
                              }),
                menu_items_.end());
        }
        markMenuForRebuild();
    }

    void clearMenu() {
        menu_items_.clear();
        markMenuForRebuild();
    }
    
    void updateWindowStateMenu(bool window_visible) {
        // 根据窗口状态更新显示/隐藏菜单项的可见性
        for (auto& item : menu_items_) {
            if (item.id == "show") {
                item.visible = !window_visible;  // 窗口隐藏时显示"显示窗口"
            } else if (item.id == "hide") {
                item.visible = window_visible;   // 窗口显示时显示"隐藏窗口"
            }
        }
        markMenuForRebuild();
    }

    const std::vector<TrayMenuItem>& getMenuItems() const {
        return menu_items_;
    }

    bool hasMenuItem(const std::string& item_id) const {
        return std::any_of(menu_items_.begin(), menu_items_.end(),
                          [&item_id](const TrayMenuItem& item) {
                              return item.id == item_id;
                          });
    }

    bool addSubMenuItem(const std::string& parent_id, const TrayMenuItem& item) {
        auto* parent = findMenuItem(parent_id);
        if (parent) {
            parent->submenu.push_back(item);
            markMenuForRebuild();
            return true;
        }
        return false;
    }

    bool removeSubMenuItem(const std::string& parent_id, const std::string& item_id) {
        auto* parent = findMenuItem(parent_id);
        if (parent) {
            auto it = std::remove_if(parent->submenu.begin(), parent->submenu.end(),
                                   [&item_id](const TrayMenuItem& item) {
                                       return item.id == item_id;
                                   });
            if (it != parent->submenu.end()) {
                parent->submenu.erase(it, parent->submenu.end());
                markMenuForRebuild();
                return true;
            }
        }
        return false;
    }

    TrayMenuItem* findSubMenuItem(const std::string& parent_id, const std::string& item_id) {
        auto* parent = findMenuItem(parent_id);
        if (parent) {
            for (auto& subitem : parent->submenu) {
                if (subitem.id == item_id) {
                    return &subitem;
                }
            }
        }
        return nullptr;
    }

    bool setSubMenu(const std::string& parent_id, const std::vector<TrayMenuItem>& submenu_items) {
        auto* parent = findMenuItem(parent_id);
        if (parent) {
            parent->submenu = submenu_items;
            markMenuForRebuild();
            return true;
        }
        return false;
    }

    bool clearSubMenu(const std::string& parent_id) {
        auto* parent = findMenuItem(parent_id);
        if (parent) {
            parent->submenu.clear();
            markMenuForRebuild();
            return true;
        }
        return false;
    }

    // Menu item state management methods
    void setMenuItemVisible(const std::string& itemId, bool visible) {
        auto* item = findMenuItem(itemId);
        if (item && item->visible != visible) {
            item->visible = visible;
            markMenuForRebuild();
        }
    }
    
    bool isMenuItemVisible(const std::string& itemId) const {
        auto* item = const_cast<Impl*>(this)->findMenuItem(itemId);
        return item ? item->visible : false;
    }
    
    void setMenuItemChecked(const std::string& itemId, bool checked) {
        auto* item = findMenuItem(itemId);
        if (item && item->checked != checked) {
            item->checked = checked;
            markMenuForRebuild();
        }
    }
    
    bool isMenuItemChecked(const std::string& itemId) const {
        auto* item = const_cast<Impl*>(this)->findMenuItem(itemId);
        return item ? item->checked : false;
    }
    
    void setMenuItemBadge(const std::string& itemId, const std::string& badge) {
        auto* item = findMenuItem(itemId);
        if (item && item->badge != badge) {
            item->badge = badge;
            markMenuForRebuild();
        }
    }
    
    std::string getMenuItemBadge(const std::string& itemId) const {
        auto* item = const_cast<Impl*>(this)->findMenuItem(itemId);
        return item ? item->badge : "";
    }
    
    void setMenuItemTooltip(const std::string& itemId, const std::string& tooltip) {
        auto* item = findMenuItem(itemId);
        if (item && item->tooltip != tooltip) {
            item->tooltip = tooltip;
            markMenuForRebuild();
        }
    }
    
    std::string getMenuItemTooltip(const std::string& itemId) const {
        auto* item = const_cast<Impl*>(this)->findMenuItem(itemId);
        return item ? item->tooltip : "";
    }
    
    // Shortcut and priority management methods
    void setMenuItemShortcut(const std::string& itemId, const std::string& shortcut) {
        auto* item = findMenuItem(itemId);
        if (item && item->shortcut != shortcut) {
            item->shortcut = shortcut;
            markMenuForRebuild();
        }
    }
    
    std::string getMenuItemShortcut(const std::string& itemId) const {
        auto* item = const_cast<Impl*>(this)->findMenuItem(itemId);
        return item ? item->shortcut : "";
    }
    
    void setMenuItemPriority(const std::string& itemId, int priority) {
        auto* item = findMenuItem(itemId);
        if (item && item->priority != priority) {
            item->priority = priority;
            sortMenuItemsByPriority();
            markMenuForRebuild();
        }
    }
    
    int getMenuItemPriority(const std::string& itemId) const {
        auto* item = const_cast<Impl*>(this)->findMenuItem(itemId);
        return item ? item->priority : 0;
    }
    
    void setMenuItemToggleCallback(const std::string& itemId, std::function<void(bool)> callback) {
        auto* item = findMenuItem(itemId);
        if (item) {
            item->toggle_callback = callback;
        }
    }
    
    void sortMenuItemsByPriority() {
        std::sort(menu_items_.begin(), menu_items_.end(), 
                  [](const TrayMenuItem& a, const TrayMenuItem& b) {
                      return a.priority > b.priority; // 高优先级在前
                  });
        
        // 递归排序子菜单
        for (auto& item : menu_items_) {
            if (!item.submenu.empty()) {
                std::sort(item.submenu.begin(), item.submenu.end(),
                          [](const TrayMenuItem& a, const TrayMenuItem& b) {
                              return a.priority > b.priority;
                          });
            }
        }
    }
    
    // Performance optimization methods
    void batchUpdateMenuItems(std::function<void()> updates) {
        bool old_rebuild_flag = menu_needs_rebuild_;
        menu_needs_rebuild_ = false; // 暂时禁用重建
        
        updates(); // 执行批量更新
        
        if (old_rebuild_flag || menu_needs_rebuild_) {
            markMenuForRebuild();
        }
    }
    
    void forceRebuildMenu() {
        rebuildMenu();
    }
    
    bool needsMenuRebuild() const {
        return menu_needs_rebuild_;
    }

    void showNotification(const std::string& title, const std::string& message,
                         const std::string& icon_name, int timeout_ms) {
        // 在控制台显示通知（简化实现）
        std::cout << "[NOTIFICATION] " << title << ": " << message << std::endl;
        
        // 这里可以集成libnotify或其他通知系统
        // 暂时使用简单的控制台输出
    }

    void updateProgress(double progress, const std::string& text) {
        progress_ = std::max(0.0, std::min(1.0, progress));
        progress_text_ = text;
        
        // 更新提示文本
        std::string tooltip = app_name_;
        if (!progress_text_.empty()) {
            tooltip += " - " + progress_text_;
        }
        setTooltip(tooltip);
    }

    void clearProgress() {
        progress_ = 0.0;
        progress_text_.clear();
        setTooltip(app_name_);
    }

    // 回调函数设置
    std::function<void()> left_click_callback_;
    std::function<void()> right_click_callback_;
    std::function<void()> double_click_callback_;
    std::function<void(TrayStatus)> status_change_callback_;
    std::function<void()> quit_callback_;  // 退出回调函数
    bool menu_needs_rebuild_;

private:
    void createDefaultMenu() {
        menu_items_.clear();
        
        // Show Window 菜单项
        TrayMenuItem show_window_item;
        show_window_item.id = "show_window";
        show_window_item.label = "Show Window";
        show_window_item.icon = "";
        show_window_item.enabled = true;
        show_window_item.separator = false;
        show_window_item.visible = true;
        show_window_item.callback = [this]() {
            if (left_click_callback_) left_click_callback_();
        };
        menu_items_.push_back(show_window_item);
        
        TrayMenuItem show_item;
        show_item.id = "show";
        show_item.label = "显示主窗口";
        show_item.icon = "";
        show_item.enabled = true;
        show_item.separator = false;
        show_item.visible = true;
        show_item.callback = [this]() {
            if (left_click_callback_) left_click_callback_();
        };
        menu_items_.push_back(show_item);
        
        TrayMenuItem hide_item;
        hide_item.id = "hide";
        hide_item.label = "隐藏窗口";
        hide_item.icon = "";
        hide_item.enabled = true;
        hide_item.separator = false;
        hide_item.visible = false;  // 初始时隐藏，根据窗口状态动态显示
        hide_item.callback = [this]() {
            if (right_click_callback_) right_click_callback_();
        };
        menu_items_.push_back(hide_item);
        
        TrayMenuItem sep1;
        sep1.id = "separator1";
        sep1.label = "";
        sep1.icon = "";
        sep1.enabled = true;
        sep1.separator = true;
        menu_items_.push_back(sep1);
        
        TrayMenuItem status_item;
        status_item.id = "status";
        status_item.label = "状态: 空闲";
        status_item.icon = "";
        status_item.enabled = false;
        status_item.separator = false;
        menu_items_.push_back(status_item);
        
        TrayMenuItem sep2;
        sep2.id = "separator2";
        sep2.label = "";
        sep2.icon = "";
        sep2.enabled = true;
        sep2.separator = true;
        menu_items_.push_back(sep2);
        
        TrayMenuItem quit_item;
        quit_item.id = "quit";
        quit_item.label = "Quit Duorou";
        quit_item.icon = "";
        quit_item.enabled = true;
        quit_item.separator = false;
        quit_item.callback = [this]() {
            // 调用退出回调函数，而不是直接调用std::exit
            if (quit_callback_) {
                quit_callback_();
            } else {
                // 如果没有设置退出回调，则使用默认退出方式
                std::exit(0);
            }
        };
        menu_items_.push_back(quit_item);
        
        rebuildMenu();
    }

    void markMenuForRebuild() {
        menu_needs_rebuild_ = true;
    }
    
    void rebuildMenuIfNeeded() {
        if (menu_needs_rebuild_) {
            rebuildMenu();
            menu_needs_rebuild_ = false;
        }
    }
    
    void rebuildMenu() {
        // 清理旧菜单
        if (menu_) {
            g_object_unref(menu_);
        }

        // GTK4: 创建简化菜单
        menu_ = gtk_popover_menu_new_from_model(nullptr);
        menu_item_map_.clear();

        buildMenuItems(menu_items_, menu_);
        
        gtk_widget_show(menu_);
        menu_needs_rebuild_ = false;
    }

    void buildMenuItems(const std::vector<TrayMenuItem>& items, GtkWidget* parent_menu) {
        for (const auto& item : items) {
            // Skip invisible items
            if (!item.visible) {
                continue;
            }
            
            GtkWidget* menu_item;
            
            if (item.separator) {
                // GTK4: 简化分隔符处理
                continue;
            } else {
                std::string label = item.label;
                 
                 // Add badge if present
                 if (!item.badge.empty()) {
                     label += " [" + item.badge + "]";
                 }
                 
                 // Add check mark if checked
                if (item.checked) {
                    label = "[✓] " + label;
                }
                 
                 // Add shortcut if present
                 if (!item.shortcut.empty()) {
                     label += "\t" + item.shortcut;
                 }
                
                menu_item = gtk_button_new_with_label(label.c_str());
                
                // 设置启用状态
                gtk_widget_set_sensitive(menu_item, item.enabled);
                
                // Set tooltip if present
                if (!item.tooltip.empty()) {
                    gtk_widget_set_tooltip_text(menu_item, item.tooltip.c_str());
                }
                
                // 如果有子菜单，创建子菜单
                if (!item.submenu.empty()) {
                    GtkWidget* submenu = gtk_popover_menu_new_from_model(nullptr);
                    buildMenuItems(item.submenu, submenu);
                    // GTK4: 简化子菜单设置
                    g_object_set_data(G_OBJECT(menu_item), "submenu", submenu);
                } else if (item.callback) {
                    // 连接回调（只有叶子节点才有回调）
                    g_object_set_data(G_OBJECT(menu_item), "callback_ptr", 
                                     const_cast<std::function<void()>*>(&item.callback));
                    g_signal_connect(menu_item, "clicked", 
                                   G_CALLBACK(onMenuItemActivatedStatic), this);
                }
            }
            
            menu_item_map_[item.id] = menu_item;
        }
    }

    void updateStatusIcon() {
        if (!tray_widget_) return;
        
        std::string icon_name;
        switch (status_) {
            case TrayStatus::Idle:
                icon_name = "application-x-executable";
                break;
            case TrayStatus::Working:
                icon_name = "system-run";
                break;
            case TrayStatus::Error:
                icon_name = "dialog-error";
                break;
            case TrayStatus::Generating:
                icon_name = "image-x-generic";
                break;
            case TrayStatus::Loading:
                icon_name = "view-refresh";
                break;
        }
        
        setIconFromTheme(icon_name);
        
        // 更新状态菜单项
        updateStatusMenuItem();
    }

    void updateStatusMenuItem() {
        std::string status_text = "状态: ";
        switch (status_) {
            case TrayStatus::Idle:
                status_text += "空闲";
                break;
            case TrayStatus::Working:
                status_text += "工作中";
                break;
            case TrayStatus::Error:
                status_text += "错误";
                break;
            case TrayStatus::Generating:
                status_text += "生成中";
                break;
            case TrayStatus::Loading:
                status_text += "加载中";
                break;
        }
        
        // 查找并更新状态菜单项
        for (auto& item : menu_items_) {
            if (item.id == "status") {
                item.label = status_text;
                break;
            }
        }
        
        rebuildMenu();
    }

    void cleanup() {
        if (menu_) {
            g_object_unref(menu_);
            menu_ = nullptr;
        }
        
        if (tray_widget_) {
            g_object_unref(tray_widget_);
            tray_widget_ = nullptr;
        }
    }

    // 静态回调函数
    static void onActivateStatic(GtkWidget* widget, gpointer user_data) {
        auto* impl = static_cast<Impl*>(user_data);
        if (impl && impl->left_click_callback_) {
            impl->left_click_callback_();
        }
    }

    static void onPopupMenuStatic(GtkWidget* widget, gpointer user_data) {
        auto* impl = static_cast<Impl*>(user_data);
        if (impl && impl->menu_) {
            // GTK4: 使用popover menu替代popup menu
            gtk_widget_show(impl->menu_);
        }
    }

    static void onMenuItemActivatedStatic(GtkWidget* menu_item, gpointer user_data) {
        auto* callback = static_cast<std::function<void()>*>(
            g_object_get_data(G_OBJECT(menu_item), "callback_ptr"));
        if (callback) {
            (*callback)();
        }
    }

    // 成员变量
    std::string app_name_;
    GtkWidget* tray_widget_;  // GTK4: 使用普通widget替代status icon
    GtkWidget* menu_;
    std::vector<TrayMenuItem> menu_items_;
    std::map<std::string, GtkWidget*> menu_item_map_;
    TrayStatus status_;
    bool visible_;
    double progress_;
    std::string progress_text_;
};

// SystemTray实现
SystemTray::SystemTray() : pimpl_(std::make_unique<Impl>()) {}

SystemTray::~SystemTray() = default;

bool SystemTray::initialize(const std::string& app_name, const std::string& icon_path) {
    return pimpl_->initialize(app_name, icon_path);
}

void SystemTray::show() {
    pimpl_->show();
}

void SystemTray::hide() {
    pimpl_->hide();
}

bool SystemTray::isVisible() const {
    return pimpl_->isVisible();
}

bool SystemTray::setIcon(const std::string& icon_path) {
    return pimpl_->setIcon(icon_path);
}

bool SystemTray::setIconFromTheme(const std::string& icon_name) {
    return pimpl_->setIconFromTheme(icon_name);
}

void SystemTray::setTooltip(const std::string& tooltip) {
    pimpl_->setTooltip(tooltip);
}

void SystemTray::setStatus(TrayStatus status) {
    pimpl_->setStatus(status);
}

TrayStatus SystemTray::getStatus() const {
    return pimpl_->getStatus();
}

void SystemTray::setMenu(const std::vector<TrayMenuItem>& menu_items) {
    pimpl_->setMenu(menu_items);
}

void SystemTray::addMenuItem(const TrayMenuItem& item) {
    pimpl_->addMenuItem(item);
}

void SystemTray::removeMenuItem(const std::string& item_id) {
    pimpl_->removeMenuItem(item_id);
}

void SystemTray::setMenuItemEnabled(const std::string& item_id, bool enabled) {
    pimpl_->setMenuItemEnabled(item_id, enabled);
}

TrayMenuItem* SystemTray::findMenuItem(const std::string& item_id) {
    return pimpl_->findMenuItem(item_id);
}

bool SystemTray::updateMenuItemLabel(const std::string& item_id, const std::string& label) {
    return pimpl_->updateMenuItemLabel(item_id, label);
}

bool SystemTray::updateMenuItemIcon(const std::string& item_id, const std::string& icon_name) {
    return pimpl_->updateMenuItemIcon(item_id, icon_name);
}

bool SystemTray::updateMenuItemCallback(const std::string& item_id, std::function<void()> callback) {
    return pimpl_->updateMenuItemCallback(item_id, callback);
}

void SystemTray::addMenuItems(const std::vector<TrayMenuItem>& items) {
    pimpl_->addMenuItems(items);
}

void SystemTray::removeMenuItems(const std::vector<std::string>& item_ids) {
    pimpl_->removeMenuItems(item_ids);
}

void SystemTray::clearMenu() {
    pimpl_->clearMenu();
}

const std::vector<TrayMenuItem>& SystemTray::getMenuItems() const {
    return pimpl_->getMenuItems();
}

bool SystemTray::hasMenuItem(const std::string& itemId) const {
    return pimpl_->hasMenuItem(itemId);
}

// Menu item state management
void SystemTray::setMenuItemVisible(const std::string& itemId, bool visible) {
    pimpl_->setMenuItemVisible(itemId, visible);
}

bool SystemTray::isMenuItemVisible(const std::string& itemId) const {
    return pimpl_->isMenuItemVisible(itemId);
}

void SystemTray::setMenuItemChecked(const std::string& itemId, bool checked) {
    pimpl_->setMenuItemChecked(itemId, checked);
}

bool SystemTray::isMenuItemChecked(const std::string& itemId) const {
    return pimpl_->isMenuItemChecked(itemId);
}

void SystemTray::setMenuItemBadge(const std::string& itemId, const std::string& badge) {
    pimpl_->setMenuItemBadge(itemId, badge);
}

std::string SystemTray::getMenuItemBadge(const std::string& itemId) const {
    return pimpl_->getMenuItemBadge(itemId);
}

void SystemTray::setMenuItemTooltip(const std::string& itemId, const std::string& tooltip) {
    pimpl_->setMenuItemTooltip(itemId, tooltip);
}

std::string SystemTray::getMenuItemTooltip(const std::string& itemId) const {
    return pimpl_->getMenuItemTooltip(itemId);
}

// Shortcut and priority management
void SystemTray::setMenuItemShortcut(const std::string& itemId, const std::string& shortcut) {
    pimpl_->setMenuItemShortcut(itemId, shortcut);
}

std::string SystemTray::getMenuItemShortcut(const std::string& itemId) const {
    return pimpl_->getMenuItemShortcut(itemId);
}

void SystemTray::setMenuItemPriority(const std::string& itemId, int priority) {
    pimpl_->setMenuItemPriority(itemId, priority);
}

int SystemTray::getMenuItemPriority(const std::string& itemId) const {
    return pimpl_->getMenuItemPriority(itemId);
}

void SystemTray::setMenuItemToggleCallback(const std::string& itemId, std::function<void(bool)> callback) {
    pimpl_->setMenuItemToggleCallback(itemId, callback);
}

void SystemTray::sortMenuItemsByPriority() {
    pimpl_->sortMenuItemsByPriority();
}

// Performance optimization methods
void SystemTray::batchUpdateMenuItems(std::function<void()> updates) {
    pimpl_->batchUpdateMenuItems(updates);
}

void SystemTray::forceRebuildMenu() {
    pimpl_->forceRebuildMenu();
}

bool SystemTray::needsMenuRebuild() const {
    return pimpl_->needsMenuRebuild();
}

bool SystemTray::addSubMenuItem(const std::string& parent_id, const TrayMenuItem& item) {
    return pimpl_->addSubMenuItem(parent_id, item);
}

bool SystemTray::removeSubMenuItem(const std::string& parent_id, const std::string& item_id) {
    return pimpl_->removeSubMenuItem(parent_id, item_id);
}

TrayMenuItem* SystemTray::findSubMenuItem(const std::string& parent_id, const std::string& item_id) {
    return pimpl_->findSubMenuItem(parent_id, item_id);
}

bool SystemTray::setSubMenu(const std::string& parent_id, const std::vector<TrayMenuItem>& submenu_items) {
    return pimpl_->setSubMenu(parent_id, submenu_items);
}

bool SystemTray::clearSubMenu(const std::string& parent_id) {
    return pimpl_->clearSubMenu(parent_id);
}

void SystemTray::showNotification(const std::string& title, const std::string& message,
                                 const std::string& icon_name, int timeout_ms) {
    pimpl_->showNotification(title, message, icon_name, timeout_ms);
}

void SystemTray::setLeftClickCallback(std::function<void()> callback) {
    pimpl_->left_click_callback_ = callback;
}

void SystemTray::setRightClickCallback(std::function<void()> callback) {
    pimpl_->right_click_callback_ = callback;
}

void SystemTray::setDoubleClickCallback(std::function<void()> callback) {
    pimpl_->double_click_callback_ = callback;
}

void SystemTray::setStatusChangeCallback(std::function<void(TrayStatus)> callback) {
    pimpl_->status_change_callback_ = callback;
}

void SystemTray::setQuitCallback(std::function<void()> callback) {
    pimpl_->quit_callback_ = callback;
}

void SystemTray::updateWindowStateMenu(bool window_visible) {
    pimpl_->updateWindowStateMenu(window_visible);
}

void SystemTray::updateProgress(double progress, const std::string& text) {
    pimpl_->updateProgress(progress, text);
}

void SystemTray::clearProgress() {
    pimpl_->clearProgress();
}

bool SystemTray::isSystemTraySupported() {
#ifdef HAVE_GTK
    // 检查GTK和系统托盘支持
    return true; // 简化实现，假设支持
#else
    return false;
#endif
}

std::string SystemTray::getDefaultIconPath(TrayStatus status) {
    switch (status) {
        case TrayStatus::Idle:
            return "application-x-executable";
        case TrayStatus::Working:
            return "system-run";
        case TrayStatus::Error:
            return "dialog-error";
        case TrayStatus::Generating:
            return "image-x-generic";
        case TrayStatus::Loading:
            return "view-refresh";
        default:
            return "application-x-executable";
    }
}



// SystemTrayManager实现
SystemTrayManager& SystemTrayManager::getInstance() {
    static SystemTrayManager instance;
    return instance;
}

bool SystemTrayManager::initialize(const std::string& app_name, const std::string& icon_path) {
    if (tray_) {
        return true; // 已经初始化
    }
    
    tray_ = std::make_unique<SystemTray>();
    return tray_->initialize(app_name, icon_path);
}

SystemTray* SystemTrayManager::getTray() {
    return tray_.get();
}

void SystemTrayManager::shutdown() {
    tray_.reset();
}

} // namespace gui
} // namespace duorou