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
        rebuildMenu();
    }

    void addMenuItem(const TrayMenuItem& item) {
        menu_items_.push_back(item);
        rebuildMenu();
    }

    void removeMenuItem(const std::string& item_id) {
        menu_items_.erase(
            std::remove_if(menu_items_.begin(), menu_items_.end(),
                          [&item_id](const TrayMenuItem& item) {
                              return item.id == item_id;
                          }),
            menu_items_.end());
        rebuildMenu();
    }

    void setMenuItemEnabled(const std::string& item_id, bool enabled) {
        for (auto& item : menu_items_) {
            if (item.id == item_id) {
                item.enabled = enabled;
                break;
            }
        }
        rebuildMenu();
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

private:
    void createDefaultMenu() {
        menu_items_ = {
            {"show", "显示主窗口", "", true, false, [this]() {
                if (left_click_callback_) left_click_callback_();
            }},
            {"separator1", "", "", true, true, nullptr},
            {"status", "状态: 空闲", "", false, false, nullptr},
            {"separator2", "", "", true, true, nullptr},
            {"quit", "退出", "", true, false, [this]() {
                // 退出应用程序
                std::exit(0);
            }}
        };
        rebuildMenu();
    }

    void rebuildMenu() {
        // 清理旧菜单
        if (menu_) {
            g_object_unref(menu_);
        }

        // GTK4: 创建简化菜单
        menu_ = gtk_popover_menu_new_from_model(nullptr);
        menu_item_map_.clear();

        for (const auto& item : menu_items_) {
            GtkWidget* menu_item;
            
            if (item.separator) {
                // GTK4: 简化分隔符处理
                continue;
            } else {
                menu_item = gtk_button_new_with_label(item.label.c_str());
                
                // 设置启用状态
                gtk_widget_set_sensitive(menu_item, item.enabled);
                
                // 连接回调
                if (item.callback) {
                    g_object_set_data(G_OBJECT(menu_item), "callback_ptr", 
                                     const_cast<std::function<void()>*>(&item.callback));
                    g_signal_connect(menu_item, "clicked", 
                                   G_CALLBACK(onMenuItemActivatedStatic), this);
                }
            }
            
            menu_item_map_[item.id] = menu_item;
        }
        
        gtk_widget_show(menu_);
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