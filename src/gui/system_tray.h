#pragma once

#include <memory>
#include <functional>
#include <string>
#include <vector>

#if (defined(HAVE_GTK) && __has_include(<gtk/gtk.h>)) || __has_include(<gtk/gtk.h>)
#include <gtk/gtk.h>
#define DUOROU_HAS_GTK 1
#else
#define DUOROU_HAS_GTK 0
// GTK 类型占位符定义（无 GTK 开发头文件时避免诊断与编译错误）
typedef void* GtkWidget;
typedef void* GtkMenu;
typedef void* GtkPopoverMenu;
typedef void* GdkPixbuf;
typedef void* gpointer;
typedef unsigned int guint;
typedef int gboolean;
typedef int gint;
typedef double gdouble;
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef G_CALLBACK
#define G_CALLBACK(f) ((void(*)(void))(f))
#endif
#ifndef g_signal_connect
#define g_signal_connect(instance, detailed_signal, c_handler, data) ((void)0)
#endif
#endif

namespace duorou {
namespace gui {

/**
 * System tray status enumeration
 */
enum class TrayStatus {
    Idle,           // Idle state
    Working,        // Working state
    Error,          // Error state
    Generating,     // Generating state
    Loading         // Loading state
};

/**
 * System tray icon type
 */
enum class TrayIconType {
    SYSTEM,     // System icon
    CUSTOM,     // Custom icon
    TEXT        // Text icon
};

/**
 * Tray menu item structure
 */
struct TrayMenuItem {
        std::string id;                           // Menu item ID
        std::string label;                        // Display text
        std::string icon;                         // Icon path or name
        bool enabled = true;                      // Whether enabled
        bool separator = false;                   // Whether is separator
        bool visible = true;                      // Whether visible
        bool checked = false;                     // Whether checked
        std::string badge;                        // Badge text
        std::string tooltip;                      // Tooltip text
        std::string shortcut;                     // Shortcut key (e.g. "Ctrl+N", "Cmd+Q")
        int priority = 0;                         // Menu item priority (for sorting)
        std::function<void()> callback;           // Click callback
         std::function<void(bool)> toggle_callback; // Toggle callback (for checkbox menu items)
         std::vector<TrayMenuItem> submenu;        // Submenu
     };

/**
 * System tray class
 * Provides system tray icon, menu and status indication functionality
 */
class SystemTray {
public:
    SystemTray();
    ~SystemTray();

    // Disable copy constructor and assignment
    SystemTray(const SystemTray&) = delete;
    SystemTray& operator=(const SystemTray&) = delete;

    /**
     * Initialize system tray
     * @param app_name Application name
     * @param icon_path Tray icon path
     * @return Whether initialization succeeded
     */
    bool initialize(const std::string& app_name, const std::string& icon_path = "");

    /**
     * Show system tray icon
     */
    void show();

    /**
     * Hide system tray icon
     */
    void hide();

    /**
     * Check if tray is visible
     * @return Whether visible
     */
    bool isVisible() const;

    /**
     * Set tray icon
     * @param icon_path Icon file path
     * @return Whether setting succeeded
     */
    bool setIcon(const std::string& icon_path);

    /**
     * Set tray icon (from resource)
     * @param icon_name Icon name
     * @return Whether setting succeeded
     */
    bool setIconFromTheme(const std::string& icon_name);

    /**
     * Set tray tooltip text
     * @param tooltip Tooltip text
     */
    void setTooltip(const std::string& tooltip);

    /**
     * Set tray status
     * @param status Status
     */
    void setStatus(TrayStatus status);

    /**
     * Get current status
     * @return Current status
     */
    TrayStatus getStatus() const;

    /**
     * Set tray menu
     * @param menu_items Menu item list
     */
    void setMenu(const std::vector<TrayMenuItem>& menu_items);

    /**
     * Add menu item
     * @param item Menu item
     */
    void addMenuItem(const TrayMenuItem& item);

    /**
     * Remove menu item
     * @param item_id Menu item ID
     */
    void removeMenuItem(const std::string& item_id);

    /**
     * Enable/disable menu item
     * @param item_id Menu item ID
     * @param enabled Whether to enable
     */
    void setMenuItemEnabled(const std::string& item_id, bool enabled);

    /**
     * Find menu item
     * @param item_id Menu item ID
     * @return Menu item pointer, returns nullptr if not found
     */
    TrayMenuItem* findMenuItem(const std::string& item_id);

    /**
     * Update menu item label
     * @param item_id Menu item ID
     * @param label New label
     * @return Whether update succeeded
     */
    bool updateMenuItemLabel(const std::string& item_id, const std::string& label);

    /**
     * Update menu item icon
     * @param item_id Menu item ID
     * @param icon_name Icon name
     * @return Whether update succeeded
     */
    bool updateMenuItemIcon(const std::string& item_id, const std::string& icon_name);

    /**
     * Update menu item callback
     * @param item_id Menu item ID
     * @param callback New callback function
     * @return Whether update succeeded
     */
    bool updateMenuItemCallback(const std::string& item_id, std::function<void()> callback);

    /**
     * Batch add menu items
     * @param items Menu item list
     */
    void addMenuItems(const std::vector<TrayMenuItem>& items);

    /**
     * Batch remove menu items
     * @param item_ids Menu item ID list
     */
    void removeMenuItems(const std::vector<std::string>& item_ids);

    /**
     * Clear all menu items
     */
    void clearMenu();

    /**
     * Get all menu items
     * @return Constant reference to menu item list
     */
    const std::vector<TrayMenuItem>& getMenuItems() const;

    /**
     * Check if menu item exists
     * @param item_id Menu item ID
     * @return Whether exists
     */
    bool hasMenuItem(const std::string& item_id) const;

    /**
     * Add submenu item
     * @param parent_id Parent menu item ID
     * @param item Submenu item
     * @return Whether addition succeeded
     */
    bool addSubMenuItem(const std::string& parent_id, const TrayMenuItem& item);

    /**
     * Remove submenu item
     * @param parent_id Parent menu item ID
     * @param item_id Submenu item ID
     * @return Whether removal succeeded
     */
    bool removeSubMenuItem(const std::string& parent_id, const std::string& item_id);

    /**
     * Find submenu item
     * @param parent_id Parent menu item ID
     * @param item_id Submenu item ID
     * @return Submenu item pointer, returns nullptr if not found
     */
    TrayMenuItem* findSubMenuItem(const std::string& parent_id, const std::string& item_id);

    /**
     * Set submenu
     * @param parent_id Parent menu item ID
     * @param submenu_items Submenu item list
     * @return Whether setting succeeded
     */
    bool setSubMenu(const std::string& parent_id, const std::vector<TrayMenuItem>& submenu_items);

    /**
     * Clear submenu
     * @param parent_id Parent menu item ID
     * @return Whether clearing succeeded
     */
    bool clearSubMenu(const std::string& parent_id);

    /**
     * Set menu item visibility
     * @param item_id Menu item ID
     * @param visible Whether visible
     */
    void setMenuItemVisible(const std::string& item_id, bool visible);

    /**
     * Check if menu item is visible
     * @param item_id Menu item ID
     * @return Whether visible
     */
    bool isMenuItemVisible(const std::string& item_id) const;

    /**
     * Set menu item checked state
     * @param item_id Menu item ID
     * @param checked Whether checked
     */
    void setMenuItemChecked(const std::string& item_id, bool checked);

    /**
     * Check if menu item is checked
     * @param item_id Menu item ID
     * @return Whether checked
     */
    bool isMenuItemChecked(const std::string& item_id) const;

    /**
     * Set menu item badge
     * @param item_id Menu item ID
     * @param badge Badge text
     */
    void setMenuItemBadge(const std::string& item_id, const std::string& badge);

    /**
     * Get menu item badge
     * @param item_id Menu item ID
     * @return Badge text
     */
    std::string getMenuItemBadge(const std::string& item_id) const;

    /**
     * Set menu item tooltip text
     * @param item_id Menu item ID
     * @param tooltip Tooltip text
     */
    void setMenuItemTooltip(const std::string& item_id, const std::string& tooltip);

    /**
     * Get menu item tooltip text
     * @param item_id Menu item ID
     * @return Tooltip text
     */
    std::string getMenuItemTooltip(const std::string& item_id) const;

     /**
      * Set menu item shortcut
      * @param item_id Menu item ID
      * @param shortcut Shortcut string (e.g. "Ctrl+N", "Cmd+Q")
      */
     void setMenuItemShortcut(const std::string& item_id, const std::string& shortcut);

     /**
      * Get menu item shortcut
      * @param item_id Menu item ID
      * @return Shortcut string
      */
     std::string getMenuItemShortcut(const std::string& item_id) const;

     /**
      * Set menu item priority
      * @param item_id Menu item ID
      * @param priority Priority value
      */
     void setMenuItemPriority(const std::string& item_id, int priority);

     /**
      * Get menu item priority
      * @param item_id Menu item ID
      * @return Priority value
      */
     int getMenuItemPriority(const std::string& item_id) const;

     /**
      * Set menu item toggle callback
      * @param item_id Menu item ID
      * @param callback Toggle callback function
      */
     void setMenuItemToggleCallback(const std::string& item_id, std::function<void(bool)> callback);

     /**
      * Sort menu items by priority
      */
     void sortMenuItemsByPriority();

     /**
      * Batch update menu items (performance optimization)
      * @param updates Function for update operations
      */
     void batchUpdateMenuItems(std::function<void()> updates);

     /**
      * Force rebuild menu
      */
     void forceRebuildMenu();

     /**
      * Check if menu needs rebuild
      * @return Whether rebuild is needed
      */
     bool needsMenuRebuild() const;

     /**
      * Show notification message
     * @param title Title
     * @param message Message content
     * @param icon_name Icon name (optional)
     * @param timeout_ms Timeout in milliseconds (0 means no auto-close)
     */
    void showNotification(const std::string& title, 
                         const std::string& message,
                         const std::string& icon_name = "",
                         int timeout_ms = 5000);

    /**
     * Set left click callback
     * @param callback Callback function
     */
    void setLeftClickCallback(std::function<void()> callback);

    /**
     * Set right click callback
     * @param callback Callback function
     */
    void setRightClickCallback(std::function<void()> callback);

    /**
     * Set double click callback
     * @param callback Callback function
     */
    void setDoubleClickCallback(std::function<void()> callback);

    /**
     * Set status change callback
     * @param callback Callback function for status changes
     */
    void setStatusChangeCallback(std::function<void(TrayStatus)> callback);
    
    /**
     * Set quit callback
     * @param callback Callback function for quit
     */
    void setQuitCallback(std::function<void()> callback);
    
    /**
     * Update menu based on window state
     * @param window_visible Whether window is visible
     */
    void updateWindowStateMenu(bool window_visible);

    /**
     * Update progress display
     * @param progress Progress value (0.0-1.0)
     * @param text Progress text
     */
    void updateProgress(double progress, const std::string& text = "");

    /**
     * Clear progress display
     */
    void clearProgress();

    /**
     * Check if system supports tray
     * @return Whether supported
     */
    static bool isSystemTraySupported();

    /**
     * Get default icon path
     * @param status Status
     * @return Icon path
     */
    static std::string getDefaultIconPath(TrayStatus status);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;

    // GTK callback functions
    static void onActivate(GtkWidget* widget, gpointer user_data);
    static void onPopupMenu(GtkWidget* widget, gpointer user_data);
    static void onMenuItemActivated(GtkWidget* menu_item, gpointer user_data);
    static gboolean onButtonPress(GtkWidget* widget, gpointer user_data);
};

/**
 * System tray manager
 * Singleton pattern, manages global system tray instance
 */
class SystemTrayManager {
public:
    /**
     * Get singleton instance
     * @return System tray manager instance
     */
    static SystemTrayManager& getInstance();

    /**
     * Initialize system tray
     * @param app_name Application name
     * @param icon_path Icon path
     * @return Whether initialization succeeded
     */
    bool initialize(const std::string& app_name, const std::string& icon_path = "");

    /**
     * Get system tray instance
     * @return System tray pointer
     */
    SystemTray* getTray();

    /**
     * Shutdown system tray
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