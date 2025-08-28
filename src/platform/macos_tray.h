#ifndef MACOS_TRAY_H
#define MACOS_TRAY_H

#ifdef __APPLE__

#include "../gui/system_tray.h"
#include <functional>
#include <memory>
#include <string>
#include <map>
#include <list>

#ifdef __OBJC__
@class NSStatusItem;
@class NSMenu;
@class NSImage;
@class NSMenuItem;
#else
typedef void NSStatusItem;
typedef void NSMenu;
typedef void NSImage;
typedef void NSMenuItem;
#endif

namespace duorou {

class MacOSTray : public gui::SystemTray {
public:
  MacOSTray();
  ~MacOSTray();

  bool initialize();
  void show();
  void hide();
  void setIcon(const std::string &icon_path);
  void setIconFromFile(const std::string &imagePath);
  void setTooltip(const std::string &tooltip);
  void addMenuItem(const std::string &label, std::function<void()> callback);
  void addSeparator();
  void clearMenu();
  void setSystemIcon();
  bool isAvailable() const;

  // 增强的菜单管理功能
  void removeMenuItem(const std::string& item_id);
  void setMenuItemEnabled(const std::string& item_id, bool enabled);
  bool updateMenuItemLabel(const std::string& item_id, const std::string& label);
  bool hasMenuItem(const std::string& item_id) const;
  void addMenuItemWithId(const std::string& item_id, const std::string& label, std::function<void()> callback);
  
  // 子菜单支持
  void addSubMenu(const std::string& parent_id, const std::string& label);
  void addSubMenuItem(const std::string& parent_id, const std::string& item_id, 
                     const std::string& label, std::function<void()> callback);
  
  // 菜单项状态管理
  void setMenuItemVisible(const std::string& item_id, bool visible);
  void setMenuItemIcon(const std::string& item_id, const std::string& icon_name);
  
  // 快捷键支持
  void setMenuItemShortcut(const std::string& item_id, const std::string& shortcut);
  
  // 窗口状态管理
  void updateWindowStateMenu(bool window_visible);
  
  // 回调函数设置
  void setLeftClickCallback(std::function<void()> callback);
  void setRightClickCallback(std::function<void()> callback);
  void setQuitCallback(std::function<void()> callback);

private:
  NSStatusItem *statusItem_;
  NSMenu *menu_;
  bool initialized_;

  // 菜单项管理
  struct MenuItemInfo {
    std::string id;
    std::string label;
    std::function<void()> callback;
    NSMenuItem* menuItem;
    bool enabled;
    bool visible;
    std::string icon_name;
    std::string shortcut;
    NSMenu* submenu;
  };
  
  std::list<MenuItemInfo> menu_items_;
  std::map<std::string, MenuItemInfo*> menu_item_map_;
  
  // 回调函数
  std::function<void()> left_click_callback_;
  std::function<void()> right_click_callback_;
  std::function<void()> quit_callback_;

  NSImage *createImageFromText(const std::string &text);
  void setupMenu();
  void rebuildMenu();
  MenuItemInfo* findMenuItemInfo(const std::string& item_id);
  NSMenuItem* createNSMenuItem(const MenuItemInfo& info);
};

} // namespace duorou

#endif // __APPLE__

#endif // MACOS_TRAY_H