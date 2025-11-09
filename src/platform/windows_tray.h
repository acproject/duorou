#ifndef WINDOWS_TRAY_H
#define WINDOWS_TRAY_H

#ifdef _WIN32

#include "../gui/system_tray.h"
#include <windows.h>
#include <string>
#include <functional>
#include <map>
#include <list>

namespace duorou {

class WindowsTray : public gui::SystemTray {
public:
  WindowsTray();
  ~WindowsTray();

  bool initialize();
  void show();
  void hide();
  void setIcon(const std::string &icon_name);
  void setIconFromFile(const std::string &imagePath);
  void setTooltip(const std::string &tooltip);
  void addMenuItem(const std::string &label, std::function<void()> callback);
  void addMenuItemWithId(const std::string &item_id, const std::string &label, std::function<void()> callback);
  void addSeparator();
  void clearMenu();
  void setSystemIcon();
  bool isAvailable() const;

  void removeMenuItem(const std::string& item_id);
  void setMenuItemEnabled(const std::string& item_id, bool enabled);
  bool updateMenuItemLabel(const std::string& item_id, const std::string& label);
  bool hasMenuItem(const std::string& item_id) const;

  void setMenuItemVisible(const std::string& item_id, bool visible);
  void setMenuItemIcon(const std::string& item_id, const std::string& icon_name);
  void setMenuItemShortcut(const std::string& item_id, const std::string& shortcut);

  void updateWindowStateMenu(bool window_visible);

  void setLeftClickCallback(std::function<void()> callback);
  void setRightClickCallback(std::function<void()> callback);
  void setQuitCallback(std::function<void()> callback);

private:
  HWND hwnd_;
  HMENU menu_;
  NOTIFYICONDATAW nid_;
  bool initialized_;
  bool icon_added_;

  struct MenuItemInfo {
    std::string id;
    std::string label;
    std::function<void()> callback;
    UINT command_id;
    bool enabled;
    bool visible;
    bool checked;
    bool separator;
    std::string icon_name;
    std::string shortcut;
  };

  std::list<MenuItemInfo> menu_items_;
  std::map<std::string, MenuItemInfo*> menu_item_map_;
  std::map<UINT, MenuItemInfo*> command_map_;
  UINT next_command_id_;

  std::function<void()> left_click_callback_;
  std::function<void()> right_click_callback_;
  std::function<void()> quit_callback_;

  static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
  void handleTrayMessage(WPARAM wParam, LPARAM lParam);
  void handleCommand(WPARAM wParam);
  void rebuildMenu();
  HICON iconForName(const std::string &name);
  static std::wstring toWide(const std::string &s);
};

} // namespace duorou

#endif // _WIN32

#endif // WINDOWS_TRAY_H