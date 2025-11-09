#ifdef _WIN32

#include "windows_tray.h"
#include <shellapi.h>
#include <commctrl.h>
#include <unordered_map>
#include <iostream>

#ifndef WM_TRAYICON
#define WM_TRAYICON (WM_APP + 1)
#endif

namespace duorou {

static const wchar_t* kTrayClassName = L"DuorouTrayWindowClass";

WindowsTray::WindowsTray()
  : hwnd_(nullptr), menu_(nullptr), initialized_(false), icon_added_(false), next_command_id_(1000) {
  ZeroMemory(&nid_, sizeof(nid_));
}

WindowsTray::~WindowsTray() {
  if (icon_added_) {
    Shell_NotifyIconW(NIM_DELETE, &nid_);
    icon_added_ = false;
  }
  if (menu_) {
    DestroyMenu(menu_);
    menu_ = nullptr;
  }
  if (hwnd_) {
    DestroyWindow(hwnd_);
    hwnd_ = nullptr;
  }
}

LRESULT CALLBACK WindowsTray::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  WindowsTray* self = reinterpret_cast<WindowsTray*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
  switch (msg) {
    case WM_TRAYICON:
      if (self) self->handleTrayMessage(wParam, lParam);
      return 0;
    case WM_COMMAND:
      if (self) self->handleCommand(wParam);
      return 0;
    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
    default:
      return DefWindowProc(hwnd, msg, wParam, lParam);
  }
}

bool WindowsTray::initialize() {
  HINSTANCE hInst = GetModuleHandle(nullptr);

  WNDCLASSEXW wc = {0};
  wc.cbSize = sizeof(WNDCLASSEXW);
  wc.lpfnWndProc = WindowsTray::WndProc;
  wc.hInstance = hInst;
  wc.lpszClassName = kTrayClassName;
  wc.hIcon = LoadIconW(nullptr, MAKEINTRESOURCEW(IDI_APPLICATION));
  wc.hIconSm = LoadIconW(nullptr, MAKEINTRESOURCEW(IDI_APPLICATION));
  if (!RegisterClassExW(&wc)) {
    // Already registered? try continuing
    // If registration fails because already registered, we ignore.
  }

  hwnd_ = CreateWindowExW(0, kTrayClassName, L"Duorou Tray", WS_OVERLAPPED,
                         CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
                         nullptr, nullptr, hInst, nullptr);
  if (!hwnd_) {
    std::cerr << "Failed to create tray window" << std::endl;
    return false;
  }
  SetWindowLongPtr(hwnd_, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

  // Create menu
  menu_ = CreatePopupMenu();
  if (!menu_) {
    std::cerr << "Failed to create tray menu" << std::endl;
    return false;
  }

  // Setup notify icon
  nid_.cbSize = sizeof(NOTIFYICONDATAW);
  nid_.hWnd = hwnd_;
  nid_.uID = 1;
  nid_.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
  nid_.uCallbackMessage = WM_TRAYICON;
  nid_.hIcon = LoadIconW(nullptr, MAKEINTRESOURCEW(IDI_APPLICATION));
  lstrcpynW(nid_.szTip, L"Duorou - AI Desktop Assistant", ARRAYSIZE(nid_.szTip));

  initialized_ = true;
  return true;
}

void WindowsTray::show() {
  if (!initialized_ || icon_added_) return;
  if (Shell_NotifyIconW(NIM_ADD, &nid_)) {
    icon_added_ = true;
  }
}

void WindowsTray::hide() {
  if (!initialized_ || !icon_added_) return;
  Shell_NotifyIconW(NIM_DELETE, &nid_);
  icon_added_ = false;
}

void WindowsTray::setSystemIcon() {
  if (!initialized_) return;
  nid_.hIcon = LoadIconW(nullptr, MAKEINTRESOURCEW(IDI_APPLICATION));
  Shell_NotifyIconW(icon_added_ ? NIM_MODIFY : NIM_ADD, &nid_);
  icon_added_ = true;
}

HICON WindowsTray::iconForName(const std::string &name) {
  if (name == "Error") return LoadIconW(nullptr, MAKEINTRESOURCEW(IDI_ERROR));
  if (name == "Success") return LoadIconW(nullptr, MAKEINTRESOURCEW(IDI_INFORMATION));
  if (name == "Lightning") return LoadIconW(nullptr, MAKEINTRESOURCEW(IDI_WARNING));
  if (name == "Flower") return LoadIconW(nullptr, MAKEINTRESOURCEW(IDI_APPLICATION));
  // Default
  return LoadIconW(nullptr, MAKEINTRESOURCEW(IDI_APPLICATION));
}

void WindowsTray::setIcon(const std::string &icon_name) {
  if (!initialized_) return;
  nid_.hIcon = iconForName(icon_name);
  Shell_NotifyIconW(icon_added_ ? NIM_MODIFY : NIM_ADD, &nid_);
  icon_added_ = true;
}

std::wstring WindowsTray::toWide(const std::string &s) {
  if (s.empty()) return L"";
  int count = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
  std::wstring ws(count, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &ws[0], count);
  return ws;
}

void WindowsTray::setIconFromFile(const std::string &imagePath) {
  if (!initialized_) return;
  std::wstring wpath = toWide(imagePath);
  HICON hIcon = (HICON)LoadImageW(nullptr, wpath.c_str(), IMAGE_ICON, 0, 0, LR_LOADFROMFILE);
  if (hIcon) {
    nid_.hIcon = hIcon;
    Shell_NotifyIconW(icon_added_ ? NIM_MODIFY : NIM_ADD, &nid_);
    icon_added_ = true;
  }
}

void WindowsTray::setTooltip(const std::string &tooltip) {
  if (!initialized_) return;
  std::wstring wtip = toWide(tooltip);
  lstrcpynW(nid_.szTip, wtip.c_str(), ARRAYSIZE(nid_.szTip));
  Shell_NotifyIconW(icon_added_ ? NIM_MODIFY : NIM_ADD, &nid_);
  icon_added_ = true;
}

void WindowsTray::addMenuItem(const std::string &label, std::function<void()> callback) {
  addMenuItemWithId(label, label, callback);
}

void WindowsTray::addMenuItemWithId(const std::string &item_id, const std::string &label, std::function<void()> callback) {
  MenuItemInfo info;
  info.id = item_id;
  info.label = label;
  info.callback = callback;
  info.command_id = next_command_id_++;
  info.enabled = true;
  info.visible = true;
  info.checked = false;
  info.separator = false;
  menu_items_.push_back(info);
  menu_item_map_[item_id] = &menu_items_.back();
  command_map_[info.command_id] = &menu_items_.back();
  rebuildMenu();
}

void WindowsTray::addSeparator() {
  MenuItemInfo info;
  info.id = "";
  info.label = "";
  info.callback = nullptr;
  info.command_id = 0;
  info.enabled = true;
  info.visible = true;
  info.checked = false;
  info.separator = true;
  menu_items_.push_back(info);
  rebuildMenu();
}

void WindowsTray::clearMenu() {
  if (menu_) {
    DestroyMenu(menu_);
  }
  menu_ = CreatePopupMenu();
  menu_items_.clear();
  menu_item_map_.clear();
  command_map_.clear();
}

bool WindowsTray::isAvailable() const {
  return initialized_;
}

void WindowsTray::removeMenuItem(const std::string& item_id) {
  auto it = menu_item_map_.find(item_id);
  if (it != menu_item_map_.end()) {
    UINT cmd = it->second->command_id;
    for (auto lit = menu_items_.begin(); lit != menu_items_.end(); ++lit) {
      if (lit->id == item_id) { menu_items_.erase(lit); break; }
    }
    menu_item_map_.erase(it);
    command_map_.erase(cmd);
    rebuildMenu();
  }
}

void WindowsTray::setMenuItemEnabled(const std::string& item_id, bool enabled) {
  auto it = menu_item_map_.find(item_id);
  if (it != menu_item_map_.end()) {
    it->second->enabled = enabled;
    rebuildMenu();
  }
}

bool WindowsTray::updateMenuItemLabel(const std::string& item_id, const std::string& label) {
  auto it = menu_item_map_.find(item_id);
  if (it != menu_item_map_.end()) {
    it->second->label = label;
    rebuildMenu();
    return true;
  }
  return false;
}

bool WindowsTray::hasMenuItem(const std::string& item_id) const {
  return menu_item_map_.find(item_id) != menu_item_map_.end();
}

void WindowsTray::setMenuItemVisible(const std::string& item_id, bool visible) {
  auto it = menu_item_map_.find(item_id);
  if (it != menu_item_map_.end()) {
    it->second->visible = visible;
    rebuildMenu();
  }
}

void WindowsTray::setMenuItemIcon(const std::string& item_id, const std::string& icon_name) {
  auto it = menu_item_map_.find(item_id);
  if (it != menu_item_map_.end()) {
    it->second->icon_name = icon_name;
    rebuildMenu();
  }
}

void WindowsTray::setMenuItemShortcut(const std::string& item_id, const std::string& shortcut) {
  auto it = menu_item_map_.find(item_id);
  if (it != menu_item_map_.end()) {
    it->second->shortcut = shortcut;
    rebuildMenu();
  }
}

void WindowsTray::updateWindowStateMenu(bool window_visible) {
  // Toggle visibility/enabled of show/hide menu items
  if (hasMenuItem("show_window")) {
    setMenuItemEnabled("show_window", !window_visible);
  }
  if (hasMenuItem("hide_window")) {
    setMenuItemEnabled("hide_window", window_visible);
  }
}

void WindowsTray::setLeftClickCallback(std::function<void()> callback) {
  left_click_callback_ = std::move(callback);
}

void WindowsTray::setRightClickCallback(std::function<void()> callback) {
  right_click_callback_ = std::move(callback);
}

void WindowsTray::setQuitCallback(std::function<void()> callback) {
  quit_callback_ = std::move(callback);
}

void WindowsTray::rebuildMenu() {
  if (!menu_) return;
  // Recreate menu items based on current list
  // First, empty menu
  while (GetMenuItemCount(menu_) > 0) {
    RemoveMenu(menu_, 0, MF_BYPOSITION);
  }

  for (const auto &item : menu_items_) {
    if (!item.visible) continue;
    if (item.separator) {
      AppendMenu(menu_, MF_SEPARATOR, 0, nullptr);
      continue;
    }
    std::wstring wlabel = toWide(item.label);
    UINT flags = MF_STRING | (item.enabled ? MF_ENABLED : MF_GRAYED);
    AppendMenuW(menu_, flags, item.command_id, wlabel.c_str());
  }
}

void WindowsTray::handleTrayMessage(WPARAM wParam, LPARAM lParam) {
  if (wParam != nid_.uID) return;
  switch (lParam) {
    case WM_LBUTTONUP:
    case WM_LBUTTONDBLCLK:
      if (left_click_callback_) left_click_callback_();
      break;
    case WM_RBUTTONUP:
    case WM_CONTEXTMENU: {
      if (right_click_callback_) right_click_callback_();
      POINT pt; GetCursorPos(&pt);
      SetForegroundWindow(hwnd_);
      TrackPopupMenu(menu_, TPM_RIGHTALIGN | TPM_BOTTOMALIGN, pt.x, pt.y, 0, hwnd_, nullptr);
      PostMessage(hwnd_, WM_NULL, 0, 0);
      break;
    }
    default:
      break;
  }
}

void WindowsTray::handleCommand(WPARAM wParam) {
  UINT cmd = LOWORD(wParam);
  auto it = command_map_.find(cmd);
  if (it != command_map_.end() && it->second) {
    auto &cb = it->second->callback;
    if (cb) cb();
  }
}

} // namespace duorou

#endif // _WIN32