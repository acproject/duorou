#ifndef MACOS_TRAY_H
#define MACOS_TRAY_H

#ifdef __APPLE__

#include "../gui/system_tray.h"
#include <string>
#include <functional>

#ifdef __OBJC__
@class NSStatusItem;
@class NSMenu;
@class NSImage;
#else
typedef void NSStatusItem;
typedef void NSMenu;
typedef void NSImage;
#endif

namespace duorou {

class MacOSTray : public gui::SystemTray {
public:
    MacOSTray();
    ~MacOSTray();

    bool initialize();
    void show();
    void hide();
    void setIcon(const std::string& icon_path);
    void setIconFromFile(const std::string& imagePath);
    void setTooltip(const std::string& tooltip);
    void addMenuItem(const std::string& label, std::function<void()> callback);
    void addSeparator();
    void clearMenu();
    void setSystemIcon();
    bool isAvailable() const;

private:
    NSStatusItem* statusItem_;
    NSMenu* menu_;
    bool initialized_;
    
    NSImage* createImageFromText(const std::string& text);
    void setupMenu();
};

} // namespace duorou

#endif // __APPLE__

#endif // MACOS_TRAY_H