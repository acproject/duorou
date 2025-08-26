#ifndef MACOS_TRAY_H
#define MACOS_TRAY_H

#ifdef __APPLE__

#include <functional>
#include <string>

// Forward declarations for Objective-C types
#ifdef __OBJC__
@class NSStatusItem;
@class NSMenu;
@class NSMenuItem;
@class NSImage;
#else
typedef struct objc_object NSStatusItem;
typedef struct objc_object NSMenu;
typedef struct objc_object NSMenuItem;
typedef struct objc_object NSImage;
#endif

class MacOSTray {
public:
    MacOSTray();
    ~MacOSTray();
    
    // Initialize the system tray
    bool initialize();
    
    // Set the tray icon (using emoji or image path)
    void setIcon(const std::string& iconText);
    void setIconFromFile(const std::string& imagePath);
    
    // Set tooltip text
    void setTooltip(const std::string& tooltip);
    
    // Menu management
    void clearMenu();
    void addMenuItem(const std::string& title, std::function<void()> callback);
    void addSeparator();
    
    // Show/hide the tray icon
    void show();
    void hide();
    
    // Check if tray is available
    bool isAvailable() const;
    
private:
    NSStatusItem* statusItem_;
    NSMenu* menu_;
    bool initialized_;
    
    // Helper methods
    NSImage* createImageFromText(const std::string& text);
    void setupMenu();
};

#endif // __APPLE__

#endif // MACOS_TRAY_H