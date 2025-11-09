#ifdef __APPLE__

#import "macos_tray.h"
#import <Cocoa/Cocoa.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

// Try to load a custom status bar icon named
// "seo_page_browser_web_window_view_icon" from bundle resources or project paths.
static NSImage *LoadCustomStatusIcon() {
  @autoreleasepool {
    // Preferred search order: ICNS -> PNG -> ICO
    // Locations: app bundle resources -> project relative paths
    NSArray<NSString *> *extensions = @[ @"icns", @"png", @"ico" ];

    // 1) Search in the app bundle resources
    NSBundle *bundle = [NSBundle mainBundle];
    if (bundle) {
      for (NSString *ext in extensions) {
        NSString *resPath = [bundle pathForResource:@"seo_page_browser_web_window_view_icon" ofType:ext];
        if (resPath != nil) {
          NSImage *img = [[NSImage alloc] initWithContentsOfFile:resPath];
          if (img) {
            // Return retained image; caller will release after assignment
            return img;
          }
        }
      }
    }

    // 2) Fallback: search relative to current working directory
    NSString *cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    NSArray<NSString *> *relPaths = @[
      @"src/gui/seo_page_browser_web_window_view_icon.icns",
      @"src/gui/seo_page_browser_web_window_view_icon.png",
      @"src/gui/seo_page_browser_web_window_view_icon.ico"
    ];
    for (NSString *rel in relPaths) {
      NSString *full = [cwd stringByAppendingPathComponent:rel];
      if ([[NSFileManager defaultManager] fileExistsAtPath:full]) {
        NSImage *img = [[NSImage alloc] initWithContentsOfFile:full];
        if (img) {
          return img;
        }
      }
    }

    return nil;
  }
}

// Helper class to handle menu item callbacks
@interface MenuItemTarget : NSObject {
  std::function<void()> callback_;
}
- (instancetype)initWithCallback:(const std::function<void()> &)callback;
- (void)menuItemClicked:(id)sender;
@end

@implementation MenuItemTarget

- (instancetype)initWithCallback:(const std::function<void()> &)callback {
  self = [super init];
  if (self) {
    callback_ = callback;
  }
  return self;
}

- (void)menuItemClicked:(id)sender {
  if (callback_) {
    callback_();
  }
}

- (void)dealloc {
  [super dealloc];
}

@end

duorou::MacOSTray::MacOSTray()
    : statusItem_(nullptr), menu_(nullptr), initialized_(false) {}

duorou::MacOSTray::~MacOSTray() {
  @autoreleasepool {
    // Release all menuItems
    for (auto &info : menu_items_) {
      if (info.menuItem) {
        [info.menuItem release];
      }
      // If a strong-associated target is stored via representedObject,
      // it will be released when menuItem is released. Nothing else needed.
    }
    menu_items_.clear();
    menu_item_map_.clear();

    // Clean up status item
    if (statusItem_) {
      [[NSStatusBar systemStatusBar] removeStatusItem:statusItem_];
      [statusItem_ release];
      statusItem_ = nil;
    }

    // Clean up menu_ (menu items will be automatically released)
    if (menu_) {
      [menu_ release];
      menu_ = nil;
    }
  }
}

bool duorou::MacOSTray::initialize() {
  if (initialized_) {
    return true;
  }

  @autoreleasepool {
    // Get the system status bar
    NSStatusBar *systemStatusBar = [NSStatusBar systemStatusBar];
    if (!systemStatusBar) {
      std::cerr << "Failed to get system status bar" << std::endl;
      return false;
    }

    // Create a status item with variable length
    statusItem_ =
        [systemStatusBar statusItemWithLength:NSVariableStatusItemLength];
    if (!statusItem_) {
      std::cerr << "Failed to create status item" << std::endl;
      return false;
    }

    [statusItem_ retain];

    // Create the menu
    menu_ = [[NSMenu alloc] init];
    [menu_ setAutoenablesItems:NO];

    // Set up the status item
    statusItem_.menu = menu_;
    std::cout << "[MacOSTray] statusItem_=" << statusItem_ << " menu_=" << menu_ << std::endl;

    // Check if button is available
    if (!statusItem_.button) {
      std::cerr << "Failed to get status item button" << std::endl;
      [[NSStatusBar systemStatusBar] removeStatusItem:statusItem_];
      [statusItem_ release];
      statusItem_ = nil;
      [menu_ release];
      menu_ = nil;
      return false;
    }
    std::cout << "[MacOSTray] button=" << statusItem_.button << std::endl;

    // Set default properties
    statusItem_.button.toolTip = @"Duorou";

    initialized_ = true;
    std::cout << "macOS system tray initialized successfully" << std::endl;

    // Skip icon setting for now to avoid crashes
    // setSystemIcon();

    return true;
  }
}

void duorou::MacOSTray::setIcon(const std::string &iconText) {
  if (!initialized_ || !statusItem_ || !statusItem_.button) {
    std::cerr << "setIcon failed: statusItem or button is nil" << std::endl;
    return;
  }

  @autoreleasepool {
    // Treat parameter as file path first; fallback to text rendering
    NSString *path = [NSString stringWithUTF8String:iconText.c_str()];
    NSImage *fileImg = [[NSImage alloc] initWithContentsOfFile:path];
    if (fileImg) {
      [fileImg setSize:NSMakeSize(18, 18)];
      [fileImg setTemplate:YES];
      statusItem_.button.image = fileImg;
      [fileImg release];
      return;
    }

    NSImage *textImg = createImageFromText(iconText);
    if (textImg && [textImg isKindOfClass:[NSImage class]]) {
      [textImg setTemplate:YES];
      statusItem_.button.image = textImg;
    } else {
      std::cerr << "Failed to load icon from path or text: " << iconText
                << std::endl;
    }
  }
}

void duorou::MacOSTray::setSystemIcon() {
  if (!initialized_ || !statusItem_ || !statusItem_.button) {
    std::cerr << "setSystemIcon failed: statusItem or button is nil"
              << std::endl;
    return;
  }

  @autoreleasepool {
    std::cout << "[MacOSTray] setSystemIcon start" << std::endl;
    NSStatusBarButton *btn = statusItem_.button;
    if (!btn) {
      std::cerr << "[MacOSTray] button unexpectedly nil" << std::endl;
      return;
    }
    std::cout << "[MacOSTray] button class = "
              << [[btn className] UTF8String] << std::endl;

    // First, attempt to use the custom icon if present
    NSImage *custom = LoadCustomStatusIcon();
    if (custom) {
      std::cout << "[MacOSTray] using custom icon" << std::endl;
      // Standard status bar icon size is 18×18 points
      [custom setSize:NSMakeSize(18, 18)];
      // Template image lets macOS tint for light/dark mode automatically
      [custom setTemplate:YES];
      btn.image = custom;
      [btn setImageScaling:NSImageScaleProportionallyDown];
      [btn setImagePosition:NSImageOnly];
      [custom release];
      std::cout << "[MacOSTray] custom icon applied (template+18x18)" << std::endl;
      return;
    }

    // Final fallback: draw a simple circle (avoid SF Symbols to be safe)
    std::cout << "[MacOSTray] using fallback circle icon" << std::endl;
    NSImage *fallbackImage = [[NSImage alloc] initWithSize:NSMakeSize(16, 16)];
    [fallbackImage lockFocus];
    [[NSColor systemBlueColor] set];
    NSBezierPath *circle =
        [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(2, 2, 12, 12)];
    [circle fill];
    [fallbackImage unlockFocus];
    [fallbackImage setTemplate:YES];
    btn.image = fallbackImage;
    std::cout << "[MacOSTray] fallback icon applied" << std::endl;
    [fallbackImage release];
  }
}

void duorou::MacOSTray::setIconFromFile(const std::string &imagePath) {
  if (!initialized_ || !statusItem_) {
    return;
  }

  @autoreleasepool {
    NSString *path = [NSString stringWithUTF8String:imagePath.c_str()];
    NSImage *image = [[NSImage alloc] initWithContentsOfFile:path];

    if (image) {
      // Resize image to appropriate size for status bar <mcreference
      // link="https://developer.apple.com/documentation/appkit/nsstatusbar"
      // index="1">1</mcreference>
      NSSize newSize = NSMakeSize(18, 18);
      [image setSize:newSize];

      statusItem_.button.image = image;
      // Set as template image for automatic color adaptation
      [image setTemplate:YES];
      [image release];
    }
  }
}

void duorou::MacOSTray::setTooltip(const std::string &tooltip) {
  if (!initialized_ || !statusItem_) {
    return;
  }

  @autoreleasepool {
    NSString *tooltipStr = [NSString stringWithUTF8String:tooltip.c_str()];
    statusItem_.button.toolTip = tooltipStr;
  }
}

void duorou::MacOSTray::clearMenu() {
  if (!initialized_ || !menu_) {
    return;
  }

  @autoreleasepool {
    // Remove all menu items (targets will be automatically released)
    [menu_ removeAllItems];
  }
}

void duorou::MacOSTray::addMenuItem(const std::string &title,
                                    std::function<void()> callback) {
  if (!initialized_ || !menu_) {
    return;
  }

  @autoreleasepool {
    NSString *titleStr = [NSString stringWithUTF8String:title.c_str()];
    NSMenuItem *menuItem =
        [[NSMenuItem alloc] initWithTitle:titleStr
                                   action:@selector(menuItemClicked:)
                            keyEquivalent:@""];

    // Create target object to handle the callback
    MenuItemTarget *target = [[MenuItemTarget alloc] initWithCallback:callback];

    [menuItem setTarget:target];
    // Ensure strong association so target isn't deallocated prematurely
    [menuItem setRepresentedObject:target];
    [menu_ addItem:menuItem];

    // representedObject is retained by NSMenuItem; safe to release locals
    [target release];
    [menuItem release];
  }
}

void duorou::MacOSTray::addSeparator() {
  if (!initialized_ || !menu_) {
    return;
  }

  @autoreleasepool {
    NSMenuItem *separator = [NSMenuItem separatorItem];
    [menu_ addItem:separator];
  }
}

void duorou::MacOSTray::show() {
  if (!initialized_) {
    return;
  }

  // If statusItem_ already exists, return directly
  if (statusItem_) {
    return;
  }

  // Recreate statusItem
  @autoreleasepool {
    NSStatusBar *systemStatusBar = [NSStatusBar systemStatusBar];
    statusItem_ =
        [systemStatusBar statusItemWithLength:NSVariableStatusItemLength];
    if (statusItem_) {
      [statusItem_ retain];
      statusItem_.menu = menu_;
      statusItem_.button.toolTip = @"Duorou";
      // Set flower emoji as icon
      setSystemIcon();
    }
  }
}

void duorou::MacOSTray::hide() {
  if (!initialized_ || !statusItem_) {
    return;
  }

  @autoreleasepool {
    [[NSStatusBar systemStatusBar] removeStatusItem:statusItem_];
    [statusItem_ release]; // Release the retained statusItem_
    statusItem_ = nil;
  }
}

bool duorou::MacOSTray::isAvailable() const {
  return initialized_ && statusItem_ != nullptr;
}

// 增强的菜单管理功能实现
void duorou::MacOSTray::removeMenuItem(const std::string &item_id) {
  if (!initialized_ || !menu_) {
    return;
  }

  auto it = std::find_if(
      menu_items_.begin(), menu_items_.end(),
      [&item_id](const MenuItemInfo &info) { return info.id == item_id; });

  if (it != menu_items_.end()) {
    @autoreleasepool {
      if (it->menuItem) {
        [menu_ removeItem:it->menuItem];
        [it->menuItem release]; // Release menuItem
      }
    }
    menu_item_map_.erase(item_id);
    menu_items_.erase(it);
  }
}

void duorou::MacOSTray::setMenuItemEnabled(const std::string &item_id,
                                           bool enabled) {
  auto *info = findMenuItemInfo(item_id);
  if (info && info->menuItem) {
    info->enabled = enabled;
    @autoreleasepool {
      [info->menuItem setEnabled:enabled];
    }
  }
}

bool duorou::MacOSTray::updateMenuItemLabel(const std::string &item_id,
                                            const std::string &label) {
  auto *info = findMenuItemInfo(item_id);
  if (info && info->menuItem) {
    info->label = label;
    @autoreleasepool {
      NSString *labelStr = [NSString stringWithUTF8String:label.c_str()];
      [info->menuItem setTitle:labelStr];
    }
    return true;
  }
  return false;
}

bool duorou::MacOSTray::hasMenuItem(const std::string &item_id) const {
  return menu_item_map_.find(item_id) != menu_item_map_.end();
}

void duorou::MacOSTray::addMenuItemWithId(const std::string &item_id,
                                          const std::string &label,
                                          std::function<void()> callback) {
  if (!initialized_ || !menu_) {
    return;
  }

  // Check if menu item with same ID already exists
  if (hasMenuItem(item_id)) {
    std::cerr << "Menu item with ID '" << item_id << "' already exists"
              << std::endl;
    return;
  }

  MenuItemInfo info;
  info.id = item_id;
  info.label = label;
  info.callback = callback;
  info.enabled = true;
  info.visible = true;
  info.submenu = nullptr;

  @autoreleasepool {
    info.menuItem = createNSMenuItem(info);
    [info.menuItem retain]; // Keep reference to menuItem
    [menu_ addItem:info.menuItem];
  }

  // Use list, pointers won't become invalid due to container reallocation
  menu_items_.push_back(info);
  menu_item_map_[item_id] = &menu_items_.back();
}

void duorou::MacOSTray::setMenuItemVisible(const std::string &item_id,
                                           bool visible) {
  auto *info = findMenuItemInfo(item_id);
  if (info && info->menuItem) {
    info->visible = visible;
    @autoreleasepool {
      [info->menuItem setHidden:!visible];
    }
  }
}

void duorou::MacOSTray::setMenuItemIcon(const std::string &item_id,
                                        const std::string &icon_name) {
  auto *info = findMenuItemInfo(item_id);
  if (info && info->menuItem) {
    info->icon_name = icon_name;
    @autoreleasepool {
      NSImage *icon = [NSImage
          imageNamed:[NSString stringWithUTF8String:icon_name.c_str()]];
      if (icon) {
        [info->menuItem setImage:icon];
      }
    }
  }
}

void duorou::MacOSTray::setMenuItemShortcut(const std::string &item_id,
                                            const std::string &shortcut) {
  auto *info = findMenuItemInfo(item_id);
  if (info && info->menuItem) {
    info->shortcut = shortcut;
    @autoreleasepool {
      NSString *shortcutStr = [NSString stringWithUTF8String:shortcut.c_str()];
      [info->menuItem setKeyEquivalent:shortcutStr];
    }
  }
}

// 辅助方法实现
duorou::MacOSTray::MenuItemInfo *
duorou::MacOSTray::findMenuItemInfo(const std::string &item_id) {
  auto it = menu_item_map_.find(item_id);
  return (it != menu_item_map_.end()) ? it->second : nullptr;
}

NSMenuItem *duorou::MacOSTray::createNSMenuItem(const MenuItemInfo &info) {
  @autoreleasepool {
    NSString *titleStr = [NSString stringWithUTF8String:info.label.c_str()];
    NSMenuItem *menuItem =
        [[NSMenuItem alloc] initWithTitle:titleStr
                                   action:@selector(menuItemClicked:)
                            keyEquivalent:@""];

    // Create target object to handle callback
    if (info.callback) {
      MenuItemTarget *target =
          [[MenuItemTarget alloc] initWithCallback:info.callback];
      [menuItem setTarget:target];
      // Strongly associate to the menu item to prevent dangling target
      [menuItem setRepresentedObject:target];
      [target release];
    }

    [menuItem setEnabled:info.enabled];
    [menuItem setHidden:!info.visible];

    // 设置快捷键
    if (!info.shortcut.empty()) {
      NSString *shortcutStr =
          [NSString stringWithUTF8String:info.shortcut.c_str()];
      [menuItem setKeyEquivalent:shortcutStr];
    }

    // 设置图标
    if (!info.icon_name.empty()) {
      NSImage *icon = [NSImage
          imageNamed:[NSString stringWithUTF8String:info.icon_name.c_str()]];
      if (icon) {
        [menuItem setImage:icon];
      }
    }

    return menuItem; // 不要autorelease，让调用者管理
  }
}

void duorou::MacOSTray::updateWindowStateMenu(bool window_visible) {
  if (!initialized_ || !menu_) {
    return;
  }

  // 根据窗口状态更新显示/隐藏菜单项的可见性
  // 只有在菜单项存在时才更新
  if (hasMenuItem("show_window")) {
    setMenuItemVisible("show_window",
                       !window_visible); // 窗口隐藏时显示"显示窗口"
  }
  if (hasMenuItem("hide_window")) {
    setMenuItemVisible("hide_window",
                       window_visible); // 窗口显示时显示"隐藏窗口"
  }
}

void duorou::MacOSTray::setLeftClickCallback(std::function<void()> callback) {
  left_click_callback_ = callback;
}

void duorou::MacOSTray::setRightClickCallback(std::function<void()> callback) {
  right_click_callback_ = callback;
}

void duorou::MacOSTray::setQuitCallback(std::function<void()> callback) {
  std::cout << "[MacOSTray] Setting quit callback" << std::endl;
  quit_callback_ = callback;

  // 如果已经有quit菜单项，更新其回调函数
  if (hasMenuItem("quit")) {
    std::cout << "[MacOSTray] Found existing quit menu item, updating callback"
              << std::endl;
    auto *info = findMenuItemInfo("quit");
    if (info) {
      info->callback = [this]() {
        std::cout << "[MacOSTray] Quit menu item clicked!" << std::endl;
        if (quit_callback_) {
          std::cout << "[MacOSTray] Calling quit callback function"
                    << std::endl;
          quit_callback_();
        } else {
          std::cout << "[MacOSTray] Warning: quit_callback_ is null"
                    << std::endl;
        }
      };

      // 更新NSMenuItem的target
      @autoreleasepool {
        if (info->menuItem) {
          MenuItemTarget *target = [[MenuItemTarget alloc] initWithCallback:info->callback];
          [info->menuItem setTarget:target];
          [info->menuItem setRepresentedObject:target];
          [target release];
          std::cout << "[MacOSTray] Updated NSMenuItem target for quit menu" << std::endl;
        }
      }
    }
  } else {
    std::cout << "[MacOSTray] Warning: quit menu item not found" << std::endl;
  }
}

NSImage *duorou::MacOSTray::createImageFromText(const std::string &text) {
  @autoreleasepool {
    NSString *textStr = [NSString stringWithUTF8String:text.c_str()];

    // Create a simple text-based image
    NSFont *font = [NSFont systemFontOfSize:12];
    NSDictionary *attributes = @{
      NSFontAttributeName : font,
      NSForegroundColorAttributeName : [NSColor blackColor]
    };

    NSSize textSize = [textStr sizeWithAttributes:attributes];
    NSSize imageSize = NSMakeSize(textSize.width + 4, textSize.height + 4);

    NSImage *image = [[NSImage alloc] initWithSize:imageSize];
    [image lockFocus];

    // Clear background
    [[NSColor clearColor] set];
    NSRectFill(NSMakeRect(0, 0, imageSize.width, imageSize.height));

    // Draw text
    [textStr drawAtPoint:NSMakePoint(2, 2) withAttributes:attributes];

    [image unlockFocus];

    return [image autorelease];
  }
}

#endif // __APPLE__