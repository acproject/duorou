#ifdef __APPLE__

#import "macos_tray.h"
#import <Cocoa/Cocoa.h>
#import "macos_tray.h"
#include <iostream>
#include <functional>
#include <map>

// Helper class to handle menu item callbacks
@interface MenuItemTarget : NSObject {
    std::function<void()>* callback_;
}
- (instancetype)initWithCallback:(std::function<void()>*)callback;
- (void)menuItemClicked:(id)sender;
@end

@implementation MenuItemTarget

- (instancetype)initWithCallback:(std::function<void()>*)callback {
    self = [super init];
    if (self) {
        callback_ = new std::function<void()>(*callback);
    }
    return self;
}

- (void)menuItemClicked:(id)sender {
    if (callback_) {
        (*callback_)();
    }
}

- (void)dealloc {
    delete callback_;
    callback_ = nullptr;
    [super dealloc];
}

@end

// Static storage for menu item targets to prevent deallocation
static std::map<NSMenuItem*, MenuItemTarget*> menuTargets;

MacOSTray::MacOSTray() 
    : statusItem_(nullptr), menu_(nullptr), initialized_(false) {
}

MacOSTray::~MacOSTray() {
    // Remove status item
    if (statusItem_) {
        [[NSStatusBar systemStatusBar] removeStatusItem:statusItem_];
        statusItem_ = nil;
    }
    
    // 清理menuTargets中的对象
    @autoreleasepool {
        for (auto& pair : menuTargets) {
            MenuItemTarget* target = pair.second;
            if (target) {
                [target release];
            }
        }
        menuTargets.clear();
    }
}

bool MacOSTray::initialize() {
    if (initialized_) {
        return true;
    }
    
    @autoreleasepool {
        // Get the system status bar
        NSStatusBar* systemStatusBar = [NSStatusBar systemStatusBar];
        if (!systemStatusBar) {
            std::cerr << "Failed to get system status bar" << std::endl;
            return false;
        }
        
        // Create a status item with variable length
        statusItem_ = [systemStatusBar statusItemWithLength:NSVariableStatusItemLength];
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
        
        // Set default properties
        statusItem_.button.toolTip = @"Duorou";
        
        initialized_ = true;
        std::cout << "macOS system tray initialized successfully" << std::endl;
        return true;
    }
}

void MacOSTray::setIcon(const std::string& iconText) {
    if (!initialized_ || !statusItem_) {
        return;
    }
    
    @autoreleasepool {
        NSImage* image = createImageFromText(iconText);
        if (image) {
            statusItem_.button.image = image;
            // Set as template image for automatic color adaptation <mcreference link="https://crosspaste.com/en/blog/mac-system-tray" index="3">3</mcreference>
            [image setTemplate:YES];
        }
    }
}

void MacOSTray::setIconFromFile(const std::string& imagePath) {
    if (!initialized_ || !statusItem_) {
        return;
    }
    
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:imagePath.c_str()];
        NSImage* image = [[NSImage alloc] initWithContentsOfFile:path];
        
        if (image) {
            // Resize image to appropriate size for status bar <mcreference link="https://developer.apple.com/documentation/appkit/nsstatusbar" index="1">1</mcreference>
            NSSize newSize = NSMakeSize(18, 18);
            [image setSize:newSize];
            
            statusItem_.button.image = image;
            // Set as template image for automatic color adaptation
            [image setTemplate:YES];
            [image release];
        }
    }
}

void MacOSTray::setTooltip(const std::string& tooltip) {
    if (!initialized_ || !statusItem_) {
        return;
    }
    
    @autoreleasepool {
        NSString* tooltipStr = [NSString stringWithUTF8String:tooltip.c_str()];
        statusItem_.button.toolTip = tooltipStr;
    }
}

void MacOSTray::clearMenu() {
    if (!initialized_ || !menu_) {
        return;
    }
    
    @autoreleasepool {
        // Clean up existing menu targets
        for (auto& pair : menuTargets) {
            [pair.second release];
        }
        menuTargets.clear();
        
        [menu_ removeAllItems];
    }
}

void MacOSTray::addMenuItem(const std::string& title, std::function<void()> callback) {
    if (!initialized_ || !menu_) {
        return;
    }
    
    @autoreleasepool {
        NSString* titleStr = [NSString stringWithUTF8String:title.c_str()];
        NSMenuItem* menuItem = [[NSMenuItem alloc] initWithTitle:titleStr
                                                          action:@selector(menuItemClicked:)
                                                   keyEquivalent:@""];
        
        // Create target object to handle the callback
        MenuItemTarget* target = [[MenuItemTarget alloc] initWithCallback:new std::function<void()>(callback)];
        
        [menuItem setTarget:target];
        [menu_ addItem:menuItem];
        
        // 保存target的引用，不需要额外retain因为menuItem会持有target
        menuTargets[menuItem] = target;
        
        [menuItem release];
        // target会被menuItem持有，不需要在这里release
    }
}

void MacOSTray::addSeparator() {
    if (!initialized_ || !menu_) {
        return;
    }
    
    @autoreleasepool {
        NSMenuItem* separator = [NSMenuItem separatorItem];
        [menu_ addItem:separator];
    }
}

void MacOSTray::show() {
    if (!initialized_) {
        return;
    }
    
    // 如果statusItem_已存在，直接返回
    if (statusItem_) {
        return;
    }
    
    // 重新创建statusItem
    @autoreleasepool {
        NSStatusBar* systemStatusBar = [NSStatusBar systemStatusBar];
        statusItem_ = [systemStatusBar statusItemWithLength:NSVariableStatusItemLength];
        if (statusItem_) {
            [statusItem_ retain];
            statusItem_.menu = menu_;
            statusItem_.button.toolTip = @"Duorou";
        }
    }
}

void MacOSTray::hide() {
    if (!initialized_ || !statusItem_) {
        return;
    }
    
    // 移除statusItem而不是设置visible属性
    @autoreleasepool {
        [[NSStatusBar systemStatusBar] removeStatusItem:statusItem_];
        [statusItem_ release];
        statusItem_ = nil;
    }
}

bool MacOSTray::isAvailable() const {
    return initialized_ && statusItem_ != nullptr;
}

NSImage* MacOSTray::createImageFromText(const std::string& text) {
    @autoreleasepool {
        NSString* textStr = [NSString stringWithUTF8String:text.c_str()];
        
        // Create font and attributes for the text
        NSFont* font = [NSFont systemFontOfSize:16];
        NSDictionary* attributes = @{
            NSFontAttributeName: font,
            NSForegroundColorAttributeName: [NSColor blackColor]
        };
        
        // Calculate text size
        NSSize textSize = [textStr sizeWithAttributes:attributes];
        
        // Create image with appropriate size
        NSSize imageSize = NSMakeSize(18, 18);
        NSImage* image = [[NSImage alloc] initWithSize:imageSize];
        
        [image lockFocus];
        
        // Clear the background
        [[NSColor clearColor] set];
        NSRectFill(NSMakeRect(0, 0, imageSize.width, imageSize.height));
        
        // Calculate position to center the text
        NSPoint drawPoint = NSMakePoint(
            (imageSize.width - textSize.width) / 2,
            (imageSize.height - textSize.height) / 2
        );
        
        // Draw the text
        [textStr drawAtPoint:drawPoint withAttributes:attributes];
        
        [image unlockFocus];
        
        return [image autorelease];
    }
}

#endif // __APPLE__