#ifdef __APPLE__

#import "macos_tray.h"
#import <Cocoa/Cocoa.h>
#import "macos_tray.h"
#include <iostream>
#include <functional>
#include <map>

// Helper class to handle menu item callbacks
@interface MenuItemTarget : NSObject {
    std::function<void()> callback_;
}
- (instancetype)initWithCallback:(const std::function<void()>&)callback;
- (void)menuItemClicked:(id)sender;
@end

@implementation MenuItemTarget

- (instancetype)initWithCallback:(const std::function<void()>&)callback {
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
    : statusItem_(nullptr), menu_(nullptr), initialized_(false) {
}

duorou::MacOSTray::~MacOSTray() {
    @autoreleasepool {
        // Remove status item first
        if (statusItem_) {
            [[NSStatusBar systemStatusBar] removeStatusItem:statusItem_];
            [statusItem_ release];
            statusItem_ = nil;
        }
        
        // 清理menu_ (menu items will be automatically released)
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
        
        // Set default properties
        statusItem_.button.toolTip = @"Duorou";
        
        initialized_ = true;
        std::cout << "macOS system tray initialized successfully" << std::endl;
        
        // Skip icon setting for now to avoid crashes
        // setSystemIcon();
        
        return true;
    }
}

void duorou::MacOSTray::setIcon(const std::string& iconText) {
    if (!initialized_ || !statusItem_ || !statusItem_.button) {
        std::cerr << "setIcon failed: statusItem or button is nil" << std::endl;
        return;
    }
    
    @autoreleasepool {
        NSImage* image = createImageFromText(iconText);
        if (image && [image isKindOfClass:[NSImage class]]) {
            // Set as template image before assigning
            [image setTemplate:YES];
            
            // Assign to button (button will retain the image)
            statusItem_.button.image = image;
        } else {
            std::cerr << "Failed to create image from text: " << iconText << std::endl;
        }
    }
}

void duorou::MacOSTray::setSystemIcon() {
    if (!initialized_ || !statusItem_ || !statusItem_.button) {
        std::cerr << "setSystemIcon failed: statusItem or button is nil" << std::endl;
        return;
    }
    
    @autoreleasepool {
        // Use tree.circle system icon
        NSImage* image = [NSImage imageWithSystemSymbolName:@"tree.circle" accessibilityDescription:nil];
        if (image) {
            [image setTemplate:YES];
            statusItem_.button.image = image;
        } else {
            // Fallback: create a simple colored circle
            NSImage* fallbackImage = [[NSImage alloc] initWithSize:NSMakeSize(16, 16)];
            [fallbackImage lockFocus];
            [[NSColor systemBlueColor] set];
            NSBezierPath* circle = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(2, 2, 12, 12)];
            [circle fill];
            [fallbackImage unlockFocus];
            [fallbackImage setTemplate:YES];
            statusItem_.button.image = fallbackImage;
            [fallbackImage release];
        }
    }
}

void duorou::MacOSTray::setIconFromFile(const std::string& imagePath) {
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

void duorou::MacOSTray::setTooltip(const std::string& tooltip) {
    if (!initialized_ || !statusItem_) {
        return;
    }
    
    @autoreleasepool {
        NSString* tooltipStr = [NSString stringWithUTF8String:tooltip.c_str()];
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

void duorou::MacOSTray::addMenuItem(const std::string& title, std::function<void()> callback) {
    if (!initialized_ || !menu_) {
        return;
    }
    
    @autoreleasepool {
        NSString* titleStr = [NSString stringWithUTF8String:title.c_str()];
        NSMenuItem* menuItem = [[NSMenuItem alloc] initWithTitle:titleStr
                                                          action:@selector(menuItemClicked:)
                                                   keyEquivalent:@""];
        
        // Create target object to handle the callback
        MenuItemTarget* target = [[MenuItemTarget alloc] initWithCallback:callback];
        
        [menuItem setTarget:target];
        [menu_ addItem:menuItem];
        
        [menuItem release];
        [target release]; // menuItem will retain target
    }
}

void duorou::MacOSTray::addSeparator() {
    if (!initialized_ || !menu_) {
        return;
    }
    
    @autoreleasepool {
        NSMenuItem* separator = [NSMenuItem separatorItem];
        [menu_ addItem:separator];
    }
}

void duorou::MacOSTray::show() {
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
            // Set flower emoji as icon
            setIcon("🌸");
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

NSImage* duorou::MacOSTray::createImageFromText(const std::string& text) {
    @autoreleasepool {
        NSString* textStr = [NSString stringWithUTF8String:text.c_str()];
        
        // Create a simple text-based image
        NSFont* font = [NSFont systemFontOfSize:12];
        NSDictionary* attributes = @{
            NSFontAttributeName: font,
            NSForegroundColorAttributeName: [NSColor blackColor]
        };
        
        NSSize textSize = [textStr sizeWithAttributes:attributes];
        NSSize imageSize = NSMakeSize(textSize.width + 4, textSize.height + 4);
        
        NSImage* image = [[NSImage alloc] initWithSize:imageSize];
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