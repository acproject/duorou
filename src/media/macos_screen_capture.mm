#import <Foundation/Foundation.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#include <iostream>
#include <functional>
#include "video_frame.h"

static std::function<void(const duorou::media::VideoFrame&)> g_frame_callback;
static SCStream* g_stream = nil;
static bool g_is_capturing = false;
static id<SCStreamDelegate> g_delegate = nil;

@interface ScreenCaptureDelegate : NSObject <SCStreamDelegate>
@end

@implementation ScreenCaptureDelegate

- (void)stream:(SCStream *)stream didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer ofType:(SCStreamOutputType)type {
    if (type != SCStreamOutputTypeScreen) {
        return;
    }
    
    if (!g_frame_callback) {
        std::cout << "ScreenCaptureKit: 没有回调函数" << std::endl;
        return;
    }
    
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (!imageBuffer) {
        std::cout << "ScreenCaptureKit: 无法获取图像缓冲区" << std::endl;
        return;
    }
    
    CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    void* baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
    
    duorou::media::VideoFrame frame;
    frame.width = static_cast<int>(width);
    frame.height = static_cast<int>(height);
    frame.channels = 4; // BGRA
    frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    
    size_t dataSize = height * bytesPerRow;
    frame.data.resize(dataSize);
    std::memcpy(frame.data.data(), baseAddress, dataSize);
    
    CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    
    std::cout << "ScreenCaptureKit: 收到真实屏幕帧 " << width << "x" << height << std::endl;
    g_frame_callback(frame);
}

- (void)stream:(SCStream *)stream didStopWithError:(NSError *)error {
    if (error) {
        std::cout << "ScreenCaptureKit stream stopped with error: " << [[error localizedDescription] UTF8String] << std::endl;
    } else {
        std::cout << "ScreenCaptureKit stream stopped" << std::endl;
    }
    g_is_capturing = false;
}

@end

namespace duorou {
namespace media {

bool initialize_macos_screen_capture() {
    if (@available(macOS 12.3, *)) {
        g_delegate = [[ScreenCaptureDelegate alloc] init];
        std::cout << "ScreenCaptureKit 初始化成功" << std::endl;
        return true;
    } else {
        std::cout << "ScreenCaptureKit 需要 macOS 12.3 或更高版本" << std::endl;
        return false;
    }
}

bool start_macos_screen_capture(std::function<void(const VideoFrame&)> callback) {
    if (@available(macOS 12.3, *)) {
        if (g_is_capturing) {
            std::cout << "ScreenCaptureKit 已在运行" << std::endl;
            return true;
        }
        
        std::cout << "开始启动 ScreenCaptureKit..." << std::endl;
        g_frame_callback = callback;
        
        // 获取可用的显示器
        [SCShareableContent getShareableContentWithCompletionHandler:^(SCShareableContent * _Nullable content, NSError * _Nullable error) {
            if (error) {
                std::cout << "获取可共享内容失败: " << [[error localizedDescription] UTF8String] << std::endl;
                return;
            }
            
            if (content.displays.count == 0) {
                std::cout << "没有找到可用的显示器" << std::endl;
                return;
            }
            
            std::cout << "找到 " << content.displays.count << " 个显示器" << std::endl;
            SCDisplay* display = content.displays.firstObject;
            SCContentFilter* filter = [[SCContentFilter alloc] initWithDisplay:display excludingWindows:@[]];
            
            SCStreamConfiguration* config = [[SCStreamConfiguration alloc] init];
            config.width = 1920;
            config.height = 1080;
            config.minimumFrameInterval = CMTimeMake(1, 30); // 30 FPS
            config.pixelFormat = kCVPixelFormatType_32BGRA;
            
            std::cout << "创建 ScreenCaptureKit 流..." << std::endl;
            g_stream = [[SCStream alloc] initWithFilter:filter configuration:config delegate:g_delegate];
            
            [g_stream startCaptureWithCompletionHandler:^(NSError * _Nullable error) {
                if (error) {
                    std::cout << "启动 ScreenCaptureKit 失败: " << [[error localizedDescription] UTF8String] << std::endl;
                    g_is_capturing = false;
                } else {
                    std::cout << "ScreenCaptureKit 启动成功，开始接收屏幕数据" << std::endl;
                    g_is_capturing = true;
                }
            }];
        }];
        
        return true;
    } else {
        std::cout << "ScreenCaptureKit 需要 macOS 12.3 或更高版本" << std::endl;
        return false;
    }
}

void stop_macos_screen_capture() {
    if (@available(macOS 12.3, *)) {
        if (g_stream && g_is_capturing) {
            g_is_capturing = false; // 立即设置为false避免重复调用
            [g_stream stopCaptureWithCompletionHandler:^(NSError * _Nullable error) {
                if (error) {
                    std::cout << "停止 ScreenCaptureKit 时出错: " << [[error localizedDescription] UTF8String] << std::endl;
                } else {
                    std::cout << "ScreenCaptureKit 已停止" << std::endl;
                }
            }];
        }
    }
}

bool is_macos_screen_capture_running() {
    return g_is_capturing;
}

void cleanup_macos_screen_capture() {
    if (@available(macOS 12.3, *)) {
        // 确保停止捕获
        if (g_stream && g_is_capturing) {
            g_is_capturing = false;
            [g_stream stopCaptureWithCompletionHandler:^(NSError * _Nullable error) {
                // 在完成回调中清理资源
                dispatch_async(dispatch_get_main_queue(), ^{
                    g_stream = nil;
                });
            }];
        } else {
            g_stream = nil;
        }
        
        // 清理其他资源
        g_delegate = nil;
        g_frame_callback = nullptr;
    }
}

} // namespace media
} // namespace duorou