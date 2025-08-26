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

@interface ScreenCaptureDelegate : NSObject <SCStreamDelegate, SCStreamOutput>
@end

@implementation ScreenCaptureDelegate

- (void)stream:(SCStream *)stream didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer ofType:(SCStreamOutputType)type {
    if (type != SCStreamOutputTypeScreen) {
        return;
    }
    
    // 检查是否仍在捕获状态
    if (!g_is_capturing || !g_frame_callback) {
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
    
    // 视频帧处理在后台线程完成，但UI更新需要切换到主线程
    // 复制frame数据避免线程安全问题
    auto frameCopy = std::make_shared<duorou::media::VideoFrame>(std::move(frame));
    
    dispatch_async(dispatch_get_main_queue(), ^{
        if (g_frame_callback && g_is_capturing) {
            g_frame_callback(*frameCopy);
        }
    });
}

- (void)stream:(SCStream *)stream didStopWithError:(NSError *)error {
    if (error) {
        std::cout << "ScreenCaptureKit stream stopped with error: " << [[error localizedDescription] UTF8String] << std::endl;
    } else {
        std::cout << "ScreenCaptureKit stream stopped" << std::endl;
    }
    g_is_capturing = false;
}

// SCStreamOutput和SCStreamDelegate共享同一个方法实现

@end

namespace duorou {
namespace media {

bool check_screen_recording_permission() {
    if (@available(macOS 12.3, *)) {
        // 尝试获取可共享内容来检查权限
        __block bool permission_granted = false;
        __block bool check_completed = false;
        
        [SCShareableContent getShareableContentWithCompletionHandler:^(SCShareableContent * _Nullable content, NSError * _Nullable error) {
            if (error) {
                std::cout << "屏幕录制权限检查失败: " << [[error localizedDescription] UTF8String] << std::endl;
                permission_granted = false;
            } else if (content.displays.count > 0) {
                std::cout << "屏幕录制权限检查通过" << std::endl;
                permission_granted = true;
            } else {
                std::cout << "没有找到可用的显示器，可能权限不足" << std::endl;
                permission_granted = false;
            }
            check_completed = true;
        }];
        
        // 等待权限检查完成（最多等待2秒）
        int wait_count = 0;
        while (!check_completed && wait_count < 200) {
            [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
            wait_count++;
        }
        
        return permission_granted;
    } else {
        std::cout << "ScreenCaptureKit 需要 macOS 12.3 或更高版本" << std::endl;
        return false;
    }
}

bool initialize_macos_screen_capture() {
    if (@available(macOS 12.3, *)) {
        // 先检查权限
        if (!check_screen_recording_permission()) {
            std::cout << "ScreenCaptureKit 权限检查失败，请在系统偏好设置 > 安全性与隐私 > 隐私 > 屏幕录制中允许此应用" << std::endl;
            return false;
        }
        
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
        
        // 确保清理之前的stream
        if (g_stream) {
            std::cout << "清理之前的 ScreenCaptureKit 流..." << std::endl;
            g_stream = nil;
        }
        
        std::cout << "开始启动 ScreenCaptureKit..." << std::endl;
        g_frame_callback = callback;
        
        // 确保在主线程上执行ScreenCaptureKit操作
        dispatch_async(dispatch_get_main_queue(), ^{
            @try {
                // 获取可用的显示器
                std::cout << "正在获取可共享内容..." << std::endl;
                [SCShareableContent getShareableContentWithCompletionHandler:^(SCShareableContent * _Nullable content, NSError * _Nullable error) {
                    @try {
                        std::cout << "getShareableContentWithCompletionHandler 回调被调用" << std::endl;
                        if (error) {
                            std::cout << "获取可共享内容失败: " << [[error localizedDescription] UTF8String] << std::endl;
                            std::cout << "错误代码: " << error.code << std::endl;
                            std::cout << "错误域: " << [error.domain UTF8String] << std::endl;
                            g_is_capturing = false;
                            return;
                        }
                        
                        if (!content || content.displays.count == 0) {
                            std::cout << "没有找到可用的显示器" << std::endl;
                            g_is_capturing = false;
                            return;
                        }
                        
                        std::cout << "找到 " << content.displays.count << " 个显示器" << std::endl;
                        SCDisplay* display = content.displays.firstObject;
                        if (!display) {
                            std::cout << "无法获取第一个显示器" << std::endl;
                            g_is_capturing = false;
                            return;
                        }
                        
                        std::cout << "使用显示器: " << display.displayID << ", 尺寸: " << display.width << "x" << display.height << std::endl;
                        
                        SCContentFilter* filter = [[SCContentFilter alloc] initWithDisplay:display excludingWindows:@[]];
                        if (!filter) {
                            std::cout << "创建内容过滤器失败" << std::endl;
                            g_is_capturing = false;
                            return;
                        }
                        std::cout << "创建内容过滤器成功" << std::endl;
                        
                        SCStreamConfiguration* config = [[SCStreamConfiguration alloc] init];
                        config.width = 1920;
                        config.height = 1080;
                        config.minimumFrameInterval = CMTimeMake(1, 30); // 30 FPS
                        config.pixelFormat = kCVPixelFormatType_32BGRA;
                        std::cout << "创建流配置成功" << std::endl;
                        
                        std::cout << "创建 ScreenCaptureKit 流..." << std::endl;
                        g_stream = [[SCStream alloc] initWithFilter:filter configuration:config delegate:g_delegate];
                        if (!g_stream) {
                            std::cout << "创建 ScreenCaptureKit 流失败" << std::endl;
                            g_is_capturing = false;
                            return;
                        }
                        std::cout << "ScreenCaptureKit 流创建成功" << std::endl;
                        
                        // 创建专门的后台队列处理视频帧，避免阻塞主线程
                        dispatch_queue_t videoQueue = dispatch_queue_create("com.duorou.video_capture", DISPATCH_QUEUE_SERIAL);
                        
                        // 添加stream output以接收视频数据
                        NSError* outputError = nil;
                        BOOL outputAdded = [g_stream addStreamOutput:(id<SCStreamOutput>)g_delegate type:SCStreamOutputTypeScreen sampleHandlerQueue:videoQueue error:&outputError];
                        if (!outputAdded || outputError) {
                            std::cout << "添加 ScreenCaptureKit 输出失败: " << (outputError ? [[outputError localizedDescription] UTF8String] : "未知错误") << std::endl;
                            g_is_capturing = false;
                            return;
                        }
                        std::cout << "ScreenCaptureKit 输出添加成功" << std::endl;
                        
                        std::cout << "开始启动捕获..." << std::endl;
                        
                        // 设置一个标志来跟踪回调是否被调用
                        __block BOOL callbackCalled = NO;
                        
                        [g_stream startCaptureWithCompletionHandler:^(NSError * _Nullable error) {
                            @try {
                                std::cout << "startCaptureWithCompletionHandler 回调被调用" << std::endl;
                                callbackCalled = YES;
                                if (error) {
                                    std::cout << "启动 ScreenCaptureKit 失败: " << [[error localizedDescription] UTF8String] << std::endl;
                                    std::cout << "错误代码: " << error.code << std::endl;
                                    std::cout << "错误域: " << [error.domain UTF8String] << std::endl;
                                    g_is_capturing = false;
                                } else {
                                    std::cout << "ScreenCaptureKit 启动成功，开始接收屏幕数据" << std::endl;
                                    g_is_capturing = true;
                                }
                            } @catch (NSException *exception) {
                                std::cout << "startCaptureWithCompletionHandler 异常: " << [[exception description] UTF8String] << std::endl;
                                g_is_capturing = false;
                            }
                        }];
                        
                        // 等待回调或超时
                        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(3.0 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
                            if (!callbackCalled) {
                                std::cout << "ScreenCaptureKit 启动超时（3秒），假设启动成功" << std::endl;
                                g_is_capturing = true;
                            }
                        });
                    } @catch (NSException *exception) {
                        std::cout << "getShareableContentWithCompletionHandler 异常: " << [[exception description] UTF8String] << std::endl;
                        g_is_capturing = false;
                    }
                }];
            } @catch (NSException *exception) {
                std::cout << "ScreenCaptureKit 启动异常: " << [[exception description] UTF8String] << std::endl;
                g_is_capturing = false;
            }
        });
        
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
            dispatch_async(dispatch_get_main_queue(), ^{
                [g_stream stopCaptureWithCompletionHandler:^(NSError * _Nullable error) {
                    if (error) {
                        std::cout << "停止 ScreenCaptureKit 时出错: " << [[error localizedDescription] UTF8String] << std::endl;
                    } else {
                        std::cout << "ScreenCaptureKit 已停止" << std::endl;
                    }
                }];
            });
        }
    }
}

bool is_macos_screen_capture_running() {
    return g_is_capturing;
}

void cleanup_macos_screen_capture() {
    if (@available(macOS 12.3, *)) {
        // 先清除回调函数，防止在清理过程中被调用
        g_frame_callback = nullptr;
        
        // 确保停止捕获
        if (g_stream && g_is_capturing) {
            g_is_capturing = false;
            
            // 在主线程上执行清理操作
            dispatch_sync(dispatch_get_main_queue(), ^{
                // 使用同步方式等待停止完成
                __block bool stop_completed = false;
                [g_stream stopCaptureWithCompletionHandler:^(NSError * _Nullable error) {
                    if (error) {
                        std::cout << "停止 ScreenCaptureKit 时出错: " << [[error localizedDescription] UTF8String] << std::endl;
                    }
                    stop_completed = true;
                }];
                
                // 等待停止完成（最多等待1秒）
                int wait_count = 0;
                while (!stop_completed && wait_count < 100) {
                    [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
                    wait_count++;
                }
                
                // 清理资源
                g_stream = nil;
            });
        }
        
        g_delegate = nil;
    }
}

} // namespace media
} // namespace duorou