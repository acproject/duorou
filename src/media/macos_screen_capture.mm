#include "video_frame.h"
#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#include <functional>
#include <iostream>
#include <thread>

static std::function<void(const duorou::media::VideoFrame &)> g_frame_callback;
static SCStream *g_stream = nil;
static bool g_is_capturing = false;
static id<SCStreamDelegate> g_delegate = nil;
static int g_consecutive_failures = 0;
static const int MAX_CONSECUTIVE_FAILURES = 10; // 连续失败10次后重启流
static std::chrono::steady_clock::time_point g_last_success_time =
    std::chrono::steady_clock::now();
static bool g_needs_restart = false;

@interface ScreenCaptureDelegate : NSObject <SCStreamDelegate, SCStreamOutput>
@end

@implementation ScreenCaptureDelegate

- (void)stream:(SCStream *)stream
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
                   ofType:(SCStreamOutputType)type {
  @try {
    if (type != SCStreamOutputTypeScreen) {
      return;
    }

    // 检查是否仍在捕获状态和回调函数有效性
    if (!g_is_capturing || !g_frame_callback) {
      return;
    }

    // 检查stream是否仍然有效
    if (!stream || stream != g_stream) {
      std::cout << "ScreenCaptureKit: 收到来自无效stream的回调" << std::endl;
      return;
    }

    // 检查CMSampleBuffer的有效性
    if (!sampleBuffer) {
      std::cout << "ScreenCaptureKit: CMSampleBuffer为空" << std::endl;
      return;
    }

    // 检查CMSampleBuffer是否有效
    if (!CMSampleBufferIsValid(sampleBuffer)) {
    std::cout << "ScreenCaptureKit: CMSampleBuffer无效" << std::endl;
    return;
  }

  // 获取CMSampleBuffer的详细信息用于调试
  CMFormatDescriptionRef formatDesc =
      CMSampleBufferGetFormatDescription(sampleBuffer);
  if (formatDesc) {
    CMMediaType mediaType = CMFormatDescriptionGetMediaType(formatDesc);
    if (mediaType != kCMMediaType_Video) {
      std::cout << "ScreenCaptureKit: 收到非视频媒体类型: " << mediaType
                << std::endl;
      return;
    }
  }

  CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
  if (!imageBuffer) {
    g_consecutive_failures++;
    std::cout << "ScreenCaptureKit: 无法获取图像缓冲区 (连续失败: "
              << g_consecutive_failures << "/" << MAX_CONSECUTIVE_FAILURES
              << ")" << std::endl;

    // 检查是否有附件数据
    CFArrayRef attachments =
        CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, false);
    if (attachments && CFArrayGetCount(attachments) > 0) {
      std::cout
          << "ScreenCaptureKit: CMSampleBuffer有附件数据，但没有图像缓冲区"
          << std::endl;
    }

    // 检查数据缓冲区
    CMBlockBufferRef blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer);
    if (blockBuffer) {
      size_t dataLength = CMBlockBufferGetDataLength(blockBuffer);
      std::cout << "ScreenCaptureKit: CMSampleBuffer有数据缓冲区，长度: "
                << dataLength << " 字节" << std::endl;
    } else {
      std::cout
          << "ScreenCaptureKit: CMSampleBuffer既没有图像缓冲区也没有数据缓冲区"
          << std::endl;
    }

    // 检查是否需要重启流
    if (g_consecutive_failures >= MAX_CONSECUTIVE_FAILURES) {
      std::cout << "ScreenCaptureKit: 连续失败次数过多，标记需要重启流..."
                << std::endl;
      g_needs_restart = true;
      g_consecutive_failures = 0; // 重置计数器
    }

    return;
  }

  // 成功获取图像缓冲区，重置失败计数器
  g_consecutive_failures = 0;
  g_last_success_time = std::chrono::steady_clock::now();

  // 检查像素缓冲区的有效性
  if (!CVPixelBufferIsPlanar(imageBuffer)) {
    // 非平面格式，检查像素格式
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(imageBuffer);
    if (pixelFormat != kCVPixelFormatType_32BGRA) {
      std::cout << "ScreenCaptureKit: 意外的像素格式: " << pixelFormat
                << ", 期望: " << kCVPixelFormatType_32BGRA << std::endl;
      return;
    }
  } else {
    std::cout << "ScreenCaptureKit: 收到平面像素缓冲区，当前不支持"
              << std::endl;
    return;
  }

  // 检查像素缓冲区尺寸
  size_t width = CVPixelBufferGetWidth(imageBuffer);
  size_t height = CVPixelBufferGetHeight(imageBuffer);
  if (width == 0 || height == 0) {
    std::cout << "ScreenCaptureKit: 像素缓冲区尺寸无效: " << width << "x"
              << height << std::endl;
    return;
  }

  // 尝试锁定像素缓冲区
  CVReturn lockResult =
      CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
  if (lockResult != kCVReturnSuccess) {
    std::cout << "ScreenCaptureKit: 无法锁定像素缓冲区，错误代码: "
              << lockResult << std::endl;
    return;
  }
  size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
  void *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);

  duorou::media::VideoFrame frame;
  frame.width = static_cast<int>(width);
  frame.height = static_cast<int>(height);
  frame.channels = 4; // BGRA
  frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count() /
                    1000.0;

  size_t dataSize = height * bytesPerRow;
  frame.data.resize(dataSize);
  std::memcpy(frame.data.data(), baseAddress, dataSize);

  CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

  // 直接在当前线程调用回调，减少线程切换开销
  // ScreenCaptureKit已经在专门的队列中处理帧数据
  if (g_frame_callback && g_is_capturing) {
    g_frame_callback(frame);
  }
  } @catch (NSException *exception) {
    std::cout << "ScreenCaptureKit 回调异常: " << [[exception description] UTF8String] << std::endl;
  }
}

- (void)stream:(SCStream *)stream didStopWithError:(NSError *)error {
  if (error) {
    std::cout << "ScreenCaptureKit stream stopped with error: " <<
        [[error localizedDescription] UTF8String] << std::endl;
  } else {
    std::cout << "ScreenCaptureKit stream stopped" << std::endl;
  }
  g_is_capturing = false;
}

// SCStreamOutput和SCStreamDelegate共享同一个方法实现

@end

namespace duorou {
namespace media {

// 前向声明
bool start_macos_screen_capture(
    std::function<void(const VideoFrame &)> callback);
void stop_macos_screen_capture();

bool check_screen_recording_permission() {
  if (@available(macOS 12.3, *)) {
    // 尝试获取可共享内容来检查权限
    __block bool permission_granted = false;
    __block bool check_completed = false;

    [SCShareableContent
        getShareableContentWithCompletionHandler:^(
            SCShareableContent *_Nullable content, NSError *_Nullable error) {
          if (error) {
            std::cout << "屏幕录制权限检查失败: " <<
                [[error localizedDescription] UTF8String] << std::endl;
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
      [[NSRunLoop currentRunLoop]
          runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
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
      std::cout << "ScreenCaptureKit 权限检查失败，请在系统偏好设置 > "
                   "安全性与隐私 > 隐私 > 屏幕录制中允许此应用"
                << std::endl;
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

bool start_macos_screen_capture(
    std::function<void(const VideoFrame &)> callback) {
  if (@available(macOS 12.3, *)) {
    // 检查是否需要重启
    if (g_needs_restart && g_is_capturing) {
      std::cout << "ScreenCaptureKit: 检测到需要重启，先停止当前流..."
                << std::endl;
      stop_macos_screen_capture();
      g_needs_restart = false;
      // 等待一小段时间让停止操作完成
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    if (g_is_capturing && !g_needs_restart) {
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
        [SCShareableContent getShareableContentWithCompletionHandler:^(
                                SCShareableContent *_Nullable content,
                                NSError *_Nullable error) {
          @try {
            std::cout << "getShareableContentWithCompletionHandler 回调被调用"
                      << std::endl;
            if (error) {
              std::cout << "获取可共享内容失败: " <<
                  [[error localizedDescription] UTF8String] << std::endl;
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

            std::cout << "找到 " << content.displays.count << " 个显示器"
                      << std::endl;
            SCDisplay *display = content.displays.firstObject;
            if (!display) {
              std::cout << "无法获取第一个显示器" << std::endl;
              g_is_capturing = false;
              return;
            }

            std::cout << "使用显示器: " << display.displayID
                      << ", 尺寸: " << display.width << "x" << display.height
                      << std::endl;

            SCContentFilter *filter =
                [[SCContentFilter alloc] initWithDisplay:display
                                        excludingWindows:@[]];
            if (!filter) {
              std::cout << "创建内容过滤器失败" << std::endl;
              g_is_capturing = false;
              return;
            }
            std::cout << "创建内容过滤器成功" << std::endl;

            SCStreamConfiguration *config =
                [[SCStreamConfiguration alloc] init];
            // 使用更保守的分辨率设置，避免过高的数据量
            config.width = std::min((int)display.width, 1280);
            config.height = std::min((int)display.height, 720);
            config.minimumFrameInterval =
                CMTimeMake(1, 30); // 进一步降低到3 FPS，大幅减少缓冲区压力
            config.pixelFormat = kCVPixelFormatType_32BGRA;
            config.queueDepth = 3;      // 进一步增加缓存深度，提供更多缓冲空间
            config.showsCursor = YES;   // 显示鼠标光标
            config.capturesAudio = YES; // 明确禁用音频捕获
            config.scalesToFit = YES;   // 允许缩放以适应配置的尺寸
            config.includeChildWindows = YES;          // 包含子窗口
            config.colorSpaceName = kCGColorSpaceSRGB; // 明确指定颜色空间
            config.backgroundColor = [NSColor blackColor].CGColor; // 设置背景色
            std::cout << "创建流配置成功" << std::endl;

            std::cout << "创建 ScreenCaptureKit 流..." << std::endl;
            g_stream = [[SCStream alloc] initWithFilter:filter
                                          configuration:config
                                               delegate:g_delegate];
            if (!g_stream) {
              std::cout << "创建 ScreenCaptureKit 流失败" << std::endl;
              g_is_capturing = false;
              return;
            }
            std::cout << "ScreenCaptureKit 流创建成功" << std::endl;

            // 创建专门的后台队列处理视频帧，避免阻塞主线程
            dispatch_queue_t videoQueue = dispatch_queue_create(
                "com.duorou.video_capture", DISPATCH_QUEUE_SERIAL);

            // 添加stream output以接收视频数据
            NSError *outputError = nil;
            BOOL outputAdded =
                [g_stream addStreamOutput:(id<SCStreamOutput>)g_delegate
                                     type:SCStreamOutputTypeScreen
                       sampleHandlerQueue:videoQueue
                                    error:&outputError];
            if (!outputAdded || outputError) {
              std::cout << "添加 ScreenCaptureKit 输出失败: "
                        << (outputError ? [[outputError localizedDescription]
                                              UTF8String]
                                        : "未知错误")
                        << std::endl;
              g_is_capturing = false;
              return;
            }
            std::cout << "ScreenCaptureKit 输出添加成功" << std::endl;

            std::cout << "开始启动捕获..." << std::endl;

            // 设置一个标志来跟踪回调是否被调用
            __block BOOL callbackCalled = NO;

            [g_stream
                startCaptureWithCompletionHandler:^(NSError *_Nullable error) {
                  @try {
                    std::cout << "startCaptureWithCompletionHandler 回调被调用"
                              << std::endl;
                    callbackCalled = YES;
                    if (error) {
                      std::cout << "启动 ScreenCaptureKit 失败: " <<
                          [[error localizedDescription] UTF8String]
                                << std::endl;
                      std::cout << "错误代码: " << error.code << std::endl;
                      std::cout << "错误域: " << [error.domain UTF8String]
                                << std::endl;
                      g_is_capturing = false;
                    } else {
                      std::cout << "ScreenCaptureKit 启动成功，开始接收屏幕数据"
                                << std::endl;
                      g_is_capturing = true;
                    }
                  } @catch (NSException *exception) {
                    std::cout << "startCaptureWithCompletionHandler 异常: " <<
                        [[exception description] UTF8String] << std::endl;
                    g_is_capturing = false;
                  }
                }];

            // 等待回调或超时
            dispatch_after(
                dispatch_time(DISPATCH_TIME_NOW, (int64_t)(3.0 * NSEC_PER_SEC)),
                dispatch_get_main_queue(), ^{
                  if (!callbackCalled) {
                    std::cout
                        << "ScreenCaptureKit 启动超时（3秒），假设启动成功"
                        << std::endl;
                    g_is_capturing = true;
                  }
                });
          } @catch (NSException *exception) {
            std::cout << "getShareableContentWithCompletionHandler 异常: " <<
                [[exception description] UTF8String] << std::endl;
            g_is_capturing = false;
          }
        }];
      } @catch (NSException *exception) {
        std::cout << "ScreenCaptureKit 启动异常: " <<
            [[exception description] UTF8String] << std::endl;
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
        [g_stream stopCaptureWithCompletionHandler:^(NSError *_Nullable error) {
          if (error) {
            std::cout << "停止 ScreenCaptureKit 时出错: " <<
                [[error localizedDescription] UTF8String] << std::endl;
          } else {
            std::cout << "ScreenCaptureKit 已停止" << std::endl;
          }
        }];
      });
    }
  }
}

bool is_macos_screen_capture_running() { return g_is_capturing; }

void cleanup_macos_screen_capture() {
  if (@available(macOS 12.3, *)) {
    std::cout << "开始清理macOS屏幕捕获资源..." << std::endl;
    
    // 使用静态互斥锁防止并发清理
    static std::mutex cleanup_mutex;
    static bool cleanup_completed = false;
    
    std::lock_guard<std::mutex> lock(cleanup_mutex);
    
    // 检查是否已经清理过
    if (cleanup_completed) {
      std::cout << "macOS屏幕捕获资源已经清理，跳过" << std::endl;
      return;
    }
    
    // 先清除回调函数，防止在清理过程中被调用
    g_frame_callback = nullptr;

    // 确保停止捕获
    if (g_stream && g_is_capturing) {
      std::cout << "正在停止屏幕捕获流..." << std::endl;
      
      // 使用信号量等待停止完成
      dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
      __block bool stop_completed = false;
      
      dispatch_async(dispatch_get_main_queue(), ^{
        if (g_stream) { // 再次检查，防止竞态条件
          [g_stream stopCaptureWithCompletionHandler:^(NSError * _Nullable error) {
            if (error) {
              std::cout << "停止捕获时出错: " << [[error localizedDescription] UTF8String] << std::endl;
            } else {
              std::cout << "屏幕捕获流已停止" << std::endl;
            }
            g_is_capturing = false;
            stop_completed = true;
            dispatch_semaphore_signal(semaphore);
          }];
        } else {
          // 如果g_stream已经为nil，直接完成
          g_is_capturing = false;
          stop_completed = true;
          dispatch_semaphore_signal(semaphore);
        }
      });
      
      // 等待最多2秒，给更多时间完成清理
      dispatch_time_t timeout = dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC);
      if (dispatch_semaphore_wait(semaphore, timeout) != 0) {
        std::cout << "警告: 停止捕获超时，强制设置状态" << std::endl;
        g_is_capturing = false;
      }
      
      dispatch_release(semaphore);
      
      // 额外等待一小段时间，确保所有回调完成
      if (stop_completed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    } else {
      // 即使没有在捕获，也要确保状态正确
      g_is_capturing = false;
    }

    // 安全地清理资源，避免使用dispatch_sync可能导致的死锁
    if ([NSThread isMainThread]) {
      // 如果已经在主线程，直接清理
      if (g_stream) {
        g_stream = nil;
        std::cout << "屏幕捕获流已清理" << std::endl;
      }
      
      if (g_delegate) {
        g_delegate = nil;
        std::cout << "屏幕捕获代理已清理" << std::endl;
      }
    } else {
      // 如果不在主线程，使用异步方式清理
      dispatch_semaphore_t cleanup_semaphore = dispatch_semaphore_create(0);
      
      dispatch_async(dispatch_get_main_queue(), ^{
        if (g_stream) {
          g_stream = nil;
          std::cout << "屏幕捕获流已清理" << std::endl;
        }
        
        if (g_delegate) {
          g_delegate = nil;
          std::cout << "屏幕捕获代理已清理" << std::endl;
        }
        
        dispatch_semaphore_signal(cleanup_semaphore);
      });
      
      // 等待清理完成，最多等待1秒
      dispatch_time_t cleanup_timeout = dispatch_time(DISPATCH_TIME_NOW, 1 * NSEC_PER_SEC);
      if (dispatch_semaphore_wait(cleanup_semaphore, cleanup_timeout) != 0) {
        std::cout << "警告: 资源清理超时" << std::endl;
      }
      
      dispatch_release(cleanup_semaphore);
    }
    
    // 标记清理完成
    cleanup_completed = true;
    std::cout << "macOS屏幕捕获资源清理完成" << std::endl;
  }
}

bool is_macos_camera_available() {
  @try {
    // 使用 AVFoundation 检测摄像头设备
    NSArray<AVCaptureDevice *> *devices =
        [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];

    if (devices.count > 0) {
      std::cout << "检测到 " << devices.count << " 个摄像头设备" << std::endl;
      for (AVCaptureDevice *device in devices) {
        std::cout << "摄像头设备: " << [device.localizedName UTF8String]
                  << std::endl;
      }
      return true;
    } else {
      std::cout << "未检测到摄像头设备" << std::endl;
      return false;
    }
  } @catch (NSException *exception) {
    std::cout << "检测摄像头时发生异常: " <<
        [[exception description] UTF8String] << std::endl;
    return false;
  }
}

} // namespace media
} // namespace duorou