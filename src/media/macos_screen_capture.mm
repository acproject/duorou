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
static const int MAX_CONSECUTIVE_FAILURES = 10; // Restart stream after 10 consecutive failures
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

    // Check if still capturing and callback function validity
    if (!g_is_capturing || !g_frame_callback) {
      return;
    }

    // Check if stream is still valid
    if (!stream || stream != g_stream) {
      std::cout << "ScreenCaptureKit: 收到来自无效stream的回调" << std::endl;
      return;
    }

    // Check CMSampleBuffer validity
    if (!sampleBuffer) {
      std::cout << "ScreenCaptureKit: CMSampleBuffer为空" << std::endl;
      return;
    }

    // Check if CMSampleBuffer is valid
    if (!CMSampleBufferIsValid(sampleBuffer)) {
      std::cout << "ScreenCaptureKit: CMSampleBuffer无效" << std::endl;
      return;
    }

    // Get CMSampleBuffer details for debugging
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

      // Check for attachment data
      CFArrayRef attachments =
          CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, false);
      if (attachments && CFArrayGetCount(attachments) > 0) {
        std::cout
            << "ScreenCaptureKit: CMSampleBuffer有附件数据，但没有图像缓冲区"
            << std::endl;
      }

      // Check data buffer
      CMBlockBufferRef blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer);
      if (blockBuffer) {
        size_t dataLength = CMBlockBufferGetDataLength(blockBuffer);
        std::cout << "ScreenCaptureKit: CMSampleBuffer有数据缓冲区，长度: "
                  << dataLength << " 字节" << std::endl;
      } else {
        std::cout << "ScreenCaptureKit: "
                     "CMSampleBuffer既没有图像缓冲区也没有数据缓冲区"
                  << std::endl;
      }

      // Check if stream restart is needed
      if (g_consecutive_failures >= MAX_CONSECUTIVE_FAILURES) {
        std::cout << "ScreenCaptureKit: 连续失败次数过多，标记需要重启流..."
                  << std::endl;
        g_needs_restart = true;
        g_consecutive_failures = 0; // Reset counter
      }

      return;
    }

    // Successfully obtained image buffer, reset failure counter
    g_consecutive_failures = 0;
    g_last_success_time = std::chrono::steady_clock::now();

    // Check pixel buffer validity
    if (!CVPixelBufferIsPlanar(imageBuffer)) {
      // Non-planar format, check pixel format
      OSType pixelFormat = CVPixelBufferGetPixelFormatType(imageBuffer);
      if (pixelFormat != kCVPixelFormatType_32BGRA) {
        std::cout << "ScreenCaptureKit: Unexpected pixel format: " << pixelFormat
                  << ", expected: " << kCVPixelFormatType_32BGRA << std::endl;
        return;
      }
    } else {
      std::cout << "ScreenCaptureKit: Received planar pixel buffer, currently not supported"
                << std::endl;
      return;
    }

    // Check pixel buffer dimensions
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    if (width == 0 || height == 0) {
      std::cout << "ScreenCaptureKit: Invalid pixel buffer dimensions: " << width << "x"
                << height << std::endl;
      return;
    }

    // Try to lock pixel buffer
    CVReturn lockResult =
        CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    if (lockResult != kCVReturnSuccess) {
      std::cout << "ScreenCaptureKit: Unable to lock pixel buffer, error code: "
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

    // Call callback directly in current thread to reduce thread switching overhead
    // ScreenCaptureKit already processes frame data in dedicated queue
    if (g_frame_callback && g_is_capturing) {
      g_frame_callback(frame);
    }
  } @catch (NSException *exception) {
    std::cout << "ScreenCaptureKit 回调异常: " <<
        [[exception description] UTF8String] << std::endl;
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

// SCStreamOutput and SCStreamDelegate share the same method implementation

@end

namespace duorou {
namespace media {

// Forward declarations
bool start_macos_screen_capture(
    std::function<void(const VideoFrame &)> callback);
void stop_macos_screen_capture();

bool check_screen_recording_permission() {
  if (@available(macOS 12.3, *)) {
    // Try to get shareable content to check permissions
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

    // Wait for permission check to complete (maximum 2 seconds)
    int wait_count = 0;
    while (!check_completed && wait_count < 200) {
      [[NSRunLoop currentRunLoop]
          runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
      wait_count++;
    }

    return permission_granted;
  } else {
    std::cout << "ScreenCaptureKit requires macOS 12.3 or higher" << std::endl;
    return false;
  }
}

bool initialize_macos_screen_capture() {
  if (@available(macOS 12.3, *)) {
    // Check permissions first
    if (!check_screen_recording_permission()) {
      std::cout << "ScreenCaptureKit permission check failed, please grant permission in "
                   "System Preferences > Security & Privacy > Privacy > Screen Recording"
                << std::endl;
      return false;
    }

    g_delegate = [[ScreenCaptureDelegate alloc] init];
    std::cout << "ScreenCaptureKit initialization successful" << std::endl;
    return true;
  } else {
    std::cout << "ScreenCaptureKit requires macOS 12.3 or higher" << std::endl;
    return false;
  }
}

bool start_macos_screen_capture(
    std::function<void(const VideoFrame &)> callback, int window_id) {
  if (@available(macOS 12.3, *)) {
    // Check if restart is needed
    if (g_needs_restart && g_is_capturing) {
      std::cout << "ScreenCaptureKit: Restart needed, stopping current stream..."
                << std::endl;
      stop_macos_screen_capture();
      g_needs_restart = false;
      // Wait a short time for the stop operation to complete
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    if (g_is_capturing && !g_needs_restart) {
      std::cout << "ScreenCaptureKit is already running" << std::endl;
      return true;
    }

    // Ensure cleanup of previous stream
    if (g_stream) {
      std::cout << "Cleaning up previous ScreenCaptureKit stream..." << std::endl;
      g_stream = nil;
    }

    std::cout << "Starting ScreenCaptureKit..." << std::endl;
    g_frame_callback = callback;

    // Ensure ScreenCaptureKit operations are executed on main thread
    dispatch_async(dispatch_get_main_queue(), ^{
      @try {
        // Get available displays
        std::cout << "Getting shareable content..." << std::endl;
        [SCShareableContent getShareableContentWithCompletionHandler:^(
                                SCShareableContent *_Nullable content,
                                NSError *_Nullable error) {
          @try {
            std::cout << "getShareableContentWithCompletionHandler callback called"
                      << std::endl;
            if (error) {
              std::cout << "Failed to get shareable content: " <<
                  [[error localizedDescription] UTF8String] << std::endl;
              std::cout << "Error code: " << error.code << std::endl;
              std::cout << "Error domain: " << [error.domain UTF8String] << std::endl;
              g_is_capturing = false;
              return;
            }

            if (!content || content.displays.count == 0) {
              std::cout << "No available displays found" << std::endl;
              g_is_capturing = false;
              return;
            }

            std::cout << "Found " << content.displays.count << " displays"
                      << std::endl;
            SCDisplay *display = content.displays.firstObject;
            if (!display) {
              std::cout << "Unable to get first display" << std::endl;
              g_is_capturing = false;
              return;
            }

            std::cout << "Using display: " << display.displayID
                      << ", size: " << display.width << "x" << display.height
                      << std::endl;

            SCContentFilter *filter = nil;
            
            // Determine capture target based on window_id parameter
            if (window_id <= 0) {
              // Capture entire desktop
              std::cout << "Creating desktop capture filter..." << std::endl;
              filter = [[SCContentFilter alloc] initWithDisplay:display
                                              excludingWindows:@[]];
            } else {
              // Capture specific window
              std::cout << "Looking for window ID: " << window_id << std::endl;
              SCWindow *targetWindow = nil;
              for (SCWindow *window in content.windows) {
                if (window.windowID == window_id) {
                  targetWindow = window;
                  break;
                }
              }
              
              if (targetWindow) {
                std::cout << "Found target window: " << [targetWindow.title UTF8String]
                          << " (" << [targetWindow.owningApplication.applicationName UTF8String] << ")" << std::endl;
                filter = [[SCContentFilter alloc] initWithDesktopIndependentWindow:targetWindow];
              } else {
                std::cout << "Window ID " << window_id << " not found, falling back to desktop capture" << std::endl;
                filter = [[SCContentFilter alloc] initWithDisplay:display
                                                excludingWindows:@[]];
              }
            }
            
            if (!filter) {
              std::cout << "Failed to create content filter" << std::endl;
              g_is_capturing = false;
              return;
            }
            std::cout << "Content filter created successfully" << std::endl;

            SCStreamConfiguration *config =
                [[SCStreamConfiguration alloc] init];
            // Use more conservative resolution settings to avoid excessive data volume
            config.width = std::min((int)display.width, 1280);
            config.height = std::min((int)display.height, 720);
            config.minimumFrameInterval =
                CMTimeMake(1, 30); // Further reduce to 30 FPS to significantly reduce buffer pressure
            config.pixelFormat = kCVPixelFormatType_32BGRA;
            config.queueDepth = 3;      // Increase cache depth for more buffer space
            config.showsCursor = YES;   // Show mouse cursor
            config.capturesAudio = YES; // Enable audio capture
            config.scalesToFit = YES;   // Allow scaling to fit configured dimensions
            config.includeChildWindows = YES;          // Include child windows
            config.colorSpaceName = kCGColorSpaceSRGB; // Explicitly specify color space
            config.backgroundColor = [NSColor blackColor].CGColor; // Set background color
            std::cout << "Stream configuration created successfully" << std::endl;

            std::cout << "Creating ScreenCaptureKit stream..." << std::endl;
            g_stream = [[SCStream alloc] initWithFilter:filter
                                          configuration:config
                                               delegate:g_delegate];
            if (!g_stream) {
              std::cout << "Failed to create ScreenCaptureKit stream" << std::endl;
              g_is_capturing = false;
              return;
            }
            std::cout << "ScreenCaptureKit stream created successfully" << std::endl;

            // Create dedicated background queue for video frame processing to avoid blocking main thread
            dispatch_queue_t videoQueue = dispatch_queue_create(
                "com.duorou.video_capture", DISPATCH_QUEUE_SERIAL);

            // Add stream output to receive video data
            NSError *outputError = nil;
            BOOL outputAdded =
                [g_stream addStreamOutput:(id<SCStreamOutput>)g_delegate
                                     type:SCStreamOutputTypeScreen
                       sampleHandlerQueue:videoQueue
                                    error:&outputError];
            if (!outputAdded || outputError) {
              std::cout << "Failed to add ScreenCaptureKit output: "
                        << (outputError ? [[outputError localizedDescription]
                                              UTF8String]
                                        : "Unknown error")
                        << std::endl;
              g_is_capturing = false;
              return;
            }
            std::cout << "ScreenCaptureKit output added successfully" << std::endl;

            std::cout << "Starting capture..." << std::endl;

            // Set a flag to track whether callback is called
            __block BOOL callbackCalled = NO;

            [g_stream
                startCaptureWithCompletionHandler:^(NSError *_Nullable error) {
                  @try {
                    std::cout << "startCaptureWithCompletionHandler callback called"
                              << std::endl;
                    callbackCalled = YES;
                    if (error) {
                      std::cout << "Failed to start ScreenCaptureKit: " <<
                          [[error localizedDescription] UTF8String]
                                << std::endl;
                      std::cout << "Error code: " << error.code << std::endl;
                      std::cout << "Error domain: " << [error.domain UTF8String]
                                << std::endl;
                      g_is_capturing = false;
                    } else {
                      std::cout << "ScreenCaptureKit started successfully, receiving screen data"
                                << std::endl;
                      g_is_capturing = true;
                    }
                  } @catch (NSException *exception) {
                    std::cout << "startCaptureWithCompletionHandler exception: " <<
                        [[exception description] UTF8String] << std::endl;
                    g_is_capturing = false;
                  }
                }];

            // Wait for callback or timeout
            dispatch_after(
                dispatch_time(DISPATCH_TIME_NOW, (int64_t)(3.0 * NSEC_PER_SEC)),
                dispatch_get_main_queue(), ^{
                  if (!callbackCalled) {
                    std::cout
                        << "ScreenCaptureKit startup timeout (3 seconds), assuming successful start"
                        << std::endl;
                    g_is_capturing = true;
                  }
                });
          } @catch (NSException *exception) {
            std::cout << "getShareableContentWithCompletionHandler exception: " <<
                [[exception description] UTF8String] << std::endl;
            g_is_capturing = false;
          }
        }];
      } @catch (NSException *exception) {
        std::cout << "ScreenCaptureKit startup exception: " <<
            [[exception description] UTF8String] << std::endl;
        g_is_capturing = false;
      }
    });

    return true;
  } else {
    std::cout << "ScreenCaptureKit requires macOS 12.3 or higher" << std::endl;
    return false;
  }
}

void stop_macos_screen_capture() {
  if (@available(macOS 12.3, *)) {
    if (g_stream && g_is_capturing) {
      g_is_capturing = false; // Immediately set to false to avoid duplicate calls
      dispatch_async(dispatch_get_main_queue(), ^{
        [g_stream stopCaptureWithCompletionHandler:^(NSError *_Nullable error) {
          if (error) {
            std::cout << "Error stopping ScreenCaptureKit: " <<
                [[error localizedDescription] UTF8String] << std::endl;
          } else {
            std::cout << "ScreenCaptureKit stopped" << std::endl;
          }
        }];
      });
    }
  }
}

bool is_macos_screen_capture_running() { return g_is_capturing; }

void cleanup_macos_screen_capture() {
  if (@available(macOS 12.3, *)) {
    std::cout << "Starting cleanup of macOS screen capture resources..." << std::endl;

    // Use static mutex to prevent concurrent cleanup
    static std::mutex cleanup_mutex;
    static bool cleanup_completed = false;

    std::lock_guard<std::mutex> lock(cleanup_mutex);

    // Check if cleanup has already been performed
    if (cleanup_completed) {
      std::cout << "macOS screen capture resources already cleaned up, skipping" << std::endl;
      return;
    }

    // Clear callback function first to prevent calls during cleanup
    g_frame_callback = nullptr;

    // Ensure capture is stopped
    if (g_stream && g_is_capturing) {
      std::cout << "Stopping screen capture stream..." << std::endl;

      if ([NSThread isMainThread]) {
        // If on main thread, call stop method directly
        if (g_stream) {
          [g_stream
              stopCaptureWithCompletionHandler:^(NSError *_Nullable error) {
                if (error) {
                  std::cout << "Error stopping capture: " <<
                      [[error localizedDescription] UTF8String] << std::endl;
                } else {
                  std::cout << "Screen capture stream stopped" << std::endl;
                }
                g_is_capturing = false;
              }];
        }
        // Give a brief delay to allow stop operation to execute
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      } else {
        // If not on main thread, stop asynchronously but don't wait for completion
        dispatch_async(dispatch_get_main_queue(), ^{
          if (g_stream) {
            [g_stream
                stopCaptureWithCompletionHandler:^(NSError *_Nullable error) {
                  if (error) {
                    std::cout << "Error stopping capture: " <<
                        [[error localizedDescription] UTF8String] << std::endl;
                  } else {
                    std::cout << "Screen capture stream stopped" << std::endl;
                  }
                  g_is_capturing = false;
                }];
          }
        });
        // Give a brief delay to allow stop operation to execute
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
    
    // Ensure state is correct
    g_is_capturing = false;

    // Safely clean up resources, avoiding deadlock
    if ([NSThread isMainThread]) {
      // If already on main thread, clean up directly
      if (g_stream) {
        g_stream = nil;
        std::cout << "Screen capture stream cleaned up" << std::endl;
      }

      if (g_delegate) {
        g_delegate = nil;
        std::cout << "Screen capture delegate cleaned up" << std::endl;
      }
    } else {
      // If not on main thread, use async cleanup but don't wait for completion to avoid deadlock
      dispatch_async(dispatch_get_main_queue(), ^{
        if (g_stream) {
          g_stream = nil;
          std::cout << "Screen capture stream cleaned up" << std::endl;
        }

        if (g_delegate) {
          g_delegate = nil;
          std::cout << "Screen capture delegate cleaned up" << std::endl;
        }
      });
      
      // Give a brief delay to allow cleanup task to execute, but don't block
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Mark cleanup as completed
    cleanup_completed = true;
    std::cout << "macOS screen capture resource cleanup completed" << std::endl;
  }
}

void update_macos_screen_capture_window(int window_id) {
  if (@available(macOS 12.3, *)) {
    if (!g_stream || !g_is_capturing) {
      std::cout << "Screen capture not running, cannot update window" << std::endl;
      return;
    }

    std::cout << "Updating screen capture window ID: " << window_id << std::endl;

    dispatch_async(dispatch_get_main_queue(), ^{
      @try {
        [SCShareableContent getShareableContentWithCompletionHandler:^(
            SCShareableContent *_Nullable content, NSError *_Nullable error) {
          @try {
            if (error || !content) {
              std::cout << "Failed to get shareable content: " <<
                  (error ? [[error localizedDescription] UTF8String] : "Unknown error")
                        << std::endl;
              return;
            }

            SCDisplay *display = content.displays.firstObject;
            if (!display) {
              std::cout << "Display not found" << std::endl;
              return;
            }

            SCContentFilter *filter = nil;
            
            // Determine capture target based on window_id parameter
            if (window_id <= 0) {
              // Capture entire desktop
              std::cout << "Updating to desktop capture..." << std::endl;
              filter = [[SCContentFilter alloc] initWithDisplay:display
                                              excludingWindows:@[]];
            } else {
              // Capture specific window
              std::cout << "Looking for window ID: " << window_id << std::endl;
              SCWindow *targetWindow = nil;
              for (SCWindow *window in content.windows) {
                if (window.windowID == window_id) {
                  targetWindow = window;
                  break;
                }
              }
              
              if (targetWindow) {
                std::cout << "Found target window: " << [targetWindow.title UTF8String]
                          << " (" << [targetWindow.owningApplication.applicationName UTF8String] << ")" << std::endl;
                filter = [[SCContentFilter alloc] initWithDesktopIndependentWindow:targetWindow];
              } else {
                std::cout << "Window ID " << window_id << " not found, keeping current settings" << std::endl;
                return;
              }
            }

            if (!filter) {
              std::cout << "Failed to create content filter" << std::endl;
              return;
            }

            // Update stream filter
            [g_stream updateContentFilter:filter completionHandler:^(NSError *_Nullable error) {
              if (error) {
                std::cout << "Failed to update content filter: " <<
                    [[error localizedDescription] UTF8String] << std::endl;
              } else {
                std::cout << "Content filter updated successfully" << std::endl;
              }
            }];
          } @catch (NSException *exception) {
            std::cout << "Update window filter exception: " <<
                [[exception description] UTF8String] << std::endl;
          }
        }];
      } @catch (NSException *exception) {
        std::cout << "Update screen capture window exception: " <<
            [[exception description] UTF8String] << std::endl;
      }
    });
  } else {
    std::cout << "ScreenCaptureKit requires macOS 12.3 or higher" << std::endl;
  }
}

bool is_macos_camera_available() {
  @try {
    // Use AVFoundation to detect camera devices
    NSArray<AVCaptureDevice *> *devices =
        [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];

    if (devices.count > 0) {
      std::cout << "Detected " << devices.count << " camera devices" << std::endl;
      for (AVCaptureDevice *device in devices) {
        std::cout << "Camera device: " << [device.localizedName UTF8String]
                  << std::endl;
      }
      return true;
    } else {
      std::cout << "No camera devices detected" << std::endl;
      return false;
    }
  } @catch (NSException *exception) {
    std::cout << "Exception occurred while detecting camera: " <<
        [[exception description] UTF8String] << std::endl;
    return false;
  }
}

} // namespace media
} // namespace duorou