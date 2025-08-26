#include "video_capture.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <algorithm>

#ifdef __APPLE__
#include "macos_screen_capture.h"
#endif
#ifdef __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#ifdef HAVE_GSTREAMER
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#endif

#ifdef __APPLE__
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
// AVFoundation 头文件在 Objective-C++ 文件中包含
#endif

#ifdef __linux__
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

namespace duorou {
namespace media {

class VideoCapture::Impl {
public:
    VideoSource source = VideoSource::NONE;
    int device_id = 0;
    bool capturing = false;
    std::function<void(const VideoFrame&)> frame_callback;
    std::thread capture_thread;
    std::mutex mutex;
    
    ~Impl() {
#ifdef HAVE_GSTREAMER
        cleanup_gstreamer();
#endif
#ifdef HAVE_OPENCV
        if (opencv_capture.isOpened()) {
            opencv_capture.release();
        }
#endif
    }
    
#ifdef HAVE_GSTREAMER
    GstElement* pipeline = nullptr;
    GstElement* appsink = nullptr;
    GMainLoop* loop = nullptr;
    std::thread gst_thread;
    bool gst_initialized = false;
#endif
    
#ifdef HAVE_OPENCV
    cv::VideoCapture opencv_capture;
#endif
    
    bool initialize_gstreamer() {
#ifdef HAVE_GSTREAMER
        if (!gst_initialized) {
            gst_init(nullptr, nullptr);
            gst_initialized = true;
            std::cout << "GStreamer 初始化成功" << std::endl;
        }
        return true;
#else
        std::cout << "GStreamer 未启用" << std::endl;
        return false;
#endif
    }
    
    void cleanup_gstreamer() {
#ifdef HAVE_GSTREAMER
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(pipeline);
            pipeline = nullptr;
            appsink = nullptr;
        }
        if (loop) {
            g_main_loop_quit(loop);
            if (gst_thread.joinable()) {
                gst_thread.join();
            }
            g_main_loop_unref(loop);
            loop = nullptr;
        }
#endif
    }
    
    bool try_gstreamer_desktop_capture() {
#ifdef HAVE_GSTREAMER
        if (!initialize_gstreamer()) {
            std::cout << "GStreamer 初始化失败" << std::endl;
            return false;
        }
        
        // 创建桌面捕获管道
#ifdef __APPLE__
        // macOS 使用 avfvideosrc 捕获屏幕，避免GTK依赖
        std::string pipeline_str = "avfvideosrc capture-screen=true ! "
                                  "video/x-raw,format=BGRA,width=1280,height=720,framerate=15/1 ! "
                                  "videoconvert ! "
                                  "video/x-raw,format=RGB ! "
                                  "appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true";
#elif defined(__linux__)
        // Linux 使用 ximagesrc 捕获屏幕
        std::string pipeline_str = "ximagesrc ! "
                                  "videoconvert ! "
                                  "video/x-raw,format=RGB,width=1280,height=720,framerate=30/1 ! "
                                  "appsink name=sink";
#else
        std::cout << "当前平台不支持 GStreamer 桌面捕获" << std::endl;
        return false;
#endif
        
        GError* error = nullptr;
        pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
        
        if (!pipeline || error) {
            std::cout << "创建 GStreamer 管道失败: " << (error ? error->message : "未知错误") << std::endl;
            std::cout << "这可能是由于权限问题或设备不可用" << std::endl;
            if (error) g_error_free(error);
            return false;
        }
        
        appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
        if (!appsink) {
            std::cout << "获取 appsink 失败" << std::endl;
            gst_object_unref(pipeline);
            pipeline = nullptr;
            return false;
        }
        
        // appsink 已在管道中配置，无需额外设置
        
        // 测试管道是否可以启动到PLAYING状态来检查权限
        GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cout << "GStreamer 管道无法启动，可能需要屏幕录制权限" << std::endl;
            std::cout << "请在系统偏好设置 > 安全性与隐私 > 隐私 > 屏幕录制中允许此应用" << std::endl;
            cleanup_gstreamer();
            return false;
        }
        
        // 等待状态变化完成
        GstState state;
        ret = gst_element_get_state(pipeline, &state, nullptr, GST_CLOCK_TIME_NONE);
        if (state != GST_STATE_PLAYING) {
            std::cout << "GStreamer 管道状态异常，当前状态: " << state << std::endl;
            std::cout << "这通常表示没有屏幕录制权限" << std::endl;
            cleanup_gstreamer();
            return false;
        }
        
        // 恢复到 NULL 状态，等待实际启动
        gst_element_set_state(pipeline, GST_STATE_NULL);
        
        std::cout << "屏幕录制权限检查通过" << std::endl;
        
        std::cout << "GStreamer 桌面捕获初始化成功" << std::endl;
        return true;
#else
        return false;
#endif
    }
    
    bool initialize_desktop_capture() {
        // 首先尝试 GStreamer
        if (try_gstreamer_desktop_capture()) {
            return true;
        }
        
        // 回退到原始实现
        std::cout << "回退到简化实现" << std::endl;
#ifdef __APPLE__
        // 尝试使用 ScreenCaptureKit
        if (duorou::media::initialize_macos_screen_capture()) {
            std::cout << "ScreenCaptureKit 桌面捕获初始化成功" << std::endl;
            return true;
        }
        std::cout << "ScreenCaptureKit 初始化失败，使用模拟数据" << std::endl;
        std::cout << "初始化 macOS 桌面捕获 (简化版本)" << std::endl;
        return true;
#elif defined(__linux__)
        std::cout << "初始化 Linux 桌面捕获 (简化版本)" << std::endl;
        return true;
#else
        std::cout << "当前平台不支持桌面捕获" << std::endl;
        return false;
#endif
    }
    
    bool initialize_camera_capture(int device_id) {
#ifdef HAVE_GSTREAMER
        if (!initialize_gstreamer()) {
            return false;
        }
        
        // 创建摄像头捕获管道
#ifdef __APPLE__
        // macOS 使用 avfvideosrc 捕获摄像头
        std::string pipeline_str = "avfvideosrc device-index=" + std::to_string(device_id) + " ! "
                                  "videoconvert ! "
                                  "video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! "
                                  "appsink name=sink";
#elif defined(__linux__)
        // Linux 使用 v4l2src 捕获摄像头
        std::string pipeline_str = "v4l2src device=/dev/video" + std::to_string(device_id) + " ! "
                                  "videoconvert ! "
                                  "video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! "
                                  "appsink name=sink";
#else
        std::cout << "当前平台不支持 GStreamer 摄像头捕获" << std::endl;
        return false;
#endif
        
        GError* error = nullptr;
        pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
        
        if (!pipeline || error) {
            std::cout << "创建摄像头 GStreamer 管道失败: " << (error ? error->message : "未知错误") << std::endl;
            if (error) g_error_free(error);
            return false;
        }
        
        appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
        if (!appsink) {
            std::cout << "获取摄像头 appsink 失败" << std::endl;
            gst_object_unref(pipeline);
            pipeline = nullptr;
            return false;
        }
        
        // 配置 appsink
        g_object_set(appsink, "emit-signals", TRUE, "sync", FALSE, nullptr);
        
        std::cout << "GStreamer 摄像头捕获初始化成功，设备: " << device_id << std::endl;
        return true;
#elif defined(HAVE_OPENCV)
        // 回退到 OpenCV 实现
        opencv_capture.open(device_id);
        if (!opencv_capture.isOpened()) {
            std::cout << "无法打开摄像头设备 " << device_id << std::endl;
            return false;
        }
        
        // 设置摄像头参数
        opencv_capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        opencv_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        opencv_capture.set(cv::CAP_PROP_FPS, 30);
        
        std::cout << "成功初始化摄像头设备 " << device_id << " (OpenCV)" << std::endl;
        return true;
#else
        std::cout << "GStreamer 和 OpenCV 均未启用，无法使用摄像头捕获" << std::endl;
        return false;
#endif
    }
    
    void capture_loop() {
#ifdef HAVE_GSTREAMER
        if (pipeline && appsink) {
            // GStreamer 捕获循环
            while (capturing) {
                GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
                if (sample) {
                    GstBuffer* buffer = gst_sample_get_buffer(sample);
                    GstCaps* caps = gst_sample_get_caps(sample);
                    
                    if (buffer && caps) {
                        GstMapInfo map;
                        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
                            // 获取视频信息
                            GstStructure* structure = gst_caps_get_structure(caps, 0);
                            int width, height;
                            gst_structure_get_int(structure, "width", &width);
                            gst_structure_get_int(structure, "height", &height);
                            
                            // 创建视频帧
                            VideoFrame frame;
                            frame.width = width;
                            frame.height = height;
                            frame.channels = 3; // RGB
                            frame.data.resize(map.size);
                            std::memcpy(frame.data.data(), map.data, map.size);
                            frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
                            
                            std::cout << "收到视频帧: " << width << "x" << height << ", 大小: " << map.size << " 字节" << std::endl;
                            
                            if (frame_callback) {
                                frame_callback(frame);
                            }
                            
                            gst_buffer_unmap(buffer, &map);
                        }
                    }
                    
                    gst_sample_unref(sample);
                } else {
                    // 没有可用的样本，稍微等待
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        } else {
#endif
            // 原始捕获循环（OpenCV 或模拟数据）
            while (capturing) {
                VideoFrame frame;
                
                if (source == VideoSource::DESKTOP_CAPTURE) {
#ifdef __APPLE__
                    // 在 macOS 上，如果 ScreenCaptureKit 正在运行，完全依赖回调机制
                    // 不生成任何后备数据，避免与真实屏幕数据混合造成闪烁
                    if (duorou::media::is_macos_screen_capture_running()) {
                        // ScreenCaptureKit 正在运行，只等待回调数据，不生成测试图案
                        std::this_thread::sleep_for(std::chrono::milliseconds(33));
                        continue;
                    }
                    // 只有在 ScreenCaptureKit 完全未运行时才使用后备实现
                    // 这种情况下生成测试图案是安全的，因为不会与真实数据混合
                    if (capture_desktop_frame(frame)) {
                        if (frame_callback) {
                            frame_callback(frame);
                        }
                    }
#else
                    // 非 macOS 平台使用标准桌面捕获
                    if (capture_desktop_frame(frame)) {
                        if (frame_callback) {
                            frame_callback(frame);
                        }
                    }
#endif
                } else if (source == VideoSource::CAMERA) {
                    if (capture_camera_frame(frame)) {
                        if (frame_callback) {
                            frame_callback(frame);
                        }
                    }
                }
                
                // 控制帧率 (30 FPS)
                std::this_thread::sleep_for(std::chrono::milliseconds(33));
            }
#ifdef HAVE_GSTREAMER
        }
#endif
    }
    
    bool capture_desktop_frame(VideoFrame& frame) {
#ifdef __APPLE__
        // macOS 桌面捕获实现 - 使用 ScreenCaptureKit
        // 注意：这个函数在新的实现中不再直接使用，因为ScreenCaptureKit使用回调机制
        // 保留作为后备方案，使用简化的测试图案
        frame.width = 1920;
        frame.height = 1080;
        frame.channels = 4; // RGBA
        frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
        
        // 创建简单的测试图案
        frame.data.resize(1920 * 1080 * 4);
        for (int y = 0; y < 1080; ++y) {
            for (int x = 0; x < 1920; ++x) {
                int index = (y * 1920 + x) * 4;
                // 创建彩色渐变图案
                frame.data[index] = static_cast<uint8_t>((x * 255) / 1920);     // R
                frame.data[index + 1] = static_cast<uint8_t>((y * 255) / 1080); // G
                frame.data[index + 2] = 128;                                    // B
                frame.data[index + 3] = 255;                                    // A
            }
        }
        
        std::cout << "使用后备桌面数据 (彩色测试图案)" << std::endl;
        return true;
#elif defined(__linux__)
        // Linux X11 桌面截图实现
        frame.width = 640;
        frame.height = 480;
        frame.channels = 3; // RGB
        frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
        
        // 简化实现：创建空白帧数据
        frame.data.resize(640 * 480 * 3);
        std::fill(frame.data.begin(), frame.data.end(), 128); // 灰色填充
        return true;
#else
        return false;
#endif
    }
    
    bool capture_camera_frame(VideoFrame& frame) {
#ifdef HAVE_OPENCV
        cv::Mat opencv_frame;
        if (!opencv_capture.read(opencv_frame)) {
            return false;
        }
        
        frame.width = opencv_frame.cols;
        frame.height = opencv_frame.rows;
        frame.channels = opencv_frame.channels();
        frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
        
        // 转换 OpenCV Mat 到字节数组
        size_t data_size = opencv_frame.total() * opencv_frame.elemSize();
        frame.data.resize(data_size);
        std::memcpy(frame.data.data(), opencv_frame.data, data_size);
        
        return true;
#else
        return false;
#endif
    }
};

VideoCapture::VideoCapture() : pImpl(std::make_unique<Impl>()) {}

VideoCapture::~VideoCapture() {
    stop_capture();
}

bool VideoCapture::initialize(VideoSource source, int device_id) {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    
    if (pImpl->capturing) {
        std::cout << "视频捕获已在运行，请先停止" << std::endl;
        return false;
    }
    
    pImpl->source = source;
    pImpl->device_id = device_id;
    
    switch (source) {
        case VideoSource::DESKTOP_CAPTURE:
            return pImpl->initialize_desktop_capture();
        case VideoSource::CAMERA:
            return pImpl->initialize_camera_capture(device_id);
        default:
            std::cout << "未知的视频源类型" << std::endl;
            return false;
    }
}

bool VideoCapture::start_capture() {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    
    if (pImpl->capturing) {
        std::cout << "视频捕获已在运行" << std::endl;
        return true;
    }
    
    if (pImpl->source == VideoSource::NONE) {
        std::cout << "请先初始化视频源" << std::endl;
        return false;
    }
    
#ifdef HAVE_GSTREAMER
    if (pImpl->pipeline) {
        // 启动 GStreamer 管道
        GstStateChangeReturn ret = gst_element_set_state(pImpl->pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cout << "启动 GStreamer 管道失败" << std::endl;
            return false;
        }
        std::cout << "GStreamer 管道已启动" << std::endl;
    }
#endif
    
#ifdef __APPLE__
    // 如果是桌面捕获，总是尝试启动ScreenCaptureKit
    if (pImpl->source == VideoSource::DESKTOP_CAPTURE) {
        std::cout << "尝试启动 ScreenCaptureKit..." << std::endl;
        if (duorou::media::start_macos_screen_capture(pImpl->frame_callback)) {
            std::cout << "ScreenCaptureKit 已成功启动" << std::endl;
        } else {
            std::cout << "启动 ScreenCaptureKit 失败，将使用后备实现" << std::endl;
            // 继续执行，使用后备的测试图案
        }
    }
#endif
    
    pImpl->capturing = true;
    pImpl->capture_thread = std::thread(&VideoCapture::Impl::capture_loop, pImpl.get());
    
    std::cout << "开始视频捕获" << std::endl;
    return true;
}

void VideoCapture::stop_capture() {
    bool was_capturing = false;
    {
        std::lock_guard<std::mutex> lock(pImpl->mutex);
        was_capturing = pImpl->capturing;
        pImpl->capturing = false;
    }
    
    if (!was_capturing) {
        return; // 已经停止，避免重复操作
    }
    
#ifdef __APPLE__
    // 停止 ScreenCaptureKit
    if (pImpl->source == VideoSource::DESKTOP_CAPTURE) {
        duorou::media::stop_macos_screen_capture();
        std::cout << "ScreenCaptureKit 已停止" << std::endl;
    }
#endif
    
    if (pImpl->capture_thread.joinable()) {
        pImpl->capture_thread.join();
    }
    
#ifdef HAVE_GSTREAMER
    if (pImpl->pipeline) {
        // 停止 GStreamer 管道
        gst_element_set_state(pImpl->pipeline, GST_STATE_NULL);
        std::cout << "GStreamer 管道已停止" << std::endl;
    }
#endif
    
#ifdef HAVE_OPENCV
    if (pImpl->opencv_capture.isOpened()) {
        pImpl->opencv_capture.release();
    }
#endif
    
    // 在所有资源清理完成后再清除回调函数
    {
        std::lock_guard<std::mutex> lock(pImpl->mutex);
        pImpl->frame_callback = nullptr;
    }
    
    std::cout << "停止视频捕获" << std::endl;
}

bool VideoCapture::is_capturing() const {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    return pImpl->capturing;
}

bool VideoCapture::get_next_frame(VideoFrame& frame) {
    // 这个方法在当前实现中不使用，因为我们使用回调机制
    return false;
}

void VideoCapture::set_frame_callback(std::function<void(const VideoFrame&)> callback) {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    pImpl->frame_callback = callback;
}

std::vector<std::string> VideoCapture::get_camera_devices() {
    std::vector<std::string> devices;
    
#ifdef HAVE_OPENCV
    // 尝试打开前几个设备
    for (int i = 0; i < 5; ++i) {
        cv::VideoCapture test_cap(i);
        if (test_cap.isOpened()) {
            devices.push_back("摄像头设备 " + std::to_string(i));
            test_cap.release();
        }
    }
#endif
    
    return devices;
}

bool VideoCapture::is_camera_available() {
#ifdef HAVE_OPENCV
    cv::VideoCapture test_cap(0);
    bool available = test_cap.isOpened();
    if (available) {
        test_cap.release();
    }
    return available;
#else
    return false;
#endif
}

std::pair<int, int> VideoCapture::get_desktop_resolution() {
#ifdef __APPLE__
    CGDirectDisplayID display = CGMainDisplayID();
    size_t width = CGDisplayPixelsWide(display);
    size_t height = CGDisplayPixelsHigh(display);
    return {static_cast<int>(width), static_cast<int>(height)};
#elif defined(__linux__)
    // Linux X11 实现
    Display* display = XOpenDisplay(nullptr);
    if (display) {
        Screen* screen = DefaultScreenOfDisplay(display);
        int width = WidthOfScreen(screen);
        int height = HeightOfScreen(screen);
        XCloseDisplay(display);
        return {width, height};
    }
#endif
    return {1920, 1080}; // 默认分辨率
}

} // namespace media
} // namespace duorou

#ifdef __APPLE__
#pragma clang diagnostic pop
#endif