#include "src/media/video_capture.h"
#include "src/media/macos_screen_capture.h"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "测试 ScreenCaptureKit 桌面捕获..." << std::endl;
    
    // 初始化 ScreenCaptureKit
    if (duorou::media::initialize_macos_screen_capture()) {
        std::cout << "ScreenCaptureKit 初始化成功" << std::endl;
        
        // 设置帧回调
        auto frame_callback = [](const duorou::media::VideoFrame& frame) {
            std::cout << "收到视频帧: " << frame.width << "x" << frame.height 
                      << ", 时间戳: " << frame.timestamp << std::endl;
        };
        
        // 启动捕获
        if (duorou::media::start_macos_screen_capture(frame_callback)) {
            std::cout << "ScreenCaptureKit 启动成功，等待5秒..." << std::endl;
            
            // 等待5秒接收帧
            std::this_thread::sleep_for(std::chrono::seconds(5));
            
            // 停止捕获
            duorou::media::stop_macos_screen_capture();
            std::cout << "ScreenCaptureKit 已停止" << std::endl;
        } else {
            std::cout << "ScreenCaptureKit 启动失败" << std::endl;
        }
        
        // 清理
        duorou::media::cleanup_macos_screen_capture();
    } else {
        std::cout << "ScreenCaptureKit 初始化失败" << std::endl;
    }
    
    return 0;
}