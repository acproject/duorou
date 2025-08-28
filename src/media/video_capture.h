#ifndef DUOROU_MEDIA_VIDEO_CAPTURE_H
#define DUOROU_MEDIA_VIDEO_CAPTURE_H

#ifdef __cplusplus
#include <string>
#include <functional>
#include <memory>
#include <vector>
#include "video_frame.h"

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#ifdef HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
}
#endif

namespace duorou {
namespace media {

enum class VideoSource {
    DESKTOP_CAPTURE,
    CAMERA,
    NONE
};



class VideoCapture {
public:
    VideoCapture();
    ~VideoCapture();
    
    // 初始化视频捕获
    bool initialize(VideoSource source, int device_id = 0);
    
    // 开始捕获
    bool start_capture();
    
    // 停止捕获
    void stop_capture();
    
    // 检查是否正在捕获
    bool is_capturing() const;
    
    // 获取下一帧
    bool get_next_frame(VideoFrame& frame);
    
    // 设置帧回调函数
    void set_frame_callback(std::function<void(const VideoFrame&)> callback);
    
    // 获取可用的摄像头设备列表
    static std::vector<std::string> get_camera_devices();
    
    // 检查摄像头是否可用
    static bool is_camera_available();
    
    // 获取桌面分辨率
    static std::pair<int, int> get_desktop_resolution();
    
    // 设置桌面捕获的窗口ID（仅在桌面捕获模式下有效）
    void set_capture_window_id(int window_id);
    
    // 设置摄像头设备索引（仅在摄像头模式下有效）
    void set_camera_device_index(int device_index);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace media
} // namespace duorou

#endif // __cplusplus
#endif // DUOROU_MEDIA_VIDEO_CAPTURE_H