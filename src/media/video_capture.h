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
    
    // Initialize video capture
    bool initialize(VideoSource source, int device_id = 0);
    
    // Start capture
    bool start_capture();
    
    // Stop capture
    void stop_capture();
    
    // Check if capturing
    bool is_capturing() const;
    
    // Get next frame
    bool get_next_frame(VideoFrame& frame);
    
    // Set frame callback function
    void set_frame_callback(std::function<void(const VideoFrame&)> callback);
    
    // Get available camera devices list
    static std::vector<std::string> get_camera_devices();
    
    // Check if camera is available
    static bool is_camera_available();
    
    // Get desktop resolution
    static std::pair<int, int> get_desktop_resolution();
    
    // Set capture window ID (only valid in desktop capture mode)
    void set_capture_window_id(int window_id);
    
    // Set camera device index (only valid in camera mode)
    void set_camera_device_index(int device_index);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace media
} // namespace duorou

#endif // __cplusplus
#endif // DUOROU_MEDIA_VIDEO_CAPTURE_H