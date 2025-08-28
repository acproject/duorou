#ifndef DUOROU_MEDIA_MACOS_SCREEN_CAPTURE_H
#define DUOROU_MEDIA_MACOS_SCREEN_CAPTURE_H

#ifdef __APPLE__
#ifdef __cplusplus

#include "video_frame.h"
#include <functional>

namespace duorou {
namespace media {

// macOS ScreenCaptureKit 接口函数
bool check_screen_recording_permission();
bool initialize_macos_screen_capture();
bool start_macos_screen_capture(
    std::function<void(const VideoFrame &)> callback, int window_id = -1);
void stop_macos_screen_capture();
bool is_macos_screen_capture_running();
void cleanup_macos_screen_capture();
void update_macos_screen_capture_window(int window_id);

// macOS 摄像头检测函数
bool is_macos_camera_available();

} // namespace media
} // namespace duorou

#endif // __cplusplus
#endif // __APPLE__

#endif // DUOROU_MEDIA_MACOS_SCREEN_CAPTURE_H