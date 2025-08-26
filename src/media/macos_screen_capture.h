#ifndef DUOROU_MEDIA_MACOS_SCREEN_CAPTURE_H
#define DUOROU_MEDIA_MACOS_SCREEN_CAPTURE_H

#ifdef __APPLE__
#ifdef __cplusplus

#include <functional>
#include "video_frame.h"

namespace duorou {
namespace media {

// macOS ScreenCaptureKit 接口函数
bool check_screen_recording_permission();
bool initialize_macos_screen_capture();
bool start_macos_screen_capture(std::function<void(const VideoFrame&)> callback);
void stop_macos_screen_capture();
bool is_macos_screen_capture_running();
void cleanup_macos_screen_capture();

} // namespace media
} // namespace duorou

#endif // __cplusplus
#endif // __APPLE__

#endif // DUOROU_MEDIA_MACOS_SCREEN_CAPTURE_H