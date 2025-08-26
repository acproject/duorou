#ifndef DUOROU_MEDIA_AUDIO_CAPTURE_H
#define DUOROU_MEDIA_AUDIO_CAPTURE_H

#include <string>
#include <functional>
#include <memory>
#include <vector>

#ifdef HAVE_PORTAUDIO
#include <portaudio.h>
#endif

namespace duorou {
namespace media {

enum class AudioSource {
    MICROPHONE,
    SYSTEM_AUDIO,
    NONE
};

struct AudioFrame {
    std::vector<float> data;
    int sample_rate;
    int channels;
    int frame_count;
    double timestamp;
};

class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();
    
    // 初始化音频捕获
    bool initialize(AudioSource source, int device_id = -1);
    
    // 开始捕获
    bool start_capture();
    
    // 停止捕获
    void stop_capture();
    
    // 检查是否正在捕获
    bool is_capturing() const;
    
    // 设置音频帧回调函数
    void set_frame_callback(std::function<void(const AudioFrame&)> callback);
    
    // 获取可用的音频输入设备列表
    static std::vector<std::string> get_input_devices();
    
    // 检查麦克风是否可用
    static bool is_microphone_available();
    
    // 设置音频参数
    void set_sample_rate(int sample_rate);
    void set_channels(int channels);
    void set_frames_per_buffer(int frames);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace media
} // namespace duorou

#endif // DUOROU_MEDIA_AUDIO_CAPTURE_H