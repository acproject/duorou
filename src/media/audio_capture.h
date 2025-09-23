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
    
    // Initialize audio capture
    bool initialize(AudioSource source, int device_id = -1);
    
    // Start capture
    bool start_capture();
    
    // Stop capture
    void stop_capture();
    
    // Check if capturing
    bool is_capturing() const;
    
    // Set audio frame callback function
    void set_frame_callback(std::function<void(const AudioFrame&)> callback);
    
    // Get list of available audio input devices
    static std::vector<std::string> get_input_devices();
    
    // Check if microphone is available
    static bool is_microphone_available();
    
    // Set audio parameters
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