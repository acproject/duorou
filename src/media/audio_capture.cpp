#include "audio_capture.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <cstring>
#include <atomic>

namespace duorou {
namespace media {

// 全局PortAudio管理
static std::mutex g_pa_mutex;
static std::atomic<int> g_pa_ref_count{0};
static bool g_pa_initialized = false;

static bool ensure_portaudio_initialized() {
#ifdef HAVE_PORTAUDIO
    std::lock_guard<std::mutex> lock(g_pa_mutex);
    if (!g_pa_initialized) {
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            std::cout << "PortAudio 初始化失败: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }
        g_pa_initialized = true;
        std::cout << "PortAudio 初始化成功" << std::endl;
    }
    g_pa_ref_count++;
    return true;
#else
    return false;
#endif
}

static void release_portaudio() {
#ifdef HAVE_PORTAUDIO
    std::lock_guard<std::mutex> lock(g_pa_mutex);
    if (g_pa_ref_count > 0) {
        g_pa_ref_count--;
        if (g_pa_ref_count == 0 && g_pa_initialized) {
            Pa_Terminate();
            g_pa_initialized = false;
            std::cout << "PortAudio 已终止" << std::endl;
        }
    }
#endif
}

class AudioCapture::Impl {
public:
    AudioSource source = AudioSource::NONE;
    int device_id = -1;
    bool capturing = false;
    std::function<void(const AudioFrame&)> frame_callback;
    std::mutex mutex;
    
    // 音频参数
    int sample_rate = 44100;
    int channels = 2;
    int frames_per_buffer = 1024;
    
#ifdef HAVE_PORTAUDIO
    PaStream* pa_stream = nullptr;
    
    static int audio_callback(const void* input_buffer,
                             void* output_buffer,
                             unsigned long frame_count,
                             const PaStreamCallbackTimeInfo* time_info,
                             PaStreamCallbackFlags status_flags,
                             void* user_data) {
        
        Impl* impl = static_cast<Impl*>(user_data);
        
        if (impl->frame_callback && input_buffer) {
            AudioFrame frame;
            frame.sample_rate = impl->sample_rate;
            frame.channels = impl->channels;
            frame.frame_count = static_cast<int>(frame_count);
            frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
            
            // 复制音频数据
            const float* input = static_cast<const float*>(input_buffer);
            size_t sample_count = frame_count * impl->channels;
            frame.data.resize(sample_count);
            std::memcpy(frame.data.data(), input, sample_count * sizeof(float));
            
            impl->frame_callback(frame);
        }
        
        return paContinue;
    }
#endif
    
    // initialize_portaudio方法已移除，现在使用全局PortAudio管理
    
    bool setup_input_stream() {
#ifdef HAVE_PORTAUDIO
        PaStreamParameters input_params;
        
        if (device_id >= 0) {
            input_params.device = device_id;
        } else {
            input_params.device = Pa_GetDefaultInputDevice();
        }
        
        if (input_params.device == paNoDevice) {
            std::cout << "没有找到可用的音频输入设备" << std::endl;
            return false;
        }
        
        // 获取设备信息并检查支持的通道数
        const PaDeviceInfo* device_info = Pa_GetDeviceInfo(input_params.device);
        if (!device_info) {
            std::cout << "无法获取音频设备信息" << std::endl;
            return false;
        }
        
        // 调整通道数以匹配设备支持的最大输入通道数
        int max_input_channels = device_info->maxInputChannels;
        if (channels > max_input_channels) {
            channels = max_input_channels;
            std::cout << "调整音频通道数为: " << channels << " (设备最大支持: " << max_input_channels << ")" << std::endl;
        }
        
        // 确保至少有1个通道
        if (channels <= 0) {
            channels = 1;
        }
        
        input_params.channelCount = channels;
        input_params.sampleFormat = paFloat32;
        input_params.suggestedLatency = device_info->defaultLowInputLatency;
        input_params.hostApiSpecificStreamInfo = nullptr;
        
        PaError err = Pa_OpenStream(
            &pa_stream,
            &input_params,
            nullptr, // 没有输出
            sample_rate,
            frames_per_buffer,
            paClipOff,
            audio_callback,
            this
        );
        
        if (err != paNoError) {
            std::cout << "打开音频流失败: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }
        
        std::cout << "音频输入流设置成功" << std::endl;
        return true;
#else
        return false;
#endif
    }
};

AudioCapture::AudioCapture() : pImpl(std::make_unique<Impl>()) {
    // 在构造时确保PortAudio已初始化
    ensure_portaudio_initialized();
}

AudioCapture::~AudioCapture() {
    stop_capture();
    
#ifdef HAVE_PORTAUDIO
    // 关闭流并释放PortAudio引用
    if (pImpl && pImpl->pa_stream) {
        Pa_CloseStream(pImpl->pa_stream);
        pImpl->pa_stream = nullptr;
    }
    release_portaudio();
#endif
}

bool AudioCapture::initialize(AudioSource source, int device_id) {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    
    if (pImpl->capturing) {
        std::cout << "音频捕获已在运行，请先停止" << std::endl;
        return false;
    }
    
    pImpl->source = source;
    pImpl->device_id = device_id;
    
    switch (source) {
        case AudioSource::MICROPHONE:
        case AudioSource::SYSTEM_AUDIO:
            return pImpl->setup_input_stream();
        default:
            std::cout << "未知的音频源类型" << std::endl;
            return false;
    }
}

bool AudioCapture::start_capture() {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    
    if (pImpl->capturing) {
        std::cout << "音频捕获已在运行" << std::endl;
        return true;
    }
    
    if (pImpl->source == AudioSource::NONE) {
        std::cout << "请先初始化音频源" << std::endl;
        return false;
    }
    
#ifdef HAVE_PORTAUDIO
    if (pImpl->pa_stream) {
        PaError err = Pa_StartStream(pImpl->pa_stream);
        if (err != paNoError) {
            std::cout << "启动音频流失败: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }
        
        pImpl->capturing = true;
        std::cout << "开始音频捕获" << std::endl;
        return true;
    }
#endif
    
    return false;
}

void AudioCapture::stop_capture() {
    {
        std::lock_guard<std::mutex> lock(pImpl->mutex);
        pImpl->capturing = false;
    }
    
#ifdef HAVE_PORTAUDIO
    if (pImpl->pa_stream) {
        // 安全地停止和关闭流
        PaError err = Pa_IsStreamActive(pImpl->pa_stream);
        if (err == 1) { // 流正在运行
            Pa_StopStream(pImpl->pa_stream);
        }
        
        Pa_CloseStream(pImpl->pa_stream);
        pImpl->pa_stream = nullptr;
    }
#endif
    
    std::cout << "停止音频捕获" << std::endl;
}

bool AudioCapture::is_capturing() const {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    return pImpl->capturing;
}

void AudioCapture::set_frame_callback(std::function<void(const AudioFrame&)> callback) {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    pImpl->frame_callback = callback;
}

std::vector<std::string> AudioCapture::get_input_devices() {
    std::vector<std::string> devices;
    
#ifdef HAVE_PORTAUDIO
    if (!ensure_portaudio_initialized()) {
        return devices;
    }
    
    int device_count = Pa_GetDeviceCount();
    for (int i = 0; i < device_count; ++i) {
        const PaDeviceInfo* device_info = Pa_GetDeviceInfo(i);
        if (device_info && device_info->maxInputChannels > 0) {
            std::string device_name = device_info->name;
            devices.push_back(device_name + " (设备 " + std::to_string(i) + ")");
        }
    }
    
    release_portaudio();
#endif
    
    return devices;
}

bool AudioCapture::is_microphone_available() {
#ifdef HAVE_PORTAUDIO
    if (!ensure_portaudio_initialized()) {
        return false;
    }
    
    PaDeviceIndex default_input = Pa_GetDefaultInputDevice();
    bool available = (default_input != paNoDevice);
    
    release_portaudio();
    return available;
#else
    return false;
#endif
}

void AudioCapture::set_sample_rate(int sample_rate) {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    if (!pImpl->capturing) {
        pImpl->sample_rate = sample_rate;
    }
}

void AudioCapture::set_channels(int channels) {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    if (!pImpl->capturing) {
        pImpl->channels = channels;
    }
}

void AudioCapture::set_frames_per_buffer(int frames) {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    if (!pImpl->capturing) {
        pImpl->frames_per_buffer = frames;
    }
}

} // namespace media
} // namespace duorou