#include "audio_capture.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

namespace duorou {
namespace media {

// Global PortAudio management
static std::mutex g_pa_mutex;
static bool g_pa_initialized = false;

static bool ensure_portaudio_initialized() {
  // Initialize PortAudio
#ifdef HAVE_PORTAUDIO
  std::lock_guard<std::mutex> lock(g_pa_mutex);
  if (!g_pa_initialized) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
      std::cout << "PortAudio initialization failed: " << Pa_GetErrorText(err)
                << std::endl;
      return false;
    }
    g_pa_initialized = true;
    std::cout << "PortAudio initialized successfully" << std::endl;
  }
  return true;
#else
  return false;
#endif
}

static void release_portaudio() {
  // Release PortAudio
#ifdef HAVE_PORTAUDIO
  (void)g_pa_mutex;
  (void)g_pa_initialized;
#endif
}

class AudioCapture::Impl {
public:
  AudioSource source = AudioSource::NONE;
  int device_id = -1;
  bool capturing = false;
  std::function<void(const AudioFrame &)> frame_callback;
  std::mutex mutex;

  // Audio parameters
  int sample_rate = 44100;
  int channels = 2;
  int frames_per_buffer = 1024;

#ifdef HAVE_PORTAUDIO
  PaStream *pa_stream = nullptr;

  static int audio_callback(const void *input_buffer, void *output_buffer,
                            unsigned long frame_count,
                            const PaStreamCallbackTimeInfo *time_info,
                            PaStreamCallbackFlags status_flags,
                            void *user_data) {

    Impl *impl = static_cast<Impl *>(user_data);

    if (impl->frame_callback && input_buffer) {
      AudioFrame frame;
      frame.sample_rate = impl->sample_rate;
      frame.channels = impl->channels;
      frame.frame_count = static_cast<int>(frame_count);
      frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count() /
                        1000.0;

      // Copy audio data
      const float *input = static_cast<const float *>(input_buffer);
      size_t sample_count = frame_count * impl->channels;
      frame.data.resize(sample_count);
      std::memcpy(frame.data.data(), input, sample_count * sizeof(float));

      impl->frame_callback(frame);
    }

    return paContinue;
  }
#endif

  // initialize_portaudio method removed, now using global PortAudio management

  bool setup_input_stream() {
#ifdef HAVE_PORTAUDIO
    PaStreamParameters input_params;

    if (device_id >= 0) {
      input_params.device = device_id;
    } else {
      input_params.device = Pa_GetDefaultInputDevice();
    }

    if (input_params.device == paNoDevice) {
      std::cout << "No available audio input device found" << std::endl;
      return false;
    }

    // Get device info and check supported channel count
    const PaDeviceInfo *device_info = Pa_GetDeviceInfo(input_params.device);
    if (!device_info) {
      std::cout << "Unable to get audio device information" << std::endl;
      return false;
    }

    // Adjust channel count to match device's maximum input channels
    int max_input_channels = device_info->maxInputChannels;
    if (channels > max_input_channels) {
      channels = max_input_channels;
      std::cout << "Adjusted audio channel count to: " << channels
                << " (device maximum support: " << max_input_channels << ")" << std::endl;
    }

    // Ensure at least 1 channel
    if (channels <= 0) {
      channels = 1;
    }

    input_params.channelCount = channels;
    input_params.sampleFormat = paFloat32;
    input_params.suggestedLatency = device_info->defaultLowInputLatency;
    input_params.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(&pa_stream, &input_params,
                                nullptr, // No output
                                sample_rate, frames_per_buffer, paClipOff,
                                audio_callback, this);

    if (err != paNoError) {
      std::cout << "Failed to open audio stream: " << Pa_GetErrorText(err) << std::endl;
      return false;
    }

    std::cout << "Audio input stream setup successful" << std::endl;
    return true;
#else
    return false;
#endif
  }
};

AudioCapture::AudioCapture() : pImpl(std::make_unique<Impl>()) {
  // Ensure PortAudio is initialized during construction
  ensure_portaudio_initialized();
}

AudioCapture::~AudioCapture() {
  stop_capture();

#ifdef HAVE_PORTAUDIO
  release_portaudio();
#endif
}

bool AudioCapture::initialize(AudioSource source, int device_id) {
  PaStream *old_stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(pImpl->mutex);

    if (pImpl->capturing) {
      std::cout << "Audio capture is already running, please stop first" << std::endl;
      return false;
    }

#ifdef HAVE_PORTAUDIO
    old_stream = pImpl->pa_stream;
    pImpl->pa_stream = nullptr;
#endif

    pImpl->source = source;
    pImpl->device_id = device_id;
  }

#ifdef HAVE_PORTAUDIO
  if (old_stream) {
    Pa_CloseStream(old_stream);
  }
#endif

  switch (source) {
  case AudioSource::MICROPHONE:
  case AudioSource::SYSTEM_AUDIO:
    return pImpl->setup_input_stream();
  default:
    std::cout << "Unknown audio source type" << std::endl;
    return false;
  }
}

bool AudioCapture::start_capture() {
  std::lock_guard<std::mutex> lock(pImpl->mutex);

  if (pImpl->capturing) {
    std::cout << "Audio capture is already running" << std::endl;
    return true;
  }

  if (pImpl->source == AudioSource::NONE) {
    std::cout << "Please initialize audio source first" << std::endl;
    return false;
  }

#ifdef HAVE_PORTAUDIO
  if (pImpl->pa_stream) {
    PaError err = Pa_StartStream(pImpl->pa_stream);
    if (err != paNoError) {
      std::cout << "Failed to start audio stream: " << Pa_GetErrorText(err) << std::endl;
      return false;
    }

    pImpl->capturing = true;
    std::cout << "Starting audio capture" << std::endl;
    return true;
  }
#endif

  return false;
}

void AudioCapture::stop_capture() {
  PaStream *stream = nullptr;
  bool should_print = false;
  {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    should_print = pImpl->capturing;
    pImpl->capturing = false;
#ifdef HAVE_PORTAUDIO
    stream = pImpl->pa_stream;
    pImpl->pa_stream = nullptr;
#endif
  }

#ifdef HAVE_PORTAUDIO
  if (stream) {
    should_print = true;
    PaError err = Pa_IsStreamActive(stream);
    if (err == 1) {
      Pa_StopStream(stream);
    }
    Pa_CloseStream(stream);
  }
#endif

  if (should_print) {
    std::cout << "Stopping audio capture" << std::endl;
  }
}

bool AudioCapture::is_capturing() const {
  std::lock_guard<std::mutex> lock(pImpl->mutex);
  return pImpl->capturing;
}

void AudioCapture::set_frame_callback(
    std::function<void(const AudioFrame &)> callback) {
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
    const PaDeviceInfo *device_info = Pa_GetDeviceInfo(i);
    if (device_info && device_info->maxInputChannels > 0) {
      std::string device_name = device_info->name;
      devices.push_back(device_name + " (Device " + std::to_string(i) + ")");
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
