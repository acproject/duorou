#include "video_capture.h"
#include "video_frame.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

#ifdef __APPLE__
#include "macos_screen_capture.h"
#endif
#ifdef __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#ifdef HAVE_GSTREAMER
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
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
  int window_id = -1;  // Window ID for desktop capture, -1 means entire desktop
  int camera_device_index = 0;  // Camera device index
  bool capturing = false;
  std::function<void(const VideoFrame &)> frame_callback;
  std::thread capture_thread;
  std::mutex mutex;

  ~Impl() {
    std::cout << "VideoCapture::Impl destructor started" << std::endl;

    // 1. First set stop flag
    capturing = false;

    // 2. Wait for capture thread to completely finish
    if (capture_thread.joinable()) {
      std::cout << "Waiting for capture thread to finish..." << std::endl;
      capture_thread.join();
      std::cout << "Capture thread finished" << std::endl;
    }

    // 3. Clean up platform-specific resources
#ifdef __APPLE__
    duorou::media::cleanup_macos_screen_capture();
#endif

    // 4. Clean up GStreamer resources
#ifdef HAVE_GSTREAMER
    cleanup_gstreamer();
#endif

    // 5. Clean up OpenCV resources
#ifdef HAVE_OPENCV
    if (opencv_capture.isOpened()) {
      opencv_capture.release();
      std::cout << "OpenCV capture released" << std::endl;
    }
#endif

    std::cout << "VideoCapture::Impl destructor completed" << std::endl;
  }

#ifdef HAVE_GSTREAMER
  GstElement *pipeline = nullptr;
  GstElement *appsink = nullptr;
  GMainLoop *loop = nullptr;
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
      std::cout << "GStreamer initialized successfully" << std::endl;
    }
    return true;
#else
    std::cout << "GStreamer not enabled" << std::endl;
    return false;
#endif
  }

  void cleanup_gstreamer() {
#ifdef HAVE_GSTREAMER
    if (pipeline) {
      // Safely stop the pipeline
      GstStateChangeReturn ret =
          gst_element_set_state(pipeline, GST_STATE_NULL);

      // Wait for state change to complete, avoiding race conditions
      if (ret != GST_STATE_CHANGE_FAILURE) {
        GstState state;
        GstState pending;
        // Wait up to 2 seconds for pipeline to completely stop
        ret = gst_element_get_state(pipeline, &state, &pending, 2 * GST_SECOND);
        if (ret == GST_STATE_CHANGE_ASYNC) {
          std::cout << "Warning: GStreamer pipeline stop still in progress" << std::endl;
        }
      }

      // Clean up appsink reference (it's part of pipeline, will be auto-released)
      appsink = nullptr;

      // Release pipeline
      gst_object_unref(pipeline);
      pipeline = nullptr;
      std::cout << "GStreamer pipeline safely cleaned up" << std::endl;
    }

    if (loop) {
      // Safely exit main loop
      if (g_main_loop_is_running(loop)) {
        g_main_loop_quit(loop);
      }

      // Wait for GStreamer thread to finish
      if (gst_thread.joinable()) {
        gst_thread.join();
      }

      // Release main loop
      g_main_loop_unref(loop);
      loop = nullptr;
      std::cout << "GStreamer main loop safely cleaned up" << std::endl;
    }
#endif
  }

  bool try_gstreamer_desktop_capture() {
#ifdef HAVE_GSTREAMER
    if (!initialize_gstreamer()) {
      std::cout << "GStreamer initialization failed" << std::endl;
      return false;
    }

    // Create desktop capture pipeline
#ifdef __APPLE__
    // macOS uses avfvideosrc to capture screen, avoiding GTK dependency
    std::string pipeline_str;
    if (window_id == -1 || window_id == 0) {
      // Capture entire desktop
      pipeline_str =
          "avfvideosrc capture-screen=true ! "
          "video/x-raw,format=BGRA,width=1280,height=720,framerate=15/1 ! "
          "videoconvert ! "
          "video/x-raw,format=RGB ! "
          "appsink name=sink emit-signals=true sync=false max-buffers=1 "
          "drop=true";
    } else {
      // avfvideosrc doesn't directly support window capture, fallback to entire desktop
      std::cout << "avfvideosrc does not support specific window capture, will capture entire desktop" << std::endl;
      pipeline_str =
          "avfvideosrc capture-screen=true ! "
          "video/x-raw,format=BGRA,width=1280,height=720,framerate=15/1 ! "
          "videoconvert ! "
          "video/x-raw,format=RGB ! "
          "appsink name=sink emit-signals=true sync=false max-buffers=1 "
          "drop=true";
    }
#elif defined(__linux__)
    // Linux uses ximagesrc to capture screen
    std::string pipeline_str;
    if (window_id == -1 || window_id == 0) {
      // Capture entire desktop
      pipeline_str =
          "ximagesrc ! "
          "videoconvert ! "
          "video/x-raw,format=RGB,width=1280,height=720,framerate=30/1 ! "
          "appsink name=sink";
    } else {
      // Capture specific window
      pipeline_str =
          "ximagesrc xid=" + std::to_string(window_id) + " ! "
          "videoconvert ! "
          "video/x-raw,format=RGB,width=1280,height=720,framerate=30/1 ! "
          "appsink name=sink";
    }
#else
    std::cout << "Current platform does not support GStreamer desktop capture" << std::endl;
    return false;
#endif

    GError *error = nullptr;
    pipeline = gst_parse_launch(pipeline_str.c_str(), &error);

    if (!pipeline || error) {
      std::cout << "Failed to create GStreamer pipeline: "
                << (error ? error->message : "unknown error") << std::endl;
      std::cout << "This may be due to permission issues or device unavailability" << std::endl;
      if (error)
        g_error_free(error);
      return false;
    }

    appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!appsink) {
      std::cout << "Failed to get appsink" << std::endl;
      gst_object_unref(pipeline);
      pipeline = nullptr;
      return false;
    }

    // appsink is already configured in pipeline, no additional setup needed

    // Test if pipeline can start to PLAYING state to check permissions
    GstStateChangeReturn ret =
        gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      std::cout << "GStreamer pipeline cannot start, may need screen recording permission" << std::endl;
      std::cout
          << "Please allow this app in System Preferences > Security & Privacy > Privacy > Screen Recording"
          << std::endl;
      cleanup_gstreamer();
      return false;
    }

    // Wait for state change to complete
    GstState state;
    ret = gst_element_get_state(pipeline, &state, nullptr, GST_CLOCK_TIME_NONE);
    if (state != GST_STATE_PLAYING) {
      std::cout << "GStreamer pipeline state abnormal, current state: " << state << std::endl;
      std::cout << "This usually indicates no screen recording permission" << std::endl;
      cleanup_gstreamer();
      return false;
    }

    // Restore to NULL state, wait for actual startup
    gst_element_set_state(pipeline, GST_STATE_NULL);

    std::cout << "Screen recording permission check passed" << std::endl;

    std::cout << "GStreamer desktop capture initialized successfully" << std::endl;
    return true;
#else
    return false;
#endif
  }

  bool initialize_desktop_capture() {
    // First try GStreamer
    if (try_gstreamer_desktop_capture()) {
      return true;
    }

    // Fallback to original implementation
    std::cout << "Fallback to simplified implementation" << std::endl;
#ifdef __APPLE__
    // Try using ScreenCaptureKit
    if (duorou::media::initialize_macos_screen_capture()) {
      std::cout << "ScreenCaptureKit desktop capture initialized successfully" << std::endl;
      return true;
    }
    std::cout << "ScreenCaptureKit initialization failed" << std::endl;
    return false;
#elif defined(__linux__)
    std::cout << "Initialize Linux desktop capture (simplified version)" << std::endl;
    return true;
#else
    std::cout << "Current platform does not support desktop capture" << std::endl;
    return false;
#endif
  }

  bool initialize_camera_capture(int device_id) {
    // If camera_device_index is -1, it means camera is disabled
    if (camera_device_index == -1) {
      return false;
    }
    
    // Use camera_device_index instead of device_id
    int actual_device_id = camera_device_index;
    
#ifdef HAVE_GSTREAMER
    if (!initialize_gstreamer()) {
      return false;
    }

    // Create camera capture pipeline
#ifdef __APPLE__
    // macOS uses avfvideosrc to capture camera
    std::string pipeline_str =
        "avfvideosrc device-index=" + std::to_string(actual_device_id) +
        " ! "
        "videoconvert ! "
        "video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! "
        "appsink name=sink";
#elif defined(__linux__)
    // Linux uses v4l2src to capture camera
    std::string pipeline_str =
        "v4l2src device=/dev/video" + std::to_string(actual_device_id) +
        " ! "
        "videoconvert ! "
        "video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! "
        "appsink name=sink";
#else
    std::cout << "Current platform does not support GStreamer camera capture" << std::endl;
    return false;
#endif

    GError *error = nullptr;
    pipeline = gst_parse_launch(pipeline_str.c_str(), &error);

    if (!pipeline || error) {
      std::cout << "Failed to create camera GStreamer pipeline: "
                << (error ? error->message : "Unknown error") << std::endl;
      if (error)
        g_error_free(error);
      return false;
    }

    appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!appsink) {
      std::cout << "Failed to get camera appsink" << std::endl;
      gst_object_unref(pipeline);
      pipeline = nullptr;
      return false;
    }

    // Configure appsink
    g_object_set(appsink, "emit-signals", TRUE, "sync", FALSE, nullptr);

    std::cout << "GStreamer camera capture initialized successfully, device: " << actual_device_id
              << std::endl;
    return true;
#elif defined(HAVE_OPENCV)
    // Fallback to OpenCV implementation
    opencv_capture.open(actual_device_id);
    if (!opencv_capture.isOpened()) {
      std::cout << "Unable to open camera device " << actual_device_id << std::endl;
      return false;
    }

    // Set camera parameters
    opencv_capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    opencv_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    opencv_capture.set(cv::CAP_PROP_FPS, 30);

    std::cout << "Successfully initialized camera device " << actual_device_id << " (OpenCV)"
              << std::endl;
    return true;
#else
    std::cout << "Neither GStreamer nor OpenCV is enabled, cannot use camera capture"
              << std::endl;
    return false;
#endif
  }

  void capture_loop() {
#ifdef HAVE_GSTREAMER
    if (pipeline && appsink) {
      // GStreamer capture loop
      while (capturing) {
        try {
          // Check basic state
          if (!capturing) {
            break;
          }

          // Add stricter null pointer checks
          if (!pipeline || !appsink ||
              gst_element_get_state(pipeline, nullptr, nullptr, 0) ==
                  GST_STATE_CHANGE_FAILURE) {
            std::cout << "GStreamer: Pipeline state abnormal, exiting capture loop" << std::endl;
            break;
          }

          GstSample *sample = nullptr;
          sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));

          if (sample) {
            GstBuffer *buffer = gst_sample_get_buffer(sample);
            GstCaps *caps = gst_sample_get_caps(sample);

            if (buffer && caps && GST_IS_BUFFER(buffer) && GST_IS_CAPS(caps)) {
              GstMapInfo map;
              if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
                // Check data validity
                if (map.data && map.size > 0) {
                  // Get video information
                  GstStructure *structure = gst_caps_get_structure(caps, 0);
                  if (structure) {
                    int width, height;
                    if (gst_structure_get_int(structure, "width", &width) &&
                        gst_structure_get_int(structure, "height", &height) &&
                        width > 0 && height > 0) {
                      // Create video frame
                      VideoFrame frame;
                      frame.width = width;
                      frame.height = height;
                      frame.channels = 3; // RGB
                      frame.data.resize(map.size);
                      std::memcpy(frame.data.data(), map.data, map.size);
                      frame.timestamp =
                          std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now()
                                  .time_since_epoch())
                              .count() /
                          1000.0;

                      std::cout << "Received video frame: " << width << "x" << height
                                << ", size: " << map.size << " bytes"
                                << std::endl;

                      if (frame_callback) {
                        frame_callback(frame);
                      }
                    } else {
                      std::cout << "GStreamer: Invalid frame dimensions" << std::endl;
                    }
                  } else {
                    std::cout << "GStreamer: Unable to get caps structure" << std::endl;
                  }
                } else {
                  std::cout << "GStreamer: Invalid buffer data" << std::endl;
                }
                gst_buffer_unmap(buffer, &map);
              } else {
                std::cout << "GStreamer: Unable to map buffer" << std::endl;
              }
            } else {
              std::cout << "GStreamer: Invalid buffer or caps" << std::endl;
            }

            gst_sample_unref(sample);
          } else {
            // No available samples, wait a bit
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
        } catch (const std::exception &e) {
          std::cout << "GStreamer capture exception: " << e.what() << std::endl;
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (...) {
          std::cout << "GStreamer capture unknown exception" << std::endl;
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      }
    } else {
#endif
      // Original capture loop (OpenCV or simulated data)
      while (capturing) {
        VideoFrame frame;

        if (source == VideoSource::DESKTOP_CAPTURE) {
#ifdef __APPLE__
          // On macOS, if ScreenCaptureKit is running, rely completely on callback mechanism
          // Don't generate any fallback data to avoid flickering from mixing with real screen data
          if (duorou::media::is_macos_screen_capture_running()) {
            // ScreenCaptureKit is running, only wait for callback data, don't generate test patterns
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
            continue;
          }
          // Only use fallback implementation when ScreenCaptureKit is completely not running
          // In this case generating test patterns is safe because it won't mix with real data
          if (capture_desktop_frame(frame)) {
            if (frame_callback) {
              frame_callback(frame);
            }
          }
#else
        // Non-macOS platforms use standard desktop capture
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

        // Control frame rate (30 FPS)
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
      }
#ifdef HAVE_GSTREAMER
    }
#endif
  }

  bool capture_desktop_frame(VideoFrame &frame) {
#ifdef __APPLE__
    // macOS desktop capture implementation - using ScreenCaptureKit
    // Note: This function is no longer directly used in the new implementation, because ScreenCaptureKit uses callback mechanism
    // Kept as fallback solution, using simplified test patterns
    frame.width = 1920;
    frame.height = 1080;
    frame.channels = 4; // RGBA
    frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count() /
                      1000.0;

    // Create simple test pattern
    frame.data.resize(1920 * 1080 * 4);
    for (int y = 0; y < 1080; ++y) {
      for (int x = 0; x < 1920; ++x) {
        int index = (y * 1920 + x) * 4;
        // Create color gradient pattern
        frame.data[index] = static_cast<uint8_t>((x * 255) / 1920);     // R
        frame.data[index + 1] = static_cast<uint8_t>((y * 255) / 1080); // G
        frame.data[index + 2] = 128;                                    // B
        frame.data[index + 3] = 255;                                    // A
      }
    }

    std::cout << "Using fallback desktop data (color test pattern)" << std::endl;
    return true;
#elif defined(__linux__)
    // Linux X11 desktop screenshot implementation
    frame.width = 640;
    frame.height = 480;
    frame.channels = 3; // RGB
    frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count() /
                      1000.0;

    // Simplified implementation: create blank frame data
    frame.data.resize(640 * 480 * 3);
    std::fill(frame.data.begin(), frame.data.end(), 128); // Gray fill
    return true;
#else
    return false;
#endif
  }

  bool capture_camera_frame(VideoFrame &frame) {
#ifdef HAVE_OPENCV
    cv::Mat opencv_frame;
    if (!opencv_capture.read(opencv_frame)) {
      return false;
    }

    frame.width = opencv_frame.cols;
    frame.height = opencv_frame.rows;
    frame.channels = opencv_frame.channels();
    frame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count() /
                      1000.0;

    // Convert OpenCV Mat to byte array
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
  std::cout << "VideoCapture destructor started" << std::endl;

  // 1. First stop capture, this will set stop flag
  stop_capture();

  std::cout << "VideoCapture destructor completed" << std::endl;
}

bool VideoCapture::initialize(VideoSource source, int device_id) {
  std::lock_guard<std::mutex> lock(pImpl->mutex);

  if (pImpl->capturing) {
    std::cout << "Video capture is already running, please stop first" << std::endl;
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
    std::cout << "Unknown video source type" << std::endl;
    return false;
  }
}

bool VideoCapture::start_capture() {
  std::lock_guard<std::mutex> lock(pImpl->mutex);

  if (pImpl->capturing) {
    std::cout << "Video capture is already running" << std::endl;
    return true;
  }

  if (pImpl->source == VideoSource::NONE) {
    std::cout << "Please initialize video source first" << std::endl;
    return false;
  }

#ifdef HAVE_GSTREAMER
  if (pImpl->pipeline) {
    // Start GStreamer pipeline
    GstStateChangeReturn ret =
        gst_element_set_state(pImpl->pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      std::cout << "Failed to start GStreamer pipeline" << std::endl;
      return false;
    }
    std::cout << "GStreamer pipeline started" << std::endl;
  }
#endif

#ifdef __APPLE__
  // If desktop capture, always try to start ScreenCaptureKit
  if (pImpl->source == VideoSource::DESKTOP_CAPTURE) {
    std::cout << "Attempting to start ScreenCaptureKit..." << std::endl;
    if (duorou::media::start_macos_screen_capture(pImpl->frame_callback, pImpl->window_id)) {
      std::cout << "ScreenCaptureKit started successfully" << std::endl;
    } else {
      std::cout << "Failed to start ScreenCaptureKit, will use fallback implementation" << std::endl;
      // Continue execution, use fallback test pattern
    }
  }
#endif

  pImpl->capturing = true;
  pImpl->capture_thread =
      std::thread(&VideoCapture::Impl::capture_loop, pImpl.get());

  std::cout << "Starting video capture" << std::endl;
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
    return; // Already stopped, avoid duplicate operations
  }

#ifdef __APPLE__
  // Stop ScreenCaptureKit
  if (pImpl->source == VideoSource::DESKTOP_CAPTURE) {
    duorou::media::stop_macos_screen_capture();
    std::cout << "ScreenCaptureKit stopped" << std::endl;
  }
#endif

  if (pImpl->capture_thread.joinable()) {
    pImpl->capture_thread.join();
  }

#ifdef HAVE_GSTREAMER
  if (pImpl->pipeline) {
    // Stop GStreamer pipeline
    gst_element_set_state(pImpl->pipeline, GST_STATE_NULL);
    std::cout << "GStreamer pipeline stopped" << std::endl;
  }
#endif

#ifdef HAVE_OPENCV
  if (pImpl->opencv_capture.isOpened()) {
    pImpl->opencv_capture.release();
  }
#endif

  // Clear callback function after all resources are cleaned up
  {
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    pImpl->frame_callback = nullptr;
  }

  std::cout << "Stopping video capture" << std::endl;
}

bool VideoCapture::is_capturing() const {
  std::lock_guard<std::mutex> lock(pImpl->mutex);
  return pImpl->capturing;
}

bool VideoCapture::get_next_frame(VideoFrame &frame) {
  // This method is not used in current implementation because we use callback mechanism
  return false;
}

void VideoCapture::set_frame_callback(
    std::function<void(const VideoFrame &)> callback) {
  std::lock_guard<std::mutex> lock(pImpl->mutex);
  pImpl->frame_callback = callback;
}

std::vector<std::string> VideoCapture::get_camera_devices() {
  std::vector<std::string> devices;

#ifdef HAVE_OPENCV
  // Try to open first few devices
  for (int i = 0; i < 5; ++i) {
    cv::VideoCapture test_cap(i);
    if (test_cap.isOpened()) {
      devices.push_back("Camera device " + std::to_string(i));
      test_cap.release();
    }
  }
#endif

  return devices;
}

bool VideoCapture::is_camera_available() {
#ifdef __APPLE__
  // On macOS use AVFoundation to detect camera
  return duorou::media::is_macos_camera_available();
#elif defined(HAVE_OPENCV)
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
  // Linux X11 implementation
  Display *display = XOpenDisplay(nullptr);
  if (display) {
    Screen *screen = DefaultScreenOfDisplay(display);
    int width = WidthOfScreen(screen);
    int height = HeightOfScreen(screen);
    XCloseDisplay(display);
    return {width, height};
  }
#endif
  return {1920, 1080}; // Default resolution
}

void VideoCapture::set_capture_window_id(int window_id) {
  std::lock_guard<std::mutex> lock(pImpl->mutex);
  pImpl->window_id = window_id;
  std::cout << "Set desktop capture window ID: " << window_id << std::endl;
}

void VideoCapture::set_camera_device_index(int device_index) {
  std::lock_guard<std::mutex> lock(pImpl->mutex);
  pImpl->camera_device_index = device_index;
  std::cout << "Set camera device index: " << device_index << std::endl;
}

} // namespace media
} // namespace duorou

#ifdef __APPLE__
#pragma clang diagnostic pop
#endif
