#include "video_display_window.h"
#include <cstring>
#include <iostream>

namespace duorou {
namespace gui {

VideoDisplayWindow::VideoDisplayWindow()
    : window_(nullptr), video_area_(nullptr), info_label_(nullptr),
      frame_data_(nullptr), frame_width_(0), frame_height_(0),
      frame_channels_(4), cached_surface_(nullptr), cached_rgba_data_(nullptr),
      cached_width_(0), cached_height_(0) {
  init_ui();
}

VideoDisplayWindow::~VideoDisplayWindow() {
  // Clean up cached Cairo surface
  if (cached_surface_) {
    cairo_surface_destroy(cached_surface_);
    cached_surface_ = nullptr;
  }

  if (window_) {
    gtk_window_destroy(GTK_WINDOW(window_));
  }
}

void VideoDisplayWindow::init_ui() {
  // Create window
  window_ = gtk_window_new();
  gtk_window_set_title(GTK_WINDOW(window_), "Video Preview");
  gtk_window_set_default_size(GTK_WINDOW(window_), 640, 480);
  gtk_window_set_resizable(GTK_WINDOW(window_), TRUE);

  // Set window level in GTK4, ensure video window is above normal windows but below modal dialogs
  gtk_window_set_modal(GTK_WINDOW(window_), FALSE);
  gtk_window_set_transient_for(GTK_WINDOW(window_), nullptr);

  // Set window type hint to make it behave as a tool window
  gtk_window_set_decorated(GTK_WINDOW(window_), TRUE);
  gtk_window_set_deletable(GTK_WINDOW(window_), TRUE);

  // Create main container
  GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
  gtk_widget_set_margin_start(vbox, 10);
  gtk_widget_set_margin_end(vbox, 10);
  gtk_widget_set_margin_top(vbox, 10);
  gtk_widget_set_margin_bottom(vbox, 10);

  // Create info label
  info_label_ = gtk_label_new("Waiting for video data...");
  gtk_widget_set_halign(info_label_, GTK_ALIGN_CENTER);

  // Create video display area
  video_area_ = gtk_drawing_area_new();
  gtk_widget_set_size_request(video_area_, 320, 240);
  gtk_widget_set_hexpand(video_area_, TRUE);
  gtk_widget_set_vexpand(video_area_, TRUE);

  // Set draw callback
  gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(video_area_), on_draw_area,
                                 this, nullptr);

  // Add to container
  gtk_box_append(GTK_BOX(vbox), info_label_);
  gtk_box_append(GTK_BOX(vbox), video_area_);

  // Set window content
  gtk_window_set_child(GTK_WINDOW(window_), vbox);

  // Connect window close signal
  g_signal_connect(window_, "close-request", G_CALLBACK(on_window_close), this);
}

void VideoDisplayWindow::show() {
  if (window_) {
    gtk_widget_set_visible(window_, TRUE);
    // Show window without forcing focus to avoid interfering with dialogs
    gtk_window_present(GTK_WINDOW(window_));
  }
}

void VideoDisplayWindow::hide() {
  if (window_) {
    gtk_widget_set_visible(window_, FALSE);
  }
}

void VideoDisplayWindow::set_close_callback(std::function<void()> callback) {
  close_callback_ = callback;
}

bool VideoDisplayWindow::is_visible() const {
  return window_ && gtk_widget_get_visible(window_);
}

void VideoDisplayWindow::update_frame(const media::VideoFrame &frame) {
  // Check if cached surface needs to be recreated
  bool need_recreate_surface =
      (frame.width != cached_width_ || frame.height != cached_height_);

  // Update frame data
  frame_width_ = frame.width;
  frame_height_ = frame.height;
  frame_channels_ = frame.channels;

  // Allocate memory to store frame data
  size_t data_size = frame.width * frame.height * frame.channels;
  frame_data_ = std::make_unique<guchar[]>(data_size);
  std::memcpy(frame_data_.get(), frame.data.data(), data_size);

  // If size changed, recreate cached surface
  if (need_recreate_surface) {
    // Clean up old surface
    if (cached_surface_) {
      cairo_surface_destroy(cached_surface_);
      cached_surface_ = nullptr;
    }

    // Update cached dimensions
    cached_width_ = frame.width;
    cached_height_ = frame.height;

    // Create new RGBA data buffer
    int stride = cairo_format_stride_for_width(CAIRO_FORMAT_RGB24, frame.width);
    cached_rgba_data_ = std::make_unique<guchar[]>(stride * frame.height);

    // Create new Cairo surface
    cached_surface_ = cairo_image_surface_create_for_data(
        cached_rgba_data_.get(), CAIRO_FORMAT_RGB24, frame.width, frame.height,
        stride);
  }

  // Update cached surface data
  if (cached_surface_ && cached_rgba_data_) {
    int stride = cairo_format_stride_for_width(CAIRO_FORMAT_RGB24, frame.width);
    int channels = frame.channels;

    // Get surface data pointer before modifying surface data
    cairo_surface_flush(cached_surface_);

    for (int y = 0; y < frame.height; y++) {
      for (int x = 0; x < frame.width; x++) {
        int src_idx = (y * frame.width + x) * channels;
        int dst_idx = y * stride + x * 4;

        if (src_idx + (channels - 1) <
            (int)(frame.width * frame.height * channels)) {
          if (channels == 4) {
            // ScreenCaptureKit uses BGRA format, convert to Cairo RGB24 format
            cached_rgba_data_[dst_idx + 0] = frame.data[src_idx + 0]; // B
            cached_rgba_data_[dst_idx + 1] = frame.data[src_idx + 1]; // G
            cached_rgba_data_[dst_idx + 2] = frame.data[src_idx + 2]; // R
            cached_rgba_data_[dst_idx + 3] = frame.data[src_idx + 3]; // A
          } else {
            // Convert RGB format to Cairo RGB24
            cached_rgba_data_[dst_idx + 0] = frame.data[src_idx + 2]; // B
            cached_rgba_data_[dst_idx + 1] = frame.data[src_idx + 1]; // G
            cached_rgba_data_[dst_idx + 2] = frame.data[src_idx + 0]; // R
            cached_rgba_data_[dst_idx + 3] = 255;                     // A
          }
        }
      }
    }

    // Mark surface data as updated, ensure Cairo knows data has been modified
    cairo_surface_mark_dirty(cached_surface_);
  }

  // Update info label - format timestamp to readable format
  char info_text[256];

  // Convert timestamp to readable time format
  auto timestamp_ms = static_cast<int64_t>(frame.timestamp * 1000);
  auto timestamp_sec = timestamp_ms / 1000;
  auto ms_part = timestamp_ms % 1000;
  auto hours = timestamp_sec / 3600;
  auto minutes = (timestamp_sec % 3600) / 60;
  auto seconds = timestamp_sec % 60;

  char timestamp_str[64];
  snprintf(timestamp_str, sizeof(timestamp_str), "%02lld:%02lld:%02lld.%03lld",
           hours, minutes, seconds, ms_part);

  snprintf(info_text, sizeof(info_text), "分辨率: %dx%d, 通道: %d, 时间戳: %s",
           frame.width, frame.height, frame.channels, timestamp_str);
  gtk_label_set_text(GTK_LABEL(info_label_), info_text);

  // Trigger redraw
  if (video_area_) {
    gtk_widget_queue_draw(video_area_);
  }
}

void VideoDisplayWindow::on_draw_area(GtkDrawingArea *area, cairo_t *cr,
                                      int width, int height,
                                      gpointer user_data) {
  VideoDisplayWindow *window = static_cast<VideoDisplayWindow *>(user_data);

  // Set background color
  cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
  cairo_paint(cr);

  // If there's a cached surface, draw it directly
  if (window->cached_surface_ && window->frame_width_ > 0 &&
      window->frame_height_ > 0) {
    // Calculate scale ratio to fit display area
    double scale_x = (double)width / window->frame_width_;
    double scale_y = (double)height / window->frame_height_;
    double scale = std::min(scale_x, scale_y);

    // Calculate centered position
    int scaled_width = (int)(window->frame_width_ * scale);
    int scaled_height = (int)(window->frame_height_ * scale);
    int x_offset = (width - scaled_width) / 2;
    int y_offset = (height - scaled_height) / 2;

    // Save current state
    cairo_save(cr);

    // Move to centered position and scale
    cairo_translate(cr, x_offset, y_offset);
    cairo_scale(cr, scale, scale);

    // Draw cached surface directly
    cairo_set_source_surface(cr, window->cached_surface_, 0, 0);
    cairo_paint(cr);

    // Restore state
    cairo_restore(cr);
  } else {
    // Show hint text when no video data
    cairo_set_source_rgb(cr, 0.7, 0.7, 0.7);
    cairo_select_font_face(cr, "Arial", CAIRO_FONT_SLANT_NORMAL,
                           CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, 16);

    const char *text = "等待视频数据...";
    cairo_text_extents_t extents;
    cairo_text_extents(cr, text, &extents);

    cairo_move_to(cr, (width - extents.width) / 2,
                  (height + extents.height) / 2);
    cairo_show_text(cr, text);
  }
}

gboolean VideoDisplayWindow::on_window_close(GtkWidget *widget,
                                             gpointer user_data) {
  // Safety check: ensure user_data is not null
  if (!user_data) {
    std::cout << "VideoDisplayWindow::on_window_close: user_data is null"
              << std::endl;
    return TRUE;
  }

  VideoDisplayWindow *window = static_cast<VideoDisplayWindow *>(user_data);

  // Safety check: ensure window object is valid
  if (!window) {
    std::cout << "VideoDisplayWindow::on_window_close: window object is null"
              << std::endl;
    return TRUE;
  }

  std::cout << "VideoDisplayWindow close event triggered" << std::endl;

  // If close callback is set, call it to stop recording
  if (window->close_callback_) {
    try {
      std::cout << "Calling video window close callback..." << std::endl;
      window->close_callback_();
      std::cout << "Video window close callback execution completed" << std::endl;
    } catch (const std::exception &e) {
      std::cout << "Video window close callback exception: " << e.what() << std::endl;
    } catch (...) {
      std::cout << "Unknown exception occurred in video window close callback" << std::endl;
    }
  }

  // Safely hide window
  try {
    window->hide();
  } catch (const std::exception &e) {
    std::cout << "Hide video window exception: " << e.what() << std::endl;
  }

  return TRUE; // Prevent window destruction, just hide
}

} // namespace gui
} // namespace duorou