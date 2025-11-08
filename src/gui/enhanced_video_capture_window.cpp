#include "enhanced_video_capture_window.h"
#include "../core/logger.h"
#include <iostream>
#include <cstring>

#ifdef __APPLE__
#include <CoreGraphics/CoreGraphics.h>
#include <ApplicationServices/ApplicationServices.h>
#endif

namespace duorou {
namespace gui {

EnhancedVideoCaptureWindow::EnhancedVideoCaptureWindow()
    : window_(nullptr), main_paned_(nullptr), left_frame_(nullptr), right_frame_(nullptr),
      video_area_(nullptr), info_label_(nullptr), source_list_(nullptr), source_scrolled_(nullptr),
      mode_label_(nullptr), refresh_button_(nullptr), current_mode_(CaptureMode::DESKTOP),
      frame_data_(nullptr), frame_width_(0), frame_height_(0), frame_channels_(4),
      cached_surface_(nullptr), cached_rgba_data_(nullptr), cached_width_(0), cached_height_(0) {
}

EnhancedVideoCaptureWindow::~EnhancedVideoCaptureWindow() {
    // Clean up cached Cairo surface
    if (cached_surface_) {
        cairo_surface_destroy(cached_surface_);
        cached_surface_ = nullptr;
    }

    if (window_) {
        gtk_window_destroy(GTK_WINDOW(window_));
    }
}

bool EnhancedVideoCaptureWindow::initialize() {
    init_ui();
    setup_styling();
    return true;
}

void EnhancedVideoCaptureWindow::show(CaptureMode mode) {
    current_mode_ = mode;
    
    // Ensure window is initialized
    if (!window_) {
        std::cerr << "Error: Window not initialized before show()" << std::endl;
        return;
    }
    
    // Update mode label
    if (mode_label_) {
        const char* mode_text = (mode == CaptureMode::DESKTOP) ? "Desktop Capture Mode" : "Camera Mode";
        gtk_label_set_text(GTK_LABEL(mode_label_), mode_text);
    }
    
    // Show window
    gtk_widget_set_visible(window_, TRUE);
    gtk_window_present(GTK_WINDOW(window_));
    
    // Refresh source list after window is shown
    update_source_list();
}

void EnhancedVideoCaptureWindow::hide() {
    if (window_) {
        gtk_widget_set_visible(window_, FALSE);
    }
}

bool EnhancedVideoCaptureWindow::is_visible() const {
    return window_ && gtk_widget_get_visible(window_);
}

void EnhancedVideoCaptureWindow::set_close_callback(std::function<void()> callback) {
    close_callback_ = callback;
}

void EnhancedVideoCaptureWindow::set_window_selection_callback(std::function<void(const WindowInfo&)> callback) {
    window_selection_callback_ = callback;
}

void EnhancedVideoCaptureWindow::set_device_selection_callback(std::function<void(const DeviceInfo&)> callback) {
    device_selection_callback_ = callback;
}

void EnhancedVideoCaptureWindow::update_frame(const media::VideoFrame& frame) {
    // Check if cached surface needs to be recreated
    bool need_recreate_surface = (frame.width != cached_width_ || frame.height != cached_height_);

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
            cached_rgba_data_.get(), CAIRO_FORMAT_RGB24, frame.width, frame.height, stride);
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

                if (src_idx + (channels - 1) < (int)(frame.width * frame.height * channels)) {
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

        // Mark surface data as updated
        cairo_surface_mark_dirty(cached_surface_);
    }

    // Update info label
    if (info_label_) {
        char info_text[256];
        auto timestamp_ms = static_cast<int64_t>(frame.timestamp * 1000);
        auto timestamp_sec = timestamp_ms / 1000;
        auto ms_part = timestamp_ms % 1000;
        auto hours = timestamp_sec / 3600;
        auto minutes = (timestamp_sec % 3600) / 60;
        auto seconds = timestamp_sec % 60;

        char timestamp_str[64];
        snprintf(timestamp_str, sizeof(timestamp_str), "%02lld:%02lld:%02lld.%03lld",
                 hours, minutes, seconds, ms_part);

        snprintf(info_text, sizeof(info_text), "Resolution: %dx%d, Channels: %d, Timestamp: %s",
                 frame.width, frame.height, frame.channels, timestamp_str);
        gtk_label_set_text(GTK_LABEL(info_label_), info_text);
    }

    // Trigger redraw
    if (video_area_) {
        gtk_widget_queue_draw(video_area_);
    }
}

void EnhancedVideoCaptureWindow::init_ui() {
    // Create main window
    window_ = gtk_window_new();
    gtk_window_set_title(GTK_WINDOW(window_), "Enhanced Video Capture Window");
    gtk_window_set_default_size(GTK_WINDOW(window_), 1000, 600);
    gtk_window_set_resizable(GTK_WINDOW(window_), TRUE);

    // Create main paned panel (left-right split)
    main_paned_ = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_paned_set_position(GTK_PANED(main_paned_), 650); // Left 650px, right 350px
    gtk_window_set_child(GTK_WINDOW(window_), main_paned_);

    // Create left and right areas
    create_video_area();
    create_source_list();

    // Connect window close signal
    g_signal_connect(window_, "close-request", G_CALLBACK(on_window_close), this);
}

void EnhancedVideoCaptureWindow::create_video_area() {
    // Create left frame
    left_frame_ = gtk_frame_new("Video Preview");
    gtk_widget_set_margin_start(left_frame_, 10);
    gtk_widget_set_margin_end(left_frame_, 5);
    gtk_widget_set_margin_top(left_frame_, 10);
    gtk_widget_set_margin_bottom(left_frame_, 10);

    // Create left content container
    GtkWidget* left_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_widget_set_margin_start(left_vbox, 10);
    gtk_widget_set_margin_end(left_vbox, 10);
    gtk_widget_set_margin_top(left_vbox, 10);
    gtk_widget_set_margin_bottom(left_vbox, 10);

    // Create info label
    info_label_ = gtk_label_new("Waiting for video data...");
    gtk_widget_set_halign(info_label_, GTK_ALIGN_CENTER);
    gtk_widget_add_css_class(info_label_, "info-label");

    // Create video display area
    video_area_ = gtk_drawing_area_new();
    gtk_widget_set_size_request(video_area_, 320, 240);
    gtk_widget_set_hexpand(video_area_, TRUE);
    gtk_widget_set_vexpand(video_area_, TRUE);

    // Set draw callback
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(video_area_), on_draw_area, this, nullptr);

    // Add to left container
    gtk_box_append(GTK_BOX(left_vbox), info_label_);
    gtk_box_append(GTK_BOX(left_vbox), video_area_);
    gtk_frame_set_child(GTK_FRAME(left_frame_), left_vbox);

    // Add to main paned panel
    gtk_paned_set_start_child(GTK_PANED(main_paned_), left_frame_);
}

void EnhancedVideoCaptureWindow::create_source_list() {
    // Create right frame
    right_frame_ = gtk_frame_new("Source Selection");
    gtk_widget_set_margin_start(right_frame_, 5);
    gtk_widget_set_margin_end(right_frame_, 10);
    gtk_widget_set_margin_top(right_frame_, 10);
    gtk_widget_set_margin_bottom(right_frame_, 10);

    // Create right content container
    GtkWidget* right_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(right_vbox, 10);
    gtk_widget_set_margin_end(right_vbox, 10);
    gtk_widget_set_margin_top(right_vbox, 10);
    gtk_widget_set_margin_bottom(right_vbox, 10);

    // Create mode label
    mode_label_ = gtk_label_new("Desktop Capture Mode");
    gtk_widget_set_halign(mode_label_, GTK_ALIGN_START);
    gtk_widget_add_css_class(mode_label_, "mode-label");

    // Create refresh button
    refresh_button_ = gtk_button_new_with_label("Refresh");
    gtk_widget_set_halign(refresh_button_, GTK_ALIGN_END);
    g_signal_connect(refresh_button_, "clicked", G_CALLBACK(on_refresh_button_clicked), this);

    // Create top horizontal container (mode label and refresh button)
    GtkWidget* top_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_box_append(GTK_BOX(top_hbox), mode_label_);
    gtk_box_append(GTK_BOX(top_hbox), refresh_button_);
    gtk_widget_set_hexpand(mode_label_, TRUE);

    // Create source list
    source_list_ = gtk_list_box_new();
    gtk_widget_add_css_class(source_list_, "source-list");
    g_signal_connect(source_list_, "row-selected", G_CALLBACK(on_source_selection_changed), this);

    // Create scroll container
    source_scrolled_ = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(source_scrolled_),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(source_scrolled_), source_list_);
    gtk_widget_set_vexpand(source_scrolled_, TRUE);

    // Add to right container
    gtk_box_append(GTK_BOX(right_vbox), top_hbox);
    gtk_box_append(GTK_BOX(right_vbox), source_scrolled_);
    gtk_frame_set_child(GTK_FRAME(right_frame_), right_vbox);

    // Add to main paned panel
    gtk_paned_set_end_child(GTK_PANED(main_paned_), right_frame_);
}

void EnhancedVideoCaptureWindow::update_source_list() {
    // Check if source_list_ is initialized
    if (!source_list_ || !GTK_IS_LIST_BOX(source_list_)) {
        std::cerr << "Error: source_list_ is not properly initialized" << std::endl;
        return;
    }
    
    // Clear existing list
    GtkWidget* child = gtk_widget_get_first_child(source_list_);
    while (child) {
        GtkWidget* next = gtk_widget_get_next_sibling(child);
        gtk_list_box_remove(GTK_LIST_BOX(source_list_), child);
        child = next;
    }

    if (current_mode_ == CaptureMode::DESKTOP) {
        refresh_window_list();
        
        // Add window list items
        for (const auto& window_info : available_windows_) {
            GtkWidget* row = gtk_list_box_row_new();
            GtkWidget* hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
            gtk_widget_set_margin_start(hbox, 10);
            gtk_widget_set_margin_end(hbox, 10);
            gtk_widget_set_margin_top(hbox, 5);
            gtk_widget_set_margin_bottom(hbox, 5);

            // Icon
            const char* icon = window_info.is_desktop ? "Desktop" : "Window";
            GtkWidget* icon_label = gtk_label_new(icon);
            gtk_widget_set_size_request(icon_label, 30, -1);

            // Window info
            GtkWidget* info_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
            GtkWidget* title_label = gtk_label_new(window_info.title.c_str());
            GtkWidget* app_label = gtk_label_new(window_info.app_name.c_str());
            
            gtk_widget_set_halign(title_label, GTK_ALIGN_START);
            gtk_widget_set_halign(app_label, GTK_ALIGN_START);
            gtk_widget_add_css_class(title_label, "window-title");
            gtk_widget_add_css_class(app_label, "app-name");

            gtk_box_append(GTK_BOX(info_vbox), title_label);
            gtk_box_append(GTK_BOX(info_vbox), app_label);

            gtk_box_append(GTK_BOX(hbox), icon_label);
            gtk_box_append(GTK_BOX(hbox), info_vbox);
            gtk_widget_set_hexpand(info_vbox, TRUE);

            gtk_list_box_row_set_child(GTK_LIST_BOX_ROW(row), hbox);
            
            // Store window info to row data
            g_object_set_data(G_OBJECT(row), "window_id", GINT_TO_POINTER(window_info.window_id));
            g_object_set_data_full(G_OBJECT(row), "window_title", g_strdup(window_info.title.c_str()), g_free);
            g_object_set_data_full(G_OBJECT(row), "app_name", g_strdup(window_info.app_name.c_str()), g_free);
            g_object_set_data(G_OBJECT(row), "is_desktop", GINT_TO_POINTER(window_info.is_desktop ? 1 : 0));

            gtk_list_box_append(GTK_LIST_BOX(source_list_), row);
        }
    } else {
        refresh_device_list();
        
        // Add device list items
        for (const auto& device_info : available_devices_) {
            GtkWidget* row = gtk_list_box_row_new();
            GtkWidget* hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
            gtk_widget_set_margin_start(hbox, 10);
            gtk_widget_set_margin_end(hbox, 10);
            gtk_widget_set_margin_top(hbox, 5);
            gtk_widget_set_margin_bottom(hbox, 5);

            // Icon - choose different icon based on device type
            const char* icon = (device_info.device_index == -1) ? "N/A" : "Camera";
            GtkWidget* icon_label = gtk_label_new(icon);
            gtk_widget_set_size_request(icon_label, 30, -1);

            // Device info
            GtkWidget* info_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
            GtkWidget* name_label = gtk_label_new(device_info.name.c_str());
            GtkWidget* id_label = gtk_label_new(device_info.id.c_str());
            
            gtk_widget_set_halign(name_label, GTK_ALIGN_START);
            gtk_widget_set_halign(id_label, GTK_ALIGN_START);
            gtk_widget_add_css_class(name_label, "device-name");
            gtk_widget_add_css_class(id_label, "device-id");

            gtk_box_append(GTK_BOX(info_vbox), name_label);
            gtk_box_append(GTK_BOX(info_vbox), id_label);

            gtk_box_append(GTK_BOX(hbox), icon_label);
            gtk_box_append(GTK_BOX(hbox), info_vbox);
            gtk_widget_set_hexpand(info_vbox, TRUE);

            gtk_list_box_row_set_child(GTK_LIST_BOX_ROW(row), hbox);
            
            // Store device info to row data
            g_object_set_data(G_OBJECT(row), "device_index", GINT_TO_POINTER(device_info.device_index));
            g_object_set_data_full(G_OBJECT(row), "device_name", g_strdup(device_info.name.c_str()), g_free);
            g_object_set_data_full(G_OBJECT(row), "device_id", g_strdup(device_info.id.c_str()), g_free);

            gtk_list_box_append(GTK_LIST_BOX(source_list_), row);
        }
    }
}

void EnhancedVideoCaptureWindow::refresh_window_list() {
    available_windows_.clear();
    
    // Add desktop option
    WindowInfo desktop_info;
    desktop_info.title = "Entire Desktop";
    desktop_info.app_name = "System Desktop";
    desktop_info.window_id = 0;
    desktop_info.is_desktop = true;
    available_windows_.push_back(desktop_info);

#ifdef __APPLE__
    // macOS get window list
    CFArrayRef window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID);
    if (window_list) {
        CFIndex count = CFArrayGetCount(window_list);
        for (CFIndex i = 0; i < count; i++) {
            CFDictionaryRef window_info = (CFDictionaryRef)CFArrayGetValueAtIndex(window_list, i);
            
            // Get window title
            CFStringRef title_ref = (CFStringRef)CFDictionaryGetValue(window_info, kCGWindowName);
            CFStringRef owner_ref = (CFStringRef)CFDictionaryGetValue(window_info, kCGWindowOwnerName);
            CFNumberRef window_id_ref = (CFNumberRef)CFDictionaryGetValue(window_info, kCGWindowNumber);
            
            if (title_ref && owner_ref && window_id_ref) {
                char title[256] = {0};
                char owner[256] = {0};
                int window_id = 0;
                
                CFStringGetCString(title_ref, title, sizeof(title), kCFStringEncodingUTF8);
                CFStringGetCString(owner_ref, owner, sizeof(owner), kCFStringEncodingUTF8);
                CFNumberGetValue(window_id_ref, kCFNumberIntType, &window_id);
                
                // Filter out empty titles and system windows
                if (strlen(title) > 0 && strcmp(owner, "Window Server") != 0) {
                    WindowInfo info;
                    info.title = title;
                    info.app_name = owner;
                    info.window_id = window_id;
                    info.is_desktop = false;
                    available_windows_.push_back(info);
                }
            }
        }
        CFRelease(window_list);
    }
#else
    // Implementation for other platforms can be added here
    WindowInfo example_window;
    example_window.title = "Example Window";
    example_window.app_name = "Example App";
    example_window.window_id = 1;
    example_window.is_desktop = false;
    available_windows_.push_back(example_window);
#endif
}

void EnhancedVideoCaptureWindow::refresh_device_list() {
    available_devices_.clear();
    
    // Add disable camera option
    DeviceInfo disable_info;
    disable_info.name = "Disable Camera";
    disable_info.id = "disable_camera";
    disable_info.device_index = -1;  // Use -1 to indicate disabled
    available_devices_.push_back(disable_info);
    
    // Get camera device list
    auto camera_devices = media::VideoCapture::get_camera_devices();
    for (size_t i = 0; i < camera_devices.size(); i++) {
        DeviceInfo info;
        info.name = camera_devices[i];
        info.id = "camera_" + std::to_string(i);
        info.device_index = static_cast<int>(i);
        available_devices_.push_back(info);
    }
    
    // If no camera found, add default item
    if (camera_devices.empty()) {
        DeviceInfo default_info;
        default_info.name = "Default Camera";
        default_info.id = "camera_0";
        default_info.device_index = 0;
        available_devices_.push_back(default_info);
    }
}

void EnhancedVideoCaptureWindow::setup_styling() {
    const char* css_data = 
        ".info-label { "
        "  font-size: 12px; "
        "  color: #666; "
        "  margin-bottom: 5px; "
        "} "
        ".mode-label { "
        "  font-size: 14px; "
        "  font-weight: bold; "
        "  color: #333; "
        "} "
        ".source-list { "
        "  background: #f8f9fa; "
        "  border: 1px solid #dee2e6; "
        "  border-radius: 6px; "
        "} "
        ".source-list row { "
        "  border-bottom: 1px solid #e9ecef; "
        "} "
        ".source-list row:hover { "
        "  background: #e3f2fd; "
        "} "
        ".source-list row:selected { "
        "  background: #2196f3; "
        "  color: white; "
        "} "
        ".window-title { "
        "  font-size: 13px; "
        "  font-weight: bold; "
        "} "
        ".app-name, .device-id { "
        "  font-size: 11px; "
        "  color: #666; "
        "} "
        ".device-name { "
        "  font-size: 13px; "
        "  font-weight: bold; "
        "} "
        "frame { "
        "  border: 1px solid #dee2e6; "
        "  border-radius: 8px; "
        "} "
        "frame > label { "
        "  font-weight: bold; "
        "  color: #495057; "
        "} ";

    GtkCssProvider* css_provider = gtk_css_provider_new();
    gtk_css_provider_load_from_string(css_provider, css_data);
    
    GtkStyleContext* style_context = gtk_widget_get_style_context(window_);
    gtk_style_context_add_provider(style_context, 
                                   GTK_STYLE_PROVIDER(css_provider), 
                                   GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    
    g_object_unref(css_provider);
}

// Static callback function implementations
void EnhancedVideoCaptureWindow::on_draw_area(GtkDrawingArea* area, cairo_t* cr, int width, int height, gpointer user_data) {
    EnhancedVideoCaptureWindow* window = static_cast<EnhancedVideoCaptureWindow*>(user_data);

    // Set background color
    cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
    cairo_paint(cr);

    // If cached surface exists, draw video frame
    if (window->cached_surface_) {
        // Calculate scale ratio to fit display area
        double scale_x = (double)width / window->cached_width_;
        double scale_y = (double)height / window->cached_height_;
    double scale = (std::min)(scale_x, scale_y);

        // Calculate center position
        double scaled_width = window->cached_width_ * scale;
        double scaled_height = window->cached_height_ * scale;
        double x = (width - scaled_width) / 2;
        double y = (height - scaled_height) / 2;

        // Apply transformation
        cairo_save(cr);
        cairo_translate(cr, x, y);
        cairo_scale(cr, scale, scale);

        // Draw video frame
        cairo_set_source_surface(cr, window->cached_surface_, 0, 0);
        cairo_paint(cr);

        cairo_restore(cr);
    } else {
        // Show waiting text
        cairo_set_source_rgb(cr, 0.7, 0.7, 0.7);
        cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
        cairo_set_font_size(cr, 16);

        const char* text = "Waiting for video data...";
        cairo_text_extents_t extents;
        cairo_text_extents(cr, text, &extents);

        double x = (width - extents.width) / 2;
        double y = (height + extents.height) / 2;

        cairo_move_to(cr, x, y);
        cairo_show_text(cr, text);
    }
}

gboolean EnhancedVideoCaptureWindow::on_window_close(GtkWidget* widget, gpointer user_data) {
    EnhancedVideoCaptureWindow* window = static_cast<EnhancedVideoCaptureWindow*>(user_data);
    if (window->close_callback_) {
        window->close_callback_();
    }
    // Hide window instead of destroying it for reuse
    window->hide();
    return TRUE; // Prevent default destroy behavior
}

void EnhancedVideoCaptureWindow::on_source_selection_changed(GtkListBox* list_box, GtkListBoxRow* row, gpointer user_data) {
    if (!row) return;
    
    EnhancedVideoCaptureWindow* window = static_cast<EnhancedVideoCaptureWindow*>(user_data);
    
    if (window->current_mode_ == CaptureMode::DESKTOP) {
        // Handle window selection
        int window_id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(row), "window_id"));
        const char* title = (const char*)g_object_get_data(G_OBJECT(row), "window_title");
        const char* app_name = (const char*)g_object_get_data(G_OBJECT(row), "app_name");
        gboolean is_desktop = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(row), "is_desktop"));
        
        if (title && app_name && window->window_selection_callback_) {
            WindowInfo info;
            info.window_id = window_id;
            info.title = title;
            info.app_name = app_name;
            info.is_desktop = (is_desktop != 0);
            window->window_selection_callback_(info);
        }
    } else {
        // Handle device selection
        int device_index = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(row), "device_index"));
        const char* name = (const char*)g_object_get_data(G_OBJECT(row), "device_name");
        const char* id = (const char*)g_object_get_data(G_OBJECT(row), "device_id");
        
        if (name && id && window->device_selection_callback_) {
            DeviceInfo info;
            info.device_index = device_index;
            info.name = name;
            info.id = id;
            window->device_selection_callback_(info);
        }
    }
}

void EnhancedVideoCaptureWindow::on_refresh_button_clicked(GtkWidget* widget, gpointer user_data) {
    EnhancedVideoCaptureWindow* window = static_cast<EnhancedVideoCaptureWindow*>(user_data);
    window->update_source_list();
}

} // namespace gui
} // namespace duorou