#ifndef DUOROU_GUI_ENHANCED_VIDEO_CAPTURE_WINDOW_H
#define DUOROU_GUI_ENHANCED_VIDEO_CAPTURE_WINDOW_H

#include <gtk/gtk.h>
#include <memory>
#include <functional>
#include "../media/video_capture.h"
#include "../media/video_frame.h"

#ifdef __cplusplus
#include <vector>
#include <string>
#endif

namespace duorou {
namespace gui {

/**
 * Enhanced video capture window
 * Implements layout with left video display area and right window/device selection list
 */
class EnhancedVideoCaptureWindow {
public:
    enum class CaptureMode {
        DESKTOP,    // Desktop capture mode
        CAMERA      // Camera mode
    };

    struct WindowInfo {
        std::string title;      // Window title
        std::string app_name;   // Application name
        int window_id;          // Window ID
        bool is_desktop;        // Whether it's desktop
    };

    struct DeviceInfo {
        std::string name;       // Device name
        std::string id;         // Device ID
        int device_index;       // Device index
    };

    EnhancedVideoCaptureWindow();
    ~EnhancedVideoCaptureWindow();

    // Disable copy constructor and assignment
    EnhancedVideoCaptureWindow(const EnhancedVideoCaptureWindow&) = delete;
    EnhancedVideoCaptureWindow& operator=(const EnhancedVideoCaptureWindow&) = delete;

    /**
     * Initialize window
     * @return Returns true on success, false on failure
     */
    bool initialize();

    /**
     * Show window
     * @param mode Capture mode
     */
    void show(CaptureMode mode);

    /**
     * Hide window
     */
    void hide();

    /**
     * Check if window is visible
     */
    bool is_visible() const;

    /**
     * Update video frame
     * @param frame Video frame data
     */
    void update_frame(const media::VideoFrame& frame);

    /**
     * Set window close callback
     * @param callback Close callback function
     */
    void set_close_callback(std::function<void()> callback);

    /**
     * Set window selection callback
     * @param callback Window selection callback function
     */
    void set_window_selection_callback(std::function<void(const WindowInfo&)> callback);

    /**
     * Set device selection callback
     * @param callback Device selection callback function
     */
    void set_device_selection_callback(std::function<void(const DeviceInfo&)> callback);

private:
    // GTK components
    GtkWidget* window_;                 // Main window
    GtkWidget* main_paned_;             // Main paned panel
    GtkWidget* left_frame_;             // Left frame
    GtkWidget* right_frame_;            // Right frame
    GtkWidget* video_area_;             // Video display area
    GtkWidget* info_label_;             // Info label
    GtkWidget* source_list_;            // Source list (windows or devices)
    GtkWidget* source_scrolled_;        // Source list scroll container
    GtkWidget* mode_label_;             // Mode label
    GtkWidget* refresh_button_;         // Refresh button

    // State variables
    CaptureMode current_mode_;
    std::vector<WindowInfo> available_windows_;
    std::vector<DeviceInfo> available_devices_;

    // Video frame data
    std::unique_ptr<guchar[]> frame_data_;
    int frame_width_;
    int frame_height_;
    int frame_channels_;

    // Cached Cairo surface
    cairo_surface_t* cached_surface_;
    std::unique_ptr<guchar[]> cached_rgba_data_;
    int cached_width_;
    int cached_height_;

    // Callback functions
    std::function<void()> close_callback_;
    std::function<void(const WindowInfo&)> window_selection_callback_;
    std::function<void(const DeviceInfo&)> device_selection_callback_;

    /**
     * Initialize UI components
     */
    void init_ui();

    /**
     * Create left video display area
     */
    void create_video_area();

    /**
     * Create right source selection list
     */
    void create_source_list();

    /**
     * Update source list content
     */
    void update_source_list();

    /**
     * Get available window list
     */
    void refresh_window_list();

    /**
     * Get available device list
     */
    void refresh_device_list();

    /**
     * Set window styling
     */
    void setup_styling();

    // Static callback functions
    static void on_draw_area(GtkDrawingArea* area, cairo_t* cr, int width, int height, gpointer user_data);
    static gboolean on_window_close(GtkWidget* widget, gpointer user_data);
    static void on_source_selection_changed(GtkListBox* list_box, GtkListBoxRow* row, gpointer user_data);
    static void on_refresh_button_clicked(GtkWidget* widget, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_ENHANCED_VIDEO_CAPTURE_WINDOW_H