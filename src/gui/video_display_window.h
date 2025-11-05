#ifndef DUOROU_GUI_VIDEO_DISPLAY_WINDOW_H
#define DUOROU_GUI_VIDEO_DISPLAY_WINDOW_H

#include <gtk/gtk.h>
#include <memory>
#include <functional>
#include "../media/video_capture.h"

namespace duorou {
namespace gui {

class VideoDisplayWindow {
public:
    VideoDisplayWindow();
    ~VideoDisplayWindow();
    
    // Show window
    void show();
    
    // Hide window
    void hide();
    
    // Update video frame
    void update_frame(const media::VideoFrame& frame);
    
    // Check if window is visible
    bool is_visible() const;
    
    // Set window close callback
    void set_close_callback(std::function<void()> callback);
    
private:
    GtkWidget* window_;
    GtkWidget* video_area_;
    GtkWidget* info_label_;
    
    // Video frame data
    std::unique_ptr<guchar[]> frame_data_;
    int frame_width_;
    int frame_height_;
    int frame_channels_;
    
    // Cache Cairo surface to avoid repeated creation
    cairo_surface_t* cached_surface_;
    std::unique_ptr<guchar[]> cached_rgba_data_;
    int cached_width_;
    int cached_height_;
    
    // Window close callback
    std::function<void()> close_callback_;
    
    // Initialize UI
    void init_ui();
    
    // Draw callback
    static void on_draw_area(GtkDrawingArea* area, cairo_t* cr, int width, int height, gpointer user_data);
    
    // Window close callback
    static gboolean on_window_close(GtkWidget* widget, gpointer user_data);
};

} // namespace gui
} // namespace duorou

#endif // DUOROU_GUI_VIDEO_DISPLAY_WINDOW_H